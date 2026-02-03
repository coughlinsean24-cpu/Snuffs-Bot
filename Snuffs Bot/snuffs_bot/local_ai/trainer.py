"""
Model Trainer

Handles model training, retraining schedules, and performance evaluation.
Manages the learning loop for continuous improvement.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .data_collector import DataCollector
from .feature_engineer import FeatureEngineer
from .trading_model import LocalTradingModel


class ModelTrainer:
    """
    Manages the training lifecycle for the local AI trading model.
    
    Responsibilities:
    1. Collect and prepare training data from trade history
    2. Trigger retraining when sufficient new data is available
    3. Evaluate model performance over time
    4. Manage model versioning
    """
    
    # Training thresholds - STATISTICALLY SOUND LEARNING
    # With 68 features, need at least 100 samples to avoid severe overfitting
    # Research: "15 trades with 68 features is statistically indefensible"
    MIN_TRADES_FOR_TRAINING = 100  # Minimum for valid ML with 68 features
    RETRAIN_EVERY_N_TRADES = 10  # Retrain more frequently
    MIN_HOURS_BETWEEN_RETRAIN = 0.5  # Retrain every 30 min if needed
    
    # Performance tracking for self-improvement
    TARGET_WIN_RATE = 0.55  # Aim for 55% win rate
    PERFORMANCE_WINDOW = 20  # Look at last 20 trades
    
    def __init__(
        self,
        data_collector: DataCollector,
        model: LocalTradingModel,
    ):
        """
        Initialize the trainer
        
        Args:
            data_collector: DataCollector instance for historical data
            model: LocalTradingModel instance to train
        """
        self.data_collector = data_collector
        self.model = model
        self.feature_engineer = FeatureEngineer()
        
        # Training state
        self.last_training_trade_count: int = 0
        self.training_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.prediction_log: List[Dict[str, Any]] = []
        
        # Self-improvement tracking
        self.recent_win_rate: float = 0.0
        self.confidence_adjustment: float = 0.0  # +/- adjustment to base confidence
        self.learned_patterns: Dict[str, Any] = {}
        
        # Load any saved performance data
        self._load_performance_state()
        
        logger.info("ModelTrainer initialized - AGGRESSIVE LEARNING MODE ACTIVE")
    
    def should_retrain(self) -> bool:
        """
        Check if the model should be retrained
        
        Returns:
            True if retraining is needed
        """
        current_trade_count = self.data_collector.get_trade_count()
        
        # Not enough data yet
        if current_trade_count < self.MIN_TRADES_FOR_TRAINING:
            logger.debug(f"Not enough trades for training: {current_trade_count}/{self.MIN_TRADES_FOR_TRAINING}")
            return False
        
        # Check if enough new trades since last training
        new_trades = current_trade_count - self.last_training_trade_count
        if new_trades < self.RETRAIN_EVERY_N_TRADES:
            return False
        
        # Check time since last training
        if self.model.last_training_time:
            hours_since = (datetime.now() - self.model.last_training_time).total_seconds() / 3600
            if hours_since < self.MIN_HOURS_BETWEEN_RETRAIN:
                logger.debug(f"Too soon to retrain: {hours_since:.1f}h < {self.MIN_HOURS_BETWEEN_RETRAIN}h")
                return False
        
        logger.info(f"Retraining triggered: {new_trades} new trades since last training")
        return True
    
    def train_model(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Train or retrain the model
        
        Args:
            force: Force training even if thresholds not met
            
        Returns:
            Training metrics or None if training was skipped
        """
        if not force and not self.should_retrain():
            return None
        
        # Get training data
        trade_records = self.data_collector.get_training_data()
        
        if len(trade_records) < self.MIN_TRADES_FOR_TRAINING:
            logger.warning(f"Insufficient trades for training: {len(trade_records)}")
            return None
        
        logger.info(f"Preparing training data from {len(trade_records)} trades")
        
        # Create features and labels
        X, y = self.feature_engineer.create_training_features(trade_records)
        
        # Train the model
        metrics = self.model.train(X, y)
        
        if 'error' not in metrics:
            self.last_training_trade_count = len(trade_records)
            
            # Log training event
            training_event = {
                'timestamp': datetime.now().isoformat(),
                'n_trades': len(trade_records),
                'metrics': metrics,
            }
            self.training_history.append(training_event)
            
            # Save training history
            self._save_training_history()
        
        return metrics
    
    def log_prediction(
        self,
        decision: Any,  # TradingDecision
        snapshot_dict: Dict[str, Any],
        actual_outcome: Optional[bool] = None,
    ) -> None:
        """
        Log a prediction for later analysis
        
        Args:
            decision: The TradingDecision made
            snapshot_dict: Market conditions at decision time
            actual_outcome: True if trade was profitable (set later)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': decision.action,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'spy_price': snapshot_dict.get('spy_price', 0),
            'spy_change_5m': snapshot_dict.get('spy_change_5m', 0),
            'vix': snapshot_dict.get('vix', 0),
            'time_decay_factor': snapshot_dict.get('time_decay_factor', 0),
            'actual_outcome': actual_outcome,
            'inference_time_ms': decision.inference_time_ms,
        }
        
        self.prediction_log.append(log_entry)
        
        # Keep only last 1000 predictions
        if len(self.prediction_log) > 1000:
            self.prediction_log = self.prediction_log[-1000:]
    
    def calculate_accuracy(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Calculate prediction accuracy over a time window
        
        Args:
            window_hours: Number of hours to look back
            
        Returns:
            Accuracy metrics
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        
        recent_predictions = [
            p for p in self.prediction_log
            if p.get('actual_outcome') is not None
            and datetime.fromisoformat(p['timestamp']) > cutoff
        ]
        
        if not recent_predictions:
            return {'accuracy': None, 'n_predictions': 0}
        
        correct = sum(1 for p in recent_predictions if p['actual_outcome'] == (p['action'] != 'HOLD'))
        
        return {
            'accuracy': correct / len(recent_predictions),
            'n_predictions': len(recent_predictions),
            'window_hours': window_hours,
        }
    
    def analyze_by_time_of_day(self) -> Dict[str, Any]:
        """
        Analyze model performance by time of day
        
        Returns:
            Performance breakdown by hour
        """
        trade_records = self.data_collector.get_training_data()
        
        if not trade_records:
            return {}
        
        # Group by hour
        hourly_stats = {}
        for trade in trade_records:
            hour = trade.get('entry_hour', 12)
            if hour not in hourly_stats:
                hourly_stats[hour] = {'profitable': 0, 'total': 0, 'avg_pnl': []}
            
            hourly_stats[hour]['total'] += 1
            if trade.get('was_profitable'):
                hourly_stats[hour]['profitable'] += 1
            hourly_stats[hour]['avg_pnl'].append(trade.get('pnl_percent', 0))
        
        # Calculate averages
        analysis = {}
        for hour, stats in hourly_stats.items():
            analysis[hour] = {
                'win_rate': stats['profitable'] / stats['total'] if stats['total'] > 0 else 0,
                'total_trades': stats['total'],
                'avg_pnl_percent': np.mean(stats['avg_pnl']) if stats['avg_pnl'] else 0,
            }
        
        return analysis
    
    def analyze_by_strategy(self) -> Dict[str, Any]:
        """
        Analyze model performance by strategy type
        
        Returns:
            Performance breakdown by strategy
        """
        trade_records = self.data_collector.get_training_data()
        
        if not trade_records:
            return {}
        
        strategy_stats = {}
        for trade in trade_records:
            strategy = trade.get('strategy', 'UNKNOWN')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'profitable': 0, 'total': 0, 'pnls': []}
            
            strategy_stats[strategy]['total'] += 1
            if trade.get('was_profitable'):
                strategy_stats[strategy]['profitable'] += 1
            strategy_stats[strategy]['pnls'].append(trade.get('pnl_percent', 0))
        
        analysis = {}
        for strategy, stats in strategy_stats.items():
            analysis[strategy] = {
                'win_rate': stats['profitable'] / stats['total'] if stats['total'] > 0 else 0,
                'total_trades': stats['total'],
                'avg_pnl_percent': np.mean(stats['pnls']) if stats['pnls'] else 0,
                'best_trade': max(stats['pnls']) if stats['pnls'] else 0,
                'worst_trade': min(stats['pnls']) if stats['pnls'] else 0,
            }
        
        return analysis
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights about what the model has learned
        
        Returns:
            Dict with learning insights
        """
        insights = {
            'model_stats': self.model.get_model_stats(),
            'data_stats': {
                'total_snapshots': self.data_collector.get_snapshot_count(),
                'total_trades': self.data_collector.get_trade_count(),
            },
            'training_history_count': len(self.training_history),
        }
        
        # Add time-of-day analysis
        time_analysis = self.analyze_by_time_of_day()
        if time_analysis:
            # Find best and worst hours
            best_hour = max(time_analysis.items(), key=lambda x: x[1]['win_rate'])[0]
            worst_hour = min(time_analysis.items(), key=lambda x: x[1]['win_rate'])[0]
            
            insights['best_trading_hour'] = {
                'hour': best_hour,
                'win_rate': time_analysis[best_hour]['win_rate'],
            }
            insights['worst_trading_hour'] = {
                'hour': worst_hour,
                'win_rate': time_analysis[worst_hour]['win_rate'],
            }
        
        # Add strategy analysis
        strategy_analysis = self.analyze_by_strategy()
        if strategy_analysis:
            best_strategy = max(strategy_analysis.items(), key=lambda x: x[1]['avg_pnl_percent'])
            insights['best_strategy'] = {
                'name': best_strategy[0],
                'avg_pnl': best_strategy[1]['avg_pnl_percent'],
            }
        
        return insights
    
    def _save_training_history(self):
        """Save training history to disk"""
        history_path = Path("data/local_ai/training_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, tuple):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item'):  # Generic numpy scalar
                return obj.item()
            return obj
        
        try:
            with open(history_path, 'w') as f:
                json.dump(convert_types(self.training_history), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
    
    def get_status_report(self) -> str:
        """
        Get a human-readable status report
        
        Returns:
            Status report string
        """
        stats = self.model.get_model_stats()
        data_stats = {
            'snapshots': self.data_collector.get_snapshot_count(),
            'trades': self.data_collector.get_trade_count(),
        }
        
        report_lines = [
            "=== Local AI Status ===",
            f"Model trained: {stats['entry_model_trained']}",
            f"Using XGBoost: {stats['using_xgboost']}",
            f"Training samples: {stats['training_samples']}",
            f"Model accuracy: {stats['model_accuracy']:.1%}" if stats['model_accuracy'] else "Model accuracy: N/A",
            f"",
            f"Data collected:",
            f"  - Market snapshots: {data_stats['snapshots']:,}",
            f"  - Completed trades: {data_stats['trades']}",
            f"",
            f"Training thresholds:",
            f"  - Min trades for training: {self.MIN_TRADES_FOR_TRAINING}",
            f"  - Retrain every N trades: {self.RETRAIN_EVERY_N_TRADES}",
            f"  - Trades since last training: {data_stats['trades'] - self.last_training_trade_count}",
        ]
        
        if stats['last_training_time']:
            report_lines.append(f"Last training: {stats['last_training_time']}")
        
        # Add self-improvement stats
        report_lines.append(f"")
        report_lines.append(f"Self-Improvement:")
        report_lines.append(f"  - Recent win rate: {self.recent_win_rate:.1%}")
        report_lines.append(f"  - Target win rate: {self.TARGET_WIN_RATE:.1%}")
        report_lines.append(f"  - Confidence adjustment: {self.confidence_adjustment:+.2f}")
        
        # Add simulation stats
        sim_stats = self.get_simulation_stats()
        if sim_stats.get('total_simulations', 0) > 0:
            report_lines.append(f"")
            report_lines.append(f"Background Simulations:")
            report_lines.append(f"  - Total simulations: {sim_stats['total_simulations']}")
            report_lines.append(f"  - Simulation win rate: {sim_stats['simulation_win_rate']:.1%}")
        
        return "\n".join(report_lines)

    def _load_performance_state(self) -> None:
        """Load saved performance state for continuity"""
        state_path = Path("data/local_ai/performance_state.json")
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.recent_win_rate = state.get('recent_win_rate', 0.0)
                self.confidence_adjustment = state.get('confidence_adjustment', 0.0)
                self.learned_patterns = state.get('learned_patterns', {})
                logger.info(f"Loaded performance state: win_rate={self.recent_win_rate:.1%}")
            except Exception as e:
                logger.debug(f"Could not load performance state: {e}")

    def _save_performance_state(self) -> None:
        """Save performance state for continuity"""
        state_path = Path("data/local_ai/performance_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'recent_win_rate': self.recent_win_rate,
            'confidence_adjustment': self.confidence_adjustment,
            'learned_patterns': self.learned_patterns,
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def update_self_improvement(self) -> Dict[str, Any]:
        """
        Analyze recent performance and adjust strategy.
        This is the "desire to do better" mechanism.
        
        Returns:
            Dict with improvement actions taken
        """
        actions_taken = []
        
        # Get recent trades
        trades = self.data_collector.get_training_data()
        if len(trades) < 5:
            return {'actions': [], 'message': 'Not enough trades for analysis'}
        
        # Calculate recent win rate (last N trades)
        recent = trades[-self.PERFORMANCE_WINDOW:]
        wins = sum(1 for t in recent if t.get('was_profitable'))
        self.recent_win_rate = wins / len(recent)
        
        # Compare to target
        performance_gap = self.recent_win_rate - self.TARGET_WIN_RATE
        
        # Adjust confidence threshold based on performance
        if performance_gap < -0.1:  # More than 10% below target
            # Being too aggressive - increase selectivity
            self.confidence_adjustment = min(0.15, self.confidence_adjustment + 0.02)
            actions_taken.append(f"Increased selectivity (conf adj: {self.confidence_adjustment:+.2f})")
            logger.warning(f"Performance below target ({self.recent_win_rate:.1%}). Increasing selectivity.")
        elif performance_gap > 0.1:  # More than 10% above target
            # Doing well - can be slightly more aggressive
            self.confidence_adjustment = max(-0.1, self.confidence_adjustment - 0.01)
            actions_taken.append(f"Decreased selectivity (conf adj: {self.confidence_adjustment:+.2f})")
            logger.info(f"Performance above target ({self.recent_win_rate:.1%}). Can be more aggressive.")
        
        # Analyze what's working and what's not
        # LEARNING MODE: Require many more trades before blocking anything
        # Need at least 50 trades per hour/strategy before making avoid decisions
        MIN_TRADES_FOR_PATTERN = 50  # Was 3 - need much more data before blocking
        
        time_analysis = self.analyze_by_time_of_day()
        if time_analysis:
            # Find hours with good/bad performance - but require lots of data
            good_hours = [h for h, s in time_analysis.items() if s['win_rate'] > 0.6 and s['total_trades'] >= MIN_TRADES_FOR_PATTERN]
            bad_hours = [h for h, s in time_analysis.items() if s['win_rate'] < 0.3 and s['total_trades'] >= MIN_TRADES_FOR_PATTERN]
            
            if good_hours:
                self.learned_patterns['preferred_hours'] = good_hours
                actions_taken.append(f"Identified good hours: {good_hours}")
            if bad_hours:
                self.learned_patterns['avoid_hours'] = bad_hours
                actions_taken.append(f"Identified hours to avoid: {bad_hours}")
        
        # Analyze by strategy
        strategy_analysis = self.analyze_by_strategy()
        if strategy_analysis:
            good_strategies = [s for s, data in strategy_analysis.items() if data['win_rate'] > 0.55 and data['total_trades'] >= MIN_TRADES_FOR_PATTERN]
            bad_strategies = [s for s, data in strategy_analysis.items() if data['win_rate'] < 0.3 and data['total_trades'] >= MIN_TRADES_FOR_PATTERN]
            
            if good_strategies:
                self.learned_patterns['preferred_strategies'] = good_strategies
            if bad_strategies:
                self.learned_patterns['avoid_strategies'] = bad_strategies
                actions_taken.append(f"Avoiding strategies: {bad_strategies}")
        
        # Save state
        self._save_performance_state()
        
        return {
            'recent_win_rate': self.recent_win_rate,
            'target_win_rate': self.TARGET_WIN_RATE,
            'performance_gap': performance_gap,
            'confidence_adjustment': self.confidence_adjustment,
            'learned_patterns': self.learned_patterns,
            'actions': actions_taken,
        }

    def get_confidence_adjustment(self) -> float:
        """Get the current confidence threshold adjustment"""
        return self.confidence_adjustment

    def should_avoid_trade(self, hour: int, strategy: str) -> Tuple[bool, str]:
        """
        Check if the AI has learned to avoid this trade setup.
        
        Args:
            hour: Hour of day (0-23)
            strategy: Strategy name
            
        Returns:
            Tuple of (should_avoid, reason)
        """
        avoid_hours = self.learned_patterns.get('avoid_hours', [])
        avoid_strategies = self.learned_patterns.get('avoid_strategies', [])
        
        if hour in avoid_hours:
            return True, f"Hour {hour} has historically poor performance"
        
        if strategy in avoid_strategies:
            return True, f"Strategy {strategy} has poor win rate"
        
        return False, ""

    def get_simulated_trades(self) -> List[Dict[str, Any]]:
        """
        Get simulated trades from the background learner for training.
        These are "what would have happened" trades that accelerate learning.
        
        Returns:
            List of simulated trade records
        """
        import sqlite3
        
        try:
            db_path = Path("data/local_ai/market_data.db")
            if not db_path.exists():
                return []
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='simulated_trades'
            """)
            if not cursor.fetchone():
                return []
            
            cursor.execute("""
                SELECT prediction_time, action, confidence, spy_entry, spy_exit,
                       price_change_pct, was_profitable, hold_minutes
                FROM simulated_trades
                ORDER BY prediction_time DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to training format
            simulated = []
            for row in rows:
                simulated.append({
                    'timestamp': row[0],
                    'strategy': row[1],  # LONG_CALL or LONG_PUT
                    'confidence': row[2],
                    'spy_entry': row[3],
                    'spy_exit': row[4],
                    'pnl_percent': row[5],  # Using price change as rough P/L proxy
                    'was_profitable': bool(row[6]),
                    'hold_time_minutes': row[7],
                    'is_simulated': True,
                })
            
            return simulated
            
        except Exception as e:
            logger.debug(f"Could not get simulated trades: {e}")
            return []

    def get_combined_training_data(self) -> List[Dict[str, Any]]:
        """
        Get both real trades AND simulated trades for training.
        Simulated trades are weighted less than real trades.
        
        Returns:
            Combined list of training records
        """
        # Get real trades
        real_trades = self.data_collector.get_training_data()
        
        # Get simulated trades
        sim_trades = self.get_simulated_trades()
        
        # Mark real trades
        for trade in real_trades:
            trade['is_simulated'] = False
            trade['training_weight'] = 1.0  # Full weight for real trades
        
        # Simulated trades get partial weight (they're approximations)
        for trade in sim_trades:
            trade['training_weight'] = 0.5  # Half weight for simulations
        
        combined = real_trades + sim_trades
        
        logger.info(
            f"Training data: {len(real_trades)} real trades + "
            f"{len(sim_trades)} simulated = {len(combined)} total"
        )
        
        return combined

    def train_with_simulations(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Train using both real and simulated data.
        Falls back to regular training if no simulations available.
        
        Returns:
            Training metrics or None
        """
        # Get combined data
        combined = self.get_combined_training_data()
        
        # Count real trades for threshold check
        real_count = sum(1 for t in combined if not t.get('is_simulated'))
        sim_count = len(combined) - real_count
        
        # Need either: enough real trades, OR real + simulated >= threshold
        min_for_training = self.MIN_TRADES_FOR_TRAINING
        effective_count = real_count + (sim_count * 0.5)  # Sims count half
        
        if not force and effective_count < min_for_training:
            logger.debug(f"Not enough data: {effective_count:.0f} effective < {min_for_training}")
            return None
        
        if not combined:
            return None
        
        logger.info(f"Training with {real_count} real + {sim_count} simulated trades")
        
        # Create features
        X, y = self.feature_engineer.create_training_features(combined)
        
        # Train
        metrics = self.model.train(X, y)
        
        if 'error' not in metrics:
            self.last_training_trade_count = real_count
            metrics['included_simulations'] = sim_count
            
            training_event = {
                'timestamp': datetime.now().isoformat(),
                'n_real_trades': real_count,
                'n_simulated': sim_count,
                'metrics': metrics,
            }
            self.training_history.append(training_event)
            self._save_training_history()
        
        return metrics

    def get_simulation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about simulated trades.
        
        Returns:
            Dict with simulation statistics
        """
        sim_trades = self.get_simulated_trades()
        
        if not sim_trades:
            return {'total': 0, 'message': 'No simulations yet'}
        
        wins = sum(1 for t in sim_trades if t.get('was_profitable'))
        
        # Analyze by action type
        by_action = {}
        for trade in sim_trades:
            action = trade.get('strategy', 'UNKNOWN')
            if action not in by_action:
                by_action[action] = {'wins': 0, 'total': 0}
            by_action[action]['total'] += 1
            if trade.get('was_profitable'):
                by_action[action]['wins'] += 1
        
        return {
            'total_simulations': len(sim_trades),
            'profitable_simulations': wins,
            'simulation_win_rate': wins / len(sim_trades) if sim_trades else 0,
            'by_action': {
                k: {'win_rate': v['wins']/v['total'] if v['total'] > 0 else 0, 'count': v['total']}
                for k, v in by_action.items()
            },
        }
