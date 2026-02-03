"""
Local Trading Model

XGBoost-based model for entry/exit predictions.
Replaces Claude API calls with instant local inference.

LEARNING MODE: During paper trading, the bot trades aggressively to 
gather experience. It learns from outcomes to become smarter over time.
"""

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

from loguru import logger

from .feature_engineer import FeatureEngineer, FeatureSet


@dataclass
class TradingDecision:
    """A trading decision from the local AI"""
    action: str  # "LONG_CALL", "LONG_PUT", "HOLD", "EXIT"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    # For entry decisions
    suggested_strike: Optional[float] = None
    suggested_contracts: int = 1
    
    # For exit decisions
    exit_reason: Optional[str] = None
    
    # Metadata
    model_version: str = "local_v1"
    inference_time_ms: float = 0.0


class LocalTradingModel:
    """
    XGBoost-based trading model for 0DTE options.
    
    LEARNING MODE PHILOSOPHY:
    - During paper trading, we WANT to trade often to gather data
    - Every trade (win or lose) teaches the model something
    - The model gets smarter with each outcome
    - Snapshots capture market state for "what-if" analysis
    
    Two models:
    1. Entry model: Predicts whether a trade will be profitable
    2. Exit model: Predicts optimal exit timing
    
    Falls back to rule-based decisions when insufficient training data.
    """
    
    MINIMUM_TRADES_FOR_TRAINING = 100  # Need 100+ samples for 68 features (avoid p >> n overfitting)
    MODEL_DIR = "data/local_ai/models"
    
    # LEARNING MODE SETTINGS
    # Paper trading = aggressive learning (trade more to learn faster)
    # Live trading = conservative (only high-confidence trades)
    PAPER_TRADING_CONFIDENCE = 0.45  # Lower threshold during learning
    LIVE_TRADING_CONFIDENCE = 0.65   # Higher threshold with real money
    PAPER_MOMENTUM_THRESHOLD = 0.03  # Smaller moves trigger trades in learning
    LIVE_MOMENTUM_THRESHOLD = 0.10   # Require stronger momentum for live
    
    def __init__(self, model_dir: Optional[str] = None, learning_mode: bool = True):
        """
        Initialize the trading model
        
        Args:
            model_dir: Directory to store model files
            learning_mode: If True, trade aggressively to gather data (paper trading)
        """
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed. Using rule-based decisions only.")
        
        self.model_dir = Path(model_dir or self.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer = FeatureEngineer()
        
        # Learning mode - paper trading trades aggressively
        self.learning_mode = learning_mode
        
        # Models
        self.entry_model: Optional[Any] = None
        self.exit_model: Optional[Any] = None
        
        # Model metadata
        self.entry_model_trained: bool = False
        self.exit_model_trained: bool = False
        self.last_training_time: Optional[datetime] = None
        self.training_samples: int = 0
        self.model_accuracy: float = 0.0
        
        # Trade statistics for learning
        self.trades_today: int = 0
        self.wins_today: int = 0
        self.consecutive_holds: int = 0  # Track if we're being too conservative
        
        # Load existing models if available
        self._load_models()
        
        mode_str = "LEARNING (aggressive)" if learning_mode else "PRODUCTION (conservative)"
        logger.info(f"LocalTradingModel initialized. Mode: {mode_str}. Entry model trained: {self.entry_model_trained}")
    
    def _load_models(self) -> None:
        """Load trained models from disk"""
        entry_path = self.model_dir / "entry_model.pkl"
        exit_path = self.model_dir / "exit_model.pkl"
        meta_path = self.model_dir / "model_meta.json"
        
        if entry_path.exists() and HAS_XGBOOST:
            try:
                with open(entry_path, 'rb') as f:
                    self.entry_model = pickle.load(f)
                self.entry_model_trained = True
                logger.info("Loaded entry model from disk")
            except Exception as e:
                logger.error(f"Failed to load entry model: {e}")
        
        if exit_path.exists() and HAS_XGBOOST:
            try:
                with open(exit_path, 'rb') as f:
                    self.exit_model = pickle.load(f)
                self.exit_model_trained = True
                logger.info("Loaded exit model from disk")
            except Exception as e:
                logger.error(f"Failed to load exit model: {e}")
        
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.last_training_time = datetime.fromisoformat(meta.get('last_training_time', ''))
                self.training_samples = meta.get('training_samples', 0)
                self.model_accuracy = meta.get('model_accuracy', 0.0)
            except Exception as e:
                logger.debug(f"Could not load model metadata: {e}")
    
    def save_models(self) -> None:
        """Save trained models to disk"""
        if self.entry_model is not None:
            entry_path = self.model_dir / "entry_model.pkl"
            with open(entry_path, 'wb') as f:
                pickle.dump(self.entry_model, f)
            logger.info(f"Saved entry model to {entry_path}")
        
        if self.exit_model is not None:
            exit_path = self.model_dir / "exit_model.pkl"
            with open(exit_path, 'wb') as f:
                pickle.dump(self.exit_model, f)
            logger.info(f"Saved exit model to {exit_path}")
        
        # Save metadata
        meta_path = self.model_dir / "model_meta.json"
        meta = {
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_samples': self.training_samples,
            'model_accuracy': self.model_accuracy,
            'entry_model_trained': self.entry_model_trained,
            'exit_model_trained': self.exit_model_trained,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def predict_entry(self, snapshot_dict: Dict[str, Any]) -> TradingDecision:
        """
        Predict whether to enter a trade
        
        Args:
            snapshot_dict: Current market snapshot data
            
        Returns:
            TradingDecision with action and confidence
        """
        start = datetime.now()
        
        # Extract features
        feature_set = self.feature_engineer.extract_features(snapshot_dict)
        
        # Use trained model if available
        if self.entry_model_trained and self.entry_model is not None:
            decision = self._predict_with_model(feature_set, snapshot_dict)
        else:
            # Fall back to rule-based system
            decision = self._rule_based_entry(feature_set, snapshot_dict)
        
        # Calculate inference time
        inference_ms = (datetime.now() - start).total_seconds() * 1000
        decision.inference_time_ms = inference_ms
        
        return decision
    
    def _predict_with_model(self, feature_set: FeatureSet, snapshot_dict: Dict[str, Any]) -> TradingDecision:
        """Use trained XGBoost model for prediction"""
        X = feature_set.features.reshape(1, -1)
        
        # Get probability prediction
        prob = self.entry_model.predict_proba(X)[0]
        profitable_prob = prob[1]  # Probability of profitable trade
        
        # Determine action based on momentum and probability
        spy_5m = snapshot_dict.get('spy_change_5m', 0)
        spy_15m = snapshot_dict.get('spy_change_15m', 0)
        
        # LEARNING MODE vs PRODUCTION MODE thresholds
        if self.learning_mode:
            # Paper trading - be AGGRESSIVE to gather data
            confidence_threshold = self.PAPER_TRADING_CONFIDENCE  # 45%
            momentum_threshold = self.PAPER_MOMENTUM_THRESHOLD    # 0.03
            
            # If we've been holding too long, lower threshold even more
            if self.consecutive_holds > 5:
                confidence_threshold = 0.40  # Even more aggressive
                momentum_threshold = 0.02
                logger.debug(f"Too many consecutive holds ({self.consecutive_holds}), lowering thresholds")
        else:
            # Live trading - be CONSERVATIVE
            confidence_threshold = self.LIVE_TRADING_CONFIDENCE   # 65%
            momentum_threshold = self.LIVE_MOMENTUM_THRESHOLD     # 0.10
        
        # In learning mode, we WANT to trade even marginal setups
        # Every trade teaches us something, win or lose
        
        if profitable_prob < confidence_threshold:
            self.consecutive_holds += 1
            
            # In learning mode, take action!
            if self.learning_mode:
                # Check technical signals to determine direction
                rsi = snapshot_dict.get('rsi_14', 50)
                macd_cross = snapshot_dict.get('macd_crossover', 'NONE')
                vwap_signal = snapshot_dict.get('vwap_signal', 'NEUTRAL')
                
                # Count signals
                bullish_signals = 0
                bearish_signals = 0
                
                if macd_cross == 'BULLISH': bullish_signals += 2
                if macd_cross == 'BEARISH': bearish_signals += 2
                if rsi < 35: bullish_signals += 2  # Strong oversold
                elif rsi < 45: bullish_signals += 1  # Moderate oversold
                if rsi > 65: bearish_signals += 2  # Strong overbought
                elif rsi > 55: bearish_signals += 1  # Moderate overbought
                if vwap_signal == 'ABOVE': bullish_signals += 1
                if vwap_signal == 'BELOW': bearish_signals += 1
                
                # Momentum bias
                if spy_5m > 0.01: bullish_signals += 1
                if spy_5m < -0.01: bearish_signals += 1
                
                # Take trade if we have any signal OR too many holds
                if bullish_signals >= 1 and bullish_signals > bearish_signals:
                    logger.info(f"🎯 LEARNING TRADE: LONG_CALL (RSI={rsi:.1f}, MACD={macd_cross}, bull={bullish_signals}, bear={bearish_signals})")
                    self.consecutive_holds = 0
                    return TradingDecision(
                        action="LONG_CALL",
                        confidence=profitable_prob,
                        reasoning=f"Learning mode trade: bullish signals ({bullish_signals} vs {bearish_signals}). RSI={rsi:.1f}",
                        suggested_strike=round(snapshot_dict.get('spy_price', 0)),
                        suggested_contracts=1,
                    )
                elif bearish_signals >= 1 and bearish_signals > bullish_signals:
                    logger.info(f"🎯 LEARNING TRADE: LONG_PUT (RSI={rsi:.1f}, MACD={macd_cross}, bull={bullish_signals}, bear={bearish_signals})")
                    self.consecutive_holds = 0
                    return TradingDecision(
                        action="LONG_PUT",
                        confidence=profitable_prob,
                        reasoning=f"Learning mode trade: bearish signals ({bearish_signals} vs {bullish_signals}). RSI={rsi:.1f}",
                        suggested_strike=round(snapshot_dict.get('spy_price', 0)),
                        suggested_contracts=1,
                    )
                elif self.consecutive_holds >= 10:
                    # Force a trade after too many holds
                    import random
                    action = "LONG_CALL" if random.random() > 0.5 else "LONG_PUT"
                    logger.info(f"🎯 FORCED TRADE: {action} after {self.consecutive_holds} consecutive holds")
                    self.consecutive_holds = 0
                    return TradingDecision(
                        action=action,
                        confidence=profitable_prob,
                        reasoning=f"Forced learning trade after {self.consecutive_holds} holds. Need experience data.",
                        suggested_strike=round(snapshot_dict.get('spy_price', 0)),
                        suggested_contracts=1,
                    )
                else:
                    logger.debug(f"HOLD: No clear signals (bull={bullish_signals}, bear={bearish_signals}, holds={self.consecutive_holds})")
                    return TradingDecision(
                        action="HOLD",
                        confidence=1.0 - profitable_prob,
                        reasoning=f"No clear signals (bull: {bullish_signals}, bear: {bearish_signals}). Waiting.",
                    )
            else:
                return TradingDecision(
                    action="HOLD",
                    confidence=1.0 - profitable_prob,
                    reasoning=f"Model confidence too low ({profitable_prob:.1%}). Waiting for better setup.",
                )
        
        # Determine direction based on momentum (with learning mode thresholds)
        if spy_5m > momentum_threshold and spy_15m > 0:
            action = "LONG_CALL"
            reasoning = f"Bullish momentum detected. Model confidence: {profitable_prob:.1%}"
            self.consecutive_holds = 0  # Reset
        elif spy_5m < -momentum_threshold and spy_15m < 0:
            action = "LONG_PUT"
            reasoning = f"Bearish momentum detected. Model confidence: {profitable_prob:.1%}"
            self.consecutive_holds = 0  # Reset
        else:
            # In learning mode, use technical indicators to find direction
            if self.learning_mode:
                # Check technical signals from snapshot
                macd_cross = snapshot_dict.get('macd_crossover', 'NONE')
                rsi = snapshot_dict.get('rsi_14', 50)
                vwap_signal = snapshot_dict.get('vwap_signal', 'NEUTRAL')
                ma_trend = snapshot_dict.get('ma_trend', 'NEUTRAL')
                
                # Technical-based direction in learning mode (AGGRESSIVE)
                bullish_signals = 0
                bearish_signals = 0
                
                if macd_cross == 'BULLISH': bullish_signals += 2
                if macd_cross == 'BEARISH': bearish_signals += 2
                if rsi < 35: bullish_signals += 2  # Strong oversold
                elif rsi < 45: bullish_signals += 1  # Moderate oversold
                if rsi > 65: bearish_signals += 2  # Strong overbought
                elif rsi > 55: bearish_signals += 1  # Moderate overbought
                if vwap_signal == 'ABOVE': bullish_signals += 1
                if vwap_signal == 'BELOW': bearish_signals += 1
                if ma_trend == 'BULLISH': bullish_signals += 1
                if ma_trend == 'BEARISH': bearish_signals += 1
                
                # Use slight momentum bias too
                if spy_5m > 0.01: bullish_signals += 1
                if spy_5m < -0.01: bearish_signals += 1
                
                # In learning mode, only need 1 signal to act (very aggressive)
                if bullish_signals >= 1 and bullish_signals > bearish_signals:
                    action = "LONG_CALL"
                    reasoning = f"Technical signals bullish ({bullish_signals} vs {bearish_signals}). Learning mode trade."
                    self.consecutive_holds = 0
                    logger.info(f"🎯 LEARNING TRADE: LONG_CALL (RSI={rsi:.1f}, MACD={macd_cross}, bull={bullish_signals}, bear={bearish_signals})")
                elif bearish_signals >= 1 and bearish_signals > bullish_signals:
                    action = "LONG_PUT"
                    reasoning = f"Technical signals bearish ({bearish_signals} vs {bullish_signals}). Learning mode trade."
                    self.consecutive_holds = 0
                    logger.info(f"🎯 LEARNING TRADE: LONG_PUT (RSI={rsi:.1f}, MACD={macd_cross}, bull={bullish_signals}, bear={bearish_signals})")
                elif self.consecutive_holds > 10:
                    # Force a trade after too many holds to keep learning
                    import random
                    action = "LONG_CALL" if random.random() > 0.5 else "LONG_PUT"
                    reasoning = f"Forced learning trade after {self.consecutive_holds} holds. Need experience data."
                    logger.info(f"🎯 FORCED TRADE: {action} after {self.consecutive_holds} consecutive holds")
                    self.consecutive_holds = 0
                else:
                    self.consecutive_holds += 1
                    return TradingDecision(
                        action="HOLD",
                        confidence=profitable_prob,
                        reasoning=f"No clear signals (bull: {bullish_signals}, bear: {bearish_signals}). Waiting.",
                    )
            else:
                self.consecutive_holds += 1
                return TradingDecision(
                    action="HOLD",
                    confidence=profitable_prob,
                    reasoning="No clear directional momentum. Waiting.",
                )
        
        # Suggest strike (ATM)
        spy_price = snapshot_dict.get('spy_price', 0)
        suggested_strike = round(spy_price)
        
        return TradingDecision(
            action=action,
            confidence=profitable_prob,
            reasoning=reasoning,
            suggested_strike=suggested_strike,
            suggested_contracts=1,
        )
    
    def _rule_based_entry(self, feature_set: FeatureSet, snapshot_dict: Dict[str, Any]) -> TradingDecision:
        """
        Rule-based entry decision when no model is trained.
        Uses time-of-day patterns and momentum.
        """
        features = feature_set.features
        
        # Extract key features by name
        spy_5m = snapshot_dict.get('spy_change_5m', 0)
        spy_15m = snapshot_dict.get('spy_change_15m', 0)
        spy_1m = snapshot_dict.get('spy_change_1m', 0)
        vix = snapshot_dict.get('vix', 20)
        time_decay = snapshot_dict.get('time_decay_factor', 0.5)
        minutes_since_open = snapshot_dict.get('minutes_since_open', 0)
        is_paper_mode = snapshot_dict.get('paper_mode', True)  # Default to paper for learning
        
        # RULE 1: Avoid first 5 minutes (too volatile) - reduced from 10 for more learning
        if minutes_since_open < 5:
            return TradingDecision(
                action="HOLD",
                confidence=0.9,
                reasoning="Market just opened. Waiting for price discovery.",
            )
        
        # RULE 2: Avoid last 30 minutes for new entries (theta crush)
        if time_decay > 0.92:  # ~30 min before close
            return TradingDecision(
                action="HOLD",
                confidence=0.9,
                reasoning="Too close to market close. Theta decay too aggressive.",
            )
        
        # RULE 3: Need momentum - VERY LOW threshold in paper mode for maximum learning
        momentum_threshold = 0.001 if is_paper_mode else 0.15  # 0.001% for paper (nearly any movement), 0.15% for live
        logger.info(f"[RULES] Momentum check: 5m={spy_5m:.4f}%, threshold={momentum_threshold}%, paper_mode={is_paper_mode}")
        if abs(spy_5m) < momentum_threshold:
            return TradingDecision(
                action="HOLD",
                confidence=0.7,
                reasoning=f"Insufficient momentum ({spy_5m:.3f}%). Need >{momentum_threshold}%.",
            )
        
        # RULE 4: Trend consistency - SUPER RELAXED for paper mode (just need 5m direction)
        logger.info(f"[RULES] Trend check: 1m={spy_1m:.4f}%, 5m={spy_5m:.4f}%, 15m={spy_15m:.4f}%")
        if is_paper_mode:
            # In paper mode, just need ANY momentum in 5m
            same_direction = abs(spy_5m) > 0.0005  # Basically any movement
        else:
            same_direction = (
                (spy_1m > 0 and spy_5m > 0 and spy_15m > 0) or
                (spy_1m < 0 and spy_5m < 0 and spy_15m < 0)
            )
        
        if not same_direction:
            return TradingDecision(
                action="HOLD",
                confidence=0.6,
                reasoning="Mixed signals across timeframes. Waiting for alignment.",
            )
        
        # RULE 5: VIX check
        if vix > 35:
            return TradingDecision(
                action="HOLD",
                confidence=0.8,
                reasoning=f"VIX too high ({vix:.1f}). Market too volatile.",
            )
        
        # Calculate confidence based on momentum strength
        momentum_strength = min(1.0, abs(spy_5m) / 0.5)  # 0.5% = max confidence
        base_confidence = 0.5 + (momentum_strength * 0.3)  # 50-80% range
        
        # Time adjustments
        if 60 <= minutes_since_open <= 180:  # 10:30 AM - 12:30 PM
            base_confidence += 0.05  # Mid-morning tends to trend
        elif 270 <= minutes_since_open <= 330:  # 2:00 - 3:00 PM (power hour)
            base_confidence += 0.05
        
        # VIX adjustments
        if 15 <= vix <= 22:
            base_confidence += 0.05  # Goldilocks zone
        
        confidence = min(0.85, base_confidence)  # Cap at 85%
        
        # Determine direction
        spy_price = snapshot_dict.get('spy_price', 0)
        
        if spy_5m > 0:
            action = "LONG_CALL"
            direction = "bullish"
        else:
            action = "LONG_PUT"
            direction = "bearish"
        
        reasoning = (
            f"Rule-based: Strong {direction} momentum ({spy_5m:+.2f}% 5m). "
            f"Trend aligned across timeframes. VIX={vix:.1f}. "
            f"Time decay factor={time_decay:.2f}."
        )
        
        return TradingDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            suggested_strike=round(spy_price),
            suggested_contracts=1,
        )
    
    def predict_exit(self, position_data: Dict[str, Any], snapshot_dict: Dict[str, Any]) -> TradingDecision:
        """
        Predict whether to exit a position
        
        Args:
            position_data: Current position info (entry_price, current_price, pnl_percent, etc.)
            snapshot_dict: Current market snapshot
            
        Returns:
            TradingDecision with EXIT action or HOLD
        """
        pnl_percent = position_data.get('pnl_percent', 0)
        strategy = position_data.get('strategy', 'LONG_CALL')
        hold_seconds = position_data.get('hold_duration_seconds', 0)
        
        # Rule-based exit logic (fast scalping strategy)
        
        # PROFIT TARGET: 15%
        if pnl_percent >= 15:
            return TradingDecision(
                action="EXIT",
                confidence=0.95,
                reasoning=f"Profit target reached ({pnl_percent:.1f}%)",
                exit_reason="PROFIT_TARGET",
            )
        
        # STOP LOSS: -20%
        if pnl_percent <= -20:
            return TradingDecision(
                action="EXIT",
                confidence=0.99,
                reasoning=f"Stop loss triggered ({pnl_percent:.1f}%)",
                exit_reason="STOP_LOSS",
            )
        
        # TRAILING STOP: If was up 5%+ and dropped 15% from peak
        max_profit = position_data.get('max_profit_percent', 0)
        if max_profit >= 5:
            drawdown_from_peak = max_profit - pnl_percent
            if drawdown_from_peak >= 15:
                return TradingDecision(
                    action="EXIT",
                    confidence=0.9,
                    reasoning=f"Trailing stop: Was +{max_profit:.1f}%, now +{pnl_percent:.1f}%",
                    exit_reason="TRAILING_STOP",
                )
        
        # TIME DECAY: Exit if held too long without profit
        time_decay = snapshot_dict.get('time_decay_factor', 0.5)
        if time_decay > 0.9 and pnl_percent < 5:  # Last 40 min
            return TradingDecision(
                action="EXIT",
                confidence=0.85,
                reasoning="Approaching close without sufficient profit",
                exit_reason="TIME_DECAY",
            )
        
        # ADVERSE MOMENTUM: Price moving strongly against us
        spy_5m = snapshot_dict.get('spy_change_5m', 0)
        if strategy == 'LONG_CALL' and spy_5m < -0.3:  # Strong down move
            return TradingDecision(
                action="EXIT",
                confidence=0.75,
                reasoning=f"Strong adverse momentum ({spy_5m:.2f}%)",
                exit_reason="ADVERSE_MARKET",
            )
        elif strategy == 'LONG_PUT' and spy_5m > 0.3:  # Strong up move
            return TradingDecision(
                action="EXIT",
                confidence=0.75,
                reasoning=f"Strong adverse momentum ({spy_5m:+.2f}%)",
                exit_reason="ADVERSE_MARKET",
            )
        
        # HOLD
        return TradingDecision(
            action="HOLD",
            confidence=0.6,
            reasoning=f"Position within parameters. P&L: {pnl_percent:+.1f}%",
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the entry prediction model using time-series-aware splitting
        and early stopping to prevent overfitting.

        Args:
            X: Feature matrix (n_samples, n_features) - MUST be in time order
            y: Labels (1=profitable, 0=not)
            validation_split: Fraction of data for validation

        Returns:
            Training metrics
        """
        if not HAS_XGBOOST:
            logger.error("Cannot train: XGBoost not installed")
            return {'error': 'XGBoost not installed'}

        n_samples = len(y)
        if n_samples < self.MINIMUM_TRADES_FOR_TRAINING:
            logger.warning(f"Not enough training data: {n_samples} < {self.MINIMUM_TRADES_FOR_TRAINING}")
            return {'error': 'Insufficient training data'}

        # TIME-SERIES SPLIT: Use chronological order, never train on future data
        # This prevents data leakage that would make validation metrics unreliable
        from sklearn.model_selection import TimeSeriesSplit

        # Use 5 folds for cross-validation, take the last fold as final validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Get all splits and use the last one (most recent data for validation)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]  # Last split: train on oldest, validate on newest

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(f"TimeSeriesSplit: Training on {len(train_idx)} oldest samples, validating on {len(val_idx)} newest")

        # Create XGBoost model with STRONGER REGULARIZATION
        # These settings prevent overfitting on small datasets (p >> n problem)
        self.entry_model = xgb.XGBClassifier(
            n_estimators=500,           # More trees, but early stopping will find optimal
            max_depth=3,                # Reduced from 5 - simpler trees overfit less
            learning_rate=0.05,         # Slower learning rate for better generalization
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            # REGULARIZATION PARAMETERS (per PDF recommendation)
            reg_alpha=0.1,              # L1 regularization (feature selection)
            reg_lambda=1.0,             # L2 regularization (weight decay)
            min_child_weight=5,         # Minimum sum of instance weight in child
            subsample=0.8,              # Row subsampling to reduce overfitting
            colsample_bytree=0.8,       # Column subsampling per tree
            gamma=0.1,                  # Minimum loss reduction for split
        )

        # Train with EARLY STOPPING
        # Stops when validation loss stops improving for 20 rounds
        self.entry_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Get the best iteration (where early stopping would have stopped)
        # Note: XGBoost stores results in evals_result
        best_iteration = self.entry_model.best_iteration if hasattr(self.entry_model, 'best_iteration') else self.entry_model.n_estimators

        # Evaluate
        train_acc = (self.entry_model.predict(X_train) == y_train).mean()
        val_acc = (self.entry_model.predict(X_val) == y_val).mean()

        # Check for overfitting (train >> val accuracy is a red flag)
        overfit_gap = train_acc - val_acc
        if overfit_gap > 0.15:
            logger.warning(f"⚠️ Possible overfitting: train={train_acc:.1%}, val={val_acc:.1%}, gap={overfit_gap:.1%}")

        # Get feature importances
        importances = self.entry_model.feature_importances_
        top_features = self.feature_engineer.get_feature_importance_names(importances)[:10]

        self.entry_model_trained = True
        self.last_training_time = datetime.now()
        self.training_samples = n_samples
        self.model_accuracy = val_acc

        # Save models
        self.save_models()

        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'n_samples': n_samples,
            'top_features': top_features,
            'best_iteration': best_iteration,
            'overfit_gap': overfit_gap,
            'split_method': 'TimeSeriesSplit',
        }

        logger.info(f"Training complete (TimeSeriesSplit). Validation accuracy: {val_acc:.1%}")
        logger.info(f"Best iteration: {best_iteration}, Overfit gap: {overfit_gap:.1%}")
        logger.info(f"Top features: {[f[0] for f in top_features[:5]]}")

        return metrics
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get current model statistics"""
        return {
            'entry_model_trained': self.entry_model_trained,
            'exit_model_trained': self.exit_model_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_samples': self.training_samples,
            'model_accuracy': self.model_accuracy,
            'n_features': self.feature_engineer.n_features,
            'using_xgboost': HAS_XGBOOST,
        }
