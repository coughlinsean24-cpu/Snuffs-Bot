#!/usr/bin/env python3
"""
Background Learner

Runs independently to:
1. Collect market snapshots every minute during market hours
2. Run simulated predictions and track outcomes
3. Learn from "what would have happened" without real trades
4. Accelerates model training with simulated experience

This allows the AI to learn from 400+ opportunities per day
instead of just a few real trades.
"""

import asyncio
import sys
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    "logs/background_learner.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


class SimulatedTrade:
    """Tracks a simulated trade prediction"""
    def __init__(
        self,
        prediction_time: datetime,
        action: str,
        confidence: float,
        spy_price_at_prediction: float,
        reasoning: str,
    ):
        self.prediction_time = prediction_time
        self.action = action  # LONG_CALL, LONG_PUT, HOLD
        self.confidence = confidence
        self.spy_price_at_prediction = spy_price_at_prediction
        self.reasoning = reasoning
        
        # Outcome tracking
        self.outcome_checked = False
        self.spy_price_at_check: Optional[float] = None
        self.price_change_percent: Optional[float] = None
        self.would_have_profited: Optional[bool] = None
        self.check_time: Optional[datetime] = None
    
    def check_outcome(self, current_spy_price: float) -> bool:
        """Check if the prediction would have been profitable"""
        self.check_time = datetime.now()
        self.spy_price_at_check = current_spy_price
        self.price_change_percent = (
            (current_spy_price - self.spy_price_at_prediction) 
            / self.spy_price_at_prediction * 100
        )
        
        # Determine if prediction would have profited
        if self.action == "LONG_CALL":
            # Call profits if price went up > 0.1%
            self.would_have_profited = self.price_change_percent > 0.1
        elif self.action == "LONG_PUT":
            # Put profits if price went down > 0.1%
            self.would_have_profited = self.price_change_percent < -0.1
        elif self.action == "HOLD":
            # HOLD is correct if price didn't move much (stayed within ±0.15%)
            self.would_have_profited = abs(self.price_change_percent) < 0.15
        else:
            self.would_have_profited = False
        
        self.outcome_checked = True
        return self.would_have_profited
    
    def to_training_record(self) -> Dict[str, Any]:
        """Convert to a training record for the model"""
        return {
            'timestamp': self.prediction_time.isoformat(),
            'action': self.action,
            'confidence': self.confidence,
            'spy_entry': self.spy_price_at_prediction,
            'spy_exit': self.spy_price_at_check,
            'price_change_pct': self.price_change_percent,
            'was_profitable': self.would_have_profited,
            'is_simulated': True,
            'hold_minutes': (self.check_time - self.prediction_time).total_seconds() / 60 if self.check_time else 0,
        }


class BackgroundLearner:
    """
    Background process that collects data and runs simulations
    to accelerate AI learning.
    """
    
    # Simulation parameters
    PREDICTION_INTERVAL_MINUTES = 2  # Make a prediction every 2 minutes
    OUTCOME_CHECK_DELAY_MINUTES = 15  # Check outcome after 15 minutes
    MIN_CONFIDENCE_FOR_SIM = 0.55  # Only simulate trades above this confidence
    
    def __init__(self):
        self.est = ZoneInfo("America/New_York")
        self.running = False
        
        # Pending simulated trades awaiting outcome check
        self.pending_simulations: List[SimulatedTrade] = []
        
        # Stats
        self.snapshots_collected = 0
        self.simulations_run = 0
        self.simulations_correct = 0
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize AI and data components"""
        try:
            from snuffs_bot.local_ai import HybridOrchestrator
            from snuffs_bot.api.client import TastytradeClient
            
            self.orchestrator = HybridOrchestrator(use_local_only=True)
            self.client = TastytradeClient()
            self.client.connect()
            
            # Start streaming for SPY and VIX
            self.client.market_data.start_streaming(["SPY", "$VIX.X"])
            
            # Wait for streaming data to arrive
            import time as time_module
            logger.info("Waiting for streaming data to populate...")
            time_module.sleep(3)
            
            logger.success("Background learner initialized")
            logger.info(f"Current snapshots: {self.orchestrator.data_collector.get_snapshot_count()}")
            logger.info(f"Current trades: {self.orchestrator.data_collector.get_trade_count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    def is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.est)
        
        # Check weekday
        if now.weekday() >= 5:
            return False
        
        # Check time
        market_open = time(9, 30)
        market_close = time(16, 15)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    async def get_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetch current market data from Tastytrade streaming cache"""
        try:
            # Get SPY quote from streaming cache
            spy_quote = self.client.market_data.get_quote("SPY")
            
            spy_price = float(spy_quote.get('last', spy_quote.get('mark', 0)))
            if spy_price == 0:
                spy_price = float(spy_quote.get('bid', 0) + spy_quote.get('ask', 0)) / 2
            
            if spy_price == 0:
                # No data yet, streaming hasn't populated
                return None
            
            # Get VIX
            try:
                vix_quote = self.client.market_data.get_quote("VIX")
                vix = float(vix_quote.get('last', vix_quote.get('mark', 20)))
                if vix == 0:
                    vix = 20.0
            except:
                vix = 20.0
            
            return {
                'spy_price': spy_price,
                'spy_bid': float(spy_quote.get('bid', spy_price)),
                'spy_ask': float(spy_quote.get('ask', spy_price)),
                'vix': vix,
                'volume': int(spy_quote.get('volume', 0)),
            }
            
        except Exception as e:
            logger.debug(f"Failed to get market data: {e}")
            return None
    
    async def collect_snapshot(self, market_data: Dict[str, Any]) -> None:
        """Collect and store a market snapshot"""
        try:
            # Build snapshot
            snapshot = self.orchestrator.data_collector.build_snapshot_from_live_data(
                spy_data={
                    'mark': market_data['spy_price'],
                    'bid': market_data['spy_bid'],
                    'ask': market_data['spy_ask'],
                    'volume': market_data['volume'],
                },
                vix=market_data['vix'],
                call_option={},
                put_option={},
            )
            
            # Record it
            self.orchestrator.data_collector.record_snapshot(snapshot)
            self.snapshots_collected += 1
            
            # Log every 10 snapshots
            if self.snapshots_collected % 10 == 0:
                logger.info(f"📸 Collected {self.snapshots_collected} snapshots (total: {self.orchestrator.data_collector.get_snapshot_count()})")
            elif self.snapshots_collected <= 3:
                logger.info(f"📸 Snapshot #{self.snapshots_collected} collected | SPY: ${market_data['spy_price']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to collect snapshot: {e}")
    
    async def run_simulation(self, market_data: Dict[str, Any]) -> Optional[SimulatedTrade]:
        """Run a simulated prediction and track for later outcome check"""
        try:
            # Get AI prediction
            decision = await self.orchestrator.get_entry_decision(
                market_data=market_data,
                vix=market_data['vix'],
            )
            
            # Track ALL decisions for learning (HOLD, LONG_CALL, LONG_PUT)
            sim = SimulatedTrade(
                prediction_time=datetime.now(self.est),
                action=decision.action,
                confidence=decision.confidence,
                spy_price_at_prediction=market_data['spy_price'],
                reasoning=decision.reasoning,
            )
            
            self.pending_simulations.append(sim)
            
            if decision.action in ["LONG_CALL", "LONG_PUT"]:
                logger.info(
                    f"📊 SIM TRADE: {decision.action} @ ${market_data['spy_price']:.2f} "
                    f"(conf: {decision.confidence:.1%}) - checking in {self.OUTCOME_CHECK_DELAY_MINUTES}min"
                )
            else:
                logger.info(
                    f"📊 SIM HOLD @ ${market_data['spy_price']:.2f} "
                    f"(conf: {decision.confidence:.1%}) - will verify in {self.OUTCOME_CHECK_DELAY_MINUTES}min"
                )
            
            return sim
            
        except Exception as e:
            logger.error(f"Failed to run simulation: {e}")
            return None
    
    async def check_pending_outcomes(self, current_spy_price: float) -> None:
        """Check outcomes of pending simulations"""
        now = datetime.now(self.est)
        
        for sim in self.pending_simulations[:]:  # Copy list to allow modification
            # Check if enough time has passed
            time_elapsed = (now - sim.prediction_time).total_seconds() / 60
            
            if time_elapsed >= self.OUTCOME_CHECK_DELAY_MINUTES:
                # Check outcome
                was_correct = sim.check_outcome(current_spy_price)
                self.simulations_run += 1
                
                if was_correct:
                    self.simulations_correct += 1
                    emoji = "✅"
                else:
                    emoji = "❌"
                
                logger.info(
                    f"{emoji} SIM RESULT: {sim.action} | "
                    f"Entry: ${sim.spy_price_at_prediction:.2f} → "
                    f"Exit: ${current_spy_price:.2f} ({sim.price_change_percent:+.2f}%) | "
                    f"Would have {'profited' if was_correct else 'lost'}"
                )
                
                # Record as simulated training data
                self._record_simulation_outcome(sim)
                
                # Remove from pending
                self.pending_simulations.remove(sim)
    
    def _record_simulation_outcome(self, sim: SimulatedTrade) -> None:
        """Record simulation as training data for the model"""
        try:
            import sqlite3
            
            db_path = "data/local_ai/market_data.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create simulated trades table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulated_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_time TEXT,
                    action TEXT,
                    confidence REAL,
                    spy_entry REAL,
                    spy_exit REAL,
                    price_change_pct REAL,
                    was_profitable INTEGER,
                    hold_minutes REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert record
            cursor.execute("""
                INSERT INTO simulated_trades 
                (prediction_time, action, confidence, spy_entry, spy_exit, 
                 price_change_pct, was_profitable, hold_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sim.prediction_time.isoformat(),
                sim.action,
                sim.confidence,
                sim.spy_price_at_prediction,
                sim.spy_price_at_check,
                sim.price_change_percent,
                1 if sim.would_have_profited else 0,
                (sim.check_time - sim.prediction_time).total_seconds() / 60,
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record simulation: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        win_rate = (
            self.simulations_correct / self.simulations_run 
            if self.simulations_run > 0 else 0
        )
        
        return {
            'snapshots_collected': self.snapshots_collected,
            'simulations_run': self.simulations_run,
            'simulations_correct': self.simulations_correct,
            'simulation_win_rate': win_rate,
            'pending_simulations': len(self.pending_simulations),
            'total_snapshots': self.orchestrator.data_collector.get_snapshot_count(),
        }
    
    def _write_status_file(self) -> None:
        """Write current status to a file for monitoring"""
        import json
        
        status = {
            'last_update': datetime.now(self.est).isoformat(),
            'is_market_hours': self.is_market_hours(),
            'running': self.running,
            **self.get_stats(),
        }
        
        try:
            status_path = Path("data/local_ai/learner_status.json")
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not write status file: {e}")

    async def run(self):
        """Main loop - runs during market hours"""
        logger.info("🚀 Background Learner started")
        logger.info(f"  - Collecting snapshots every minute")
        logger.info(f"  - Running simulations every {self.PREDICTION_INTERVAL_MINUTES} minutes")
        logger.info(f"  - Checking outcomes after {self.OUTCOME_CHECK_DELAY_MINUTES} minutes")
        
        self.running = True
        last_prediction_time = datetime.now(self.est) - timedelta(minutes=self.PREDICTION_INTERVAL_MINUTES + 1)
        last_health_check = datetime.now(self.est)
        consecutive_failures = 0
        
        while self.running:
            try:
                now = datetime.now(self.est)
                
                # Only run during market hours
                if not self.is_market_hours():
                    logger.debug("Market closed, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Health check streaming every 5 minutes
                if (now - last_health_check).total_seconds() > 300:
                    if hasattr(self.client.market_data, 'check_streaming_health'):
                        self.client.market_data.check_streaming_health()
                    last_health_check = now
                
                # Get market data
                market_data = await self.get_market_data()
                if not market_data or market_data.get('spy_price', 0) == 0:
                    consecutive_failures += 1
                    logger.warning(f"Failed to get market data (attempt {consecutive_failures})")
                    
                    # After 3 failures, try to restart streaming
                    if consecutive_failures >= 3:
                        logger.warning("Restarting market data stream...")
                        try:
                            self.client.market_data.stop_streaming()
                            await asyncio.sleep(2)
                            self.client.market_data.start_streaming(["SPY", "$VIX.X"])
                            consecutive_failures = 0
                        except Exception as e:
                            logger.error(f"Failed to restart stream: {e}")
                    
                    await asyncio.sleep(30)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Always collect snapshot (every minute)
                await self.collect_snapshot(market_data)
                
                # Check pending simulation outcomes
                await self.check_pending_outcomes(market_data['spy_price'])
                
                # Run new simulation every N minutes
                minutes_since_last = (now - last_prediction_time).total_seconds() / 60
                if minutes_since_last >= self.PREDICTION_INTERVAL_MINUTES:
                    await self.run_simulation(market_data)
                    last_prediction_time = now
                
                # Log stats every 10 minutes
                if self.snapshots_collected % 10 == 0 and self.snapshots_collected > 0:
                    stats = self.get_stats()
                    logger.info(
                        f"📈 Stats: {stats['total_snapshots']} snapshots | "
                        f"{stats['simulations_run']} sims ({stats['simulation_win_rate']:.1%} correct) | "
                        f"{stats['pending_simulations']} pending"
                    )
                
                # Write status file for monitoring
                self._write_status_file()
                
                # Wait 1 minute
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                await asyncio.sleep(30)
        
        logger.info("Background Learner stopped")


async def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 BACKGROUND LEARNER - Accelerated AI Training")
    print("=" * 60)
    print()
    print("This process will:")
    print("  1. Collect market snapshots every minute")
    print("  2. Run simulated predictions every 5 minutes")
    print("  3. Check 'what would have happened' after 20 minutes")
    print("  4. Learn from 100+ daily simulated trades")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    learner = BackgroundLearner()
    await learner.run()


if __name__ == "__main__":
    asyncio.run(main())
