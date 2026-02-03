"""
Main Trading Engine

The central orchestrator that brings together all components:
- Tastytrade API connection
- AI agents for decision making
- Strategy selection and execution
- Risk management and guardrails
- Paper trading simulation
- Continuous learning loop
"""

import asyncio
import time
import traceback
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import signal
import pytz

from loguru import logger


# NYSE Holiday Calendar (2024-2026)
# Source: https://www.nyse.com/markets/hours-calendars
NYSE_HOLIDAYS: Set[date] = {
    # 2024
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # MLK Day
    date(2024, 2, 19),   # Presidents Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}


@dataclass
class SimulatedTrade:
    """Tracks a simulated prediction for later outcome checking"""
    prediction_time: datetime
    action: str  # LONG_CALL, LONG_PUT, HOLD
    confidence: float
    spy_price_at_prediction: float
    reasoning: str = ""
    
    # Filled when outcome is checked
    check_time: Optional[datetime] = None
    spy_price_at_check: Optional[float] = None
    price_change_percent: float = 0.0
    would_have_profited: bool = False
    
    def check_outcome(self, current_spy_price: float) -> bool:
        """Check if prediction would have been profitable"""
        self.check_time = datetime.now(pytz.timezone('US/Eastern'))
        self.spy_price_at_check = current_spy_price
        self.price_change_percent = ((current_spy_price - self.spy_price_at_prediction) 
                                      / self.spy_price_at_prediction) * 100
        
        # Determine if profitable (needs ~0.1% move to cover spread costs)
        if self.action == "LONG_CALL":
            self.would_have_profited = self.price_change_percent > 0.1
        elif self.action == "LONG_PUT":
            self.would_have_profited = self.price_change_percent < -0.1
        elif self.action == "HOLD":
            # HOLD is correct if price didn't move much (stayed flat)
            self.would_have_profited = abs(self.price_change_percent) < 0.15
        else:
            self.would_have_profited = False
            
        return self.would_have_profited

from snuffs_bot.config.settings import get_settings, Settings
from snuffs_bot.database.connection import init_database, create_all_tables, db_session_scope
from snuffs_bot.database.models import Trade, AIDecision
from snuffs_bot.api.client import TastytradeClient

# Import types for annotations (actual initialization happens in start())
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from snuffs_bot.ai.orchestrator import AIOrchestrator
    from snuffs_bot.strategies.zero_dte import StrategySelector
    from snuffs_bot.risk.guardrails import RiskGuardrails
    from snuffs_bot.paper_trading.execution_coordinator import ExecutionCoordinator
    from snuffs_bot.learning.scheduler import LearningScheduler
    from snuffs_bot.paper_trading.realtime_monitor import RealTimePositionMonitor
    from snuffs_bot.ai.agents.exit_agent import ExitAgent


class EngineState(Enum):
    """Trading engine states"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class TradingEngine:
    """
    Main trading engine orchestrating all components.

    Responsibilities:
    - Initialize and manage all subsystems
    - Coordinate the trading loop
    - Handle market data updates
    - Process AI decisions
    - Execute trades (paper and live)
    - Manage risk and guardrails
    - Run learning loop
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the trading engine

        Args:
            settings: Optional settings override
        """
        self.settings = settings or get_settings()
        self.state = EngineState.STOPPED

        # Component references (initialized in start())
        self.orchestrator: Optional["AIOrchestrator"] = None
        self.local_ai: Optional["HybridOrchestrator"] = None  # Local AI (XGBoost)
        self.news_collector = None  # News/context collector for market awareness
        self.strategy_selector: Optional["StrategySelector"] = None
        self.risk_guardrails: Optional["RiskGuardrails"] = None
        self.execution_coordinator: Optional["ExecutionCoordinator"] = None
        self.learning_scheduler: Optional["LearningScheduler"] = None
        self.tastytrade_client: Optional[TastytradeClient] = None
        self.realtime_monitor: Optional["RealTimePositionMonitor"] = None
        self.exit_agent: Optional["ExitAgent"] = None

        # Market data
        self.current_spy_price = 0.0
        self.current_vix = 0.0
        self.market_data: Dict[str, Any] = {}
        
        # Market events for AI learning (set these based on economic calendar)
        self.market_events = {
            "fed_speaking": 0,        # Jerome Powell speaking today
            "fomc_day": 0,            # FOMC meeting/rate decision day
            "rate_decision": 0,       # Rate policy announced today
            "earnings_major": 0,      # Major earnings (AAPL, MSFT, etc)
            "economic_data": 0,       # Major economic data (CPI, jobs, GDP)
            "event_notes": "",
        }
        
        # News-based context (updated by NewsCollector)
        self.news_context = {
            "news_sentiment": 0.0,    # -1.0 bearish to +1.0 bullish
            "war_tensions": 0,        # 1 if war/military news
            "tariff_news": 0,         # 1 if trade war/tariff news
            "fed_hawkish": 0,         # 1 if hawkish Fed news
            "fed_dovish": 0,          # 1 if dovish Fed news
            "recession_fears": 0,     # 1 if recession fears in news
            "context_summary": "",    # Why the market is moving
        }

        # Trading state
        self.trading_enabled = False
        self.paper_only_mode = True
        self.last_decision_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None  # Track last trade for cooldown
        self.trade_cooldown_seconds = 60  # Minimum 60 seconds between trades
        self.decisions_today = 0
        self.trades_today = 0

        # Quote integrity settings (prevents trading on stale/bad data)
        self.last_quote_update: Optional[datetime] = None
        self.QUOTE_STALENESS_SECONDS = 30  # Reject trades if quotes older than 30 seconds
        self.MAX_RELATIVE_SPREAD_PERCENT = 10.0  # Reject if spread > 10% of option price

        # Background learning state (merged from background_learner.py)
        self.background_learning_enabled = True
        self.pending_simulations: List[SimulatedTrade] = []
        self.snapshots_collected = 0
        self.simulations_run = 0
        self.simulations_correct = 0
        self.PREDICTION_INTERVAL_MINUTES = 2
        self.OUTCOME_CHECK_DELAY_MINUTES = 15
        self.last_prediction_time: Optional[datetime] = None
        self.est = pytz.timezone('US/Eastern')

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            "state_change": [],
            "decision_made": [],
            "trade_executed": [],
            "error": [],
        }

        # Async tasks
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Trading schedule
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)
        self.trading_start = dt_time(9, 45)  # Start 15 min after open
        self.trading_cutoff = dt_time(15, 45)  # Stop 15 min before close

        # Decision intervals
        self.decision_interval_seconds = 30  # 30 seconds for fast 0DTE scalping

    def register_handler(self, event: str, handler: Callable) -> None:
        """Register an event handler"""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    async def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to handlers"""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def _generate_simulated_option_chain(self, spy_price: float, vix: float) -> Dict[str, Any]:
        """
        Generate simulated option chain for paper trading

        Creates realistic option data based on current SPY price and VIX.
        Target delta 0.40-0.50 should be ATM or slightly OTM (1-2 strikes).
        """
        from datetime import datetime

        # Base IV from VIX (VIX represents 30-day annualized vol, 0DTE is higher)
        base_iv = vix / 100 * 1.5  # 0DTE IV is typically higher

        # Generate strikes around current price (1 dollar increments)
        strikes = []
        for offset in range(-10, 11):  # $10 up and down from current price
            strike = round(spy_price + offset)
            
            # Calculate proper delta based on moneyness
            # For 0DTE, delta drops off very quickly as you move OTM
            if offset == 0:  # ATM
                call_delta = 0.50
                put_delta = -0.50
            elif offset > 0:  # OTM calls (strike > spot)
                # Delta decreases as strike moves higher (more OTM)
                # For 0DTE, each $1 OTM drops delta by ~0.08-0.10
                call_delta = max(0.05, 0.50 - offset * 0.08)
                put_delta = min(-0.05, -0.50 - offset * 0.08)
            else:  # ITM calls / OTM puts (strike < spot)
                # Delta increases as strike moves lower (more ITM for calls)
                call_delta = min(0.95, 0.50 + abs(offset) * 0.08)
                put_delta = max(-0.95, -0.50 + abs(offset) * 0.08)

            # Price based on delta and IV (simplified Black-Scholes-ish)
            # ATM options on 0DTE are typically 0.3-0.5% of underlying
            call_price = max(0.05, spy_price * base_iv * abs(call_delta) * 0.1)
            put_price = max(0.05, spy_price * base_iv * abs(put_delta) * 0.1)
            
            # IV skew - OTM options have higher IV
            iv_skew = abs(offset) / spy_price

            strike_data = {
                "strike": strike,
                "expiration": datetime.now().strftime("%Y-%m-%d"),
                "call": {
                    "strike": strike,
                    "bid": round(call_price * 0.95, 2),
                    "ask": round(call_price * 1.05, 2),
                    "mid": round(call_price, 2),
                    "delta": round(call_delta, 3),
                    "gamma": round(0.05 / (1 + abs(offset)), 4),
                    "theta": round(-call_price * 0.1, 2),
                    "vega": round(spy_price * 0.001, 2),
                    "iv": round(base_iv + iv_skew * 0.1, 3),
                    "expiration": datetime.now().strftime("%Y-%m-%d"),
                },
                "put": {
                    "strike": strike,
                    "bid": round(put_price * 0.95, 2),
                    "ask": round(put_price * 1.05, 2),
                    "mid": round(put_price, 2),
                    "delta": round(put_delta, 3),
                    "gamma": round(0.05 / (1 + abs(offset)), 4),
                    "theta": round(-put_price * 0.1, 2),
                    "vega": round(spy_price * 0.001, 2),
                    "iv": round(base_iv + iv_skew * 0.1, 3),
                    "expiration": datetime.now().strftime("%Y-%m-%d"),
                }
            }
            strikes.append(strike_data)

        # Return in format expected by strategies
        return {
            "calls": [s["call"] for s in strikes],
            "puts": [s["put"] for s in strikes],
            "underlying_price": spy_price,
            "expiration": datetime.now().strftime("%Y-%m-%d"),
        }

    async def initialize(self) -> bool:
        """
        Initialize all engine components

        Returns:
            True if initialization successful
        """
        logger.info("Initializing trading engine...")
        self.state = EngineState.STARTING
        await self._emit_event("state_change", {"state": self.state.value})

        try:
            # Initialize database
            logger.info("Initializing database...")
            init_database()
            create_all_tables()  # Create tables if they don't exist

            # Initialize Tastytrade client for real market data
            logger.info("Connecting to Tastytrade for market data...")
            try:
                self.tastytrade_client = TastytradeClient.from_env()
                self.tastytrade_client.connect()
                logger.success("Connected to Tastytrade API")

                # Start WebSocket streaming for real-time quotes
                logger.info("Starting WebSocket market data stream...")
                if self.tastytrade_client.market_data.start_streaming():
                    logger.success("WebSocket streaming active for SPY/VIX")
                else:
                    logger.warning("WebSocket streaming failed, will use fallback")
            except Exception as e:
                logger.warning(f"Could not connect to Tastytrade: {e}")
                logger.warning("Will use simulated market data")
                self.tastytrade_client = None

            # Initialize AI orchestrator
            logger.info("Initializing AI orchestrator...")
            from snuffs_bot.ai.orchestrator import AIOrchestrator
            self.orchestrator = AIOrchestrator()

            # Initialize Local AI (XGBoost-based self-learning model)
            logger.info("Initializing Local AI system...")
            from snuffs_bot.local_ai import HybridOrchestrator
            
            # Paper trading = learning mode (trade aggressively to gather experience)
            is_paper_trading = self.settings.paper_trading
            
            self.local_ai = HybridOrchestrator(
                use_local_only=self.settings.use_local_ai,
                data_dir=self.settings.local_ai_data_dir,
                learning_mode=is_paper_trading,  # Aggressive trading in paper mode
            )
            if self.settings.use_local_ai:
                if is_paper_trading:
                    logger.success("Local AI enabled - LEARNING MODE (trading aggressively to gather experience)")
                else:
                    logger.success("Local AI enabled - PRODUCTION MODE (conservative trading)")
            else:
                logger.info("Local AI initialized in hybrid mode (recording data for training)")
            
            # Initialize News Collector for market context awareness
            logger.info("Initializing News Collector...")
            try:
                from snuffs_bot.local_ai.news_collector import NewsCollector
                self.news_collector = NewsCollector(data_dir=self.settings.local_ai_data_dir)
                logger.success("News Collector active - AI will understand market context")
            except Exception as e:
                logger.warning(f"News Collector not available: {e}")
                self.news_collector = None

            # Initialize strategy selector
            logger.info("Initializing strategy selector...")
            from snuffs_bot.strategies.zero_dte.strategy_selector import StrategySelector
            self.strategy_selector = StrategySelector()

            # Initialize risk guardrails
            logger.info("Initializing risk guardrails...")
            from snuffs_bot.risk.guardrails import RiskGuardrails
            self.risk_guardrails = RiskGuardrails()

            # Initialize execution coordinator
            logger.info("Initializing execution coordinator...")
            from snuffs_bot.paper_trading.execution_coordinator import (
                ExecutionCoordinator, ExecutionMode
            )
            self.execution_coordinator = ExecutionCoordinator(
                mode=ExecutionMode.PAPER_ONLY
            )
            
            # Pass market data provider to execution coordinator for real option prices
            if self.tastytrade_client and self.tastytrade_client.is_connected:
                self.execution_coordinator.set_market_data_provider(
                    self.tastytrade_client.market_data
                )
                logger.success("Execution coordinator connected to Tastytrade for real option prices")

            # Initialize learning scheduler
            logger.info("Initializing learning scheduler...")
            from snuffs_bot.learning.scheduler import LearningScheduler
            self.learning_scheduler = LearningScheduler()

            # Load previous learnings from paper/live trading
            logger.info("Loading learned knowledge from previous sessions...")
            if self.learning_scheduler.load_learnings_from_file():
                report = self.learning_scheduler.get_learning_report()
                patterns_count = report.get("pattern_statistics", {}).get("patterns_with_data", 0)
                threshold = report.get("current_thresholds", {}).get("confidence", 0.65)
                logger.info(f"  Loaded {patterns_count} patterns with trade data")
                logger.info(f"  Current confidence threshold: {threshold:.0%}")
            else:
                logger.info("  No previous learnings found - starting with base knowledge")

            # Initialize real-time position monitor for 0DTE exits
            logger.info("Initializing real-time position monitor...")
            from snuffs_bot.paper_trading.realtime_monitor import RealTimePositionMonitor
            from snuffs_bot.ai.agents.exit_agent import ExitAgent
            
            self.exit_agent = ExitAgent()
            self.realtime_monitor = RealTimePositionMonitor(
                exit_callback=self._handle_realtime_exit,
                ai_exit_callback=self._get_ai_exit_decision,
            )
            logger.success("Real-time position monitor initialized (sub-second monitoring)")

            logger.success("Trading engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            self.state = EngineState.ERROR
            await self._emit_event("error", {"message": str(e)})
            return False

    async def start(self) -> None:
        """Start the trading engine"""
        if self.state not in [EngineState.STOPPED, EngineState.PAUSED]:
            logger.warning(f"Cannot start engine in state {self.state}")
            return

        # Initialize if needed
        if self.orchestrator is None:
            if not await self.initialize():
                return

        logger.info("Starting trading engine...")
        self.state = EngineState.RUNNING
        self.trading_enabled = True
        await self._emit_event("state_change", {"state": self.state.value})

        # Log execution mode prominently
        mode = self.execution_coordinator.mode.value
        if self.paper_only_mode or mode == "PAPER_ONLY":
            logger.warning("=" * 60)
            logger.warning("📋 PAPER TRADING MODE - NO REAL MONEY AT RISK")
            logger.warning("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("💰 LIVE TRADING MODE - REAL MONEY!")
            logger.error("=" * 60)

        # Start learning scheduler
        await self.learning_scheduler.start()

        # Start main trading loop
        task = asyncio.create_task(self._trading_loop())
        self._tasks.append(task)

        # Start market data loop
        task = asyncio.create_task(self._market_data_loop())
        self._tasks.append(task)

        # Start position monitoring loop
        task = asyncio.create_task(self._position_monitor_loop())
        self._tasks.append(task)

        # Start real-time position monitor for sub-second exit decisions
        if self.realtime_monitor:
            await self.realtime_monitor.start()
            task = asyncio.create_task(self._realtime_price_loop())
            self._tasks.append(task)
            logger.success("Real-time position monitoring active (100ms intervals)")

        # Start background learning loop (merged from background_learner.py)
        if self.background_learning_enabled and self.local_ai:
            task = asyncio.create_task(self._background_learning_loop())
            self._tasks.append(task)
            logger.success("Background learning active (snapshots + simulations)")

        logger.success(f"Trading engine started [{mode}]")

    async def stop(self) -> None:
        """Stop the trading engine gracefully"""
        if self.state == EngineState.STOPPED:
            return

        logger.info("Stopping trading engine...")
        self.state = EngineState.STOPPING
        self.trading_enabled = False
        await self._emit_event("state_change", {"state": self.state.value})

        # Signal shutdown
        self._shutdown_event.set()

        # Stop real-time monitor
        if self.realtime_monitor:
            await self.realtime_monitor.stop()

        # Save learnings before stopping (preserves knowledge for next session)
        if self.learning_scheduler:
            logger.info("Saving learned knowledge for future sessions...")
            if self.learning_scheduler.save_learnings_to_file():
                report = self.learning_scheduler.get_learning_report()
                total_trades = sum(
                    p.get("occurrences", 0)
                    for p in report.get("pattern_statistics", {}).get("top_performing", [])
                )
                logger.info(f"  Knowledge from {self.trades_today} trades this session saved")
            await self.learning_scheduler.stop()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        self.state = EngineState.STOPPED
        await self._emit_event("state_change", {"state": self.state.value})

        logger.success("Trading engine stopped")

    async def pause(self) -> None:
        """Pause trading (keeps monitoring)"""
        if self.state != EngineState.RUNNING:
            return

        logger.info("Pausing trading engine...")
        self.trading_enabled = False
        self.state = EngineState.PAUSED
        await self._emit_event("state_change", {"state": self.state.value})

    async def resume(self) -> None:
        """Resume trading from paused state"""
        if self.state != EngineState.PAUSED:
            return

        logger.info("Resuming trading engine...")
        self.trading_enabled = True
        self.state = EngineState.RUNNING
        await self._emit_event("state_change", {"state": self.state.value})

    def is_market_day(self) -> bool:
        """Check if today is a market day (weekday, not a holiday)"""
        et = pytz.timezone('US/Eastern')
        today = datetime.now(et).date()

        # Check if it's a weekday (Monday=0, Friday=4)
        if today.weekday() > 4:
            logger.debug(f"Market closed: {today} is a weekend")
            return False

        # Check NYSE holiday calendar
        if today in NYSE_HOLIDAYS:
            logger.info(f"Market closed: {today} is an NYSE holiday")
            return False

        return True

    def is_market_hours(self) -> bool:
        """Check if within market hours"""
        if not self.is_market_day():
            return False
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close

    def is_trading_hours(self) -> bool:
        """Check if within trading hours (excluding open/close buffers)"""
        if not self.is_market_day():
            return False
        now = datetime.now().time()
        return self.trading_start <= now <= self.trading_cutoff

    def _are_quotes_fresh(self) -> bool:
        """Check if quotes are fresh enough to trade on"""
        if self.last_quote_update is None:
            logger.warning("Quote integrity: No quote update timestamp - rejecting trade")
            return False

        age_seconds = (datetime.now() - self.last_quote_update).total_seconds()
        if age_seconds > self.QUOTE_STALENESS_SECONDS:
            logger.warning(f"Quote integrity: Quotes are {age_seconds:.1f}s old (max {self.QUOTE_STALENESS_SECONDS}s) - rejecting trade")
            return False

        return True

    def _is_spread_acceptable(self, option_mid: float, option_spread: float) -> bool:
        """
        Check if option spread is acceptable (not too wide relative to price)

        Args:
            option_mid: Option mid/mark price
            option_spread: Bid-ask spread (ask - bid)

        Returns:
            True if spread is acceptable, False if too wide
        """
        if option_mid <= 0:
            return False

        spread_percent = (option_spread / option_mid) * 100
        if spread_percent > self.MAX_RELATIVE_SPREAD_PERCENT:
            logger.warning(
                f"Quote integrity: Spread {spread_percent:.1f}% exceeds max {self.MAX_RELATIVE_SPREAD_PERCENT}% "
                f"(spread=${option_spread:.2f}, mid=${option_mid:.2f}) - rejecting trade"
            )
            return False

        return True

    def _check_quote_integrity(self) -> Tuple[bool, str]:
        """
        Comprehensive quote integrity check before trading

        Returns:
            Tuple of (is_ok, reason_if_not_ok)
        """
        # Check quote freshness
        if not self._are_quotes_fresh():
            return False, "Stale quotes"

        # Check SPY spread
        spy_bid = self.market_data.get("spy_bid", 0)
        spy_ask = self.market_data.get("spy_ask", 0)
        spy_mid = self.market_data.get("spy_price", 0)
        if spy_bid > 0 and spy_ask > 0:
            spy_spread = spy_ask - spy_bid
            if spy_spread > 0.10:  # SPY spread should be very tight (< $0.10)
                logger.warning(f"Quote integrity: SPY spread ${spy_spread:.2f} too wide")
                return False, "SPY spread too wide"

        # Check option spreads from market data if available
        call_bid = self.market_data.get("call_bid", 0)
        call_ask = self.market_data.get("call_ask", 0)
        call_mid = self.market_data.get("call_price", 0)
        if call_bid > 0 and call_ask > 0 and call_mid > 0:
            call_spread = call_ask - call_bid
            if not self._is_spread_acceptable(call_mid, call_spread):
                return False, "Call option spread too wide"

        put_bid = self.market_data.get("put_bid", 0)
        put_ask = self.market_data.get("put_ask", 0)
        put_mid = self.market_data.get("put_price", 0)
        if put_bid > 0 and put_ask > 0 and put_mid > 0:
            put_spread = put_ask - put_bid
            if not self._is_spread_acceptable(put_mid, put_spread):
                return False, "Put option spread too wide"

        return True, ""

    def _is_force_flat_time(self) -> bool:
        """Check if we're past the force-flat time (3:50 PM ET) - UNCONDITIONAL"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et).time()
        force_flat_time = dt_time(15, 50)  # 3:50 PM ET
        return now >= force_flat_time

    async def _force_flat_all_positions(self) -> None:
        """Force close ALL positions - called unconditionally at 3:50 PM ET"""
        if self.execution_coordinator:
            open_positions = self.execution_coordinator.simulator.portfolio.positions
            if open_positions:
                logger.warning("=" * 60)
                logger.warning("⚠️  FORCE FLAT: Closing all positions before market close")
                logger.warning(f"⚠️  {len(open_positions)} position(s) to close")
                logger.warning("=" * 60)

                for pos_id in list(open_positions.keys()):
                    try:
                        position = open_positions.get(pos_id)
                        if position:
                            market_data = {
                                "spread_bid": position.current_price * 0.95,
                                "spread_ask": position.current_price * 1.05,
                                "vix": self.current_vix or 18
                            }
                            await self.execution_coordinator.simulator.close_trade(
                                pos_id, market_data, "FORCE_FLAT_EOD"
                            )
                            logger.warning(f"⚠️  Force closed position: {pos_id}")
                    except Exception as e:
                        logger.error(f"Failed to force close {pos_id}: {e}")

    async def _trading_loop(self) -> None:
        """Main trading decision loop"""
        logger.info("Trading loop started")

        while not self._shutdown_event.is_set():
            try:
                # UNCONDITIONAL FORCE FLAT CHECK - runs before any other logic
                # This is a safety measure to prevent holding positions into expiration
                if self._is_force_flat_time():
                    await self._force_flat_all_positions()
                    # Don't make any new decisions after force-flat time
                    await asyncio.sleep(self.decision_interval_seconds)
                    continue

                # Check if we should make decisions
                if self.trading_enabled and self.is_trading_hours():
                    spy_price = self.market_data.get("spy_price", 0)
                    logger.info(f"📊 EVALUATING TRADE OPPORTUNITY | SPY: ${spy_price:.2f}")
                    await self._make_trading_decision()
                elif not self.is_trading_hours():
                    now = datetime.now().time()
                    logger.debug(f"Outside trading hours ({now} not in {self.trading_start}-{self.trading_cutoff})")

                # Wait for next decision interval
                await asyncio.sleep(self.decision_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                error_context = {
                    "component": "trading_loop",
                    "spy_price": self.market_data.get("spy_price", 0),
                    "vix": self.market_data.get("vix", 0),
                    "decisions_today": self.decisions_today,
                    "trades_today": self.trades_today,
                    "traceback": traceback.format_exc()
                }
                logger.error(f"Error in trading loop: {e}")
                logger.error(f"Error context: {error_context}")
                await self._emit_event("error", {"message": str(e), "context": error_context})
                await asyncio.sleep(60)  # Back off on error

        logger.info("Trading loop stopped")

    async def _market_data_loop(self) -> None:
        """Market data update loop"""
        logger.info("Market data loop started")

        while not self._shutdown_event.is_set():
            try:
                await self._update_market_data()
                await asyncio.sleep(10)  # Update every 10 seconds (faster for 0DTE)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(30)

        logger.info("Market data loop stopped")

    async def _position_monitor_loop(self) -> None:
        """Position monitoring and exit management loop (backup to real-time monitor)"""
        logger.info("Position monitor loop started")

        while not self._shutdown_event.is_set():
            try:
                await self._check_positions()
                
                # Also update ALL database positions with streaming prices
                # This ensures dashboard shows correct prices for all open positions
                if self.execution_coordinator:
                    self.execution_coordinator.update_all_db_positions_from_streaming()
                
                # Faster checks: 5 seconds when positions open, 15 otherwise
                has_positions = self.execution_coordinator and len(
                    self.execution_coordinator.paper_simulator.portfolio.positions
                ) > 0
                await asyncio.sleep(5 if has_positions else 15)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitor loop: {e}")
                await asyncio.sleep(15)

        logger.info("Position monitor loop stopped")

    async def _realtime_price_loop(self) -> None:
        """
        Real-time price update loop for open positions
        
        Fetches prices every 500ms when positions are open to feed
        the real-time monitor for instant exit decisions.
        """
        logger.info("Real-time price loop started")

        while not self._shutdown_event.is_set():
            try:
                if not self.realtime_monitor or not self.realtime_monitor.positions:
                    # No positions to monitor - sleep longer
                    await asyncio.sleep(1.0)
                    continue

                # Fetch current prices for all monitored positions
                for pos_id in list(self.realtime_monitor.positions.keys()):
                    position = self.realtime_monitor.positions.get(pos_id)
                    if not position:
                        continue

                    # Get real-time quote from Tastytrade
                    bid, ask = await self._get_position_quote(pos_id)
                    
                    if bid > 0 and ask > 0:
                        # Feed to real-time monitor (instant exit check)
                        await self.realtime_monitor.process_price_tick(
                            position_id=pos_id,
                            bid=bid,
                            ask=ask,
                        )

                # 500ms between price fetches for speed
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in real-time price loop: {e}")
                await asyncio.sleep(1.0)

        logger.info("Real-time price loop stopped")

    async def _background_learning_loop(self) -> None:
        """
        Background learning loop - collects snapshots and runs simulations
        
        Merged from background_learner.py to run within the trading engine.
        This enables accelerated learning without a separate process.
        """
        logger.info("🧠 Background learning loop started")
        logger.info(f"  - Collecting snapshots every minute")
        logger.info(f"  - Running simulations every {self.PREDICTION_INTERVAL_MINUTES} minutes")
        logger.info(f"  - Checking outcomes after {self.OUTCOME_CHECK_DELAY_MINUTES} minutes")
        logger.info(f"  - News context refresh every 5 minutes")
        
        self.last_prediction_time = datetime.now(self.est) - timedelta(minutes=self.PREDICTION_INTERVAL_MINUTES + 1)
        last_news_refresh = datetime.now(self.est) - timedelta(minutes=10)  # Force initial refresh
        
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(self.est)
                
                # Only run during market hours
                if not self.is_market_hours():
                    await asyncio.sleep(60)
                    continue
                
                spy_price = self.market_data.get("spy_price", 0)
                vix = self.market_data.get("vix", 0)
                
                if spy_price == 0 or vix == 0:
                    await asyncio.sleep(30)
                    continue
                
                # Refresh news context every 5 minutes
                if (now - last_news_refresh).total_seconds() >= 300:
                    await self._refresh_news_context()
                    last_news_refresh = now
                
                # Collect snapshot every minute
                await self._collect_learning_snapshot(spy_price, vix)
                
                # Check pending simulation outcomes
                await self._check_simulation_outcomes(spy_price)
                
                # Run new simulation every N minutes
                if self.last_prediction_time:
                    minutes_since_last = (now - self.last_prediction_time).total_seconds() / 60
                    if minutes_since_last >= self.PREDICTION_INTERVAL_MINUTES:
                        await self._run_learning_simulation(spy_price, vix)
                        self.last_prediction_time = now
                
                # Log stats every 10 snapshots
                if self.snapshots_collected > 0 and self.snapshots_collected % 10 == 0:
                    win_rate = (self.simulations_correct / self.simulations_run * 100) if self.simulations_run > 0 else 0
                    logger.info(
                        f"📈 Learning Stats: {self.snapshots_collected} snapshots | "
                        f"{self.simulations_run} sims ({win_rate:.1f}% correct) | "
                        f"{len(self.pending_simulations)} pending"
                    )
                
                # Write status file for monitoring
                self._write_learning_status()
                
                # Wait 1 minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background learning loop: {e}")
                await asyncio.sleep(60)
        
        logger.info("Background learning loop stopped")

    async def _refresh_news_context(self) -> None:
        """Refresh news context for market awareness"""
        if not self.news_collector:
            return
            
        try:
            context = await asyncio.to_thread(self.news_collector.get_current_context)
            
            # Update news context state
            self.news_context = {
                "news_sentiment": context.overall_sentiment,
                "war_tensions": context.war_tensions,
                "tariff_news": context.tariff_news,
                "fed_hawkish": context.fed_hawkish,
                "fed_dovish": context.fed_dovish,
                "recession_fears": context.recession_fears,
                "context_summary": context.context_summary,
            }
            
            # Log significant context changes
            if context.war_tensions or context.tariff_news:
                logger.warning(f"⚠️ GEOPOLITICAL NEWS: {context.top_themes[:3]}")
            if context.overall_sentiment < -0.3:
                logger.warning(f"📉 BEARISH NEWS SENTIMENT: {context.overall_sentiment:.2f}")
            elif context.overall_sentiment > 0.3:
                logger.info(f"📈 BULLISH NEWS SENTIMENT: {context.overall_sentiment:.2f}")
            
            if context.news_count > 0:
                logger.debug(f"📰 News context updated: {context.news_count} articles, sentiment={context.overall_sentiment:.2f}")
                
        except Exception as e:
            logger.debug(f"News context refresh failed: {e}")

    async def _collect_learning_snapshot(self, spy_price: float, vix: float) -> None:
        """Collect and store a market snapshot for learning"""
        try:
            if not self.local_ai or not hasattr(self.local_ai, 'data_collector'):
                return
            
            # Build snapshot from current market data
            snapshot = self.local_ai.data_collector.build_snapshot_from_live_data(
                spy_data={
                    'mark': spy_price,
                    'bid': spy_price - 0.01,
                    'ask': spy_price + 0.01,
                    'volume': self.market_data.get('volume', 0),
                },
                vix=vix,
                call_option={},
                put_option={},
            )
            
            # Add news context to snapshot
            snapshot.news_sentiment = self.news_context.get("news_sentiment", 0.0)
            snapshot.war_tensions = self.news_context.get("war_tensions", 0)
            snapshot.tariff_news = self.news_context.get("tariff_news", 0)
            snapshot.fed_hawkish = self.news_context.get("fed_hawkish", 0)
            snapshot.fed_dovish = self.news_context.get("fed_dovish", 0)
            snapshot.recession_fears = self.news_context.get("recession_fears", 0)
            snapshot.context_summary = self.news_context.get("context_summary", "")[:500]
            
            # Record it
            self.local_ai.data_collector.record_snapshot(snapshot)
            self.snapshots_collected += 1
            
            if self.snapshots_collected <= 3:
                logger.info(f"📸 Snapshot #{self.snapshots_collected} collected | SPY: ${spy_price:.2f}")
                
        except Exception as e:
            logger.debug(f"Failed to collect snapshot: {e}")

    async def _run_learning_simulation(self, spy_price: float, vix: float) -> None:
        """Run a simulated prediction for later outcome checking"""
        try:
            if not self.local_ai:
                return
            
            # Get AI prediction (without executing)
            market_data = {
                'spy_price': spy_price,
                'vix': vix,
                'spy_change_5m': self.market_data.get('spy_change_5m', 0),
                'spy_change_15m': self.market_data.get('spy_change_15m', 0),
                # Include news context in decision making
                'news_sentiment': self.news_context.get("news_sentiment", 0.0),
                'war_tensions': self.news_context.get("war_tensions", 0),
                'context_summary': self.news_context.get("context_summary", ""),
            }
            
            decision = await self.local_ai.get_entry_decision(
                market_data=market_data,
                vix=vix,
            )
            
            # Create simulated trade record
            sim = SimulatedTrade(
                prediction_time=datetime.now(self.est),
                action=decision.action,
                confidence=decision.confidence,
                spy_price_at_prediction=spy_price,
                reasoning=decision.reasoning,
            )
            
            self.pending_simulations.append(sim)
            
            if decision.action in ["LONG_CALL", "LONG_PUT"]:
                logger.info(
                    f"📊 SIM TRADE: {decision.action} @ ${spy_price:.2f} "
                    f"(conf: {decision.confidence:.1%}) - checking in {self.OUTCOME_CHECK_DELAY_MINUTES}min"
                )
            else:
                logger.debug(
                    f"📊 SIM HOLD @ ${spy_price:.2f} "
                    f"(conf: {decision.confidence:.1%})"
                )
                
        except Exception as e:
            logger.debug(f"Failed to run simulation: {e}")

    async def _check_simulation_outcomes(self, current_spy_price: float) -> None:
        """Check outcomes of pending simulations"""
        now = datetime.now(self.est)
        
        for sim in self.pending_simulations[:]:  # Copy list to allow modification
            time_elapsed = (now - sim.prediction_time).total_seconds() / 60
            
            if time_elapsed >= self.OUTCOME_CHECK_DELAY_MINUTES:
                was_correct = sim.check_outcome(current_spy_price)
                self.simulations_run += 1
                
                if was_correct:
                    self.simulations_correct += 1
                    emoji = "✅"
                else:
                    emoji = "❌"
                
                if sim.action != "HOLD":
                    logger.info(
                        f"{emoji} SIM RESULT: {sim.action} | "
                        f"Entry: ${sim.spy_price_at_prediction:.2f} → "
                        f"Exit: ${current_spy_price:.2f} ({sim.price_change_percent:+.2f}%)"
                    )
                
                # Record as simulated training data
                self._record_simulation_outcome(sim)
                
                # Remove from pending
                self.pending_simulations.remove(sim)

    def _record_simulation_outcome(self, sim: SimulatedTrade) -> None:
        """Record simulation as training data for the model"""
        try:
            db_path = Path("data/local_ai/market_data.db")
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
            
            hold_minutes = (sim.check_time - sim.prediction_time).total_seconds() / 60 if sim.check_time else 0
            
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
                hold_minutes,
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Failed to record simulation: {e}")

    def _write_learning_status(self) -> None:
        """Write status file for monitoring"""
        status = {
            'last_update': datetime.now(self.est).isoformat(),
            'snapshots_collected': self.snapshots_collected,
            'simulations_run': self.simulations_run,
            'simulations_correct': self.simulations_correct,
            'pending_simulations': len(self.pending_simulations),
            'simulation_win_rate': (self.simulations_correct / self.simulations_run) if self.simulations_run > 0 else 0,
            'running': self.state == EngineState.RUNNING,
        }
        
        try:
            import json
            status_path = Path("data/local_ai/learner_status.json")
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not write status file: {e}")

    async def _get_position_quote(self, position_id: str) -> tuple[float, float]:
        """
        Get real-time bid/ask for a position
        
        Returns:
            (bid, ask) tuple
        """
        # Try to get from paper portfolio first
        if not self.execution_coordinator:
            return (0.0, 0.0)

        position = self.execution_coordinator.paper_simulator.portfolio.positions.get(position_id)
        if not position:
            return (0.0, 0.0)

        # Use current price with simulated spread
        current_price = position.current_price
        if current_price > 0:
            # Simulate bid/ask spread (typical spread for 0DTE options)
            spread = current_price * 0.02  # 2% spread
            return (current_price - spread/2, current_price + spread/2)

        return (0.0, 0.0)

    async def _handle_realtime_exit(
        self,
        position_id: str,
        reason: str,
        exit_price: float,
        recommendation: Any = None,
    ) -> None:
        """
        Handle exit triggered by real-time monitor
        
        This is called immediately when an exit condition is met.
        """
        logger.info(f"⚡ REAL-TIME EXIT TRIGGERED: {position_id} | {reason}")

        if not self.execution_coordinator:
            return

        try:
            # Close the paper position immediately
            market_data = {
                "spread_bid": exit_price * 0.98,
                "spread_ask": exit_price * 1.02,
                "vix": self.market_data.get("vix", 15),
            }

            await self.execution_coordinator.paper_simulator.close_trade(
                position_id=position_id,
                market_data=market_data,
                reason=reason,
            )

            # Remove from real-time monitor
            self.realtime_monitor.remove_position(position_id)

            # Process through learning system
            position_data = {
                "position_id": position_id,
                "exit_reason": reason,
                "exit_price": exit_price,
                "exit_time": datetime.now(),
            }
            await self._process_closed_position(position_data)

            self.trades_today += 1
            await self._emit_event("trade_executed", {
                "action": "CLOSE",
                "position_id": position_id,
                "reason": reason,
                "exit_price": exit_price,
            })

        except Exception as e:
            logger.error(f"Error handling real-time exit: {e}")

    async def _get_ai_exit_decision(
        self,
        position_id: str,
        entry_price: float,
        current_price: float,
        profit_percent: float,
        drawdown_from_high: float,
        momentum: float,
        time_held_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Get AI exit decision for a position
        
        Called by real-time monitor when it wants AI input.
        """
        if not self.exit_agent:
            return None

        try:
            result = await self.exit_agent.analyze_exit(
                position_id=position_id,
                entry_price=entry_price,
                current_price=current_price,
                profit_percent=profit_percent,
                drawdown_from_high=drawdown_from_high,
                momentum=momentum,
                time_held_seconds=time_held_seconds,
                spy_price=self.market_data.get("spy_price", 0),
                vix=self.market_data.get("vix", 0),
            )
            return result

        except Exception as e:
            logger.error(f"AI exit decision failed: {e}")
            return None

    async def _update_market_data(self) -> None:
        """Update current market data from Tastytrade"""
        spy_price = self.current_spy_price or 0.0
        vix_price = self.current_vix or 0.0
        spy_change = 0.0
        vix_change = 0.0

        # Fetch real prices from Tastytrade if connected
        if self.tastytrade_client and self.tastytrade_client.is_connected:
            try:
                # Check streaming health and reconnect if needed
                if hasattr(self.tastytrade_client.market_data, 'check_streaming_health'):
                    self.tastytrade_client.market_data.check_streaming_health()
                
                # Get SPY quote
                spy_quote = self.tastytrade_client.market_data.get_quote("SPY")
                if spy_quote:
                    spy_price = float(spy_quote.get("last", 0) or spy_quote.get("mark", 0) or 0)
                    prev_close = float(spy_quote.get("previous-close", spy_price) or spy_price)
                    if prev_close > 0:
                        spy_change = ((spy_price - prev_close) / prev_close) * 100

                # Get VIX quote (Tastytrade uses $VIX.X symbol)
                vix_quote = self.tastytrade_client.market_data.get_quote("$VIX.X")
                if vix_quote:
                    vix_price = float(vix_quote.get("last", 0) or vix_quote.get("mark", 0) or 0)
                    prev_vix = float(vix_quote.get("previous-close", vix_price) or vix_price)
                    if prev_vix > 0:
                        vix_change = vix_price - prev_vix

                self.current_spy_price = spy_price
                self.current_vix = vix_price
                
                # Update Local AI price history for momentum calculations
                if self.local_ai and spy_price > 0:
                    self.local_ai.data_collector.update_live_price(spy_price, vix_price)
                
                logger.debug(f"[LIVE DATA] SPY: ${spy_price:.2f} ({spy_change:+.2f}%), VIX: {vix_price:.2f}")

            except Exception as e:
                logger.warning(f"Could not fetch live market data: {e}")

        # Use defaults if no real data available
        if spy_price <= 0:
            spy_price = 596.89  # Last known SPY close
            logger.debug(f"[SIMULATED] Using default SPY price: ${spy_price:.2f}")
        if vix_price <= 0:
            vix_price = 15.5
            logger.debug(f"[SIMULATED] Using default VIX: {vix_price:.2f}")

        self.market_data = {
            "spy_price": spy_price,
            "spy_change_percent": spy_change,
            "vix": vix_price,
            "vix_change": vix_change,
            "current_time": datetime.now(),
            "trend_direction": "BULLISH" if spy_change > 0.3 else "BEARISH" if spy_change < -0.3 else "NEUTRAL",
            "momentum": "STRONG" if abs(spy_change) > 0.5 else "WEAK" if abs(spy_change) < 0.2 else "MODERATE",
            "volume_ratio": 1.0,
            "data_source": "TASTYTRADE" if self.tastytrade_client and self.tastytrade_client.is_connected else "SIMULATED",
        }

        # Update quote timestamp for staleness detection
        if spy_price > 0:
            self.last_quote_update = datetime.now()
        
        # Record continuous market data for Local AI training (every 10 seconds)
        # Only record during market hours (9:30 AM to 4:15 PM Eastern)
        # Includes SPY, VIX, AND option data from Tastytrade
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_cutoff = now.replace(hour=16, minute=15, second=0, microsecond=0)
        is_weekday = now.weekday() < 5
        is_market_hours = is_weekday and market_open <= now <= market_cutoff
        
        if self.local_ai and spy_price > 0 and is_market_hours:
            try:
                from snuffs_bot.local_ai.data_collector import MarketSnapshot
                
                # Get ATM option data from Tastytrade streaming cache
                call_price, call_bid, call_ask, call_strike = 0.0, 0.0, 0.0, 0.0
                call_delta, call_gamma, call_theta, call_iv = 0.0, 0.0, 0.0, 0.0
                call_volume, call_oi = 0, 0
                put_price, put_bid, put_ask, put_strike = 0.0, 0.0, 0.0, 0.0
                put_delta, put_gamma, put_theta, put_iv = 0.0, 0.0, 0.0, 0.0
                put_volume, put_oi = 0, 0
                spy_volume = 0
                
                if self.tastytrade_client and self.tastytrade_client.is_connected:
                    try:
                        today = datetime.now().strftime("%y%m%d")
                        # Search the streaming cache for any 0DTE options and SPY volume
                        with self.tastytrade_client.market_data._cache_lock:
                            # Get SPY volume from cache
                            spy_cache = self.tastytrade_client.market_data._quote_cache.get("SPY", {})
                            spy_volume = int(spy_cache.get("volume", 0) or 0)
                            
                            for symbol, data in self.tastytrade_client.market_data._quote_cache.items():
                                if f"SPY   {today}C" in symbol and call_price == 0:
                                    call_price = float(data.get("mark", 0) or ((data.get("bid", 0) + data.get("ask", 0)) / 2) or 0)
                                    call_bid = float(data.get("bid", 0) or 0)
                                    call_ask = float(data.get("ask", 0) or 0)
                                    # Extract strike from symbol (SPY   260128C00696000 -> 696)
                                    try:
                                        call_strike = int(symbol[-8:-3])
                                    except:
                                        call_strike = round(spy_price)
                                    call_delta = float(data.get("delta", 0.5) or 0.5)
                                    call_gamma = float(data.get("gamma", 0) or 0)
                                    call_theta = float(data.get("theta", 0) or 0)
                                    call_iv = float(data.get("implied-volatility", 0) or 0)
                                    call_volume = int(data.get("volume", 0) or 0)
                                    call_oi = int(data.get("open_interest", 0) or 0)
                                    
                                elif f"SPY   {today}P" in symbol and put_price == 0:
                                    put_price = float(data.get("mark", 0) or ((data.get("bid", 0) + data.get("ask", 0)) / 2) or 0)
                                    put_bid = float(data.get("bid", 0) or 0)
                                    put_ask = float(data.get("ask", 0) or 0)
                                    try:
                                        put_strike = int(symbol[-8:-3])
                                    except:
                                        put_strike = round(spy_price)
                                    put_delta = float(data.get("delta", -0.5) or -0.5)
                                    put_gamma = float(data.get("gamma", 0) or 0)
                                    put_theta = float(data.get("theta", 0) or 0)
                                    put_iv = float(data.get("implied-volatility", 0) or 0)
                                    put_volume = int(data.get("volume", 0) or 0)
                                    put_oi = int(data.get("open_interest", 0) or 0)
                                
                                # Found both, stop searching
                                if call_price > 0 and put_price > 0:
                                    break

                        # Update market_data with option prices for quote integrity checks
                        self.market_data.update({
                            "call_price": call_price,
                            "call_bid": call_bid,
                            "call_ask": call_ask,
                            "put_price": put_price,
                            "put_bid": put_bid,
                            "put_ask": put_ask,
                        })
                    except Exception as e:
                        logger.debug(f"Could not fetch option data for snapshot: {e}")
                
                now = datetime.now()
                snapshot = MarketSnapshot(
                    timestamp=now,
                    spy_price=spy_price,
                    spy_bid=spy_price - 0.01,
                    spy_ask=spy_price + 0.01,
                    spy_volume=spy_volume,
                    spy_change_today=spy_change,
                    vix=vix_price,
                    vix_change=vix_change,
                    hour=now.hour,
                    minute=now.minute,
                    minutes_since_open=max(0, (now.hour - 9) * 60 + now.minute - 30),
                    minutes_until_close=max(0, (16 - now.hour) * 60 - now.minute),
                    # Option data
                    call_strike=call_strike,
                    call_price=call_price,
                    call_bid=call_bid,
                    call_ask=call_ask,
                    call_delta=call_delta,
                    call_gamma=call_gamma,
                    call_theta=call_theta,
                    call_iv=call_iv,
                    call_volume=call_volume,
                    call_open_interest=call_oi,
                    put_strike=put_strike,
                    put_price=put_price,
                    put_bid=put_bid,
                    put_ask=put_ask,
                    put_delta=put_delta,
                    put_gamma=put_gamma,
                    put_theta=put_theta,
                    put_iv=put_iv,
                    put_volume=put_volume,
                    put_open_interest=put_oi,
                    call_spread=call_ask - call_bid if call_ask > 0 else 0,
                    put_spread=put_ask - put_bid if put_ask > 0 else 0,
                    # Market events for AI learning
                    fed_speaking=self.market_events.get("fed_speaking", 0),
                    fomc_day=self.market_events.get("fomc_day", 0),
                    rate_decision=self.market_events.get("rate_decision", 0),
                    earnings_major=self.market_events.get("earnings_major", 0),
                    economic_data=self.market_events.get("economic_data", 0),
                    event_notes=self.market_events.get("event_notes", ""),
                )
                self.local_ai.data_collector.record_snapshot(snapshot)
            except Exception as e:
                logger.debug(f"Local AI continuous data collection: {e}")

    async def _make_trading_decision(self) -> None:
        """Make a trading decision using AI orchestrator or Local AI"""
        try:
            # QUOTE INTEGRITY CHECK - reject trades on stale or wide-spread quotes
            quote_ok, quote_reason = self._check_quote_integrity()
            if not quote_ok:
                logger.warning(f"⚠️  Skipping trade decision: {quote_reason}")
                return

            # Check risk guardrails first
            portfolio_state = self._get_portfolio_state()
            risk_check = self.risk_guardrails.check_trading_hours()

            if not risk_check.passed:
                violations = [str(v) for v in risk_check.violations]
                logger.debug(f"Risk check failed: {', '.join(violations)}")
                return

            # Get option chain for execution planning
            option_chain = {}
            call_option = {}
            put_option = {}
            if self.tastytrade_client and self.tastytrade_client.is_connected:
                try:
                    option_chain = await asyncio.to_thread(
                        self.tastytrade_client.market_data.get_option_chain, "SPY"
                    )
                    # Extract ATM options for local AI
                    spy_price = self.market_data.get("spy_price", 595.0)
                    atm_strike = round(spy_price)
                    if option_chain:
                        for opt in option_chain.get("calls", []):
                            if opt.get("strike") == atm_strike:
                                call_option = opt
                                break
                        for opt in option_chain.get("puts", []):
                            if opt.get("strike") == atm_strike:
                                put_option = opt
                                break
                except Exception as e:
                    logger.warning(f"Could not fetch option chain: {e}")

            # USE LOCAL AI if enabled
            if self.settings.use_local_ai and self.local_ai:
                await self._make_local_ai_decision(portfolio_state, call_option, put_option)
                return

            # Otherwise use Claude AI orchestrator
            decision = await asyncio.to_thread(
                self.orchestrator.evaluate_trade_opportunity,
                self.market_data,
                portfolio_state,
                option_chain
            )

            # Record snapshot for local AI training (even when using Claude)
            if self.local_ai:
                try:
                    vix = self.market_data.get("vix", 20)
                    # Use await directly since get_entry_decision is async
                    await self.local_ai.get_entry_decision(
                        self.market_data, vix, call_option, put_option
                    )
                    logger.debug("Local AI: Recorded market snapshot for training")
                except Exception as e:
                    logger.debug(f"Local AI snapshot recording: {e}")

            self.last_decision_time = datetime.now()
            self.decisions_today += 1

            # Extract strategy from execution plan or market response
            suggested_strategy = "NONE"
            if decision.execution_plan:
                suggested_strategy = decision.execution_plan.get("strategy_type", "NONE")
            if suggested_strategy in ("NONE", None, ""):
                if decision.market_response:
                    suggested_strategy = decision.market_response.get("data", {}).get("recommended_strategy", "NONE")
            
            # FALLBACK: If Haiku still didn't provide a strategy, use momentum-based selection
            if suggested_strategy in ("NONE", None, ""):
                spy_change = self.market_data.get("spy_change_percent", 0)
                trend_direction = self.market_data.get("trend_direction", "NEUTRAL")
                
                if trend_direction == "BULLISH" or spy_change > 0.05:
                    suggested_strategy = "LONG_CALL"
                    logger.info(f"Strategy fallback: LONG_CALL (trend={trend_direction}, SPY {spy_change:+.2f}%)")
                elif trend_direction == "BEARISH" or spy_change < -0.05:
                    suggested_strategy = "LONG_PUT"
                    logger.info(f"Strategy fallback: LONG_PUT (trend={trend_direction}, SPY {spy_change:+.2f}%)")
                elif self.paper_only_mode:
                    # In paper trading mode, alternate for learning diversity
                    import random
                    suggested_strategy = random.choice(["LONG_CALL", "LONG_PUT"])
                    logger.info(f"Strategy fallback (paper): {suggested_strategy} (neutral market, random for learning)")
                else:
                    suggested_strategy = "NONE"
                    logger.debug(f"No clear momentum (trend={trend_direction}, {spy_change:+.2f}%), no strategy selected")

            # Log decision
            logger.info(
                f"AI Decision: {decision.consensus.value} | "
                f"Confidence: {decision.confidence}% | "
                f"Strategy: {suggested_strategy}"
            )

            # Store decision in database
            await self._store_decision(decision)

            # Emit event
            await self._emit_event("decision_made", {
                "decision": decision.consensus.value,
                "confidence": decision.confidence,
                "strategy": suggested_strategy,
            })

            # Execute if approved (EXECUTE or PAPER_ONLY consensus)
            from ..ai.orchestrator import ConsensusDecision
            should_execute = decision.consensus in (ConsensusDecision.EXECUTE, ConsensusDecision.PAPER_ONLY)

            if should_execute:
                # Check with learning system
                should_exec = self.learning_scheduler.should_execute(
                    decision.confidence / 100.0,  # Convert to 0-1 range
                    {
                        "market": decision.market_response.get("confidence", 0) / 100.0,
                        "risk": decision.risk_response.get("confidence", 0) / 100.0,
                        "execution": decision.execution_response.get("confidence", 0) / 100.0 if decision.execution_response else 0,
                    },
                    suggested_strategy
                )

                # PAPER TRADING: Override learning system blocks to ensure trades execute
                # This allows the bot to generate more training data
                if not should_exec["execute"] and self.settings.paper_trading:
                    logger.info(f"Paper trading override: Ignoring learning system block ({should_exec['reasoning']})")
                    should_exec["execute"] = True

                if should_exec["execute"]:
                    await self._execute_trade(decision, suggested_strategy)
                else:
                    logger.info(f"Learning system blocked trade: {should_exec['reasoning']}")

        except Exception as e:
            error_context = {
                "component": "make_trading_decision",
                "market_data": self.market_data,
                "portfolio_state": portfolio_state if 'portfolio_state' in dir() else None,
                "decisions_today": self.decisions_today,
                "traceback": traceback.format_exc()
            }
            logger.error(f"CRITICAL: Error making trading decision: {e}")
            logger.error(f"Decision error context: {error_context}")
            await self._emit_event("error", {"message": f"Decision error: {e}", "context": error_context})

    async def _make_local_ai_decision(
        self,
        portfolio_state: Dict[str, Any],
        call_option: Dict[str, Any],
        put_option: Dict[str, Any],
    ) -> None:
        """
        Make a trading decision using Local AI (XGBoost model).
        
        This is INSTANT (sub-millisecond) with no API costs.
        The model learns from every trade to improve over time.
        """
        try:
            vix = self.market_data.get("vix", 20)
            
            # Get decision from local AI
            decision = await self.local_ai.get_entry_decision(
                market_data=self.market_data,
                vix=vix,
                call_option=call_option,
                put_option=put_option,
            )
            
            self.last_decision_time = datetime.now()
            self.decisions_today += 1
            
            # Map action to strategy
            strategy_map = {
                "LONG_CALL": "LONG_CALL",
                "LONG_PUT": "LONG_PUT",
                "HOLD": "NONE",
            }
            suggested_strategy = strategy_map.get(decision.action, "NONE")
            
            # Log decision
            logger.info(
                f"Local AI Decision: {decision.action} | "
                f"Confidence: {decision.confidence:.1%} | "
                f"Source: {decision.source} | "
                f"Inference: {decision.inference_time_ms:.2f}ms"
            )
            
            # Store decision in database (create a compatible structure)
            await self._store_local_ai_decision(decision, suggested_strategy)
            
            # Emit event
            await self._emit_event("decision_made", {
                "decision": decision.action,
                "confidence": int(decision.confidence * 100),
                "strategy": suggested_strategy,
                "source": "local_ai",
            })
            
            # Execute if approved
            if decision.action in ("LONG_CALL", "LONG_PUT"):
                # Check trade cooldown - don't spam trades
                if self.last_trade_time:
                    seconds_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds()
                    if seconds_since_last_trade < self.trade_cooldown_seconds:
                        logger.info(f"Trade cooldown: {int(self.trade_cooldown_seconds - seconds_since_last_trade)}s remaining - skipping")
                        return
                
                # Check max position limit BEFORE opening new trade
                current_positions = portfolio_state.get("open_positions", 0)
                max_positions = self.settings.max_concurrent_positions if hasattr(self.settings, 'max_concurrent_positions') else 3
                
                if current_positions >= max_positions:
                    logger.info(f"Position limit reached: {current_positions}/{max_positions} - skipping trade")
                    return
                
                # Create a minimal decision object for _execute_local_trade
                await self._execute_local_trade(decision, suggested_strategy)
                
                # Update last trade time
                self.last_trade_time = datetime.now()
                
        except Exception as e:
            logger.error(f"Local AI decision error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _store_local_ai_decision(self, decision, strategy: str) -> None:
        """Store local AI decision to database"""
        try:
            from ..database import db_session_scope
            from ..database.models import AIDecision
            
            with db_session_scope() as session:
                ai_decision = AIDecision(
                    market_agent_response={"source": decision.source, "local_ai": True},
                    market_confidence=int(decision.confidence * 100),
                    risk_agent_response={"approved": True, "local_ai": True},
                    risk_approval=True,
                    execution_agent_response={"strategy_type": strategy, "strike": decision.suggested_strike},
                    consensus_decision=decision.action,
                    consensus_reasoning=f"[Local AI] {decision.reasoning}",
                    was_executed=decision.action in ("LONG_CALL", "LONG_PUT"),
                    total_tokens_used=0,
                    estimated_cost=0.0,
                )
                session.add(ai_decision)
                session.commit()
                logger.debug(f"Stored local AI decision: {decision.action}")
        except Exception as e:
            logger.debug(f"Could not store local AI decision: {e}")

    def _parse_option_chain(self, raw_chain: Dict, spy_price: float) -> Dict:
        """Parse Tastytrade option chain format into calls/puts lists"""
        result = {"calls": [], "puts": []}
        
        if not raw_chain:
            logger.warning("No raw option chain provided")
            return result
        
        try:
            items = raw_chain.get("items", [])
            if not items:
                items = raw_chain.get("data", {}).get("items", [])
            
            if not items:
                logger.warning("No items in option chain")
                return result
            
            # Get today's date for 0DTE filtering
            from datetime import datetime, date
            import pytz
            est = pytz.timezone("US/Eastern")
            today = datetime.now(est).strftime("%Y-%m-%d")
            
            # Find today's expiration (0DTE) or closest
            expirations = items[0].get("expirations", []) if items else []
            
            target_expiration = None
            for exp in expirations:
                exp_date = exp.get("expiration-date", "")
                if exp_date == today:
                    target_expiration = exp
                    break
            
            # If no 0DTE, use first expiration
            if not target_expiration and expirations:
                target_expiration = expirations[0]
                logger.info(f"No 0DTE, using expiration: {target_expiration.get('expiration-date')}")
            
            if not target_expiration:
                logger.warning("No expiration found in chain")
                return result
            
            strikes = target_expiration.get("strikes", [])
            exp_date = target_expiration.get("expiration-date", today)
            
            # Get market data provider for real-time prices
            market_data_provider = None
            if self.tastytrade_client and hasattr(self.tastytrade_client, 'market_data'):
                market_data_provider = self.tastytrade_client.market_data
            
            # Collect option symbols we need prices for (use OCC format, not streamer format)
            option_symbols_to_subscribe = []
            
            # Filter strikes near ATM (within $5 of current price for better pricing)
            for strike_data in strikes:
                strike_price = float(strike_data.get("strike-price", 0))
                
                # Only include strikes near current price (tighter range)
                if abs(strike_price - spy_price) > 5:
                    continue
                
                # Use OCC symbol format (SPY   260129C00688000) for subscription
                # This is what actually works with DXLink streaming
                call_symbol = strike_data.get("call", "")
                put_symbol = strike_data.get("put", "")
                
                if call_symbol:
                    option_symbols_to_subscribe.append(call_symbol)
                if put_symbol:
                    option_symbols_to_subscribe.append(put_symbol)
            
            # Subscribe to option symbols and wait briefly for prices
            if market_data_provider and option_symbols_to_subscribe:
                try:
                    logger.info(f"Subscribing to {len(option_symbols_to_subscribe)} options: {option_symbols_to_subscribe[:5]}...")
                    
                    # Use add_streaming_symbol which opens the subscription
                    for sym in option_symbols_to_subscribe[:10]:  # Limit to 10 for performance
                        market_data_provider.add_streaming_symbol(sym)
                    
                    import time
                    time.sleep(3)  # Wait for streaming data to arrive
                    
                    # Debug: check what's in cache
                    cache_count = 0
                    for sym in option_symbols_to_subscribe[:5]:
                        cached = market_data_provider.get_cached_quote(sym)
                        if cached:
                            cache_count += 1
                            logger.info(f"Cache hit for {sym}: ${cached.get('mid', cached.get('last', 0)):.2f}")
                        else:
                            logger.debug(f"Cache miss for {sym}")
                    logger.info(f"Cache hits: {cache_count}/{len(option_symbols_to_subscribe[:5])} for subscribed options")
                except Exception as e:
                    logger.debug(f"Could not subscribe to options: {e}")
            
            # Now parse with real prices
            for strike_data in strikes:
                strike_price = float(strike_data.get("strike-price", 0))
                
                # Only include strikes near current price
                if abs(strike_price - spy_price) > 5:
                    continue
                
                # Get call option data
                call_symbol = strike_data.get("call", "")
                call_streamer = strike_data.get("call-streamer-symbol", "")
                
                if call_symbol:
                    # Get real price from streaming cache - try multiple formats
                    call_price = None
                    if market_data_provider:
                        # Try streamer symbol first
                        cached = market_data_provider.get_cached_quote(call_streamer)
                        if not cached:
                            # Try OCC format (without dots, with spaces)
                            occ_format = call_symbol  # This is already OCC format like "SPY   260129C00688000"
                            cached = market_data_provider.get_cached_quote(occ_format)
                        if not cached:
                            # Try stripped OCC (no spaces)
                            stripped = call_symbol.replace(" ", "")
                            cached = market_data_provider.get_cached_quote(stripped)
                        
                        if cached:
                            call_price = cached.get("mid") or cached.get("mark") or cached.get("last")
                            if not call_price and cached.get("bid") and cached.get("ask"):
                                call_price = (cached.get("bid", 0) + cached.get("ask", 0)) / 2
                    
                    if not call_price or call_price <= 0:
                        # Skip options without valid prices
                        continue
                    
                    result["calls"].append({
                        "symbol": call_symbol,
                        "streamer_symbol": call_streamer,
                        "strike": strike_price,
                        "expiration": exp_date,
                        "ask": cached.get("ask", call_price * 1.02) if cached else call_price * 1.02,
                        "bid": cached.get("bid", call_price * 0.98) if cached else call_price * 0.98,
                        "mid": call_price,
                        "delta": 0.5 if strike_price <= spy_price else 0.3,
                    })
                
                # Get put option data
                put_symbol = strike_data.get("put", "")
                put_streamer = strike_data.get("put-streamer-symbol", "")
                
                if put_symbol:
                    # Get real price from streaming cache - try multiple formats
                    put_price = None
                    if market_data_provider:
                        # Try streamer symbol first
                        cached = market_data_provider.get_cached_quote(put_streamer)
                        if not cached:
                            # Try OCC format
                            cached = market_data_provider.get_cached_quote(put_symbol)
                        if not cached:
                            # Try stripped OCC (no spaces)
                            stripped = put_symbol.replace(" ", "")
                            cached = market_data_provider.get_cached_quote(stripped)
                        
                        if cached:
                            put_price = cached.get("mid") or cached.get("mark") or cached.get("last")
                            if not put_price and cached.get("bid") and cached.get("ask"):
                                put_price = (cached.get("bid", 0) + cached.get("ask", 0)) / 2
                    
                    if not put_price or put_price <= 0:
                        # Skip options without valid prices
                        continue
                    
                    result["puts"].append({
                        "symbol": put_symbol,
                        "streamer_symbol": put_streamer,
                        "strike": strike_price,
                        "expiration": exp_date,
                        "ask": cached.get("ask", put_price * 1.02) if cached else put_price * 1.02,
                        "bid": cached.get("bid", put_price * 0.98) if cached else put_price * 0.98,
                        "mid": put_price,
                        "delta": -0.5 if strike_price >= spy_price else -0.3,
                    })
            
            logger.info(f"Parsed option chain: {len(result['calls'])} calls, {len(result['puts'])} puts near ${spy_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error parsing option chain: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result

    async def _execute_local_trade(self, decision, strategy_name: str) -> None:
        """Execute a trade based on Local AI decision"""
        try:
            # Get strategy
            strategy = self.strategy_selector.get_strategy(strategy_name)
            if not strategy:
                logger.error(f"Unknown strategy: {strategy_name}")
                return

            spy_price = self.market_data.get("spy_price", 595.0)
            vix = self.market_data.get("vix", 15.0)

            # Get option chain
            raw_chain = None
            if self.tastytrade_client and self.tastytrade_client.is_connected:
                try:
                    raw_chain = await asyncio.to_thread(
                        self.tastytrade_client.market_data.get_option_chain, "SPY"
                    )
                except Exception as e:
                    logger.warning(f"Could not get option chain: {e}")

            # Parse option chain into calls/puts format
            option_chain = self._parse_option_chain(raw_chain, spy_price)

            # Set max option price based on account size
            # This prevents selecting options that are too expensive for the account
            account_balance = self.settings.starting_capital
            risk_percent = self.settings.risk_per_trade_percent
            max_affordable_option = (account_balance * risk_percent) / 100  # Max per contract
            # Clamp between $0.50 (minimum for reasonable trades) and $10.00 (cap)
            strategy.max_option_price = max(0.50, min(10.00, max_affordable_option / 100))
            logger.debug(
                f"Option price filter: ${strategy.min_option_price:.2f}-${strategy.max_option_price:.2f} "
                f"(based on ${account_balance:.0f} account, {risk_percent:.0%} risk)"
            )

            # STEP 1: Calculate position with 1 contract to get option price
            test_position = strategy.calculate_position(
                spy_price=spy_price,
                option_chain=option_chain,
                contracts=1,
            )

            if not test_position:
                logger.warning("Strategy returned no position")
                return

            # STEP 2: Use PositionSizer to calculate optimal contract count
            # OTM options are cheaper, so we can afford more contracts
            from snuffs_bot.paper_trading.position_sizing import PositionSizer

            option_price = test_position.entry_debit  # Premium per share (e.g., $1.50)
            if option_price <= 0:
                option_price = spy_price * 0.003  # Fallback: ~0.3% of underlying

            position_sizer = PositionSizer(
                default_risk_percent=self.settings.risk_per_trade_percent * 100,  # Convert to percentage
                max_risk_percent=15.0,  # Allow up to 15% of account per trade
                min_risk_percent=2.0,   # At least 2%
            )

            account_balance = self.settings.starting_capital
            sizing_result = position_sizer.calculate_position_size(
                account_balance=account_balance,
                option_price=option_price,
                max_contracts=10,  # Cap at 10 contracts for risk management
            )

            contracts = sizing_result.contracts if sizing_result.can_afford else 1
            contracts = max(1, contracts)  # Ensure at least 1 contract

            # Log position sizing decision
            cost_per_contract = option_price * 100
            logger.info(
                f"📊 Position sizing: ${option_price:.2f}/share = ${cost_per_contract:.2f}/contract | "
                f"Account: ${account_balance:.2f} | Contracts: {contracts} | "
                f"Total cost: ${contracts * cost_per_contract:.2f} ({sizing_result.percent_of_account:.1f}% of account)"
            )

            # STEP 3: Recalculate position with proper contract count (if > 1)
            if contracts > 1:
                position = strategy.calculate_position(
                    spy_price=spy_price,
                    option_chain=option_chain,
                    contracts=contracts,
                )
                if not position:
                    logger.warning("Could not calculate position with multiple contracts, using 1")
                    position = test_position
            else:
                position = test_position

            # Execute through paper trading coordinator
            spy_price = self.market_data.get("spy_price", 595.0)
            vix = self.market_data.get("vix", 15.0)
            
            result = await asyncio.to_thread(
                self.execution_coordinator.execute_trade,
                spread_position=position,
                spy_price=spy_price,
                vix=vix,
                ai_decision_id=f"local_ai_{int(time.time())}",
            )

            if result.get("paper_success"):
                trade_id = result.get("paper_position_id", "unknown")
                entry_price = result.get("paper_fill_price", 0) or (position.entry_price if hasattr(position, 'entry_price') else 0)
                strike = position.legs[0].strike if position.legs else 0
                
                # Record trade entry in local AI for learning (marked as local AI decision)
                self.local_ai.record_trade_entry(
                    trade_id=trade_id,
                    strategy=strategy_name,
                    strike=strike,
                    entry_price=abs(entry_price) if entry_price else 1.0,
                    contracts=position.contracts if hasattr(position, 'contracts') else 1,
                    is_local_ai=True,  # This trade was decided by local AI
                )
                
                logger.success(f"✅ Local AI paper trade opened: {strategy_name} | Strike ${strike:.0f} | Entry ${abs(entry_price) if entry_price else 1.0:.2f}")
                
                # Position monitoring is handled by the execution coordinator
            else:
                logger.warning(f"Trade execution failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Local trade execution error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _execute_trade(self, decision, strategy_name: str) -> None:
        """Execute a trade based on AI decision"""
        try:
            # Get strategy
            strategy = self.strategy_selector.get_strategy(strategy_name)
            if not strategy:
                logger.error(f"Unknown strategy: {strategy_name}")
                return

            # Build the spread position using calculate_position
            spy_price = self.market_data.get("spy_price", 478.0)
            vix = self.market_data.get("vix", 15.0)

            # Try to get real option chain from Tastytrade, fallback to simulated
            option_chain = None
            if self.tastytrade_client and self.tastytrade_client.is_connected:
                try:
                    option_chain = await asyncio.to_thread(
                        self.tastytrade_client.market_data.get_option_chain, "SPY"
                    )
                    if option_chain and option_chain.get("calls"):
                        logger.info(f"Using REAL option chain with {len(option_chain.get('calls', []))} strikes")
                except Exception as e:
                    logger.warning(f"Could not get real option chain: {e}")

            # Fallback to simulated if real chain unavailable
            if not option_chain or not option_chain.get("calls"):
                logger.info("Using simulated option chain")
                option_chain = self._generate_simulated_option_chain(spy_price, vix)

            # Calculate contracts based on capital and risk
            capital = self.settings.starting_capital
            risk_amount = capital * self.settings.risk_per_trade_percent
            estimated_premium = spy_price * 0.003  # ~0.3% of underlying
            contracts = max(1, int(risk_amount / (estimated_premium * 100)))

            spread_position = await asyncio.to_thread(
                strategy.calculate_position,
                spy_price=spy_price,
                option_chain=option_chain,
                contracts=contracts,
            )

            if not spread_position:
                logger.warning("Could not build spread position")
                return

            # CRITICAL: Get real streaming price for the option symbol
            # This ensures entry price matches what we'll use for exits
            if spread_position.legs and self.tastytrade_client and self.tastytrade_client.is_connected:
                for leg in spread_position.legs:
                    option_symbol = leg.symbol
                    real_quote = self.tastytrade_client.market_data.get_option_quote_with_wait(option_symbol, max_wait=2.0)
                    
                    if real_quote and real_quote.get("mark", 0) > 0:
                        real_price = real_quote.get("mark", leg.fill_price)
                        old_price = leg.fill_price
                        leg.fill_price = real_price
                        logger.info(f"[REAL PRICE] Updated {option_symbol}: ${old_price:.2f} -> ${real_price:.2f} (TASTYTRADE)")
                    else:
                        logger.warning(f"[REAL PRICE] Could not get streaming price for {option_symbol}, using chain price ${leg.fill_price:.2f}")
                
                # Update position entry_debit with real prices
                total_debit = sum(leg.fill_price * leg.quantity for leg in spread_position.legs)
                spread_position.entry_debit = total_debit
                spread_position.max_loss = total_debit * 100  # Per contract

            # Execute through coordinator
            result = await asyncio.to_thread(
                self.execution_coordinator.execute_trade,
                spread_position=spread_position,
                spy_price=self.market_data.get("spy_price", 478.0),
                vix=self.market_data.get("vix", 15.0),
                ai_decision_id=decision.decision_id,
            )

            if result.get("paper_success"):
                self.trades_today += 1
                position_id = result.get("paper_position_id")
                entry_price = spread_position.entry_credit or spread_position.entry_debit or 0
                strike = spread_position.legs[0].strike if spread_position.legs else 0
                
                logger.success(
                    f"Trade executed: {spread_position.strategy_type} | "
                    f"Entry: ${entry_price:.2f}"
                )
                
                # Record trade entry in local AI for XGBoost training
                if self.local_ai:
                    try:
                        self.local_ai.record_trade_entry(
                            trade_id=position_id,
                            strategy=spread_position.strategy_type,
                            strike=strike,
                            entry_price=entry_price,
                            contracts=spread_position.contracts or 1,
                            is_local_ai=False,  # This trade was decided by Claude AI, not local AI
                        )
                        logger.info(f"Local AI: Recorded trade entry {position_id} | {spread_position.strategy_type} @ ${entry_price:.2f}")
                    except Exception as e:
                        logger.debug(f"Local AI trade entry recording: {e}")

                # Register with real-time monitor for instant exit decisions
                if self.realtime_monitor and position_id:
                    from snuffs_bot.paper_trading.realtime_monitor import DynamicExitMode
                    self.realtime_monitor.register_position(
                        position_id=position_id,
                        symbol=f"SPY_{spread_position.strategy_type}",
                        entry_price=spread_position.entry_credit,
                        contracts=spread_position.contracts,
                        strategy_type=spread_position.strategy_type,
                        mode=DynamicExitMode.BALANCED,
                    )
                    logger.info(f"📡 Position {position_id} registered for real-time monitoring")

                await self._emit_event("trade_executed", {
                    "strategy": spread_position.strategy_type,
                    "entry_credit": spread_position.entry_credit,
                    "position_id": position_id,
                })

        except Exception as e:
            error_context = {
                "component": "execute_trade",
                "strategy": strategy_name,
                "decision_id": decision.decision_id if hasattr(decision, 'decision_id') else None,
                "market_data": self.market_data,
                "traceback": traceback.format_exc()
            }
            logger.error(f"CRITICAL: Error executing trade: {e}")
            logger.error(f"Trade execution error context: {error_context}")
            await self._emit_event("error", {"message": f"Trade execution error: {e}", "context": error_context})

    async def _check_positions(self) -> None:
        """Check open positions for exit conditions"""
        if not self.execution_coordinator:
            return

        try:
            # Get current prices and check exits
            result = self.execution_coordinator.check_exits(
                current_spy_price=self.market_data.get("spy_price", 478.0),
                current_vix=self.market_data.get("vix", 15.0),
            )

            # Process any closed positions through learning system
            for closed in result.get("paper_closed", []):
                await self._process_closed_position(closed)

        except Exception as e:
            logger.error(f"Error checking positions: {e}")

    async def _process_closed_position(self, position_data: Dict[str, Any]) -> None:
        """Process a closed position through learning system"""
        try:
            trade_id = position_data.get("position_id", "")
            strategy = position_data.get("strategy_type", "")
            entry_price = position_data.get("entry_price", 0)
            exit_price = position_data.get("exit_price", 0)
            pnl = position_data.get("realized_pnl", 0)
            exit_reason = position_data.get("exit_reason", "UNKNOWN")
            
            # Record trade exit in local AI for XGBoost training
            if self.local_ai:
                try:
                    # Calculate P&L percent from price change (not dollar PnL)
                    pnl_percent = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    self.local_ai.record_trade_exit(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        exit_reason=exit_reason,
                        max_profit=position_data.get("max_profit", 0),
                        max_loss=position_data.get("max_loss", 0),
                    )
                    logger.info(f"Local AI: Recorded trade exit {trade_id} | P&L: ${pnl:.2f} ({pnl_percent:+.1f}%) | Reason: {exit_reason}")
                except Exception as e:
                    logger.debug(f"Local AI trade exit recording: {e}")
            
            # Send to learning scheduler
            await self.learning_scheduler.process_trade_completion(
                trade_id=trade_id,
                strategy=strategy,
                entry_time=position_data.get("entry_time", datetime.now()),
                exit_time=position_data.get("exit_time", datetime.now()),
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=exit_reason,
                market_data=self.market_data,
                ai_confidence=position_data.get("ai_confidence", 0.7),
            )

        except Exception as e:
            logger.error(f"Error processing closed position: {e}")

    async def _store_decision(self, decision) -> None:
        """Store AI decision in database"""
        try:
            # Build response JSONs for storage
            market_response = decision.market_response or {}
            risk_response = decision.risk_response or {}
            execution_response = decision.execution_response or {}

            # Extract reasoning for consensus
            reasoning_parts = []
            if market_response.get("reasoning"):
                reasoning_parts.append(f"Market: {market_response['reasoning']}")
            if risk_response.get("reasoning"):
                reasoning_parts.append(f"Risk: {risk_response['reasoning']}")
            consensus_reasoning = " | ".join(reasoning_parts) if reasoning_parts else None

            with db_session_scope() as session:
                db_decision = AIDecision(
                    decision_time=datetime.now(),
                    market_agent_response=market_response,
                    market_confidence=int(market_response.get("confidence", 0)),
                    risk_agent_response=risk_response,
                    risk_approval=risk_response.get("decision") == "APPROVE",
                    execution_agent_response=execution_response,
                    consensus_decision=decision.consensus.value,
                    consensus_reasoning=consensus_reasoning,
                    was_executed=False,
                )
                session.add(db_decision)
                logger.info(f"Stored AI decision: {decision.consensus.value}")
        except Exception as e:
            logger.error(f"Error storing decision: {e}")

    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        if self.execution_coordinator:
            perf = self.execution_coordinator.get_paper_performance()
            portfolio = perf.get("portfolio", {})
            return {
                "account_value": portfolio.get("total_value", self.settings.starting_capital),
                "daily_pnl": portfolio.get("daily_pnl", 0),
                "open_positions": portfolio.get("open_positions", 0),
                "buying_power": portfolio.get("buying_power", self.settings.starting_capital),
                "total_exposure": portfolio.get("total_exposure", 0),
            }
        return {
            "account_value": self.settings.starting_capital,
            "daily_pnl": 0,
            "open_positions": 0,
            "buying_power": self.settings.starting_capital,
            "total_exposure": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "state": self.state.value,
            "trading_enabled": self.trading_enabled,
            "paper_only_mode": self.paper_only_mode,
            "is_market_hours": self.is_market_hours(),
            "is_trading_hours": self.is_trading_hours(),
            "decisions_today": self.decisions_today,
            "trades_today": self.trades_today,
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None,
            "market_data": {
                "spy_price": self.market_data.get("spy_price", 0),
                "vix": self.market_data.get("vix", 0),
            },
            "portfolio": self._get_portfolio_state(),
        }

    def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if self.execution_coordinator:
            return self.execution_coordinator.get_paper_performance()
        return {}

    def get_learning_report(self) -> Dict[str, Any]:
        """Get learning system report"""
        if self.learning_scheduler:
            return self.learning_scheduler.get_learning_report()
        return {}


async def run_engine():
    """Run the trading engine"""
    engine = TradingEngine()

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Start engine
    await engine.start()

    # Wait for shutdown
    while engine.state == EngineState.RUNNING:
        await asyncio.sleep(1)

    logger.info("Engine shutdown complete")


if __name__ == "__main__":
    asyncio.run(run_engine())
