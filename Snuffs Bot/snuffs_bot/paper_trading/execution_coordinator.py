"""
Execution Coordinator

Coordinates parallel execution of paper and live trades.
Allows running paper trades alongside live trades for
continuous learning and strategy validation.
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from .simulator import PaperTradingSimulator, SimulatorConfig
from .virtual_portfolio import VirtualPosition
from ..strategies.zero_dte.base_zero_dte import SpreadPosition
from ..ai.orchestrator import TradingDecision, ConsensusDecision
from ..config.settings import get_settings
from ..database.connection import db_session_scope
from ..database.models import Trade


class ExecutionMode(Enum):
    """Trading execution modes"""
    PAPER_ONLY = "PAPER_ONLY"        # Only paper trading (learning mode)
    LIVE_ONLY = "LIVE_ONLY"          # Only live trading (no paper)
    PARALLEL = "PARALLEL"            # Both paper and live
    SHADOW = "SHADOW"                # Paper shadows live decisions


@dataclass
class ExecutionResult:
    """Result of executing a trade decision"""
    decision_id: str
    execution_mode: ExecutionMode
    timestamp: datetime = field(default_factory=datetime.now)

    # Paper trading result
    paper_executed: bool = False
    paper_position_id: Optional[str] = None
    paper_fill_price: Optional[float] = None
    paper_slippage: Optional[float] = None

    # Live trading result
    live_executed: bool = False
    live_order_id: Optional[str] = None
    live_fill_price: Optional[float] = None

    # Comparison
    price_difference: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "mode": self.execution_mode.value,
            "paper_executed": self.paper_executed,
            "paper_position_id": self.paper_position_id,
            "paper_fill_price": self.paper_fill_price,
            "live_executed": self.live_executed,
            "live_fill_price": self.live_fill_price,
            "price_difference": self.price_difference,
            "timestamp": self.timestamp.isoformat(),
        }


class ExecutionCoordinator:
    """
    Coordinates paper and live trade execution

    Features:
    - Run paper trades in parallel with live trades
    - Compare paper vs live performance
    - Shadow mode: paper trades mirror live decisions
    - Automatic paper trade percentage
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER_ONLY,
        paper_trade_ratio: float = 0.5,
        live_client=None,  # Optional Tastytrade client for live trading
        market_data_provider=None  # Optional market data provider for real option prices
    ):
        """
        Initialize execution coordinator

        Args:
            mode: Execution mode
            paper_trade_ratio: Ratio of trades to run as paper (0.0-1.0)
            live_client: Optional broker client for live trades
            market_data_provider: Optional market data provider for real option prices
        """
        self.settings = get_settings()
        self.mode = mode
        self.paper_trade_ratio = paper_trade_ratio
        self.live_client = live_client
        self.market_data_provider = market_data_provider

        # Initialize paper trading simulator with capital from settings
        starting_capital = self.settings.starting_capital
        simulator_config = SimulatorConfig(
            starting_capital=starting_capital,
            realistic_fills=True,
            auto_manage_exits=True,
            save_to_database=True
        )
        self.paper_simulator = PaperTradingSimulator(simulator_config)

        # Execution history
        self.execution_history: List[ExecutionResult] = []

        # Counters
        self.total_decisions = 0
        self.paper_executions = 0
        self.live_executions = 0

        logger.info(
            f"Execution Coordinator initialized: mode={mode.value}, "
            f"paper_ratio={paper_trade_ratio}, capital=${starting_capital:,.0f}"
        )

    def _update_all_positions_in_db(self, positions_prices: Dict[str, float]) -> None:
        """Update all position prices in database for dashboard visibility
        
        Args:
            positions_prices: Dict mapping position order (by entry time) to current price
        """
        try:
            with db_session_scope() as session:
                # Get all open paper trades ordered by entry time
                open_trades = session.query(Trade).filter(
                    Trade.trade_type == "PAPER",
                    Trade.status == "OPEN"
                ).order_by(Trade.entry_time.asc()).all()

                for idx, trade in enumerate(open_trades):
                    if idx < len(positions_prices):
                        current_price = list(positions_prices.values())[idx] if isinstance(positions_prices, dict) else 0
                        if current_price > 0:
                            trade.exit_price = abs(current_price)
                            entry = float(trade.entry_price or 0)
                            if entry > 0:
                                trade.pnl = (current_price - entry) * 100  # Per contract
                
                if open_trades:
                    logger.debug(f"Updated {len(open_trades)} position prices in DB")
        except Exception as e:
            logger.debug(f"Could not update position prices in DB: {e}")

    def update_all_db_positions_from_streaming(self) -> int:
        """
        Update ALL open database positions using real streaming option prices.
        This ensures dashboard shows correct prices even if positions aren't in memory.
        
        Returns:
            Number of positions updated
        """
        if not self.market_data_provider:
            return 0
            
        updated = 0
        try:
            with db_session_scope() as session:
                open_trades = session.query(Trade).filter(
                    Trade.trade_type == "PAPER",
                    Trade.status == "OPEN"
                ).all()
                
                for trade in open_trades:
                    # Get the option symbol from entry_legs
                    if trade.entry_legs and len(trade.entry_legs) > 0:
                        option_symbol = trade.entry_legs[0].get("symbol", "")
                        if option_symbol:
                            # Get real streaming price
                            real_price = self._get_option_price(option_symbol)
                            if real_price > 0:
                                trade.exit_price = real_price
                                entry = float(trade.entry_price or 0)
                                contracts = int(trade.position_size or 1)
                                if entry > 0:
                                    # P&L = (current - entry) * 100 * contracts
                                    trade.pnl = (real_price - entry) * 100 * contracts
                                updated += 1
                                logger.debug(f"Updated trade {trade.id} ({option_symbol}): ${real_price:.2f}")
                
                if updated > 0:
                    logger.info(f"Updated {updated} DB positions with streaming prices")
                    
        except Exception as e:
            logger.debug(f"Could not update DB positions from streaming: {e}")
            
        return updated

    def _update_position_price_in_db(self, position_id: str, current_price: float) -> None:
        """Update position current price in database for dashboard visibility"""
        try:
            with db_session_scope() as session:
                # Find ALL open trades and update the one that matches by order
                open_trades = session.query(Trade).filter(
                    Trade.trade_type == "PAPER",
                    Trade.status == "OPEN"
                ).order_by(Trade.entry_time.asc()).all()

                # Get all open positions from portfolio to match order
                portfolio = self.paper_simulator.portfolio
                pos_ids = list(portfolio.positions.keys())
                
                # Find the index of this position_id
                try:
                    pos_idx = pos_ids.index(position_id)
                    if pos_idx < len(open_trades):
                        trade = open_trades[pos_idx]
                        trade.exit_price = abs(current_price)
                        entry = float(trade.entry_price or 0)
                        contracts = int(trade.position_size or 1)
                        if entry > 0:
                            trade.pnl = (current_price - entry) * 100 * contracts
                        logger.debug(f"Updated DB position {pos_idx}: ${current_price:.2f}")
                except (ValueError, IndexError):
                    # Position not found in list, update most recent
                    if open_trades:
                        trade = open_trades[-1]
                        trade.exit_price = abs(current_price)
                        entry = float(trade.entry_price or 0)
                        contracts = int(trade.position_size or 1)
                        if entry > 0:
                            trade.pnl = (current_price - entry) * 100 * contracts
        except Exception as e:
            logger.debug(f"Could not update position price in DB: {e}")

    async def start(self) -> None:
        """Start the execution coordinator"""
        await self.paper_simulator.start()
        logger.info("Execution coordinator started")

    async def stop(self) -> None:
        """Stop the execution coordinator"""
        await self.paper_simulator.stop()
        logger.info("Execution coordinator stopped")

    async def execute_decision(
        self,
        decision: TradingDecision,
        spread_position: SpreadPosition,
        market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a trading decision based on mode

        Args:
            decision: AI trading decision
            spread_position: Position to execute
            market_data: Current market data

        Returns:
            ExecutionResult with paper and/or live results
        """
        self.total_decisions += 1

        result = ExecutionResult(
            decision_id=decision.decision_id,
            execution_mode=self.mode
        )

        # Determine what to execute based on mode and decision
        should_paper, should_live = self._determine_execution(decision)

        # Execute paper trade
        if should_paper:
            paper_result = await self._execute_paper(
                decision, spread_position, market_data
            )
            result.paper_executed = paper_result is not None
            if paper_result:
                result.paper_position_id = paper_result.position_id
                result.paper_fill_price = paper_result.entry_price
                self.paper_executions += 1

        # Execute live trade
        if should_live and self.live_client:
            live_result = await self._execute_live(
                decision, spread_position, market_data
            )
            result.live_executed = live_result.get("success", False)
            if result.live_executed:
                result.live_order_id = live_result.get("order_id")
                result.live_fill_price = live_result.get("fill_price")
                self.live_executions += 1

        # Calculate price difference if both executed
        if result.paper_fill_price and result.live_fill_price:
            result.price_difference = result.paper_fill_price - result.live_fill_price

        self.execution_history.append(result)

        logger.info(
            f"Decision {decision.decision_id} executed: "
            f"paper={result.paper_executed}, live={result.live_executed}"
        )

        return result

    def _determine_execution(
        self,
        decision: TradingDecision
    ) -> Tuple[bool, bool]:
        """
        Determine what should be executed based on mode and decision

        Returns:
            Tuple of (should_paper, should_live)
        """
        if decision.consensus == ConsensusDecision.REJECT:
            return False, False

        if self.mode == ExecutionMode.PAPER_ONLY:
            return True, False

        if self.mode == ExecutionMode.LIVE_ONLY:
            return False, True

        if self.mode == ExecutionMode.SHADOW:
            # Paper shadows all live trades
            return True, True

        if self.mode == ExecutionMode.PARALLEL:
            # Use paper ratio to decide
            import random

            # EXECUTE = both paper and live
            if decision.consensus == ConsensusDecision.EXECUTE:
                should_live = True
                should_paper = random.random() < self.paper_trade_ratio
                return should_paper, should_live

            # PAPER_ONLY = only paper
            if decision.consensus == ConsensusDecision.PAPER_ONLY:
                return True, False

        return False, False

    async def _execute_paper(
        self,
        decision: TradingDecision,
        spread_position: SpreadPosition,
        market_data: Dict[str, Any]
    ) -> Optional[VirtualPosition]:
        """Execute paper trade"""
        try:
            position = await self.paper_simulator.execute_trade(
                spread_position=spread_position,
                market_data=market_data,
                ai_decision_id=decision.decision_id
            )
            return position

        except Exception as e:
            logger.error(f"Paper trade execution failed: {e}")
            return None

    async def _execute_live(
        self,
        decision: TradingDecision,
        spread_position: SpreadPosition,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute live trade through broker

        Returns dict with execution result
        """
        if not self.live_client:
            logger.warning("No live client configured")
            return {"success": False, "error": "No live client"}

        try:
            # This would integrate with the actual Tastytrade client
            # For now, return placeholder
            logger.warning("Live trading not yet implemented")
            return {"success": False, "error": "Not implemented"}

        except Exception as e:
            logger.error(f"Live trade execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def close_position(
        self,
        position_id: str,
        market_data: Dict[str, Any],
        reason: str = "MANUAL"
    ) -> Dict[str, Any]:
        """Close a position (paper or live based on ID prefix)"""
        if position_id.startswith("paper_"):
            return await self._close_paper_position(position_id, market_data, reason)
        else:
            return await self._close_live_position(position_id, reason)

    async def _close_paper_position(
        self,
        position_id: str,
        market_data: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """Close a paper position"""
        closed = await self.paper_simulator.close_trade(
            position_id=position_id,
            market_data=market_data,
            reason=reason
        )

        if closed:
            return {
                "success": True,
                "position_id": position_id,
                "pnl": closed.realized_pnl,
                "reason": reason
            }
        return {"success": False, "error": "Position not found"}

    async def _close_live_position(
        self,
        position_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """Close a live position"""
        # Would integrate with Tastytrade client
        logger.warning("Live position closing not yet implemented")
        return {"success": False, "error": "Not implemented"}

    def get_paper_performance(self) -> Dict[str, Any]:
        """Get paper trading performance metrics"""
        return self.paper_simulator.get_summary()

    def get_execution_comparison(self) -> Dict[str, Any]:
        """Compare paper vs live execution performance"""
        parallel_executions = [
            e for e in self.execution_history
            if e.paper_executed and e.live_executed
        ]

        if not parallel_executions:
            return {"parallel_trades": 0, "comparison": None}

        price_diffs = [
            e.price_difference for e in parallel_executions
            if e.price_difference is not None
        ]

        avg_diff = sum(price_diffs) / len(price_diffs) if price_diffs else 0

        return {
            "parallel_trades": len(parallel_executions),
            "avg_price_difference": avg_diff,
            "paper_better": sum(1 for d in price_diffs if d > 0),
            "live_better": sum(1 for d in price_diffs if d < 0),
            "equal": sum(1 for d in price_diffs if d == 0),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get complete execution summary"""
        return {
            "mode": self.mode.value,
            "paper_trade_ratio": self.paper_trade_ratio,
            "total_decisions": self.total_decisions,
            "paper_executions": self.paper_executions,
            "live_executions": self.live_executions,
            "paper_performance": self.get_paper_performance(),
            "execution_comparison": self.get_execution_comparison(),
        }

    def set_mode(self, mode: ExecutionMode) -> None:
        """Change execution mode"""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Execution mode changed: {old_mode.value} -> {mode.value}")

    def set_paper_ratio(self, ratio: float) -> None:
        """Change paper trade ratio"""
        self.paper_trade_ratio = max(0.0, min(1.0, ratio))
        logger.info(f"Paper trade ratio set to {self.paper_trade_ratio}")

    def set_market_data_provider(self, provider) -> None:
        """Set the market data provider for real option prices"""
        self.market_data_provider = provider
        logger.info("Market data provider set for real option price updates")

    def _get_option_price(self, symbol: str) -> float:
        """
        Get current option price from market data provider
        
        Args:
            symbol: Option symbol (e.g., 'SPY260127C00706000')
            
        Returns:
            Current option price or 0.0 if unavailable
        """
        if not self.market_data_provider:
            logger.warning(f"No market data provider for option price lookup: {symbol}")
            return 0.0
            
        try:
            # Check streaming health and reconnect if needed
            if hasattr(self.market_data_provider, 'check_streaming_health'):
                self.market_data_provider.check_streaming_health()
            
            # First, try to add this symbol to streaming if not already
            if hasattr(self.market_data_provider, 'add_streaming_symbol'):
                self.market_data_provider.add_streaming_symbol(symbol)
            
            # Try to get quote from streaming cache first
            quote = self.market_data_provider.get_quote(symbol)
            if quote:
                # Use mark price (mid of bid/ask) or last price
                mark = quote.get("mark", 0)
                if mark > 0:
                    logger.info(f"[OPTION PRICE] {symbol}: ${mark:.2f} (streaming)")
                    return mark
                last = quote.get("last", 0)
                if last > 0:
                    logger.info(f"[OPTION PRICE] {symbol}: ${last:.2f} (streaming last)")
                    return last
                # Calculate from bid/ask
                bid = quote.get("bid", 0)
                ask = quote.get("ask", 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    logger.info(f"[OPTION PRICE] {symbol}: ${mid:.2f} (streaming bid/ask)")
                    return mid
            
            # Fallback to REST API if streaming didn't work
            if hasattr(self.market_data_provider, 'get_option_quote_rest'):
                rest_quote = self.market_data_provider.get_option_quote_rest(symbol)
                if rest_quote:
                    mark = rest_quote.get("mark", 0)
                    if mark > 0:
                        logger.info(f"[OPTION PRICE] {symbol}: ${mark:.2f} (REST API)")
                        return mark
            
            logger.debug(f"No quote data available for option: {symbol}")
        except Exception as e:
            logger.error(f"Error getting option price for {symbol}: {e}")
            
        return 0.0

    def _subscribe_option_symbols(self, position) -> None:
        """Subscribe to option symbols for a position to get real-time prices"""
        if not self.market_data_provider or not position.legs:
            return
            
        for leg in position.legs:
            symbol = leg.get("symbol")
            if symbol and hasattr(self.market_data_provider, 'add_streaming_symbol'):
                self.market_data_provider.add_streaming_symbol(symbol)
                logger.info(f"Subscribed to streaming for option: {symbol}")

    def execute_trade(
        self,
        spread_position: SpreadPosition,
        spy_price: float,
        vix: float,
        ai_decision_id: str = ""
    ) -> Dict[str, Any]:
        """
        Synchronous trade execution for compatibility with trading engine

        Args:
            spread_position: Position to execute
            spy_price: Current SPY price
            vix: Current VIX
            ai_decision_id: Associated AI decision ID

        Returns:
            Dict with execution results
        """
        try:
            # Execute paper trade synchronously via the portfolio
            if self.mode in [ExecutionMode.PAPER_ONLY, ExecutionMode.PARALLEL, ExecutionMode.SHADOW]:
                position = self.paper_simulator.portfolio.open_position(
                    spread_position=spread_position,
                    spy_price=spy_price,
                    vix=vix,
                    ai_decision_id=ai_decision_id
                )
                self.paper_executions += 1

                # Subscribe to real-time option prices for this position
                if position:
                    self._subscribe_option_symbols(position)

                # Save trade to database for dashboard visibility
                if position and self.paper_simulator.config.save_to_database:
                    self.paper_simulator._save_trade_to_db(position, "OPEN")
                    logger.info(f"Paper trade saved to database: {position.position_id}")

                return {
                    "paper_success": True,
                    "paper_position_id": position.position_id if position else None,
                    "paper_fill_price": position.entry_price if position else None,
                    "live_success": False,
                }

            return {"paper_success": False, "live_success": False}

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"paper_success": False, "live_success": False, "error": str(e)}

    # Track SPY price history for momentum calculation
    _spy_price_history: List[Tuple[datetime, float]] = []
    _last_spy_price: float = 0.0

    def _calculate_spy_momentum(self, current_spy_price: float) -> Tuple[str, float]:
        """
        Calculate SPY trend and momentum from recent price history
        
        Returns:
            Tuple of (trend: 'BULLISH'|'BEARISH'|'NEUTRAL', momentum: % change)
        """
        now = datetime.now()
        
        # Add current price to history
        self._spy_price_history.append((now, current_spy_price))
        
        # Keep only last 60 seconds of history
        cutoff = now.timestamp() - 60
        self._spy_price_history = [(t, p) for t, p in self._spy_price_history if t.timestamp() > cutoff]
        
        # Need at least 10 seconds of data
        if len(self._spy_price_history) < 2:
            return "NEUTRAL", 0.0
        
        # Calculate momentum over last 30 seconds (or available history)
        first_price = self._spy_price_history[0][1]
        momentum = ((current_spy_price - first_price) / first_price) * 100 if first_price > 0 else 0.0
        
        # Determine trend based on momentum - use wider thresholds for learning mode
        # 0.05% is a more meaningful move (not just noise)
        if momentum > 0.05:  # Up more than 0.05%
            trend = "BULLISH"
        elif momentum < -0.05:  # Down more than 0.05%
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        return trend, momentum

    def check_exits(
        self,
        current_spy_price: float,
        current_vix: float
    ) -> Dict[str, Any]:
        """
        Check all open positions for exit conditions

        Args:
            current_spy_price: Current SPY price
            current_vix: Current VIX level

        Returns:
            Dict with lists of closed positions
        """
        result: Dict[str, List[Any]] = {
            "paper_closed": [],
            "live_closed": [],
        }

        try:
            # Calculate SPY trend and momentum
            spy_trend, spy_momentum = self._calculate_spy_momentum(current_spy_price)
            
            # Check paper positions in the portfolio
            portfolio = self.paper_simulator.portfolio
            num_positions = len(portfolio.positions)
            
            if num_positions > 0:
                logger.info(f"Checking {num_positions} positions (SPY: ${current_spy_price:.2f}, Trend: {spy_trend}, Momentum: {spy_momentum:+.3f}%)")
            
            positions_to_close = []

            for pos_id, pos in list(portfolio.positions.items()):  # Iterate over dict items
                new_price = 0.0
                price_source = "UNKNOWN"
                
                # Try to get real option price from Tastytrade streaming
                if self.market_data_provider and pos.legs:
                    # Get the primary leg's symbol (first leg for LONG options)
                    primary_leg = pos.legs[0] if pos.legs else None
                    if primary_leg and primary_leg.get("symbol"):
                        option_symbol = primary_leg["symbol"]
                        real_price = self._get_option_price(option_symbol)
                        if real_price > 0:
                            new_price = real_price
                            price_source = "TASTYTRADE"
                            logger.debug(f"Got real price for {option_symbol}: ${real_price:.2f}")
                
                # Fallback: simulate price based on SPY movement (always use this if no real price)
                if new_price <= 0:
                    entry_price = abs(pos.entry_price)
                    
                    if pos.spy_price_at_entry > 0:
                        spy_change_pct = (current_spy_price - pos.spy_price_at_entry) / pos.spy_price_at_entry
                        
                        # LONG_CALL gains value when SPY goes up, LONG_PUT gains when SPY goes down
                        if "CALL" in pos.strategy_type:
                            price_multiplier = 1 + (spy_change_pct * 5)  # ~5x leverage for ATM options
                        else:  # PUT
                            price_multiplier = 1 - (spy_change_pct * 5)  # Inverse for puts
                        
                        # Apply time decay (lose ~5% per hour for 0DTE)
                        hours_held = (datetime.now() - pos.entry_time).total_seconds() / 3600
                        time_decay = max(0.5, 1 - (hours_held * 0.05))
                        
                        new_price = entry_price * price_multiplier * time_decay
                        # Floor at 10% of entry price to prevent unrealistic drops
                        new_price = max(entry_price * 0.1, new_price)
                        price_source = "SIMULATED"
                        logger.debug(f"Simulated price: SPY change={spy_change_pct*100:.2f}%, mult={price_multiplier:.3f}, decay={time_decay:.3f}")
                    else:
                        # No SPY reference - just apply small time decay
                        hours_held = (datetime.now() - pos.entry_time).total_seconds() / 3600
                        time_decay = max(0.7, 1 - (hours_held * 0.03))
                        new_price = entry_price * time_decay
                        price_source = "FALLBACK"
                
                pos.update_price(new_price)
                
                # Update position price in database for dashboard visibility
                if self.paper_simulator.config.save_to_database:
                    self._update_position_price_in_db(pos_id, new_price)
                
                # Log position status with price source - use INFO level so we always see it
                entry = abs(pos.entry_price)
                pnl_pct = ((new_price / entry) - 1) * 100 if entry > 0 else 0
                pnl_dollar = (new_price - entry) * 100  # Per contract
                logger.info(f"[POSITION] {pos_id[:12]}: {pos.strategy_type} entry=${entry:.2f}, current=${new_price:.2f} ({price_source}), P&L={pnl_pct:+.1f}% (${pnl_dollar:+.2f})")

                # Check exit conditions in priority order
                # 1. Static profit target (50%+)
                if pos.should_take_profit():
                    positions_to_close.append((pos_id, pos, "PROFIT_TARGET"))
                # 2. Trailing stop (protects profits when dropping from high)
                elif pos.should_trailing_stop():
                    positions_to_close.append((pos_id, pos, "TRAILING_STOP"))
                # 3. Dynamic market direction exit (calls exit on bearish, puts on bullish)
                elif pos.should_exit_adverse_market(spy_trend, spy_momentum):
                    positions_to_close.append((pos_id, pos, "ADVERSE_MARKET"))
                # 4. Static stop loss (25%)
                elif pos.should_stop_loss():
                    positions_to_close.append((pos_id, pos, "STOP_LOSS"))

            # Close positions that hit targets
            for pos_id, pos, reason in positions_to_close:
                exit_price = pos.current_price
                pos.close(exit_price=exit_price, reason=reason)
                del portfolio.positions[pos_id]  # Remove from dict by key
                portfolio.closed_positions.append(pos)
                
                # Save closed trade to database
                if self.paper_simulator.config.save_to_database:
                    self.paper_simulator._save_trade_to_db(pos, "CLOSED")

                result["paper_closed"].append({
                    "position_id": pos.position_id,
                    "strategy_type": pos.strategy_type,
                    "entry_time": pos.entry_time,
                    "exit_time": pos.exit_time,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.exit_price,
                    "realized_pnl": pos.realized_pnl,
                    "exit_reason": reason,
                    "ai_confidence": getattr(pos, 'ai_confidence', 0.7),
                })

                logger.info(f"Closed paper position {pos.position_id}: {reason}, P&L=${pos.realized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error checking exits: {e}")

        return result
