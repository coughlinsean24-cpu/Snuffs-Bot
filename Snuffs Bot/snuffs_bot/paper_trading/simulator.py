"""
Paper Trading Simulator

Main simulator class that coordinates virtual trading
using live market data from Tastytrade.

Features adaptive exit management:
- Trailing stops that lock in profits
- Momentum-based exit signals
- Reversal detection to exit while still profitable
"""

import asyncio
import traceback
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from loguru import logger

from .virtual_portfolio import VirtualPortfolio, VirtualPosition
from .order_simulator import OrderSimulator, SimulatedFill
from .adaptive_exits import AdaptiveExitManager, ExitSignal
from ..strategies.zero_dte.base_zero_dte import SpreadPosition, ExitReason
from ..config.settings import get_settings
from ..database.connection import db_session_scope
from ..database.models import Trade


@dataclass
class SimulatorConfig:
    """Configuration for paper trading simulator"""
    starting_capital: float = 100000.0
    realistic_fills: bool = True
    auto_manage_exits: bool = True
    save_to_database: bool = True
    price_update_interval: float = 5.0  # seconds

    # Adaptive exit settings
    initial_stop_percent: float = 50.0      # Initial stop loss before trailing
    trailing_stop_percent: float = 30.0     # Trail 30% from high (tighter = more protection)
    profit_lock_threshold: float = 25.0     # Start trailing after 25% gain
    max_profit_target: float = 100.0        # Take profit at 100% gain


class PaperTradingSimulator:
    """
    Paper trading simulator using live market data

    Key features:
    - Uses real prices from Tastytrade
    - All orders are simulated internally
    - Tracks positions and P&L
    - Parallel execution with live trading
    - Saves results to database for learning
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        price_feed: Optional[Callable] = None
    ):
        """
        Initialize the paper trading simulator

        Args:
            config: Simulator configuration
            price_feed: Optional callback to get live prices
        """
        self.config = config or SimulatorConfig()
        self.settings = get_settings()

        # Initialize components
        self.portfolio = VirtualPortfolio(self.config.starting_capital)
        self.order_simulator = OrderSimulator(self.config.realistic_fills)

        # Adaptive exit manager - handles trailing stops, momentum exits, reversals
        self.exit_manager = AdaptiveExitManager(
            initial_stop_percent=self.config.initial_stop_percent,
            trailing_stop_percent=self.config.trailing_stop_percent,
            profit_lock_threshold=self.config.profit_lock_threshold,
            max_profit_target=self.config.max_profit_target,
        )

        # Price feed callback (will be set by execution coordinator)
        self.price_feed = price_feed

        # State
        self.is_running = False
        self._position_monitor_task: Optional[asyncio.Task] = None

        logger.info(
            f"Paper Trading Simulator initialized: "
            f"capital=${self.config.starting_capital:,.2f}, "
            f"adaptive exits enabled (trail {self.config.trailing_stop_percent}% from high)"
        )
        
        # Restore any open positions from database on startup
        self._restore_open_positions_from_db()

    def _restore_open_positions_from_db(self) -> int:
        """
        Restore open positions from database after restart/disconnect.
        This ensures positions aren't lost if the bot loses internet or crashes.
        
        Returns:
            Number of positions restored
        """
        restored = 0
        try:
            with db_session_scope() as session:
                open_trades = session.query(Trade).filter(
                    Trade.trade_type == "PAPER",
                    Trade.status == "OPEN"
                ).all()
                
                for trade in open_trades:
                    # Check if position already exists in memory
                    existing_ids = [p.position_id for p in self.portfolio.positions.values()]
                    
                    # Create a position ID based on trade ID
                    position_id = f"paper_db_{trade.id}"
                    if position_id in self.portfolio.positions:
                        continue
                    
                    # Reconstruct VirtualPosition from database
                    position = VirtualPosition(
                        position_id=position_id,
                        strategy_type=trade.strategy,
                        legs=trade.entry_legs or [],
                        entry_time=trade.entry_time,
                        entry_price=float(trade.entry_price or 0),
                        contracts=trade.position_size or 1,
                        max_profit=float(trade.entry_price or 0) * 100,  # Estimate
                        max_loss=float(trade.max_risk or 0),
                        spy_price_at_entry=float(trade.spy_price or 0),
                        vix_at_entry=float(trade.vix or 20),
                    )
                    
                    # Set current price from exit_price (used for streaming updates)
                    if trade.exit_price:
                        position.current_price = float(trade.exit_price)
                        position.update_price(position.current_price)
                    
                    # Add to portfolio
                    self.portfolio.positions[position_id] = position
                    
                    # Register with exit manager
                    self.exit_manager.register_position(
                        position_id,
                        float(trade.entry_price or 0),
                        trade.entry_time
                    )
                    
                    restored += 1
                    logger.info(f"Restored position from DB: {trade.strategy} (ID {trade.id})")
                
                if restored > 0:
                    logger.success(f"Restored {restored} open positions from database")
                    
        except Exception as e:
            logger.warning(f"Could not restore positions from DB: {e}")
            
        return restored

    async def start(self) -> None:
        """Start the paper trading simulator"""
        if self.is_running:
            logger.warning("Simulator already running")
            return

        self.is_running = True

        if self.config.auto_manage_exits:
            self._position_monitor_task = asyncio.create_task(
                self._monitor_positions()
            )

        logger.info("Paper trading simulator started")

    async def stop(self) -> None:
        """Stop the paper trading simulator"""
        self.is_running = False

        if self._position_monitor_task:
            self._position_monitor_task.cancel()
            try:
                await self._position_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Paper trading simulator stopped")

    async def execute_trade(
        self,
        spread_position: SpreadPosition,
        market_data: Dict[str, Any],
        ai_decision_id: Optional[str] = None
    ) -> Optional[VirtualPosition]:
        """
        Execute a paper trade

        Args:
            spread_position: Strategy position to simulate
            market_data: Current market data (SPY price, VIX, option prices)
            ai_decision_id: Optional AI decision reference

        Returns:
            VirtualPosition if successful, None otherwise
        """
        # Build order from position
        order = {
            "order_type": "LIMIT",
            "price_type": "CREDIT",
            "price": spread_position.net_credit,
            "contracts": spread_position.contracts,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "action": leg.action,
                    "quantity": leg.quantity
                }
                for leg in spread_position.legs
            ]
        }

        # Simulate the fill
        fill = self.order_simulator.simulate_entry_fill(order, market_data)

        if not fill.was_filled:
            logger.warning(f"Paper order did not fill: {fill.notes}")
            return None

        # Update position with actual fill price
        spread_position.entry_credit = fill.fill_price

        # Open position in portfolio with slippage tracking
        position = self.portfolio.open_position(
            spread_position=spread_position,
            spy_price=market_data.get("spy_price", 0),
            vix=market_data.get("vix", 0),
            ai_decision_id=ai_decision_id,
            entry_slippage=fill.slippage
        )

        # Register with adaptive exit manager for smart trailing stops
        self.exit_manager.register_position(
            position_id=position.position_id,
            entry_price=fill.fill_price,
            entry_time=position.entry_time
        )

        # Save to database if enabled
        if self.config.save_to_database:
            self._save_trade_to_db(position, "OPEN")

        logger.info(
            f"Paper trade executed: {position.position_id} "
            f"@ ${fill.fill_price:.2f} (slippage: ${fill.slippage:.4f}) "
            f"[adaptive exits enabled]"
        )

        return position

    async def close_trade(
        self,
        position_id: str,
        market_data: Dict[str, Any],
        reason: str = "MANUAL"
    ) -> Optional[VirtualPosition]:
        """
        Close a paper trade

        Args:
            position_id: Position to close
            market_data: Current market data
            reason: Reason for closing

        Returns:
            Closed position or None
        """
        position = self.portfolio.positions.get(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found")
            return None

        # Build exit order
        order = {
            "order_type": "LIMIT",
            "price_type": "DEBIT",
            "price": position.current_price,
            "contracts": position.contracts,
        }

        # Simulate exit fill
        fill = self.order_simulator.simulate_exit_fill(order, market_data)

        # Close position with slippage tracking
        closed = self.portfolio.close_position(
            position_id=position_id,
            exit_price=fill.fill_price,
            reason=reason,
            exit_slippage=fill.slippage
        )

        # Remove from adaptive exit tracking
        self.exit_manager.remove_position(position_id)

        # Update database
        if self.config.save_to_database and closed:
            self._save_trade_to_db(closed, "CLOSED")

        return closed

    def update_prices(self, price_updates: Dict[str, float]) -> List[str]:
        """
        Update positions with new prices

        Args:
            price_updates: Dict mapping position_id to current spread value

        Returns:
            List of position IDs that triggered exits
        """
        exits = self.portfolio.update_positions(price_updates)

        exit_ids = []
        for exit_info in exits:
            exit_ids.append(exit_info["position_id"])
            logger.info(
                f"Paper position {exit_info['position_id']} "
                f"triggered {exit_info['reason']}"
            )

        return exit_ids

    async def _monitor_positions(self) -> None:
        """Background task to monitor and auto-manage positions with adaptive exits"""
        while self.is_running:
            try:
                # Get current prices if price feed available
                if self.price_feed:
                    price_updates = await self._fetch_position_prices()

                    # Update portfolio prices
                    self.portfolio.update_positions(price_updates)

                    # Use adaptive exit manager for smart exit decisions
                    for pos_id, current_price in price_updates.items():
                        position = self.portfolio.positions.get(pos_id)
                        if not position:
                            continue

                        # Get adaptive exit recommendation
                        recommendation = self.exit_manager.update_price(
                            pos_id, current_price
                        )

                        if not recommendation:
                            continue

                        # Check if we should exit
                        should_exit = recommendation.signal in (
                            ExitSignal.TAKE_PROFIT,
                            ExitSignal.TRAILING_STOP,
                            ExitSignal.MOMENTUM_EXIT,
                            ExitSignal.REVERSAL,
                            ExitSignal.STOP_LOSS,
                        )

                        # For HIGH/IMMEDIATE urgency or certain signals, exit now
                        if should_exit and recommendation.urgency in ("HIGH", "IMMEDIATE"):
                            reason = recommendation.signal.value
                            logger.info(
                                f"Adaptive exit triggered for {pos_id}: "
                                f"{recommendation.signal.value} - {recommendation.reason}"
                            )

                            market_data = {
                                "spread_bid": current_price * 0.98,
                                "spread_ask": current_price * 1.02,
                                "vix": 18
                            }
                            await self.close_trade(pos_id, market_data, reason)

                        # For MEDIUM urgency momentum/reversal signals while profitable
                        elif (should_exit and
                              recommendation.urgency == "MEDIUM" and
                              recommendation.signal in (ExitSignal.MOMENTUM_EXIT, ExitSignal.REVERSAL)):
                            # Log warning but let it ride a bit longer
                            tracker = self.exit_manager.trackers.get(pos_id)
                            if tracker and tracker.profit_percent > 20:
                                # If profit is good and momentum turning, exit
                                reason = recommendation.signal.value
                                logger.info(
                                    f"Momentum exit for {pos_id}: {recommendation.reason}"
                                )
                                market_data = {
                                    "spread_bid": current_price * 0.98,
                                    "spread_ask": current_price * 1.02,
                                    "vix": 18
                                }
                                await self.close_trade(pos_id, market_data, reason)

                # Check time-based exits (handled by adaptive manager too, but keep as backup)
                await self._check_time_exits()

            except Exception as e:
                error_context = {
                    "component": "paper_trading_monitor",
                    "open_positions": len(self.portfolio.positions),
                    "traceback": traceback.format_exc()
                }
                logger.error(f"Position monitor error: {e}")
                logger.error(f"Monitor error context: {error_context}")

            await asyncio.sleep(self.config.price_update_interval)
            
            # Periodic state backup (every 30 seconds) for crash recovery
            if hasattr(self, '_last_backup_time'):
                if (datetime.now() - self._last_backup_time).total_seconds() > 30:
                    self._backup_state_to_file()
                    self._last_backup_time = datetime.now()
            else:
                self._last_backup_time = datetime.now()

    def _backup_state_to_file(self) -> None:
        """Backup portfolio state to JSON file for crash recovery"""
        import json
        from pathlib import Path
        
        try:
            backup_path = Path("data/local_ai/portfolio_backup.json")
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "starting_capital": self.portfolio.starting_capital,
                "cash": self.portfolio.cash,
                "total_trades": self.portfolio.total_trades,
                "winning_trades": self.portfolio.winning_trades,
                "losing_trades": self.portfolio.losing_trades,
                "open_positions": [
                    {
                        "position_id": p.position_id,
                        "strategy_type": p.strategy_type,
                        "entry_time": p.entry_time.isoformat(),
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "contracts": p.contracts,
                    }
                    for p in self.portfolio.positions.values()
                ],
            }
            
            with open(backup_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug(f"Portfolio state backed up: {len(state['open_positions'])} positions")
        except Exception as e:
            logger.debug(f"Could not backup portfolio state: {e}")

    async def _fetch_position_prices(self) -> Dict[str, float]:
        """Fetch current prices for all open positions"""
        price_updates = {}

        for pos_id, position in self.portfolio.positions.items():
            if self.price_feed:
                try:
                    # This would call the actual price feed
                    prices = await self.price_feed(position.legs)
                    if prices:
                        price_updates[pos_id] = prices.get("spread_value", position.current_price)
                except Exception as e:
                    logger.debug(f"Could not fetch price for {pos_id}: {e}")

        return price_updates

    async def _check_time_exits(self) -> None:
        """Check for time-based forced exits"""
        import pytz
        et = pytz.timezone('US/Eastern')
        current_time = datetime.now(et).time()

        # Force exit at 3:50 PM ET
        force_exit_time = time(15, 50)

        if current_time >= force_exit_time:
            for pos_id in list(self.portfolio.positions.keys()):
                position = self.portfolio.positions.get(pos_id)
                if position:
                    market_data = {
                        "spread_bid": position.current_price * 0.95,
                        "spread_ask": position.current_price * 1.05,
                        "vix": 18
                    }
                    await self.close_trade(pos_id, market_data, "TIME_STOP")

    def _save_trade_to_db(self, position: VirtualPosition, action: str) -> None:
        """Save paper trade to database for learning"""
        try:
            with db_session_scope() as session:
                if action == "OPEN":
                    trade = Trade(
                        trade_type="PAPER",
                        strategy=position.strategy_type,
                        entry_time=position.entry_time,
                        entry_price=abs(position.entry_price),  # Store as positive
                        entry_legs=position.legs,
                        spy_price=position.spy_price_at_entry,
                        vix=position.vix_at_entry,
                        status="OPEN",
                        position_size=position.contracts,  # Required field
                        max_risk=position.max_loss,  # Required field
                        fees=position.entry_commission,
                    )
                    session.add(trade)
                    logger.info(f"Trade saved to DB: {position.strategy_type} - {position.position_id}")

                elif action == "CLOSED":
                    # Update existing trade - match by entry_time for precise identification
                    # This prevents mixing up P&L when multiple trades of same strategy exist
                    trade = session.query(Trade).filter(
                        Trade.trade_type == "PAPER",
                        Trade.status == "OPEN",
                        Trade.strategy == position.strategy_type,
                        Trade.entry_time == position.entry_time  # Match exact entry time
                    ).first()

                    # Fallback to old method if exact match not found (for legacy trades)
                    if not trade:
                        trade = session.query(Trade).filter(
                            Trade.trade_type == "PAPER",
                            Trade.status == "OPEN",
                            Trade.strategy == position.strategy_type
                        ).order_by(Trade.entry_time.desc()).first()

                    if trade:
                        trade.exit_time = position.exit_time
                        trade.exit_price = abs(position.exit_price)
                        trade.gross_pnl = position.gross_pnl  # P&L before costs
                        trade.pnl = position.realized_pnl  # Net P&L after costs
                        trade.fees = position.total_commission  # Total commission
                        trade.slippage = position.total_slippage  # Total slippage
                        trade.status = "CLOSED"
                        trade.exit_reason = position.exit_reason

                        # Calculate P&L percent
                        if trade.entry_price and float(trade.entry_price) > 0:
                            trade.pnl_percent = (position.realized_pnl / (float(trade.entry_price) * 100 * position.contracts)) * 100

                        logger.info(
                            f"Trade closed in DB: {position.strategy_type} - "
                            f"Gross: ${position.gross_pnl:.2f}, Fees: ${position.total_commission:.2f}, "
                            f"Slippage: ${position.total_slippage:.4f}, Net P&L: ${position.realized_pnl:.2f}"
                        )

        except Exception as e:
            logger.warning(f"Could not save paper trade to DB: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get complete simulator summary including adaptive exit status"""
        # Get adaptive exit recommendations for all positions
        exit_recommendations = {}
        for pos_id in self.portfolio.positions:
            rec = self.exit_manager.analyze_exit(pos_id)
            tracker_summary = self.exit_manager.get_position_summary(pos_id)
            exit_recommendations[pos_id] = {
                "signal": rec.signal.value,
                "urgency": rec.urgency,
                "reason": rec.reason,
                "trailing_active": tracker_summary.get("trailing_active", False),
                "current_stop": tracker_summary.get("current_stop", 0),
                "profit_percent": tracker_summary.get("profit_percent", 0),
                "high_water_mark": tracker_summary.get("high_water_mark", 0),
            }

        return {
            "portfolio": self.portfolio.get_portfolio_summary(),
            "open_positions": self.portfolio.get_open_positions_summary(),
            "exit_recommendations": exit_recommendations,
            "recent_closed": self.portfolio.get_closed_positions_summary(10),
            "is_running": self.is_running,
            "config": {
                "starting_capital": self.config.starting_capital,
                "realistic_fills": self.config.realistic_fills,
                "auto_manage_exits": self.config.auto_manage_exits,
                "trailing_stop_percent": self.config.trailing_stop_percent,
                "profit_lock_threshold": self.config.profit_lock_threshold,
            }
        }

    def reset(self) -> None:
        """Reset simulator to initial state"""
        self.portfolio.reset()
        # Clear all adaptive exit trackers
        self.exit_manager.trackers.clear()
        logger.info("Paper trading simulator reset")
