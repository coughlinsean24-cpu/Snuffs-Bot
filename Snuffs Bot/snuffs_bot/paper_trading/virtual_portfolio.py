"""
Virtual Portfolio Manager

Tracks simulated positions and account state without
sending any orders to the broker.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from ..strategies.zero_dte.base_zero_dte import SpreadPosition, OptionLeg, ExitReason


class PositionState(Enum):
    """State of a virtual position"""
    PENDING = "PENDING"      # Order submitted, waiting for fill
    OPEN = "OPEN"            # Position is active
    CLOSING = "CLOSING"      # Exit order submitted
    CLOSED = "CLOSED"        # Position fully closed


@dataclass
class VirtualPosition:
    """
    A simulated position in the virtual portfolio

    Tracks all the same data as a real position but
    exists only in memory/database.
    """
    position_id: str
    strategy_type: str
    legs: List[Dict[str, Any]]  # Simplified leg data

    # Entry details
    entry_time: datetime
    entry_price: float          # Net credit/debit
    contracts: int

    # Risk parameters
    max_profit: float
    max_loss: float

    # Exit rules (for LONG options - FAST SCALPING)
    # Profit target: 15% gain = quick profit, cover commissions
    # Stop loss: 20% loss = exit quickly to minimize damage
    profit_target_percent: float = 15.0
    stop_loss_percent: float = 20.0

    # Current state
    state: PositionState = PositionState.OPEN
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    # High water mark for trailing stop
    high_water_mark: float = 0.0
    trailing_stop_active: bool = False
    trailing_stop_percent: float = 15.0  # Exit if drops 15% from high (tight trail)
    profit_lock_threshold: float = 5.0  # Activate trailing after just 5% profit

    # Market data at entry
    spy_price_at_entry: float = 0.0
    vix_at_entry: float = 0.0

    # Exit details
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: Optional[str] = None
    realized_pnl: float = 0.0

    # Commission tracking (Tastytrade: $1/contract per leg, capped at $10/leg)
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    total_commission: float = 0.0

    # Slippage tracking (difference between requested and filled price)
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    total_slippage: float = 0.0

    # Gross P&L (before costs) for transparency
    gross_pnl: float = 0.0

    # AI tracking
    ai_decision_id: Optional[str] = None

    # Metadata
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_commission(self, num_legs: int, contracts: int) -> float:
        """
        Calculate commission based on Tastytrade fee structure

        Tastytrade options fees:
        - $1.00 per contract per leg to open
        - $0.00 per contract to close (closing trades are free!)
        - Capped at $10 per leg

        Args:
            num_legs: Number of option legs (Iron Condor = 4, Credit Spread = 2)
            contracts: Number of contracts

        Returns:
            Total commission in dollars
        """
        commission_per_leg = min(contracts * 1.00, 10.00)  # $1/contract, max $10/leg
        return commission_per_leg * num_legs

    def update_price(self, new_price: float) -> None:
        """Update current price and calculate unrealized P&L (including commissions)"""
        self.current_price = new_price

        # Track high water mark for trailing stop
        if new_price > self.high_water_mark:
            old_hwm = self.high_water_mark
            self.high_water_mark = new_price
            if old_hwm > 0:  # Only log if it's actually updating
                logger.debug(f"[TRAILING] New high water mark: ${old_hwm:.2f} -> ${new_price:.2f}")
        
        # Activate trailing stop once we hit profit threshold
        entry = abs(self.entry_price)
        if entry > 0:
            profit_pct = ((new_price - entry) / entry) * 100
            if profit_pct >= self.profit_lock_threshold and not self.trailing_stop_active:
                self.trailing_stop_active = True
                logger.info(f"[TRAILING] ✅ ACTIVATED at {profit_pct:.1f}% profit (threshold: {self.profit_lock_threshold}%) - High water: ${self.high_water_mark:.2f}")

        # For LONG options: P&L = Current Value - Entry Cost
        # Positive when option value increases (good for buyer)
        # Use absolute value of entry_price since LONG options have negative entry (debit)
        gross_pnl = (new_price - entry) * 100 * self.contracts

        # Subtract commissions from unrealized P&L
        # Entry commission is already paid, exit is free at Tastytrade
        self.unrealized_pnl = gross_pnl - self.entry_commission

    def should_take_profit(self) -> bool:
        """Check if profit target is hit (for LONG options)"""
        entry = abs(self.entry_price)  # Use absolute value since LONG options have negative entry (debit)
        if entry <= 0:
            return False

        # For LONG options: take profit when price increases above target
        # e.g., 50% profit target = sell when option is worth 1.5x entry price
        target_price = entry * (1 + self.profit_target_percent / 100)
        hit_target = self.current_price >= target_price
        if hit_target:
            logger.info(f"PROFIT TARGET HIT: current ${self.current_price:.2f} >= target ${target_price:.2f}")
        return hit_target

    def should_stop_loss(self) -> bool:
        """Check if stop loss is hit (for LONG options)"""
        entry = abs(self.entry_price)  # Use absolute value since LONG options have negative entry (debit)
        if entry <= 0:
            return False

        # For LONG options: stop loss when price drops below threshold
        # e.g., 20% stop loss = exit when option loses 20% of value
        stop_price = entry * (1 - self.stop_loss_percent / 100)
        hit_stop = self.current_price <= stop_price
        
        # Debug logging to see why stop isn't triggering
        pnl_pct = ((self.current_price - entry) / entry) * 100 if entry > 0 else 0
        logger.debug(f"[STOP CHECK] Entry=${entry:.3f}, Current=${self.current_price:.3f}, Stop=${stop_price:.3f}, StopPct={self.stop_loss_percent}%, P&L={pnl_pct:.1f}%, Hit={hit_stop}")
        
        if hit_stop:
            logger.info(f"STOP LOSS HIT: current ${self.current_price:.2f} <= stop ${stop_price:.2f}")
        return hit_stop

    def should_trailing_stop(self) -> bool:
        """Check if trailing stop is hit (protects profits when price drops from high)"""
        if not self.trailing_stop_active:
            return False
            
        if self.high_water_mark <= 0:
            return False
        
        entry = abs(self.entry_price)
        
        # Calculate trailing stop price from high water mark
        # e.g., 15% trailing = exit if drops 15% from high
        trailing_stop_price = self.high_water_mark * (1 - self.trailing_stop_percent / 100)
        
        # Calculate current profit %
        current_profit_pct = ((self.current_price - entry) / entry) * 100 if entry > 0 else 0
        drop_from_high_pct = ((self.high_water_mark - self.current_price) / self.high_water_mark) * 100 if self.high_water_mark > 0 else 0
        
        # Log trailing stop status when active
        logger.debug(f"[TRAILING CHECK] Entry: ${entry:.2f}, Current: ${self.current_price:.2f}, High: ${self.high_water_mark:.2f}, "
                    f"Trail stop at: ${trailing_stop_price:.2f}, Drop from high: {drop_from_high_pct:.1f}%, Current P&L: {current_profit_pct:+.1f}%")
        
        # Only trigger if we're still in profit overall (avoid exiting at a loss via trailing)
        still_in_profit = self.current_price > entry
        hit_trailing = self.current_price <= trailing_stop_price
        
        if hit_trailing and still_in_profit:
            profit_locked = ((self.current_price - entry) / entry) * 100
            logger.info(f"TRAILING STOP HIT: current ${self.current_price:.2f} dropped from high ${self.high_water_mark:.2f} (locking {profit_locked:.1f}% profit)")
            return True
        elif hit_trailing and not still_in_profit:
            logger.debug(f"[TRAILING] Would have triggered but not in profit anymore (P&L: {current_profit_pct:.1f}%)")
        return False

    def should_exit_adverse_market(self, spy_trend: str, momentum: float) -> bool:
        """
        Check if we should exit due to adverse market direction
        
        Calls should exit when market turns bearish
        Puts should exit when market turns bullish
        
        LEARNING MODE: Be more patient - let trades develop
        We want to see how they would have played out
        
        Args:
            spy_trend: 'BULLISH', 'BEARISH', or 'NEUTRAL' based on recent SPY movement
            momentum: Recent price change % (negative = falling, positive = rising)
        
        Returns:
            True if position should exit due to adverse market
        """
        # If trailing stop is active, let it handle the exit instead
        # Don't override trailing stop logic with adverse market
        if self.trailing_stop_active:
            return False
        
        entry = abs(self.entry_price)
        if entry <= 0:
            return False
        
        # Calculate current profit/loss percentage
        pnl_pct = ((self.current_price - entry) / entry) * 100
        
        # LEARNING MODE: Be more patient - require stronger signals to exit
        # Only exit on STRONG adverse moves (not weak wiggles)
        
        # If we have good profit (10%+) and STRONG adverse momentum, protect it
        if pnl_pct > 10:
            if "CALL" in self.strategy_type and spy_trend == "BEARISH" and momentum < -0.08:
                logger.info(f"ADVERSE MARKET: CALL exit with {pnl_pct:.1f}% profit - strong bearish momentum ({momentum:.2f}%)")
                return True
            elif "PUT" in self.strategy_type and spy_trend == "BULLISH" and momentum > 0.08:
                logger.info(f"ADVERSE MARKET: PUT exit with {pnl_pct:.1f}% profit - strong bullish momentum ({momentum:.2f}%)")
                return True
        
        # If at a significant loss (-15%+), cut on strong adverse moves
        if pnl_pct < -15:
            if "CALL" in self.strategy_type and spy_trend == "BEARISH" and momentum < -0.10:
                logger.info(f"ADVERSE MARKET: CALL loss cut at {pnl_pct:.1f}% - strong bearish momentum ({momentum:.2f}%)")
                return True
            elif "PUT" in self.strategy_type and spy_trend == "BULLISH" and momentum > 0.10:
                logger.info(f"ADVERSE MARKET: PUT loss cut at {pnl_pct:.1f}% - strong bullish momentum ({momentum:.2f}%)")
                return True
        
        return False

    def close(self, exit_price: float, reason: str, exit_slippage: float = 0.0) -> None:
        """Close the position with full transaction cost tracking"""
        self.state = PositionState.CLOSED
        self.exit_time = datetime.now()
        self.exit_price = exit_price
        self.exit_reason = reason

        # Calculate gross P&L for LONG options (pure price movement)
        # Profit = exit price - entry price (buy low, sell high)
        # Use absolute value of entry_price since LONG options have negative entry (debit)
        entry = abs(self.entry_price)
        self.gross_pnl = (exit_price - entry) * 100 * self.contracts

        # Exit commission is $0 at Tastytrade (closing trades are free)
        self.exit_commission = 0.0
        self.total_commission = self.entry_commission + self.exit_commission

        # Track slippage (execution costs)
        self.exit_slippage = exit_slippage
        self.total_slippage = self.entry_slippage + self.exit_slippage

        # Net P&L after all costs (commission + slippage)
        # Note: Slippage is already reflected in fill prices, so we only subtract commission
        # The gross_pnl uses actual filled prices which already include slippage impact
        self.realized_pnl = self.gross_pnl - self.total_commission

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "position_id": self.position_id,
            "strategy_type": self.strategy_type,
            "legs": self.legs,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "contracts": self.contracts,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "state": self.state.value,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "gross_pnl": self.gross_pnl,
            "realized_pnl": self.realized_pnl,
            "exit_reason": self.exit_reason,
            "ai_decision_id": self.ai_decision_id,
            "entry_commission": self.entry_commission,
            "exit_commission": self.exit_commission,
            "total_commission": self.total_commission,
            "entry_slippage": self.entry_slippage,
            "exit_slippage": self.exit_slippage,
            "total_slippage": self.total_slippage,
            "spy_price_at_entry": self.spy_price_at_entry,
        }


class VirtualPortfolio:
    """
    Virtual portfolio that simulates account state

    Tracks:
    - Starting and current capital
    - Open and closed positions
    - Daily P&L
    - Performance metrics
    """

    def __init__(self, starting_capital: float = 100000.0):
        """
        Initialize virtual portfolio

        Args:
            starting_capital: Initial account value
        """
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: Dict[str, VirtualPosition] = {}
        self.closed_positions: List[VirtualPosition] = []

        # Daily tracking
        self.daily_pnl: Dict[str, float] = {}  # date string -> pnl
        self.session_start = datetime.now()

        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        logger.info(f"Virtual portfolio initialized with ${starting_capital:,.2f}")

    @property
    def account_value(self) -> float:
        """Current total account value (cash + positions)"""
        position_value = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + position_value

    @property
    def buying_power(self) -> float:
        """Available buying power (simplified as cash - margin used)"""
        margin_used = sum(p.max_loss for p in self.positions.values())
        return max(0, self.cash - margin_used)

    @property
    def total_exposure(self) -> float:
        """Total max loss across all positions"""
        return sum(p.max_loss for p in self.positions.values())

    @property
    def open_position_count(self) -> int:
        """Number of open positions"""
        return len(self.positions)

    @property
    def today_pnl(self) -> float:
        """Today's realized + unrealized P&L"""
        today = date.today().isoformat()

        # Realized from closed positions today
        realized = sum(
            p.realized_pnl for p in self.closed_positions
            if p.exit_time and p.exit_time.date() == date.today()
        )

        # Unrealized from open positions
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())

        return realized + unrealized

    def open_position(
        self,
        spread_position: SpreadPosition,
        spy_price: float,
        vix: float,
        ai_decision_id: Optional[str] = None,
        entry_slippage: float = 0.0
    ) -> VirtualPosition:
        """
        Open a new virtual position

        Args:
            spread_position: Strategy position to simulate
            spy_price: Current SPY price
            vix: Current VIX level
            ai_decision_id: Optional AI decision reference
            entry_slippage: Slippage on entry fill

        Returns:
            Created VirtualPosition
        """
        position_id = f"paper_{uuid.uuid4().hex[:8]}"

        # Convert legs to simple dict format
        legs_data = []
        for leg in spread_position.legs:
            legs_data.append({
                "symbol": leg.symbol,
                "option_type": leg.option_type,
                "strike": leg.strike,
                "action": leg.action,
                "quantity": leg.quantity,
                "fill_price": leg.fill_price,
            })

        position = VirtualPosition(
            position_id=position_id,
            strategy_type=spread_position.strategy_type,
            legs=legs_data,
            entry_time=datetime.now(),
            entry_price=spread_position.net_credit,
            contracts=spread_position.contracts,
            max_profit=spread_position.max_profit,
            max_loss=spread_position.max_loss,
            profit_target_percent=spread_position.profit_target_percent,
            stop_loss_percent=spread_position.stop_loss_percent,
            spy_price_at_entry=spy_price,
            vix_at_entry=vix,
            ai_decision_id=ai_decision_id,
        )

        # Calculate entry commission (Tastytrade: $1/contract/leg, max $10/leg)
        num_legs = len(legs_data)
        position.entry_commission = position.calculate_commission(num_legs, spread_position.contracts)

        # Track entry slippage
        position.entry_slippage = entry_slippage

        self.positions[position_id] = position
        self.total_trades += 1

        logger.info(
            f"Opened paper position {position_id}: "
            f"{spread_position.strategy_type} @ ${spread_position.net_credit:.2f} "
            f"(commission: ${position.entry_commission:.2f}, slippage: ${entry_slippage:.4f})"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str,
        exit_slippage: float = 0.0
    ) -> Optional[VirtualPosition]:
        """
        Close a virtual position

        Args:
            position_id: Position to close
            exit_price: Price at which position is closed
            reason: Reason for closing
            exit_slippage: Slippage on exit fill

        Returns:
            Closed position or None
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return None

        position = self.positions.pop(position_id)
        position.close(exit_price, reason, exit_slippage)

        # Update metrics
        if position.realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update cash
        self.cash += position.realized_pnl

        self.closed_positions.append(position)

        logger.info(
            f"Closed paper position {position_id}: "
            f"P&L=${position.realized_pnl:.2f} ({reason})"
        )

        return position

    def update_positions(self, price_updates: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Update all positions with new prices

        Args:
            price_updates: Dict of position_id -> current spread value

        Returns:
            List of positions that triggered exit conditions
        """
        exits = []

        for pos_id, position in self.positions.items():
            if pos_id in price_updates:
                position.update_price(price_updates[pos_id])

                # Check exit conditions
                if position.should_take_profit():
                    exits.append({
                        "position_id": pos_id,
                        "reason": "PROFIT_TARGET",
                        "price": position.current_price
                    })
                elif position.should_stop_loss():
                    exits.append({
                        "position_id": pos_id,
                        "reason": "STOP_LOSS",
                        "price": position.current_price
                    })

        return exits

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get complete portfolio summary"""
        # Calculate total commissions paid
        open_commissions = sum(p.entry_commission for p in self.positions.values())
        closed_commissions = sum(p.total_commission for p in self.closed_positions)
        total_commissions = open_commissions + closed_commissions

        return {
            "starting_capital": self.starting_capital,
            "current_value": self.account_value,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "total_return": (self.account_value - self.starting_capital) / self.starting_capital * 100,
            "today_pnl": self.today_pnl,
            "open_positions": self.open_position_count,
            "total_exposure": self.total_exposure,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / self.total_trades * 100 if self.total_trades else 0,
            "total_commissions": total_commissions,
        }

    def get_open_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all open positions"""
        return [p.to_dict() for p in self.positions.values()]

    def get_closed_positions_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get summary of recent closed positions"""
        recent = sorted(
            self.closed_positions,
            key=lambda p: p.exit_time or datetime.min,
            reverse=True
        )[:limit]
        return [p.to_dict() for p in recent]

    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.cash = self.starting_capital
        self.positions.clear()
        self.closed_positions.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.session_start = datetime.now()

        logger.info("Virtual portfolio reset")
