"""
Base class for 0DTE Options Strategies

Provides common functionality for all 0DTE strategies:
- Position tracking with Greeks
- Profit/loss calculations
- Exit rule management
- Time-based logic for expiration day
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pytz

from loguru import logger

from ...config.settings import get_settings


class PositionStatus(Enum):
    """Status of a spread position"""
    PENDING = "PENDING"      # Order not yet filled
    OPEN = "OPEN"            # Position is active
    CLOSING = "CLOSING"      # Close order submitted
    CLOSED = "CLOSED"        # Position fully closed
    EXPIRED = "EXPIRED"      # Expired worthless or ITM


class ExitReason(Enum):
    """Reason for exiting a position"""
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    TIME_STOP = "TIME_STOP"
    MANUAL = "MANUAL"
    ADJUSTMENT = "ADJUSTMENT"
    EXPIRATION = "EXPIRATION"
    RISK_LIMIT = "RISK_LIMIT"


@dataclass
class OptionLeg:
    """Represents a single option leg in a spread"""
    symbol: str                     # Full option symbol
    option_type: str                # 'CALL' or 'PUT'
    strike: float                   # Strike price
    expiration: str                 # Expiration date (YYYY-MM-DD)
    action: str                     # 'BUY' or 'SELL'
    quantity: int                   # Number of contracts
    fill_price: Optional[float] = None
    current_price: Optional[float] = None

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0

    @property
    def is_short(self) -> bool:
        """Check if this is a short leg"""
        return self.action == "SELL"

    @property
    def is_long(self) -> bool:
        """Check if this is a long leg"""
        return self.action == "BUY"

    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L for this leg"""
        if self.fill_price is None or self.current_price is None:
            return 0.0

        multiplier = 100 * self.quantity
        if self.is_short:
            return (self.fill_price - self.current_price) * multiplier
        else:
            return (self.current_price - self.fill_price) * multiplier


@dataclass
class SpreadPosition:
    """Represents a complete spread position (multiple legs)"""
    position_id: str
    strategy_type: str              # 'IRON_CONDOR', 'PUT_CREDIT_SPREAD', etc.
    legs: List[OptionLeg]
    status: PositionStatus = PositionStatus.PENDING

    # Entry details
    entry_time: Optional[datetime] = None
    entry_credit: float = 0.0       # Total credit received (positive)
    entry_debit: float = 0.0        # Total debit paid (positive)

    # Risk parameters
    max_profit: float = 0.0
    max_loss: float = 0.0
    contracts: int = 1

    # Exit rules
    profit_target_percent: float = 15.0   # Exit at 15% profit (tighter for faster learning)
    stop_loss_percent: float = 200.0      # Exit at 2x credit received
    time_stop: Optional[time] = None      # Force exit time

    # Current state
    current_value: float = 0.0
    unrealized_pnl: float = 0.0

    # Exit details
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: Optional[ExitReason] = None
    realized_pnl: float = 0.0

    # Metadata
    ai_decision_id: Optional[str] = None
    notes: str = ""

    @property
    def is_credit_spread(self) -> bool:
        """Check if this is a credit spread (received premium)"""
        return self.entry_credit > self.entry_debit

    @property
    def net_credit(self) -> float:
        """Net credit/debit for the position"""
        return self.entry_credit - self.entry_debit

    @property
    def profit_target_price(self) -> float:
        """Price at which to take profit"""
        if self.is_credit_spread:
            # For credit spreads, buy back at lower price
            target_profit = self.net_credit * (self.profit_target_percent / 100)
            return self.net_credit - target_profit
        return 0.0

    @property
    def stop_loss_price(self) -> float:
        """Price at which to stop loss"""
        if self.is_credit_spread:
            # For credit spreads, loss is when spread value increases
            max_loss_amount = self.net_credit * (self.stop_loss_percent / 100)
            return self.net_credit + max_loss_amount
        return float('inf')

    def update_greeks(self) -> Dict[str, float]:
        """Calculate aggregate Greeks for the position"""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for leg in self.legs:
            multiplier = -1 if leg.is_short else 1
            total_delta += leg.delta * multiplier * leg.quantity
            total_gamma += leg.gamma * multiplier * leg.quantity
            total_theta += leg.theta * multiplier * leg.quantity
            total_vega += leg.vega * multiplier * leg.quantity

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega
        }

    def check_exit_conditions(self, current_price: float) -> Tuple[bool, Optional[ExitReason]]:
        """
        Check if any exit conditions are met

        Args:
            current_price: Current spread value

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Profit target
        if self.is_credit_spread and current_price <= self.profit_target_price:
            return True, ExitReason.PROFIT_TARGET

        # Stop loss
        if self.is_credit_spread and current_price >= self.stop_loss_price:
            return True, ExitReason.STOP_LOSS

        # Time stop
        if self.time_stop:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et).time()
            if now >= self.time_stop:
                return True, ExitReason.TIME_STOP

        return False, None


class ZeroDTEStrategy(ABC):
    """
    Abstract base class for 0DTE options strategies

    Provides common functionality for expiration-day trading:
    - Time-window management
    - Position tracking
    - Exit rule enforcement
    - Greeks monitoring
    """

    # Trading time windows (Eastern Time)
    DEFAULT_ENTRY_START = time(9, 35)   # 9:35 AM - after initial volatility
    DEFAULT_ENTRY_END = time(12, 0)     # 12:00 PM - no new trades after noon
    DEFAULT_EXIT_TIME = time(15, 50)    # 3:50 PM - force close before expiration (unconditional)

    def __init__(self, name: str):
        """
        Initialize 0DTE strategy

        Args:
            name: Strategy name
        """
        self.name = name
        self.settings = get_settings()
        self.positions: Dict[str, SpreadPosition] = {}
        self.closed_positions: List[SpreadPosition] = []

        # Parse time settings
        self.entry_start = self._parse_time(self.settings.trading_start_time)
        self.entry_end = self._parse_time(self.settings.trading_end_time)
        self.force_exit_time = self.DEFAULT_EXIT_TIME

        # Position counter for IDs
        self._position_count = 0

        logger.info(f"Initialized {name} strategy")

    def _parse_time(self, time_str: str) -> time:
        """Parse time string (HH:MM) to time object"""
        try:
            parts = time_str.split(":")
            return time(int(parts[0]), int(parts[1]))
        except Exception:
            return self.DEFAULT_ENTRY_START

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Return strategy type identifier"""
        pass

    @abstractmethod
    def calculate_position(
        self,
        spy_price: float,
        option_chain: Dict[str, Any],
        contracts: int = 1
    ) -> Optional[SpreadPosition]:
        """
        Calculate the position parameters for this strategy

        Args:
            spy_price: Current SPY price
            option_chain: Available options data
            contracts: Number of contracts to trade

        Returns:
            SpreadPosition with all legs defined, or None if cannot construct
        """
        pass

    @abstractmethod
    def get_entry_order(self, position: SpreadPosition) -> Dict[str, Any]:
        """
        Generate entry order for the position

        Args:
            position: Position to enter

        Returns:
            Order specification dictionary
        """
        pass

    @abstractmethod
    def get_exit_order(self, position: SpreadPosition) -> Dict[str, Any]:
        """
        Generate exit order for the position

        Args:
            position: Position to exit

        Returns:
            Order specification dictionary
        """
        pass

    def is_entry_window(self) -> bool:
        """Check if current time is within entry window"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et).time()
        return self.entry_start <= now <= self.entry_end

    def is_trading_day(self) -> bool:
        """Check if today is a trading day (weekday)"""
        today = datetime.now().weekday()
        return today < 5  # Monday = 0, Friday = 4

    def should_force_exit(self) -> bool:
        """Check if we should force exit all positions"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et).time()
        return now >= self.force_exit_time

    def generate_position_id(self) -> str:
        """Generate unique position ID"""
        self._position_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.strategy_type}_{timestamp}_{self._position_count}"

    def add_position(self, position: SpreadPosition) -> None:
        """Add a position to tracking"""
        self.positions[position.position_id] = position
        logger.info(f"Added position {position.position_id}")

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: ExitReason
    ) -> Optional[SpreadPosition]:
        """
        Close a position and move to closed list

        Args:
            position_id: Position to close
            exit_price: Exit price achieved
            exit_reason: Reason for exit

        Returns:
            Closed position or None
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return None

        position = self.positions.pop(position_id)
        position.status = PositionStatus.CLOSED
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = exit_reason

        # Calculate realized P&L
        if position.is_credit_spread:
            # P&L = Credit received - Cost to close
            position.realized_pnl = (position.net_credit - exit_price) * 100 * position.contracts
        else:
            position.realized_pnl = (exit_price - position.net_credit) * 100 * position.contracts

        self.closed_positions.append(position)

        logger.info(
            f"Closed position {position_id}: "
            f"P&L=${position.realized_pnl:.2f} ({exit_reason.value})"
        )

        return position

    def get_open_positions(self) -> List[SpreadPosition]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_total_exposure(self) -> float:
        """Calculate total max loss across all open positions"""
        return sum(p.max_loss for p in self.get_open_positions())

    def get_total_delta(self) -> float:
        """Calculate total delta exposure"""
        total = 0.0
        for position in self.get_open_positions():
            greeks = position.update_greeks()
            total += greeks["delta"]
        return total

    def check_all_exit_conditions(self) -> List[Tuple[str, ExitReason]]:
        """
        Check exit conditions for all open positions

        Returns:
            List of (position_id, exit_reason) for positions that should exit
        """
        exits = []

        # Check time-based force exit
        if self.should_force_exit():
            for pos_id in self.positions:
                exits.append((pos_id, ExitReason.TIME_STOP))
            return exits

        # Check individual position conditions
        for pos_id, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue

            should_exit, reason = position.check_exit_conditions(position.current_value)
            if should_exit and reason:
                exits.append((pos_id, reason))

        return exits

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get today's trading statistics"""
        today_closed = [
            p for p in self.closed_positions
            if p.exit_time and p.exit_time.date() == datetime.now().date()
        ]

        total_pnl = sum(p.realized_pnl for p in today_closed)
        winners = sum(1 for p in today_closed if p.realized_pnl > 0)
        losers = sum(1 for p in today_closed if p.realized_pnl <= 0)

        return {
            "strategy": self.name,
            "open_positions": len(self.get_open_positions()),
            "closed_today": len(today_closed),
            "realized_pnl": total_pnl,
            "winners": winners,
            "losers": losers,
            "win_rate": winners / len(today_closed) * 100 if today_closed else 0,
            "total_exposure": self.get_total_exposure(),
            "total_delta": self.get_total_delta()
        }

    def find_strike_by_delta(
        self,
        options: List[Dict],
        target_delta: float,
        option_type: str
    ) -> Optional[Dict]:
        """
        Find option closest to target delta

        Args:
            options: List of option data
            target_delta: Target delta (positive for calls, negative for puts)
            option_type: 'CALL' or 'PUT'

        Returns:
            Option data dict or None
        """
        if not options:
            return None

        # Adjust target for puts (negative delta)
        if option_type == "PUT":
            target_delta = -abs(target_delta)
        else:
            target_delta = abs(target_delta)

        closest = None
        min_diff = float('inf')

        for opt in options:
            delta = opt.get("delta", 0)
            diff = abs(delta - target_delta)
            if diff < min_diff:
                min_diff = diff
                closest = opt

        return closest

    def __repr__(self) -> str:
        return f"{self.name}(open={len(self.positions)}, closed={len(self.closed_positions)})"
