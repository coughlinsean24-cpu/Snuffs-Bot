"""
Adaptive Exit Manager for Long Options

Implements smart exit strategies that adapt as trades move:
- Trailing stops that lock in profits
- Momentum-based exit signals
- Time-decay awareness for 0DTE
- Early exit on reversal signals
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import pytz

from loguru import logger


class ExitSignal(Enum):
    """Types of exit signals"""
    HOLD = "HOLD"                      # Keep position
    TAKE_PROFIT = "TAKE_PROFIT"        # Hit profit target
    TRAILING_STOP = "TRAILING_STOP"    # Trailing stop triggered
    MOMENTUM_EXIT = "MOMENTUM_EXIT"    # Momentum reversing
    TIME_DECAY = "TIME_DECAY"          # Theta burn risk
    REVERSAL = "REVERSAL"              # Price action reversal detected
    STOP_LOSS = "STOP_LOSS"            # Hard stop loss hit


@dataclass
class ExitRecommendation:
    """Recommendation from the exit analyzer"""
    signal: ExitSignal
    confidence: float  # 0-1
    urgency: str  # "LOW", "MEDIUM", "HIGH", "IMMEDIATE"
    reason: str
    suggested_exit_price: Optional[float] = None
    dollar_pnl: Optional[float] = None  # Actual dollar P&L


@dataclass
class PositionTracker:
    """Tracks a position's price history for adaptive exits"""
    position_id: str
    entry_price: float
    entry_time: datetime

    # Price tracking
    current_price: float = 0.0
    high_water_mark: float = 0.0  # Highest price seen
    low_water_mark: float = float('inf')  # Lowest price seen

    # Price history (last N prices for momentum)
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)
    max_history: int = 20

    # Exit parameters (tighter for faster training cycles)
    initial_stop_percent: float = 20.0  # Initial stop loss % (was 50)
    trailing_stop_percent: float = 15.0  # Trail by 15% from high (was 30)
    profit_lock_threshold: float = 10.0  # Start trailing after 10% profit (was 25)

    # Contract info for dollar-based exits
    contracts: int = 1  # Number of contracts
    
    # Dollar-based thresholds (per position, not per contract)
    min_dollar_profit: float = 30.0  # Take profit at this dollar amount
    max_dollar_loss: float = 75.0  # Stop at this dollar loss
    
    # State
    trailing_active: bool = False
    current_stop_price: float = 0.0

    def update(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update with new price"""
        timestamp = timestamp or datetime.now()
        self.current_price = price

        # Track high/low
        if price > self.high_water_mark:
            self.high_water_mark = price
        if price < self.low_water_mark:
            self.low_water_mark = price

        # Add to history
        self.price_history.append((timestamp, price))
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)

        # Update trailing stop
        self._update_trailing_stop()

    def _update_trailing_stop(self) -> None:
        """Update trailing stop based on high water mark"""
        current_profit_percent = self.profit_percent

        # Activate trailing stop once we hit profit threshold
        if current_profit_percent >= self.profit_lock_threshold:
            self.trailing_active = True

        if self.trailing_active:
            # Trail from high water mark
            trail_amount = self.high_water_mark * (self.trailing_stop_percent / 100)
            new_stop = self.high_water_mark - trail_amount

            # Stop can only go UP (never lower it)
            if new_stop > self.current_stop_price:
                self.current_stop_price = new_stop
                logger.debug(
                    f"Position {self.position_id}: Trailing stop raised to "
                    f"${self.current_stop_price:.2f} (high: ${self.high_water_mark:.2f})"
                )
        else:
            # Use initial stop loss
            self.current_stop_price = self.entry_price * (1 - self.initial_stop_percent / 100)

    @property
    def profit_percent(self) -> float:
        """Current profit as percentage"""
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def dollar_pnl(self) -> float:
        """Current P&L in dollars (per contract = 100 shares)"""
        price_diff = self.current_price - self.entry_price
        return price_diff * 100 * self.contracts  # Options are 100 shares per contract

    @property
    def is_single_contract(self) -> bool:
        """Check if this is a single-contract position"""
        return self.contracts == 1

    @property
    def drawdown_from_high(self) -> float:
        """Current drawdown from high water mark as percentage"""
        if self.high_water_mark <= 0:
            return 0.0
        return ((self.high_water_mark - self.current_price) / self.high_water_mark) * 100

    def get_momentum(self, periods: int = 5) -> float:
        """
        Calculate recent momentum (positive = moving up, negative = moving down)

        Returns: Rate of change as percentage
        """
        if len(self.price_history) < periods + 1:
            return 0.0

        old_price = self.price_history[-(periods + 1)][1]
        new_price = self.price_history[-1][1]

        if old_price <= 0:
            return 0.0

        return ((new_price - old_price) / old_price) * 100

    def is_reversing(self, threshold: float = -5.0) -> bool:
        """
        Detect if price is reversing from recent high

        Args:
            threshold: Momentum threshold to consider reversal

        Returns:
            True if momentum has turned negative significantly
        """
        if not self.trailing_active:
            return False

        # Check if we're in profit but momentum is strongly negative
        if self.profit_percent > 10 and self.get_momentum(3) < threshold:
            return True

        # Check for significant pullback from high
        if self.drawdown_from_high > 15 and self.profit_percent > 0:
            return True

        return False


class AdaptiveExitManager:
    """
    Manages adaptive exits across all positions

    Features:
    - Trailing stops that protect profits
    - Momentum-based exit signals
    - Time-of-day awareness for 0DTE
    - Reversal detection
    - Dollar-based thresholds for single-contract protection
    - OTM-aware exit logic
    """

    def __init__(
        self,
        initial_stop_percent: float = 20.0,  # Tighter for faster training (was 50)
        trailing_stop_percent: float = 15.0,  # Tighter trail (was 30)
        profit_lock_threshold: float = 10.0,  # Faster trailing activation (was 25)
        max_profit_target: float = 50.0,  # Take profit at 50% gain (was 100)
        # Dollar-based thresholds
        enable_dollar_exits: bool = True,
        min_dollar_profit: float = 30.0,  # Take profit at $30 gain
        max_dollar_loss: float = 75.0,  # Stop at $75 loss
        # Single-contract aggressive exits
        single_contract_mode: bool = True,
        single_contract_profit_pct: float = 12.0,  # 12% profit target for single
        single_contract_stop_pct: float = 15.0,  # 15% stop for single
    ):
        """
        Initialize adaptive exit manager

        Args:
            initial_stop_percent: Initial stop loss before trailing kicks in
            trailing_stop_percent: How far to trail from high (lower = tighter)
            profit_lock_threshold: Profit % needed to activate trailing
            max_profit_target: Take profit at this gain %
            enable_dollar_exits: Use dollar-based thresholds
            min_dollar_profit: Minimum dollar profit to take
            max_dollar_loss: Maximum dollar loss to allow
            single_contract_mode: Use tighter exits for single contracts
            single_contract_profit_pct: Profit target % for single contracts
            single_contract_stop_pct: Stop loss % for single contracts
        """
        self.initial_stop_percent = initial_stop_percent
        self.trailing_stop_percent = trailing_stop_percent
        self.profit_lock_threshold = profit_lock_threshold
        self.max_profit_target = max_profit_target
        
        # Dollar-based settings
        self.enable_dollar_exits = enable_dollar_exits
        self.min_dollar_profit = min_dollar_profit
        self.max_dollar_loss = max_dollar_loss
        
        # Single-contract settings
        self.single_contract_mode = single_contract_mode
        self.single_contract_profit_pct = single_contract_profit_pct
        self.single_contract_stop_pct = single_contract_stop_pct

        self.trackers: Dict[str, PositionTracker] = {}

    def register_position(
        self,
        position_id: str,
        entry_price: float,
        entry_time: Optional[datetime] = None,
        contracts: int = 1
    ) -> PositionTracker:
        """Register a new position for tracking"""
        # Use tighter thresholds for single contracts
        stop_pct = self.initial_stop_percent
        profit_pct = self.profit_lock_threshold
        
        if self.single_contract_mode and contracts == 1:
            stop_pct = self.single_contract_stop_pct
            profit_pct = min(8.0, self.profit_lock_threshold)  # Tighter activation
            logger.info(f"Single-contract mode: using tighter exits (stop: {stop_pct}%, profit lock: {profit_pct}%)")
        
        tracker = PositionTracker(
            position_id=position_id,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(),
            current_price=entry_price,
            high_water_mark=entry_price,
            initial_stop_percent=stop_pct,
            trailing_stop_percent=self.trailing_stop_percent,
            profit_lock_threshold=profit_pct,
            contracts=contracts,
            min_dollar_profit=self.min_dollar_profit,
            max_dollar_loss=self.max_dollar_loss,
        )

        self.trackers[position_id] = tracker
        logger.info(f"Registered position {position_id} for adaptive exit tracking")

        return tracker

    def update_price(
        self,
        position_id: str,
        current_price: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[ExitRecommendation]:
        """
        Update price and get exit recommendation

        Args:
            position_id: Position to update
            current_price: Current option price
            timestamp: Optional timestamp

        Returns:
            ExitRecommendation or None if position not found
        """
        tracker = self.trackers.get(position_id)
        if not tracker:
            return None

        tracker.update(current_price, timestamp)

        return self.analyze_exit(position_id)

    def analyze_exit(self, position_id: str) -> ExitRecommendation:
        """
        Analyze position and recommend exit action

        Args:
            position_id: Position to analyze

        Returns:
            ExitRecommendation with signal and reasoning
        """
        tracker = self.trackers.get(position_id)
        if not tracker:
            return ExitRecommendation(
                signal=ExitSignal.HOLD,
                confidence=0,
                urgency="LOW",
                reason="Position not found"
            )

        profit_pct = tracker.profit_percent
        current_price = tracker.current_price
        dollar_pnl = tracker.dollar_pnl
        
        # ========== DOLLAR-BASED EXITS (check first for single contracts) ==========
        if self.enable_dollar_exits:
            # Dollar profit target (especially important for single contracts)
            if dollar_pnl >= tracker.min_dollar_profit:
                return ExitRecommendation(
                    signal=ExitSignal.TAKE_PROFIT,
                    confidence=0.92,
                    urgency="HIGH",
                    reason=f"Dollar profit target hit: ${dollar_pnl:.2f} >= ${tracker.min_dollar_profit:.2f}",
                    suggested_exit_price=current_price,
                    dollar_pnl=dollar_pnl
                )
            
            # Dollar loss limit (hard stop to protect capital)
            if dollar_pnl <= -tracker.max_dollar_loss:
                return ExitRecommendation(
                    signal=ExitSignal.STOP_LOSS,
                    confidence=0.98,
                    urgency="IMMEDIATE",
                    reason=f"Dollar loss limit hit: ${dollar_pnl:.2f} exceeded ${tracker.max_dollar_loss:.2f} max loss",
                    suggested_exit_price=current_price,
                    dollar_pnl=dollar_pnl
                )
        
        # ========== SINGLE CONTRACT AGGRESSIVE EXITS ==========
        if self.single_contract_mode and tracker.is_single_contract:
            # Tighter profit target for single contracts
            if profit_pct >= self.single_contract_profit_pct:
                return ExitRecommendation(
                    signal=ExitSignal.TAKE_PROFIT,
                    confidence=0.90,
                    urgency="HIGH",
                    reason=f"Single-contract profit target: {profit_pct:.1f}% >= {self.single_contract_profit_pct}% (${dollar_pnl:.2f})",
                    suggested_exit_price=current_price,
                    dollar_pnl=dollar_pnl
                )
            
            # Tighter stop for single contracts  
            if profit_pct <= -self.single_contract_stop_pct:
                return ExitRecommendation(
                    signal=ExitSignal.STOP_LOSS,
                    confidence=0.95,
                    urgency="IMMEDIATE",
                    reason=f"Single-contract stop: {profit_pct:.1f}% loss exceeded {self.single_contract_stop_pct}% (${dollar_pnl:.2f})",
                    suggested_exit_price=current_price,
                    dollar_pnl=dollar_pnl
                )

        # ========== PERCENTAGE-BASED EXITS ==========
        # Check max profit target
        if profit_pct >= self.max_profit_target:
            return ExitRecommendation(
                signal=ExitSignal.TAKE_PROFIT,
                confidence=0.95,
                urgency="HIGH",
                reason=f"Hit {self.max_profit_target}% profit target (${dollar_pnl:.2f})",
                suggested_exit_price=current_price,
                dollar_pnl=dollar_pnl
            )

        # Check trailing stop
        if tracker.trailing_active and current_price <= tracker.current_stop_price:
            return ExitRecommendation(
                signal=ExitSignal.TRAILING_STOP,
                confidence=0.90,
                urgency="IMMEDIATE",
                reason=f"Trailing stop hit at ${tracker.current_stop_price:.2f} "
                       f"(locked in {profit_pct:.1f}% profit, ${dollar_pnl:.2f})",
                suggested_exit_price=current_price,
                dollar_pnl=dollar_pnl
            )

        # Check for reversal while in profit
        if tracker.is_reversing():
            return ExitRecommendation(
                signal=ExitSignal.REVERSAL,
                confidence=0.75,
                urgency="HIGH",
                reason=f"Momentum reversing, drawdown {tracker.drawdown_from_high:.1f}% "
                       f"from high (still {profit_pct:.1f}% profit, ${dollar_pnl:.2f})",
                suggested_exit_price=current_price,
                dollar_pnl=dollar_pnl
            )

        # Check momentum exit (strong negative momentum while profitable)
        momentum = tracker.get_momentum(3)
        if profit_pct > 15 and momentum < -8:
            return ExitRecommendation(
                signal=ExitSignal.MOMENTUM_EXIT,
                confidence=0.70,
                urgency="MEDIUM",
                reason=f"Strong negative momentum ({momentum:.1f}%), "
                       f"consider taking {profit_pct:.1f}% profit (${dollar_pnl:.2f})",
                suggested_exit_price=current_price,
                dollar_pnl=dollar_pnl
            )

        # Check hard stop loss (before trailing activates)
        if not tracker.trailing_active:
            hard_stop = tracker.entry_price * (1 - self.initial_stop_percent / 100)
            if current_price <= hard_stop:
                return ExitRecommendation(
                    signal=ExitSignal.STOP_LOSS,
                    confidence=0.95,
                    urgency="IMMEDIATE",
                    reason=f"Hard stop loss hit at ${hard_stop:.2f} (${dollar_pnl:.2f} loss)",
                    suggested_exit_price=current_price,
                    dollar_pnl=dollar_pnl
                )

        # Check time decay risk for 0DTE
        time_signal = self._check_time_decay_risk(tracker)
        if time_signal:
            # Add dollar_pnl to time decay signals
            time_signal.dollar_pnl = dollar_pnl
            return time_signal

        # Default: hold
        hold_reason = f"P&L: {profit_pct:.1f}% (${dollar_pnl:.2f})"
        if tracker.trailing_active:
            hold_reason += f", trailing stop at ${tracker.current_stop_price:.2f}"
        else:
            hold_reason += f", stop at ${tracker.current_stop_price:.2f}"

        return ExitRecommendation(
            signal=ExitSignal.HOLD,
            confidence=0.5,
            urgency="LOW",
            reason=hold_reason,
            dollar_pnl=dollar_pnl
        )

    def _check_time_decay_risk(self, tracker: PositionTracker) -> Optional[ExitRecommendation]:
        """Check for time-based exit signals on 0DTE"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        current_time = now.time()

        profit_pct = tracker.profit_percent

        # After 2:30 PM - theta accelerates dramatically
        if current_time >= time(14, 30):
            # If profitable, consider taking it
            if profit_pct > 20:
                return ExitRecommendation(
                    signal=ExitSignal.TIME_DECAY,
                    confidence=0.65,
                    urgency="MEDIUM",
                    reason=f"After 2:30 PM - theta accelerating. "
                           f"Consider taking {profit_pct:.1f}% profit",
                    suggested_exit_price=tracker.current_price
                )
            # If losing, don't hold hoping for miracle
            elif profit_pct < -20:
                return ExitRecommendation(
                    signal=ExitSignal.TIME_DECAY,
                    confidence=0.70,
                    urgency="HIGH",
                    reason="After 2:30 PM with loss - theta will make it worse",
                    suggested_exit_price=tracker.current_price
                )

        # After 3:30 PM - very high risk
        if current_time >= time(15, 30):
            if profit_pct > 0:
                return ExitRecommendation(
                    signal=ExitSignal.TIME_DECAY,
                    confidence=0.80,
                    urgency="HIGH",
                    reason=f"After 3:30 PM - take the {profit_pct:.1f}% profit now",
                    suggested_exit_price=tracker.current_price
                )

        return None

    def remove_position(self, position_id: str) -> None:
        """Remove position from tracking"""
        if position_id in self.trackers:
            del self.trackers[position_id]
            logger.debug(f"Removed position {position_id} from exit tracking")

    def get_all_recommendations(self) -> Dict[str, ExitRecommendation]:
        """Get exit recommendations for all tracked positions"""
        return {
            pos_id: self.analyze_exit(pos_id)
            for pos_id in self.trackers
        }

    def get_position_summary(self, position_id: str) -> Dict[str, Any]:
        """Get detailed summary for a position"""
        tracker = self.trackers.get(position_id)
        if not tracker:
            return {"error": "Position not found"}

        return {
            "position_id": position_id,
            "entry_price": tracker.entry_price,
            "current_price": tracker.current_price,
            "high_water_mark": tracker.high_water_mark,
            "profit_percent": tracker.profit_percent,
            "drawdown_from_high": tracker.drawdown_from_high,
            "momentum_3": tracker.get_momentum(3),
            "momentum_5": tracker.get_momentum(5),
            "trailing_active": tracker.trailing_active,
            "current_stop": tracker.current_stop_price,
            "is_reversing": tracker.is_reversing(),
        }
