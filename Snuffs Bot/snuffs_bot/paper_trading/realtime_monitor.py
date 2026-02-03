"""
Real-Time Position Monitor

Provides sub-second position monitoring for 0DTE options.
Uses WebSocket streaming for instant price updates and
AI-driven dynamic exit decisions.

NO DELAYS - Continuous scanning when positions are open.
"""

import asyncio
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
import pytz

from loguru import logger

from .adaptive_exits import ExitSignal, ExitRecommendation


class DynamicExitMode(Enum):
    """Dynamic exit behavior modes"""
    AGGRESSIVE = "AGGRESSIVE"      # Quick profits, tight stops
    BALANCED = "BALANCED"          # Standard trailing
    RUNNER = "RUNNER"              # Let winners run
    SCALP = "SCALP"                # Very quick in/out


@dataclass
class DynamicTargets:
    """Dynamically adjusted profit/loss targets"""
    current_profit_target: float        # Current profit % to take
    current_stop_loss: float            # Current stop loss %
    trailing_trigger: float             # When to start trailing
    trailing_distance: float            # How far to trail
    
    # Market-adjusted factors
    volatility_multiplier: float = 1.0  # Higher in volatile markets
    momentum_factor: float = 0.0        # Positive = let run, negative = exit fast
    time_decay_factor: float = 1.0      # Higher as expiration approaches
    
    # AI overrides
    ai_recommends_hold: bool = False
    ai_target_override: Optional[float] = None
    ai_reason: str = ""
    
    def get_effective_profit_target(self) -> float:
        """Get the current effective profit target considering all factors"""
        if self.ai_target_override is not None:
            return self.ai_target_override
            
        base_target = self.current_profit_target
        
        # Adjust for volatility (higher vol = tighter target)
        vol_adjusted = base_target / self.volatility_multiplier
        
        # Adjust for momentum (positive momentum = let it run)
        if self.momentum_factor > 2:  # Strong upward momentum
            vol_adjusted *= 1.2  # Raise target 20%
        elif self.momentum_factor < -3:  # Reversing
            vol_adjusted *= 0.8  # Lower target 20%
            
        # Adjust for time decay (later in day = lower target)
        time_adjusted = vol_adjusted / self.time_decay_factor
        
        return max(time_adjusted, 10.0)  # Minimum 10% target


@dataclass 
class MonitoredPosition:
    """Enhanced position tracking for real-time monitoring"""
    position_id: str
    symbol: str
    entry_price: float
    entry_time: datetime
    contracts: int
    strategy_type: str
    
    # Real-time price data
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_mid: float = 0.0
    last_update: Optional[datetime] = None
    
    # Price history for momentum (last 60 seconds)
    price_ticks: List[tuple] = field(default_factory=list)  # [(timestamp, price), ...]
    
    # Dynamic targets
    targets: Optional[DynamicTargets] = None
    
    # Performance tracking
    high_price: float = 0.0
    low_price: float = float('inf')
    
    # Exit mode
    mode: DynamicExitMode = DynamicExitMode.BALANCED
    
    def update_price(self, bid: float, ask: float, timestamp: Optional[datetime] = None) -> None:
        """Update with new price tick"""
        ts = timestamp or datetime.now()
        mid = (bid + ask) / 2
        
        self.current_bid = bid
        self.current_ask = ask
        self.current_mid = mid
        self.last_update = ts
        
        # Track high/low
        if mid > self.high_price:
            self.high_price = mid
        if mid < self.low_price:
            self.low_price = mid
        
        # Add to price history (keep last 60 seconds)
        self.price_ticks.append((ts, mid))
        cutoff = ts.timestamp() - 60
        self.price_ticks = [t for t in self.price_ticks if t[0].timestamp() > cutoff]
    
    @property
    def profit_percent(self) -> float:
        """Current profit as percentage"""
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_mid - self.entry_price) / self.entry_price) * 100
    
    @property
    def drawdown_from_high(self) -> float:
        """Current drawdown from high"""
        if self.high_price <= 0:
            return 0.0
        return ((self.high_price - self.current_mid) / self.high_price) * 100
    
    def get_momentum(self, seconds: int = 10) -> float:
        """Calculate momentum over last N seconds"""
        if len(self.price_ticks) < 2:
            return 0.0
        
        now = datetime.now().timestamp()
        cutoff = now - seconds
        recent = [t for t in self.price_ticks if t[0].timestamp() > cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        old_price = recent[0][1]
        new_price = recent[-1][1]
        
        if old_price <= 0:
            return 0.0
        
        return ((new_price - old_price) / old_price) * 100


class RealTimePositionMonitor:
    """
    Real-time position monitor with no delays
    
    When positions are open:
    - Processes every price tick instantly
    - Updates targets dynamically based on conditions
    - Can exit in <100ms from signal
    
    When no positions:
    - Sleeps to save resources
    """
    
    def __init__(
        self,
        exit_callback: Optional[Callable] = None,
        ai_exit_callback: Optional[Callable] = None,
        default_mode: DynamicExitMode = DynamicExitMode.BALANCED,
    ):
        """
        Initialize real-time monitor
        
        Args:
            exit_callback: Async function to call when exit triggered
                          signature: async def callback(position_id, reason, exit_price)
            ai_exit_callback: Optional AI agent to consult for exit decisions
            default_mode: Default exit behavior mode
        """
        self.exit_callback = exit_callback
        self.ai_exit_callback = ai_exit_callback
        self.default_mode = default_mode
        
        self.positions: Dict[str, MonitoredPosition] = {}
        self.pending_exits: Set[str] = set()  # Positions being exited
        
        # Base targets by mode
        self.mode_targets = {
            DynamicExitMode.SCALP: {
                "profit_target": 20.0,
                "stop_loss": 25.0,
                "trailing_trigger": 15.0,
                "trailing_distance": 10.0,
            },
            DynamicExitMode.AGGRESSIVE: {
                "profit_target": 35.0,
                "stop_loss": 30.0,
                "trailing_trigger": 20.0,
                "trailing_distance": 15.0,
            },
            DynamicExitMode.BALANCED: {
                "profit_target": 15.0,  # Tighter for faster learning (was 50)
                "stop_loss": 40.0,
                "trailing_trigger": 25.0,
                "trailing_distance": 20.0,
            },
            DynamicExitMode.RUNNER: {
                "profit_target": 100.0,
                "stop_loss": 20.0,  # Tighter for faster learning (was 50)
                "trailing_trigger": 30.0,
                "trailing_distance": 25.0,
            },
        }
        
        # Monitoring state
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._price_queue: asyncio.Queue = asyncio.Queue()
        
        # Stats
        self.ticks_processed = 0
        self.exits_triggered = 0
        
        logger.info("Real-time position monitor initialized")
    
    async def start(self) -> None:
        """Start the real-time monitor"""
        if self.is_running:
            return
        
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Real-time position monitor started")
    
    async def stop(self) -> None:
        """Stop the monitor"""
        self.is_running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time position monitor stopped")
    
    def register_position(
        self,
        position_id: str,
        symbol: str,
        entry_price: float,
        contracts: int,
        strategy_type: str,
        mode: Optional[DynamicExitMode] = None,
    ) -> MonitoredPosition:
        """Register a position for real-time monitoring"""
        pos_mode = mode or self.default_mode
        base_targets = self.mode_targets[pos_mode]
        
        position = MonitoredPosition(
            position_id=position_id,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=datetime.now(),
            contracts=contracts,
            strategy_type=strategy_type,
            current_mid=entry_price,
            high_price=entry_price,
            low_price=entry_price,
            mode=pos_mode,
            targets=DynamicTargets(
                current_profit_target=base_targets["profit_target"],
                current_stop_loss=base_targets["stop_loss"],
                trailing_trigger=base_targets["trailing_trigger"],
                trailing_distance=base_targets["trailing_distance"],
            ),
        )
        
        self.positions[position_id] = position
        logger.info(
            f"Registered position {position_id} for real-time monitoring "
            f"(mode={pos_mode.value}, target={base_targets['profit_target']}%)"
        )
        
        return position
    
    def remove_position(self, position_id: str) -> None:
        """Remove position from monitoring"""
        if position_id in self.positions:
            del self.positions[position_id]
            self.pending_exits.discard(position_id)
            logger.debug(f"Removed position {position_id} from real-time monitor")
    
    async def process_price_tick(
        self,
        position_id: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ExitRecommendation]:
        """
        Process a price tick - THIS IS THE HOT PATH
        
        Called for every price update, must be fast.
        Returns exit recommendation if action needed.
        """
        position = self.positions.get(position_id)
        if not position or position_id in self.pending_exits:
            return None
        
        # Update price
        position.update_price(bid, ask, timestamp)
        self.ticks_processed += 1
        
        # Update dynamic targets based on market conditions
        self._update_dynamic_targets(position)
        
        # Check exit conditions
        recommendation = self._check_exit_conditions(position)
        
        # If exit recommended with high urgency, trigger immediately
        if recommendation and recommendation.urgency == "IMMEDIATE":
            await self._trigger_exit(position, recommendation)
        
        return recommendation
    
    def _update_dynamic_targets(self, position: MonitoredPosition) -> None:
        """Dynamically adjust targets based on conditions"""
        if not position.targets:
            return
        
        targets = position.targets
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        current_time = now.time()
        
        # Update momentum factor
        targets.momentum_factor = position.get_momentum(10)  # Last 10 seconds
        
        # Update time decay factor (increases as day progresses)
        if current_time >= time(15, 45):
            targets.time_decay_factor = 2.0  # Very tight near close
        elif current_time >= time(15, 30):
            targets.time_decay_factor = 1.5
        elif current_time >= time(14, 30):
            targets.time_decay_factor = 1.2
        else:
            targets.time_decay_factor = 1.0
        
        # Update volatility multiplier based on price swings
        if len(position.price_ticks) > 5:
            prices = [t[1] for t in position.price_ticks[-20:]]
            if len(prices) > 1:
                price_range = max(prices) - min(prices)
                avg_price = sum(prices) / len(prices)
                if avg_price > 0:
                    vol = (price_range / avg_price) * 100
                    if vol > 10:
                        targets.volatility_multiplier = 1.5  # High volatility
                    elif vol > 5:
                        targets.volatility_multiplier = 1.2
                    else:
                        targets.volatility_multiplier = 1.0
    
    def _check_exit_conditions(self, position: MonitoredPosition) -> Optional[ExitRecommendation]:
        """Check all exit conditions - must be fast"""
        if not position.targets:
            return None
        
        targets = position.targets
        profit_pct = position.profit_percent
        drawdown = position.drawdown_from_high
        momentum = targets.momentum_factor
        effective_target = targets.get_effective_profit_target()
        
        # 1. Check profit target hit
        if profit_pct >= effective_target:
            return ExitRecommendation(
                signal=ExitSignal.TAKE_PROFIT,
                confidence=0.95,
                urgency="IMMEDIATE",
                reason=f"Hit {effective_target:.1f}% profit target (current: {profit_pct:.1f}%)",
                suggested_exit_price=position.current_mid,
            )
        
        # 2. Check trailing stop (after profit threshold)
        if profit_pct >= targets.trailing_trigger:
            trailing_stop_pct = targets.trailing_distance
            if drawdown >= trailing_stop_pct:
                locked_profit = profit_pct
                return ExitRecommendation(
                    signal=ExitSignal.TRAILING_STOP,
                    confidence=0.92,
                    urgency="IMMEDIATE",
                    reason=f"Trailing stop hit - locked in {locked_profit:.1f}% profit "
                           f"(drawdown {drawdown:.1f}% from high)",
                    suggested_exit_price=position.current_mid,
                )
        
        # 3. Check stop loss (before trailing activates)
        if profit_pct <= -targets.current_stop_loss:
            return ExitRecommendation(
                signal=ExitSignal.STOP_LOSS,
                confidence=0.95,
                urgency="IMMEDIATE",
                reason=f"Stop loss hit at {targets.current_stop_loss:.1f}%",
                suggested_exit_price=position.current_mid,
            )
        
        # 4. Check momentum reversal while profitable
        if profit_pct > 20 and momentum < -5:
            # Strong negative momentum while profitable
            if drawdown > 10:
                return ExitRecommendation(
                    signal=ExitSignal.REVERSAL,
                    confidence=0.80,
                    urgency="HIGH",
                    reason=f"Momentum reversing ({momentum:.1f}% in 10s), "
                           f"protect {profit_pct:.1f}% profit",
                    suggested_exit_price=position.current_mid,
                )
        
        # 5. Time-based exits for 0DTE
        et = pytz.timezone('US/Eastern')
        current_time = datetime.now(et).time()
        
        if current_time >= time(15, 50):
            # Force exit near close
            return ExitRecommendation(
                signal=ExitSignal.TIME_DECAY,
                confidence=0.99,
                urgency="IMMEDIATE",
                reason="Forced exit - market closing soon",
                suggested_exit_price=position.current_mid,
            )
        
        if current_time >= time(15, 30) and profit_pct > 10:
            return ExitRecommendation(
                signal=ExitSignal.TIME_DECAY,
                confidence=0.75,
                urgency="HIGH",
                reason=f"Late day profit taking - securing {profit_pct:.1f}%",
                suggested_exit_price=position.current_mid,
            )
        
        # 6. AI override check (if profitable and AI says hold)
        if targets.ai_recommends_hold and profit_pct > 15:
            return None  # AI recommends holding
        
        return None
    
    async def _trigger_exit(
        self,
        position: MonitoredPosition,
        recommendation: ExitRecommendation,
    ) -> None:
        """Trigger an immediate exit"""
        position_id = position.position_id
        
        if position_id in self.pending_exits:
            return  # Already being exited
        
        self.pending_exits.add(position_id)
        self.exits_triggered += 1
        
        logger.info(
            f"🚨 REAL-TIME EXIT: {position_id} | {recommendation.signal.value} | "
            f"P&L: {position.profit_percent:.1f}% | {recommendation.reason}"
        )
        
        if self.exit_callback:
            try:
                await self.exit_callback(
                    position_id=position_id,
                    reason=recommendation.signal.value,
                    exit_price=position.current_mid,
                    recommendation=recommendation,
                )
            except Exception as e:
                logger.error(f"Exit callback failed: {e}")
                self.pending_exits.discard(position_id)
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                if not self.positions:
                    # No positions - sleep longer
                    await asyncio.sleep(1.0)
                    continue
                
                # With positions - check more frequently
                # (Price ticks come through process_price_tick, 
                # this is backup time-based check)
                for pos_id in list(self.positions.keys()):
                    position = self.positions.get(pos_id)
                    if not position:
                        continue
                    
                    # Do periodic exit check
                    recommendation = self._check_exit_conditions(position)
                    
                    if recommendation and recommendation.urgency in ("HIGH", "IMMEDIATE"):
                        await self._trigger_exit(position, recommendation)
                
                await asyncio.sleep(0.1)  # 100ms between checks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Real-time monitor error: {e}")
                await asyncio.sleep(0.5)
    
    async def request_ai_exit_analysis(
        self,
        position_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Request AI analysis for exit decision (optional)"""
        position = self.positions.get(position_id)
        if not position or not self.ai_exit_callback:
            return None
        
        try:
            result = await self.ai_exit_callback(
                position_id=position_id,
                entry_price=position.entry_price,
                current_price=position.current_mid,
                profit_percent=position.profit_percent,
                drawdown_from_high=position.drawdown_from_high,
                momentum=position.get_momentum(10),
                time_held_seconds=(datetime.now() - position.entry_time).total_seconds(),
            )
            
            if result and position.targets:
                # Apply AI recommendations
                if result.get("recommend_hold"):
                    position.targets.ai_recommends_hold = True
                    position.targets.ai_reason = result.get("reason", "AI recommends hold")
                if result.get("adjusted_target"):
                    position.targets.ai_target_override = result["adjusted_target"]
            
            return result
            
        except Exception as e:
            logger.warning(f"AI exit analysis failed: {e}")
            return None
    
    def adjust_mode(
        self,
        position_id: str,
        new_mode: DynamicExitMode,
    ) -> None:
        """Dynamically adjust the exit mode for a position"""
        position = self.positions.get(position_id)
        if not position:
            return
        
        base_targets = self.mode_targets[new_mode]
        position.mode = new_mode
        position.targets = DynamicTargets(
            current_profit_target=base_targets["profit_target"],
            current_stop_loss=base_targets["stop_loss"],
            trailing_trigger=base_targets["trailing_trigger"],
            trailing_distance=base_targets["trailing_distance"],
            # Preserve AI overrides
            ai_recommends_hold=position.targets.ai_recommends_hold if position.targets else False,
            ai_target_override=position.targets.ai_target_override if position.targets else None,
        )
        
        logger.info(f"Adjusted {position_id} to {new_mode.value} mode")
    
    def get_position_status(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a position"""
        position = self.positions.get(position_id)
        if not position:
            return None
        
        return {
            "position_id": position_id,
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "current_mid": position.current_mid,
            "current_bid": position.current_bid,
            "current_ask": position.current_ask,
            "profit_percent": position.profit_percent,
            "drawdown_from_high": position.drawdown_from_high,
            "high_price": position.high_price,
            "momentum_10s": position.get_momentum(10),
            "mode": position.mode.value,
            "effective_target": position.targets.get_effective_profit_target() if position.targets else None,
            "trailing_active": position.profit_percent >= (position.targets.trailing_trigger if position.targets else 25),
            "time_held": str(datetime.now() - position.entry_time),
            "ticks_received": len(position.price_ticks),
            "last_update": position.last_update.isoformat() if position.last_update else None,
            "pending_exit": position_id in self.pending_exits,
        }
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status for all monitored positions"""
        return {
            "positions": [
                self.get_position_status(pos_id)
                for pos_id in self.positions
            ],
            "ticks_processed": self.ticks_processed,
            "exits_triggered": self.exits_triggered,
            "is_running": self.is_running,
        }
