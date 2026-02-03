"""
Pattern Recognition System

Identifies recurring patterns in market conditions and trade outcomes
to improve future decision-making.
"""

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import statistics

from loguru import logger


class PatternType(Enum):
    """Types of patterns the system can recognize"""
    MARKET_REGIME = "MARKET_REGIME"
    TIME_OF_DAY = "TIME_OF_DAY"
    VIX_LEVEL = "VIX_LEVEL"
    STRATEGY_CONDITION = "STRATEGY_CONDITION"
    MOMENTUM = "MOMENTUM"
    REVERSAL = "REVERSAL"
    CONSOLIDATION = "CONSOLIDATION"


@dataclass
class MarketPattern:
    """A recognized market pattern with associated outcomes"""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str

    # Pattern conditions
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Associated outcomes
    total_occurrences: int = 0
    winning_occurrences: int = 0
    losing_occurrences: int = 0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0

    # Strategy recommendations
    preferred_strategies: List[str] = field(default_factory=list)
    avoid_strategies: List[str] = field(default_factory=list)

    # Confidence
    confidence_score: float = 0.0
    last_updated: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate for this pattern"""
        if self.total_occurrences == 0:
            return 0.0
        return self.winning_occurrences / self.total_occurrences

    def update_with_outcome(self, pnl: float, strategy: str) -> None:
        """Update pattern statistics with new outcome"""
        self.total_occurrences += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_occurrences += 1
            if strategy not in self.preferred_strategies:
                # Add if consistently profitable
                if self.win_rate > 0.6:
                    self.preferred_strategies.append(strategy)
        else:
            self.losing_occurrences += 1
            if self.win_rate < 0.4 and strategy not in self.avoid_strategies:
                self.avoid_strategies.append(strategy)

        self.avg_pnl = self.total_pnl / self.total_occurrences
        self.confidence_score = min(1.0, self.total_occurrences / 20)  # Full confidence at 20 samples
        self.last_updated = datetime.now()


@dataclass
class PatternMatch:
    """A match between current conditions and a known pattern"""
    pattern: MarketPattern
    match_strength: float  # 0-1, how closely conditions match
    expected_outcome: str  # "FAVORABLE", "UNFAVORABLE", "NEUTRAL"
    recommended_action: str
    reasoning: str


class PatternRecognizer:
    """
    Recognizes market patterns and provides recommendations based on
    historical outcomes associated with those patterns.
    """

    def __init__(self):
        """Initialize the pattern recognizer"""
        self.patterns: Dict[str, MarketPattern] = {}
        self._initialize_base_patterns()

    def _initialize_base_patterns(self) -> None:
        """Initialize base patterns from domain knowledge"""
        # VIX-based patterns
        self._add_pattern(MarketPattern(
            pattern_id="low_vix_stable",
            pattern_type=PatternType.VIX_LEVEL,
            name="Low VIX Stable",
            description="VIX below 15 with stable conditions - cheap premiums for buying",
            conditions={"vix_max": 15, "vix_change_max": 1.0},
            # Low VIX = cheap options, good for buying directional plays
        ))

        self._add_pattern(MarketPattern(
            pattern_id="elevated_vix",
            pattern_type=PatternType.VIX_LEVEL,
            name="Elevated VIX",
            description="VIX between 20-30, expensive premiums but high movement potential",
            conditions={"vix_min": 20, "vix_max": 30},
            # Expensive options but could work if directional conviction is strong
        ))

        self._add_pattern(MarketPattern(
            pattern_id="high_vix_spike",
            pattern_type=PatternType.VIX_LEVEL,
            name="High VIX Spike",
            description="VIX above 30 - extremely expensive premiums, avoid buying",
            conditions={"vix_min": 30},
            avoid_strategies=["LONG_CALL", "LONG_PUT"],  # Too expensive to buy options
        ))

        # Time-based patterns
        self._add_pattern(MarketPattern(
            pattern_id="morning_volatility",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Morning Volatility Window",
            description="First 30 minutes after market open - big moves possible",
            conditions={"time_start": "09:30", "time_end": "10:00"},
            # Good time for directional plays if trend is clear
        ))

        self._add_pattern(MarketPattern(
            pattern_id="midday_lull",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Midday Lull",
            description="Lower volatility period 11:30-13:30 - limited movement",
            conditions={"time_start": "11:30", "time_end": "13:30"},
            avoid_strategies=["LONG_CALL", "LONG_PUT"],  # Theta burn with no movement
        ))

        self._add_pattern(MarketPattern(
            pattern_id="power_hour",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Power Hour",
            description="Last hour of trading with increased volume - strong moves",
            conditions={"time_start": "15:00", "time_end": "16:00"},
            # Direction determined by other factors - can be great for directional plays
        ))

        # Market regime patterns
        self._add_pattern(MarketPattern(
            pattern_id="bullish_trend",
            pattern_type=PatternType.MARKET_REGIME,
            name="Bullish Trend",
            description="SPY trending up with positive momentum - buy calls",
            conditions={"spy_direction": "UP", "momentum": "POSITIVE"},
            preferred_strategies=["LONG_CALL"],
            avoid_strategies=["LONG_PUT"],
        ))

        self._add_pattern(MarketPattern(
            pattern_id="bearish_trend",
            pattern_type=PatternType.MARKET_REGIME,
            name="Bearish Trend",
            description="SPY trending down with negative momentum - buy puts",
            conditions={"spy_direction": "DOWN", "momentum": "NEGATIVE"},
            preferred_strategies=["LONG_PUT"],
            avoid_strategies=["LONG_CALL"],
        ))

        self._add_pattern(MarketPattern(
            pattern_id="range_bound",
            pattern_type=PatternType.MARKET_REGIME,
            name="Range Bound",
            description="SPY oscillating within a defined range - no directional edge",
            conditions={"spy_direction": "NEUTRAL", "range_percent": 0.5},
            avoid_strategies=["LONG_CALL", "LONG_PUT"],  # No clear direction, theta burns you
        ))

        # Momentum patterns
        self._add_pattern(MarketPattern(
            pattern_id="morning_gap_up",
            pattern_type=PatternType.MOMENTUM,
            name="Morning Gap Up",
            description="SPY gaps up at open - bullish momentum",
            conditions={"gap_direction": "UP", "gap_percent_min": 0.3},
            preferred_strategies=["LONG_CALL"],  # Ride the bullish momentum
        ))

        self._add_pattern(MarketPattern(
            pattern_id="morning_gap_down",
            pattern_type=PatternType.MOMENTUM,
            name="Morning Gap Down",
            description="SPY gaps down at open - bearish momentum",
            conditions={"gap_direction": "DOWN", "gap_percent_min": 0.3},
            preferred_strategies=["LONG_PUT"],  # Ride the bearish momentum
        ))

        self._add_pattern(MarketPattern(
            pattern_id="reversal_after_trend",
            pattern_type=PatternType.REVERSAL,
            name="Trend Exhaustion Reversal",
            description="Signs of trend exhaustion after extended move",
            conditions={"trend_duration_min": 2, "reversal_signal": True},
            # Direction depends on which way the reversal goes
        ))

        # ========== AGGRESSIVE PATTERNS ==========

        # Gap Fill Patterns (gaps tend to fill same day)
        self._add_pattern(MarketPattern(
            pattern_id="gap_up_fade",
            pattern_type=PatternType.MOMENTUM,
            name="Gap Up Fade Setup",
            description="Large gap up (>0.5%) showing early weakness - gap fill likely",
            conditions={"gap_direction": "UP", "gap_percent_min": 0.5, "early_weakness": True},
            preferred_strategies=["LONG_PUT"],  # Fade the gap up with puts
        ))

        self._add_pattern(MarketPattern(
            pattern_id="gap_down_fade",
            pattern_type=PatternType.MOMENTUM,
            name="Gap Down Fade Setup",
            description="Large gap down (>0.5%) showing early strength - gap fill likely",
            conditions={"gap_direction": "DOWN", "gap_percent_min": 0.5, "early_strength": True},
            preferred_strategies=["LONG_CALL"],  # Fade the gap down with calls
        ))

        # V-Reversal Patterns (quick reversals after sharp moves)
        self._add_pattern(MarketPattern(
            pattern_id="v_bottom_reversal",
            pattern_type=PatternType.REVERSAL,
            name="V-Bottom Reversal",
            description="Sharp selloff followed by aggressive buying - bullish reversal",
            conditions={"selloff_percent_min": 0.5, "bounce_percent_min": 0.3, "time_minutes_max": 30},
            preferred_strategies=["LONG_CALL"],  # Buy calls on the bounce
            avoid_strategies=["LONG_PUT"],
        ))

        self._add_pattern(MarketPattern(
            pattern_id="v_top_reversal",
            pattern_type=PatternType.REVERSAL,
            name="V-Top Reversal",
            description="Sharp rally followed by aggressive selling - bearish reversal",
            conditions={"rally_percent_min": 0.5, "drop_percent_min": 0.3, "time_minutes_max": 30},
            preferred_strategies=["LONG_PUT"],  # Buy puts on the drop
            avoid_strategies=["LONG_CALL"],
        ))

        # Momentum Breakout/Breakdown Patterns
        self._add_pattern(MarketPattern(
            pattern_id="momentum_breakout",
            pattern_type=PatternType.MOMENTUM,
            name="Momentum Breakout",
            description="SPY breaks above prior high with volume - continuation likely",
            conditions={"spy_direction": "UP", "volume_ratio_min": 1.5, "new_high": True},
            preferred_strategies=["LONG_CALL"],  # Buy calls for bullish breakout
        ))

        self._add_pattern(MarketPattern(
            pattern_id="momentum_breakdown",
            pattern_type=PatternType.MOMENTUM,
            name="Momentum Breakdown",
            description="SPY breaks below prior low with volume - continuation likely",
            conditions={"spy_direction": "DOWN", "volume_ratio_min": 1.5, "new_low": True},
            preferred_strategies=["LONG_PUT"],  # Buy puts for bearish breakdown
        ))

        # VIX Spike Fade (VIX spikes often mean-revert)
        self._add_pattern(MarketPattern(
            pattern_id="vix_spike_fade",
            pattern_type=PatternType.VIX_LEVEL,
            name="VIX Spike Fade",
            description="VIX spikes >15% intraday then starts declining - fear subsiding",
            conditions={"vix_spike_percent_min": 15, "vix_declining": True},
            preferred_strategies=["LONG_CALL"],  # Fear subsiding = market bouncing = calls
        ))

        # VIX Crush After Event
        self._add_pattern(MarketPattern(
            pattern_id="vix_crush",
            pattern_type=PatternType.VIX_LEVEL,
            name="VIX Crush",
            description="VIX dropping rapidly (>10%) - bullish market sentiment",
            conditions={"vix_change_percent_min": -10},
            preferred_strategies=["LONG_CALL"],  # VIX crush usually = market up = calls
        ))

        # VWAP Patterns (institutional levels)
        self._add_pattern(MarketPattern(
            pattern_id="vwap_hold_bullish",
            pattern_type=PatternType.MARKET_REGIME,
            name="VWAP Hold Bullish",
            description="SPY testing and holding above VWAP - institutions buying",
            conditions={"above_vwap": True, "vwap_bounce": True},
            preferred_strategies=["LONG_CALL"],  # Institutions buying = calls
        ))

        self._add_pattern(MarketPattern(
            pattern_id="vwap_rejection_bearish",
            pattern_type=PatternType.MARKET_REGIME,
            name="VWAP Rejection Bearish",
            description="SPY rejected at VWAP from below - institutions selling",
            conditions={"below_vwap": True, "vwap_rejection": True},
            preferred_strategies=["LONG_PUT"],  # Institutions selling = puts
        ))

        # Time-Based Aggressive Windows
        self._add_pattern(MarketPattern(
            pattern_id="opening_range_breakout",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Opening Range Breakout",
            description="SPY breaks 9:30-10:00 range with conviction - ride the break",
            conditions={"time_start": "10:00", "time_end": "10:30", "range_break": True},
            # Direction determined by breakout direction (up = calls, down = puts)
        ))

        self._add_pattern(MarketPattern(
            pattern_id="lunch_reversal",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Lunch Hour Reversal",
            description="Trend reversal during low-volume lunch (11:30-13:00)",
            conditions={"time_start": "11:30", "time_end": "13:00", "reversal_signal": True},
            # Play the reversal direction
        ))

        self._add_pattern(MarketPattern(
            pattern_id="afternoon_trend",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Afternoon Trend Continuation",
            description="Strong trend established by 14:00 - likely continues into close",
            conditions={"time_start": "14:00", "time_end": "15:00", "strong_trend": True},
            # Play the trend direction (up = calls, down = puts)
        ))

        # Theta Burn Patterns (0DTE specific) - NOTE: Theta works AGAINST long options!
        self._add_pattern(MarketPattern(
            pattern_id="theta_burn_caution",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Theta Burn Caution",
            description="10:00-11:30 rapid theta decay window - need quick directional move",
            conditions={"time_start": "10:00", "time_end": "11:30", "vix_max": 20},
            # For long options: need strong conviction, theta burning your premium
        ))

        self._add_pattern(MarketPattern(
            pattern_id="late_day_gamma_opportunity",
            pattern_type=PatternType.TIME_OF_DAY,
            name="Late Day Gamma Opportunity",
            description="After 14:00 gamma explodes - big moves in either direction",
            conditions={"time_start": "14:00", "time_end": "16:00"},
            # Can be great for long options if direction is right (huge payoff potential)
        ))

        # Extreme Move Patterns
        self._add_pattern(MarketPattern(
            pattern_id="extreme_move_mean_revert",
            pattern_type=PatternType.MOMENTUM,
            name="Extreme Move Mean Reversion",
            description="SPY moves >1% intraday - mean reversion probability increases",
            conditions={"spy_move_percent_min": 1.0},
            # If up big, consider puts for reversion; if down big, consider calls
        ))

        self._add_pattern(MarketPattern(
            pattern_id="two_percent_day",
            pattern_type=PatternType.MOMENTUM,
            name="Two Percent Day",
            description="SPY moving >2% - extreme volatility, proceed with caution",
            conditions={"spy_move_percent_min": 2.0},
            # Risky but potentially rewarding for directional plays
        ))

    def _add_pattern(self, pattern: MarketPattern) -> None:
        """Add a pattern to the registry"""
        self.patterns[pattern.pattern_id] = pattern

    def recognize_patterns(
        self,
        market_data: Dict[str, Any]
    ) -> List[PatternMatch]:
        """
        Recognize patterns in current market conditions

        Args:
            market_data: Current market data including:
                - spy_price: Current SPY price
                - spy_change_percent: SPY change from previous close
                - vix: Current VIX level
                - vix_change: VIX change from previous close
                - current_time: Current time
                - trend_direction: Detected trend (UP, DOWN, NEUTRAL)
                - momentum: Momentum indicator value

        Returns:
            List of pattern matches sorted by relevance
        """
        matches = []

        current_time = market_data.get("current_time", datetime.now().time())
        if isinstance(current_time, datetime):
            current_time = current_time.time()

        vix = market_data.get("vix", 15.0)
        spy_change = market_data.get("spy_change_percent", 0.0)
        trend = market_data.get("trend_direction", "NEUTRAL")

        for pattern_id, pattern in self.patterns.items():
            match_strength = self._calculate_match_strength(pattern, market_data, current_time)

            if match_strength > 0.3:  # Only include meaningful matches
                expected = self._determine_expected_outcome(pattern, market_data)
                action = self._determine_recommended_action(pattern, expected)
                reasoning = self._generate_reasoning(pattern, market_data, expected)

                matches.append(PatternMatch(
                    pattern=pattern,
                    match_strength=match_strength,
                    expected_outcome=expected,
                    recommended_action=action,
                    reasoning=reasoning
                ))

        # Sort by match strength and pattern confidence
        matches.sort(
            key=lambda m: m.match_strength * m.pattern.confidence_score,
            reverse=True
        )

        return matches[:5]  # Return top 5 matches

    def _calculate_match_strength(
        self,
        pattern: MarketPattern,
        market_data: Dict[str, Any],
        current_time: dt_time
    ) -> float:
        """Calculate how closely current conditions match a pattern"""
        conditions = pattern.conditions
        match_points = 0
        total_points = len(conditions)

        if total_points == 0:
            return 0.0

        # VIX conditions
        vix = market_data.get("vix", 15.0)
        if "vix_min" in conditions:
            if vix >= conditions["vix_min"]:
                match_points += 1
            else:
                match_points += max(0, 1 - abs(vix - conditions["vix_min"]) / 10)

        if "vix_max" in conditions:
            if vix <= conditions["vix_max"]:
                match_points += 1
            else:
                match_points += max(0, 1 - abs(vix - conditions["vix_max"]) / 10)

        if "vix_change_max" in conditions:
            vix_change = abs(market_data.get("vix_change", 0.0))
            if vix_change <= conditions["vix_change_max"]:
                match_points += 1

        # Time conditions
        if "time_start" in conditions and "time_end" in conditions:
            time_start = datetime.strptime(conditions["time_start"], "%H:%M").time()
            time_end = datetime.strptime(conditions["time_end"], "%H:%M").time()

            if time_start <= current_time <= time_end:
                match_points += 1

        # Direction conditions
        if "spy_direction" in conditions:
            trend = market_data.get("trend_direction", "NEUTRAL")
            if conditions["spy_direction"] == trend:
                match_points += 1

        if "momentum" in conditions:
            momentum = market_data.get("momentum", "NEUTRAL")
            if conditions["momentum"] == momentum:
                match_points += 1

        # Gap conditions
        if "gap_direction" in conditions:
            gap = market_data.get("gap_percent", 0.0)
            gap_direction = "UP" if gap > 0 else "DOWN" if gap < 0 else "NEUTRAL"

            if conditions["gap_direction"] == gap_direction:
                gap_min = conditions.get("gap_percent_min", 0)
                if abs(gap) >= gap_min:
                    match_points += 1

        # SPY move conditions (for extreme move patterns)
        spy_change = abs(market_data.get("spy_change_percent", 0.0))
        if "spy_move_percent_min" in conditions:
            if spy_change >= conditions["spy_move_percent_min"]:
                match_points += 1

        # VIX spike/crush conditions
        vix_change_percent = market_data.get("vix_change_percent", 0.0)
        if "vix_spike_percent_min" in conditions:
            if vix_change_percent >= conditions["vix_spike_percent_min"]:
                match_points += 1

        if "vix_change_percent_min" in conditions:
            # For VIX crush (negative change)
            if vix_change_percent <= conditions["vix_change_percent_min"]:
                match_points += 1

        if "vix_declining" in conditions:
            if vix_change_percent < 0:
                match_points += 1

        # Volume conditions
        volume_ratio = market_data.get("volume_ratio", 1.0)
        if "volume_ratio_min" in conditions:
            if volume_ratio >= conditions["volume_ratio_min"]:
                match_points += 1

        # VWAP conditions
        if "above_vwap" in conditions:
            spy_price = market_data.get("spy_price", 0)
            vwap = market_data.get("vwap", spy_price)
            if spy_price > vwap:
                match_points += 1

        if "below_vwap" in conditions:
            spy_price = market_data.get("spy_price", 0)
            vwap = market_data.get("vwap", spy_price)
            if spy_price < vwap:
                match_points += 1

        # High/Low conditions
        if "new_high" in conditions:
            if market_data.get("new_intraday_high", False):
                match_points += 1

        if "new_low" in conditions:
            if market_data.get("new_intraday_low", False):
                match_points += 1

        # Reversal/bounce signals (from market data analysis)
        if "reversal_signal" in conditions:
            if market_data.get("reversal_detected", False):
                match_points += 1

        if "early_weakness" in conditions:
            if market_data.get("early_weakness", False):
                match_points += 1

        if "early_strength" in conditions:
            if market_data.get("early_strength", False):
                match_points += 1

        if "strong_trend" in conditions:
            momentum = market_data.get("momentum", "MODERATE")
            if momentum == "STRONG":
                match_points += 1

        return match_points / total_points if total_points > 0 else 0.0

    def _determine_expected_outcome(
        self,
        pattern: MarketPattern,
        market_data: Dict[str, Any]
    ) -> str:
        """Determine expected outcome based on pattern history"""
        if pattern.total_occurrences < 5:
            return "NEUTRAL"  # Not enough data

        if pattern.win_rate >= 0.65 and pattern.avg_pnl > 0:
            return "FAVORABLE"
        elif pattern.win_rate <= 0.35 or pattern.avg_pnl < -50:
            return "UNFAVORABLE"
        else:
            return "NEUTRAL"

    def _determine_recommended_action(
        self,
        pattern: MarketPattern,
        expected_outcome: str
    ) -> str:
        """Determine recommended action based on pattern and expected outcome"""
        if expected_outcome == "UNFAVORABLE":
            if pattern.avoid_strategies:
                return f"AVOID: {', '.join(pattern.avoid_strategies)}"
            return "WAIT for better conditions"
        elif expected_outcome == "FAVORABLE":
            if pattern.preferred_strategies:
                return f"CONSIDER: {', '.join(pattern.preferred_strategies)}"
            return "CONDITIONS favorable for trading"
        else:
            return "PROCEED with caution"

    def _generate_reasoning(
        self,
        pattern: MarketPattern,
        market_data: Dict[str, Any],
        expected_outcome: str
    ) -> str:
        """Generate human-readable reasoning for the pattern match"""
        parts = [f"Pattern '{pattern.name}' detected."]

        if pattern.total_occurrences >= 5:
            parts.append(
                f"Historical: {pattern.win_rate:.0%} win rate "
                f"across {pattern.total_occurrences} occurrences."
            )

        if pattern.avg_pnl != 0:
            parts.append(f"Avg outcome: ${pattern.avg_pnl:+.2f}")

        if expected_outcome == "FAVORABLE" and pattern.preferred_strategies:
            parts.append(f"Best strategies: {', '.join(pattern.preferred_strategies)}")
        elif expected_outcome == "UNFAVORABLE" and pattern.avoid_strategies:
            parts.append(f"Avoid: {', '.join(pattern.avoid_strategies)}")

        return " ".join(parts)

    def update_pattern_with_outcome(
        self,
        pattern_id: str,
        pnl: float,
        strategy: str
    ) -> None:
        """
        Update a pattern with a trade outcome

        Args:
            pattern_id: ID of the pattern to update
            pnl: P&L from the trade
            strategy: Strategy used in the trade
        """
        if pattern_id in self.patterns:
            self.patterns[pattern_id].update_with_outcome(pnl, strategy)
            logger.debug(
                f"Updated pattern {pattern_id}: "
                f"win_rate={self.patterns[pattern_id].win_rate:.1%}"
            )

    def record_trade_patterns(
        self,
        market_data: Dict[str, Any],
        pnl: float,
        strategy: str
    ) -> List[str]:
        """
        Record patterns that were present during a trade

        Args:
            market_data: Market conditions at trade entry
            pnl: Trade P&L
            strategy: Strategy used

        Returns:
            List of pattern IDs that were updated
        """
        matches = self.recognize_patterns(market_data)
        updated_patterns = []

        for match in matches:
            if match.match_strength >= 0.5:  # Only update strong matches
                self.update_pattern_with_outcome(
                    match.pattern.pattern_id,
                    pnl,
                    strategy
                )
                updated_patterns.append(match.pattern.pattern_id)

        return updated_patterns

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about all patterns"""
        stats = {
            "total_patterns": len(self.patterns),
            "patterns_with_data": 0,
            "top_performing": [],
            "worst_performing": [],
            "by_type": {},
        }

        patterns_with_outcomes = [
            p for p in self.patterns.values()
            if p.total_occurrences >= 5
        ]

        stats["patterns_with_data"] = len(patterns_with_outcomes)

        # Sort by win rate
        sorted_patterns = sorted(
            patterns_with_outcomes,
            key=lambda p: (p.win_rate, p.avg_pnl),
            reverse=True
        )

        # Top 3 performing
        stats["top_performing"] = [
            {
                "name": p.name,
                "win_rate": p.win_rate,
                "avg_pnl": p.avg_pnl,
                "occurrences": p.total_occurrences,
            }
            for p in sorted_patterns[:3]
        ]

        # Worst 3 performing
        stats["worst_performing"] = [
            {
                "name": p.name,
                "win_rate": p.win_rate,
                "avg_pnl": p.avg_pnl,
                "occurrences": p.total_occurrences,
            }
            for p in sorted_patterns[-3:]
        ]

        # By pattern type
        for pattern in self.patterns.values():
            ptype = pattern.pattern_type.value
            if ptype not in stats["by_type"]:
                stats["by_type"][ptype] = {
                    "count": 0,
                    "total_occurrences": 0,
                    "total_pnl": 0,
                }
            stats["by_type"][ptype]["count"] += 1
            stats["by_type"][ptype]["total_occurrences"] += pattern.total_occurrences
            stats["by_type"][ptype]["total_pnl"] += pattern.total_pnl

        return stats

    def get_recommendations_for_strategy(
        self,
        strategy: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get pattern-based recommendations for a specific strategy

        Args:
            strategy: Strategy to evaluate
            market_data: Current market conditions

        Returns:
            Recommendations including confidence and reasoning
        """
        matches = self.recognize_patterns(market_data)

        favorable_patterns = []
        unfavorable_patterns = []
        confidence_factors = []

        for match in matches:
            pattern = match.pattern

            if strategy in pattern.preferred_strategies:
                favorable_patterns.append({
                    "name": pattern.name,
                    "win_rate": pattern.win_rate,
                    "match_strength": match.match_strength,
                })
                confidence_factors.append(match.match_strength * pattern.win_rate)

            elif strategy in pattern.avoid_strategies:
                unfavorable_patterns.append({
                    "name": pattern.name,
                    "win_rate": pattern.win_rate,
                    "match_strength": match.match_strength,
                })
                confidence_factors.append(-match.match_strength * (1 - pattern.win_rate))

        # Calculate overall recommendation
        if not confidence_factors:
            recommendation = "NEUTRAL"
            confidence = 0.5
        else:
            avg_factor = statistics.mean(confidence_factors)
            if avg_factor > 0.2:
                recommendation = "FAVORABLE"
                confidence = 0.5 + avg_factor
            elif avg_factor < -0.2:
                recommendation = "UNFAVORABLE"
                confidence = 0.5 - avg_factor
            else:
                recommendation = "NEUTRAL"
                confidence = 0.5

        return {
            "strategy": strategy,
            "recommendation": recommendation,
            "confidence": min(1.0, max(0.0, confidence)),
            "favorable_patterns": favorable_patterns,
            "unfavorable_patterns": unfavorable_patterns,
            "reasoning": self._build_strategy_reasoning(
                strategy, favorable_patterns, unfavorable_patterns
            )
        }

    def _build_strategy_reasoning(
        self,
        strategy: str,
        favorable: List[Dict],
        unfavorable: List[Dict]
    ) -> str:
        """Build reasoning string for strategy recommendation"""
        parts = []

        if favorable:
            pattern_names = [p["name"] for p in favorable[:2]]
            parts.append(f"{strategy} favored by: {', '.join(pattern_names)}")

        if unfavorable:
            pattern_names = [p["name"] for p in unfavorable[:2]]
            parts.append(f"Caution from: {', '.join(pattern_names)}")

        if not parts:
            return f"No strong pattern signals for {strategy}"

        return "; ".join(parts)
