"""
Trade Outcome Analyzer

Analyzes completed trades to extract learnings and identify what
contributed to success or failure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import statistics

from loguru import logger


class OutcomeType(Enum):
    """Trade outcome classification"""
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    MANUAL_EXIT = "MANUAL_EXIT"
    MARKET_CLOSE = "MARKET_CLOSE"
    RISK_BREACH = "RISK_BREACH"


@dataclass
class TradeOutcome:
    """Complete trade outcome with all relevant data"""
    trade_id: str
    strategy_type: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    outcome_type: OutcomeType

    # Market conditions at entry
    spy_price_entry: float = 0.0
    vix_entry: float = 0.0
    market_regime: str = ""
    trend_direction: str = ""

    # Market conditions at exit
    spy_price_exit: float = 0.0
    vix_exit: float = 0.0
    spy_move_percent: float = 0.0

    # AI decision context
    ai_confidence: float = 0.0
    market_agent_score: float = 0.0
    risk_agent_score: float = 0.0
    execution_agent_score: float = 0.0

    # Position details
    max_profit_potential: float = 0.0
    max_loss_potential: float = 0.0
    profit_captured_percent: float = 0.0

    # Timing metrics
    hold_duration_minutes: int = 0
    time_to_profit_target_minutes: Optional[int] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeAnalysis:
    """Analysis results for a trade outcome"""
    trade_id: str
    success: bool
    key_factors: List[str]
    warnings: List[str]
    recommendations: List[str]
    score: float  # 0-100 quality score
    learnings: Dict[str, Any]


class OutcomeAnalyzer:
    """
    Analyzes trade outcomes to extract actionable learnings
    """

    def __init__(self):
        """Initialize the outcome analyzer"""
        self.outcomes: List[TradeOutcome] = []
        self.analyses: List[OutcomeAnalysis] = []

        # Performance benchmarks
        self.target_win_rate = 0.65
        self.target_profit_factor = 1.5
        self.target_avg_winner = 75.0
        self.target_avg_loser = -50.0

    def analyze_outcome(self, outcome: TradeOutcome) -> OutcomeAnalysis:
        """
        Analyze a single trade outcome

        Args:
            outcome: The trade outcome to analyze

        Returns:
            Analysis with key factors and recommendations
        """
        self.outcomes.append(outcome)

        key_factors = []
        warnings = []
        recommendations = []
        learnings = {}

        # Determine if trade was successful
        success = outcome.pnl > 0

        # Analyze entry timing
        entry_analysis = self._analyze_entry(outcome)
        key_factors.extend(entry_analysis["factors"])
        warnings.extend(entry_analysis["warnings"])
        learnings["entry"] = entry_analysis

        # Analyze exit timing
        exit_analysis = self._analyze_exit(outcome)
        key_factors.extend(exit_analysis["factors"])
        warnings.extend(exit_analysis["warnings"])
        learnings["exit"] = exit_analysis

        # Analyze market conditions
        market_analysis = self._analyze_market_conditions(outcome)
        key_factors.extend(market_analysis["factors"])
        warnings.extend(market_analysis["warnings"])
        learnings["market"] = market_analysis

        # Analyze AI decision quality
        ai_analysis = self._analyze_ai_decision(outcome)
        key_factors.extend(ai_analysis["factors"])
        warnings.extend(ai_analysis["warnings"])
        learnings["ai"] = ai_analysis

        # Generate recommendations
        recommendations = self._generate_recommendations(outcome, learnings)

        # Calculate quality score
        score = self._calculate_quality_score(outcome, learnings)

        analysis = OutcomeAnalysis(
            trade_id=outcome.trade_id,
            success=success,
            key_factors=key_factors,
            warnings=warnings,
            recommendations=recommendations,
            score=score,
            learnings=learnings
        )

        self.analyses.append(analysis)

        logger.info(
            f"Analyzed trade {outcome.trade_id}: "
            f"{'SUCCESS' if success else 'FAILURE'} | "
            f"Score: {score:.1f} | "
            f"P&L: ${outcome.pnl:+.2f}"
        )

        return analysis

    def _analyze_entry(self, outcome: TradeOutcome) -> Dict[str, Any]:
        """Analyze trade entry quality"""
        factors = []
        warnings = []

        analysis = {
            "vix_level": outcome.vix_entry,
            "market_regime": outcome.market_regime,
            "ai_confidence": outcome.ai_confidence,
        }

        # VIX analysis (for LONG options - lower VIX = cheaper premiums)
        if outcome.vix_entry < 18:
            factors.append("Low VIX environment (cheaper premiums for buying)")
            analysis["vix_favorable"] = True
        elif outcome.vix_entry > 25:
            warnings.append("High VIX at entry (expensive premiums)")
            analysis["vix_favorable"] = False
        else:
            analysis["vix_favorable"] = True

        # AI confidence analysis
        if outcome.ai_confidence >= 0.75:
            factors.append("High AI confidence at entry")
            analysis["confidence_adequate"] = True
        elif outcome.ai_confidence < 0.55:
            warnings.append("Low AI confidence at entry")
            analysis["confidence_adequate"] = False
        else:
            analysis["confidence_adequate"] = True

        # Time of day analysis
        entry_hour = outcome.entry_time.hour
        entry_minute = outcome.entry_time.minute

        if entry_hour == 9 and entry_minute < 45:
            warnings.append("Entry during first 15 minutes (high volatility)")
            analysis["timing_optimal"] = False
        elif entry_hour >= 15 and entry_minute >= 30:
            warnings.append("Late entry (limited time for profit target)")
            analysis["timing_optimal"] = False
        else:
            analysis["timing_optimal"] = True

        analysis["factors"] = factors
        analysis["warnings"] = warnings

        return analysis

    def _analyze_exit(self, outcome: TradeOutcome) -> Dict[str, Any]:
        """Analyze trade exit quality"""
        factors = []
        warnings = []

        analysis = {
            "outcome_type": outcome.outcome_type.value,
            "profit_captured_percent": outcome.profit_captured_percent,
            "hold_duration_minutes": outcome.hold_duration_minutes,
        }

        # Exit type analysis
        if outcome.outcome_type == OutcomeType.PROFIT_TARGET:
            factors.append("Clean profit target exit")
            analysis["exit_quality"] = "excellent"
        elif outcome.outcome_type == OutcomeType.STOP_LOSS:
            if outcome.pnl_percent < -100:  # Lost more than max expected
                warnings.append("Stop loss exceeded expected max loss")
                analysis["exit_quality"] = "poor"
            else:
                factors.append("Disciplined stop loss exit")
                analysis["exit_quality"] = "acceptable"
        elif outcome.outcome_type == OutcomeType.TIME_EXIT:
            if outcome.pnl > 0:
                factors.append("Profitable time-based exit")
                analysis["exit_quality"] = "good"
            else:
                warnings.append("Negative time-based exit")
                analysis["exit_quality"] = "poor"

        # Profit capture analysis
        if outcome.profit_captured_percent >= 50 and outcome.pnl > 0:
            factors.append(f"Captured {outcome.profit_captured_percent:.0f}% of max profit")
        elif outcome.pnl > 0 and outcome.profit_captured_percent < 30:
            warnings.append("Low profit capture percentage")

        # Hold duration analysis
        if outcome.hold_duration_minutes < 10 and outcome.pnl < 0:
            warnings.append("Quick loss (possible poor entry)")
        elif outcome.hold_duration_minutes > 180 and outcome.pnl < 0:
            warnings.append("Prolonged losing position")

        analysis["factors"] = factors
        analysis["warnings"] = warnings

        return analysis

    def _analyze_market_conditions(self, outcome: TradeOutcome) -> Dict[str, Any]:
        """Analyze market conditions impact"""
        factors = []
        warnings = []

        analysis = {
            "spy_move_percent": outcome.spy_move_percent,
            "vix_change": outcome.vix_exit - outcome.vix_entry,
            "trend_direction": outcome.trend_direction,
        }

        # SPY movement analysis
        spy_move = abs(outcome.spy_move_percent)

        if spy_move > 1.5:
            warnings.append(f"Large SPY move during trade ({outcome.spy_move_percent:+.2f}%)")
            analysis["market_volatile"] = True
        else:
            analysis["market_volatile"] = False

        # Strategy alignment with market direction (LONG options)
        if outcome.strategy_type == "LONG_CALL":
            # LONG_CALL profits when SPY goes UP
            if outcome.spy_move_percent > 0.3 and outcome.pnl > 0:
                factors.append("Profit from correct bullish call")
                analysis["strategy_aligned"] = True
            elif outcome.spy_move_percent < -0.3 and outcome.pnl < 0:
                factors.append("Loss from bearish move against long call")
                analysis["strategy_aligned"] = False
            elif outcome.pnl < 0 and abs(outcome.spy_move_percent) < 0.3:
                factors.append("Loss from sideways action (theta decay)")
                analysis["strategy_aligned"] = False
        elif outcome.strategy_type == "LONG_PUT":
            # LONG_PUT profits when SPY goes DOWN
            if outcome.spy_move_percent < -0.3 and outcome.pnl > 0:
                factors.append("Profit from correct bearish put")
                analysis["strategy_aligned"] = True
            elif outcome.spy_move_percent > 0.3 and outcome.pnl < 0:
                factors.append("Loss from bullish move against long put")
                analysis["strategy_aligned"] = False
            elif outcome.pnl < 0 and abs(outcome.spy_move_percent) < 0.3:
                factors.append("Loss from sideways action (theta decay)")
                analysis["strategy_aligned"] = False

        analysis["factors"] = factors
        analysis["warnings"] = warnings

        return analysis

    def _analyze_ai_decision(self, outcome: TradeOutcome) -> Dict[str, Any]:
        """Analyze AI decision quality"""
        factors = []
        warnings = []

        analysis = {
            "overall_confidence": outcome.ai_confidence,
            "market_agent_score": outcome.market_agent_score,
            "risk_agent_score": outcome.risk_agent_score,
            "execution_agent_score": outcome.execution_agent_score,
        }

        # Agent score analysis
        agents_agreed = (
            outcome.market_agent_score > 0.6 and
            outcome.risk_agent_score > 0.6 and
            outcome.execution_agent_score > 0.6
        )

        if agents_agreed:
            if outcome.pnl > 0:
                factors.append("Strong agent consensus led to profitable trade")
                analysis["consensus_quality"] = "validated"
            else:
                warnings.append("Strong consensus but trade lost - review conditions")
                analysis["consensus_quality"] = "review_needed"
        else:
            if outcome.pnl < 0:
                warnings.append("Weak consensus correlated with loss")
                analysis["consensus_quality"] = "correctly_cautious"
            else:
                factors.append("Profit despite mixed consensus - possible luck")
                analysis["consensus_quality"] = "lucky"

        # Individual agent analysis
        if outcome.market_agent_score < 0.5 and outcome.pnl < 0:
            factors.append("Market agent correctly identified unfavorable conditions")
        if outcome.risk_agent_score < 0.5 and outcome.pnl < 0:
            factors.append("Risk agent correctly flagged elevated risk")

        analysis["factors"] = factors
        analysis["warnings"] = warnings

        return analysis

    def _generate_recommendations(
        self,
        outcome: TradeOutcome,
        learnings: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations from analysis"""
        recommendations = []

        # Entry recommendations
        entry = learnings.get("entry", {})
        if not entry.get("vix_favorable"):
            recommendations.append("Consider avoiding entries when VIX > 25")
        if not entry.get("confidence_adequate"):
            recommendations.append("Increase confidence threshold for live trades")
        if not entry.get("timing_optimal"):
            recommendations.append("Prefer entries between 9:45 AM and 3:00 PM")

        # Exit recommendations
        exit_analysis = learnings.get("exit", {})
        if exit_analysis.get("exit_quality") == "poor":
            recommendations.append("Review stop loss placement and timing")
        if exit_analysis.get("profit_captured_percent", 100) < 30 and outcome.pnl > 0:
            recommendations.append("Consider tighter profit targets to capture gains")

        # Market recommendations
        market = learnings.get("market", {})
        if market.get("market_volatile"):
            recommendations.append("Reduce position size in high volatility environments")
        if not market.get("strategy_aligned", True):
            recommendations.append("Improve market direction assessment before entry")

        # AI recommendations
        ai = learnings.get("ai", {})
        if ai.get("consensus_quality") == "review_needed":
            recommendations.append("Add market conditions to context learning")

        return recommendations

    def _calculate_quality_score(
        self,
        outcome: TradeOutcome,
        learnings: Dict[str, Any]
    ) -> float:
        """Calculate overall trade quality score (0-100)"""
        score = 50.0  # Start at neutral

        # Profitability (+/- 20 points)
        if outcome.pnl > 0:
            score += min(20, outcome.pnl_percent / 5)
        else:
            score += max(-20, outcome.pnl_percent / 5)

        # Entry quality (+/- 10 points)
        entry = learnings.get("entry", {})
        if entry.get("vix_favorable"):
            score += 3
        if entry.get("confidence_adequate"):
            score += 4
        if entry.get("timing_optimal"):
            score += 3

        # Exit quality (+/- 10 points)
        exit_analysis = learnings.get("exit", {})
        exit_quality = exit_analysis.get("exit_quality", "acceptable")
        if exit_quality == "excellent":
            score += 10
        elif exit_quality == "good":
            score += 5
        elif exit_quality == "poor":
            score -= 10

        # Profit capture (+/- 10 points)
        if outcome.pnl > 0:
            capture = outcome.profit_captured_percent
            score += min(10, capture / 10)

        return max(0, min(100, score))

    def get_aggregate_learnings(self, last_n: int = 50) -> Dict[str, Any]:
        """
        Get aggregate learnings from recent trades

        Args:
            last_n: Number of recent trades to analyze

        Returns:
            Aggregated insights and patterns
        """
        recent_outcomes = self.outcomes[-last_n:] if self.outcomes else []
        recent_analyses = self.analyses[-last_n:] if self.analyses else []

        if not recent_outcomes:
            return {"error": "No trades to analyze"}

        # Calculate aggregate metrics
        pnls = [o.pnl for o in recent_outcomes]
        winners = [o for o in recent_outcomes if o.pnl > 0]
        losers = [o for o in recent_outcomes if o.pnl < 0]

        win_rate = len(winners) / len(recent_outcomes) if recent_outcomes else 0
        avg_win = statistics.mean([w.pnl for w in winners]) if winners else 0
        avg_loss = statistics.mean([l.pnl for l in losers]) if losers else 0
        profit_factor = abs(sum(w.pnl for w in winners) / sum(l.pnl for l in losers)) if losers and sum(l.pnl for l in losers) != 0 else 0

        # Strategy performance
        strategy_perf = {}
        for outcome in recent_outcomes:
            if outcome.strategy_type not in strategy_perf:
                strategy_perf[outcome.strategy_type] = {"wins": 0, "losses": 0, "pnl": 0}
            if outcome.pnl > 0:
                strategy_perf[outcome.strategy_type]["wins"] += 1
            else:
                strategy_perf[outcome.strategy_type]["losses"] += 1
            strategy_perf[outcome.strategy_type]["pnl"] += outcome.pnl

        # Collect common warnings and recommendations
        all_warnings = []
        all_recommendations = []
        for analysis in recent_analyses:
            all_warnings.extend(analysis.warnings)
            all_recommendations.extend(analysis.recommendations)

        # Count frequencies
        from collections import Counter
        warning_counts = Counter(all_warnings)
        recommendation_counts = Counter(all_recommendations)

        # Quality scores
        avg_score = statistics.mean([a.score for a in recent_analyses]) if recent_analyses else 50

        return {
            "period_trades": len(recent_outcomes),
            "total_pnl": sum(pnls),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_quality_score": avg_score,
            "strategy_performance": strategy_perf,
            "top_warnings": warning_counts.most_common(5),
            "top_recommendations": recommendation_counts.most_common(5),
            "meets_targets": {
                "win_rate": win_rate >= self.target_win_rate,
                "profit_factor": profit_factor >= self.target_profit_factor,
                "avg_win": avg_win >= self.target_avg_winner,
                "avg_loss": avg_loss >= self.target_avg_loser,
            }
        }
