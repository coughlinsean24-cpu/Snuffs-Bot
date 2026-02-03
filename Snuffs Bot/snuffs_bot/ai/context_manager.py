"""
Conversation Context Manager

Manages AI context and learning:
- Stores recent decisions and outcomes
- Provides learning insights to agents
- Maintains conversation history
- Manages token budget for context
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import json
from loguru import logger

from ..config.settings import get_settings


@dataclass
class DecisionRecord:
    """Record of a trading decision"""
    decision_id: str
    timestamp: str
    consensus: str
    confidence: int
    market_data: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    outcome: Optional[Dict[str, Any]] = None
    trade_id: Optional[int] = None
    pnl: Optional[float] = None
    lesson_learned: Optional[str] = None


@dataclass
class LearningInsight:
    """An insight derived from trade outcomes"""
    insight_id: str
    created_at: str
    category: str  # 'strategy', 'timing', 'risk', 'execution'
    insight: str
    supporting_trades: List[str]  # decision_ids
    effectiveness_score: int  # 1-10


class ConversationContext:
    """
    Manages context for AI agents to enable learning

    Features:
    - Rolling window of recent decisions
    - Outcome tracking and analysis
    - Learning insights generation
    - Token-aware context trimming
    """

    def __init__(self, max_decisions: int = 50, max_insights: int = 20):
        """
        Initialize context manager

        Args:
            max_decisions: Maximum decisions to keep in memory
            max_insights: Maximum insights to maintain
        """
        self.settings = get_settings()

        # Decision history (rolling window)
        self.decisions: deque[DecisionRecord] = deque(maxlen=max_decisions)

        # Learning insights
        self.insights: deque[LearningInsight] = deque(maxlen=max_insights)

        # Current session stats
        self.session_start = datetime.now()
        self.session_decisions = 0
        self.session_trades_executed = 0
        self.session_pnl = 0.0

        logger.info(f"Context manager initialized (max_decisions={max_decisions})")

    def add_decision(
        self,
        decision_id: str,
        consensus: str,
        confidence: int,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> None:
        """
        Record a new decision

        Args:
            decision_id: Unique identifier for the decision
            consensus: The consensus decision (EXECUTE, REJECT, etc.)
            confidence: Confidence level 0-100
            market_data: Market conditions at decision time
            portfolio_state: Portfolio state at decision time
        """
        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now().isoformat(),
            consensus=consensus,
            confidence=confidence,
            market_data=market_data,
            portfolio_state=portfolio_state
        )

        self.decisions.append(record)
        self.session_decisions += 1

        logger.debug(f"Recorded decision {decision_id}: {consensus}")

    def record_outcome(
        self,
        decision_id: str,
        trade_id: Optional[int],
        pnl: float,
        outcome_data: Dict[str, Any]
    ) -> None:
        """
        Record the outcome of a decision

        Args:
            decision_id: The decision this outcome relates to
            trade_id: Database trade ID (if executed)
            pnl: Profit/loss from the trade
            outcome_data: Additional outcome information
        """
        # Find the decision
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                decision.outcome = outcome_data
                decision.trade_id = trade_id
                decision.pnl = pnl

                if trade_id:
                    self.session_trades_executed += 1
                    self.session_pnl += pnl

                logger.info(
                    f"Recorded outcome for {decision_id}: "
                    f"P&L=${pnl:.2f}"
                )

                # Analyze for learning
                self._analyze_for_learning(decision)
                return

        logger.warning(f"Decision {decision_id} not found in context")

    def _analyze_for_learning(self, decision: DecisionRecord) -> None:
        """
        Analyze a completed decision for learning insights

        This is where we extract patterns from trade outcomes
        """
        if decision.outcome is None or decision.pnl is None:
            return

        pnl = decision.pnl
        market_data = decision.market_data
        consensus = decision.consensus

        # Simple pattern detection
        vix = market_data.get("vix", 0)
        spy_change = market_data.get("spy_change_percent", 0)

        # Generate lesson
        lesson = None

        if pnl > 0:
            # Winning trade analysis
            if vix > 25:
                lesson = f"High VIX ({vix}) trade was profitable (+${pnl:.2f})"
            elif abs(spy_change) < 0.3:
                lesson = f"Sideways market ({spy_change}%) trade was profitable (+${pnl:.2f})"
        else:
            # Losing trade analysis
            if vix < 15:
                lesson = f"Low VIX ({vix}) trade lost (${pnl:.2f}) - consider wider spreads"
            elif consensus == "EXECUTE" and decision.confidence < 60:
                lesson = f"Low confidence ({decision.confidence}%) trade lost - raise threshold"

        if lesson:
            decision.lesson_learned = lesson
            logger.info(f"Learning insight: {lesson}")

    def get_recent_insights(self, limit: int = 5) -> List[str]:
        """
        Get recent learning insights as strings

        Args:
            limit: Maximum number of insights to return

        Returns:
            List of insight strings
        """
        # Combine formal insights and recent lessons
        all_insights = []

        # Add formal insights
        for insight in list(self.insights)[-limit:]:
            all_insights.append(insight.insight)

        # Add recent lessons from decisions
        for decision in list(self.decisions)[-20:]:
            if decision.lesson_learned:
                all_insights.append(decision.lesson_learned)

        return all_insights[-limit:]

    def get_similar_decisions(
        self,
        market_data: Dict[str, Any],
        limit: int = 3
    ) -> List[DecisionRecord]:
        """
        Find decisions made in similar market conditions

        Args:
            market_data: Current market conditions
            limit: Maximum decisions to return

        Returns:
            List of similar past decisions
        """
        current_vix = market_data.get("vix", 0)
        current_spy = market_data.get("spy_price", 0)

        similar = []

        for decision in self.decisions:
            if decision.outcome is None:
                continue  # Only consider completed trades

            past_vix = decision.market_data.get("vix", 0)
            past_spy = decision.market_data.get("spy_price", 0)

            # Simple similarity: VIX within 3 points, SPY within 1%
            vix_diff = abs(current_vix - past_vix)
            spy_diff = abs(current_spy - past_spy) / current_spy * 100 if current_spy else 100

            if vix_diff <= 3 and spy_diff <= 1:
                similar.append(decision)

        # Sort by recency (most recent first)
        similar.sort(key=lambda d: d.timestamp, reverse=True)

        return similar[:limit]

    def add_insight(
        self,
        category: str,
        insight: str,
        supporting_decisions: List[str],
        effectiveness_score: int
    ) -> None:
        """
        Add a new learning insight

        Args:
            category: Category of insight
            insight: The insight text
            supporting_decisions: Decision IDs that support this insight
            effectiveness_score: How effective (1-10)
        """
        insight_obj = LearningInsight(
            insight_id=f"insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            category=category,
            insight=insight,
            supporting_trades=supporting_decisions,
            effectiveness_score=effectiveness_score
        )

        self.insights.append(insight_obj)
        logger.info(f"Added insight: {insight}")

    def get_context_window(self, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Get a token-limited context window for AI agents

        Args:
            max_tokens: Maximum tokens for the context

        Returns:
            Dictionary with relevant context
        """
        context = {
            "session_stats": {
                "decisions_made": self.session_decisions,
                "trades_executed": self.session_trades_executed,
                "session_pnl": round(self.session_pnl, 2),
                "session_duration_minutes": (
                    datetime.now() - self.session_start
                ).total_seconds() / 60
            },
            "recent_decisions": [],
            "insights": []
        }

        # Add recent decisions (most recent first)
        for decision in reversed(list(self.decisions)[-5:]):
            context["recent_decisions"].append({
                "consensus": decision.consensus,
                "confidence": decision.confidence,
                "pnl": decision.pnl,
                "vix": decision.market_data.get("vix"),
                "lesson": decision.lesson_learned
            })

        # Add insights
        for insight in list(self.insights)[-5:]:
            context["insights"].append({
                "category": insight.category,
                "insight": insight.insight,
                "score": insight.effectiveness_score
            })

        return context

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current trading session"""
        total_decisions = len(self.decisions)
        executed = sum(1 for d in self.decisions if d.trade_id is not None)
        rejected = sum(1 for d in self.decisions if d.consensus == "REJECT")

        winning = sum(1 for d in self.decisions if d.pnl and d.pnl > 0)
        losing = sum(1 for d in self.decisions if d.pnl and d.pnl < 0)

        return {
            "session_start": self.session_start.isoformat(),
            "duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "total_decisions": total_decisions,
            "trades_executed": executed,
            "trades_rejected": rejected,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": winning / executed * 100 if executed else 0,
            "total_pnl": self.session_pnl,
            "insights_generated": len(self.insights)
        }

    def clear_session(self) -> None:
        """Clear session data (keep insights)"""
        self.decisions.clear()
        self.session_start = datetime.now()
        self.session_decisions = 0
        self.session_trades_executed = 0
        self.session_pnl = 0.0
        logger.info("Session cleared")

    def export_context(self) -> str:
        """Export context as JSON string (for persistence)"""
        data = {
            "decisions": [
                {
                    "decision_id": d.decision_id,
                    "timestamp": d.timestamp,
                    "consensus": d.consensus,
                    "confidence": d.confidence,
                    "pnl": d.pnl,
                    "lesson_learned": d.lesson_learned
                }
                for d in self.decisions
            ],
            "insights": [
                {
                    "insight_id": i.insight_id,
                    "created_at": i.created_at,
                    "category": i.category,
                    "insight": i.insight,
                    "effectiveness_score": i.effectiveness_score
                }
                for i in self.insights
            ],
            "session_stats": self.get_session_summary()
        }
        return json.dumps(data, indent=2)

    def import_context(self, json_str: str) -> None:
        """Import context from JSON string"""
        try:
            data = json.loads(json_str)

            # Import insights only (decisions are session-specific)
            for insight_data in data.get("insights", []):
                insight = LearningInsight(
                    insight_id=insight_data["insight_id"],
                    created_at=insight_data["created_at"],
                    category=insight_data["category"],
                    insight=insight_data["insight"],
                    supporting_trades=[],
                    effectiveness_score=insight_data["effectiveness_score"]
                )
                self.insights.append(insight)

            logger.info(f"Imported {len(data.get('insights', []))} insights")

        except Exception as e:
            logger.error(f"Failed to import context: {e}")
