"""
Feedback Integrator

Integrates learnings from trade outcomes back into the AI decision-making
process. Updates agent context, adjusts confidence thresholds, and
modifies strategy selection criteria.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json

from loguru import logger

from .outcome_analyzer import OutcomeAnalyzer, OutcomeAnalysis, TradeOutcome
from .pattern_recognition import PatternRecognizer


@dataclass
class AgentFeedback:
    """Feedback for a specific AI agent"""
    agent_name: str
    feedback_type: str  # "POSITIVE", "NEGATIVE", "ADJUSTMENT"
    message: str
    adjustments: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningUpdate:
    """A learning update to be applied to the system"""
    update_id: str
    update_type: str
    source: str  # "OUTCOME_ANALYSIS", "PATTERN_RECOGNITION", "AGGREGATE"
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    applied_at: Optional[datetime] = None


class FeedbackIntegrator:
    """
    Integrates learnings back into the AI system.

    Responsible for:
    - Generating context updates for AI agents
    - Adjusting confidence thresholds based on performance
    - Updating strategy selection criteria
    - Maintaining learning history
    """

    def __init__(
        self,
        outcome_analyzer: OutcomeAnalyzer,
        pattern_recognizer: PatternRecognizer
    ):
        """
        Initialize the feedback integrator

        Args:
            outcome_analyzer: Analyzer for trade outcomes
            pattern_recognizer: Pattern recognition system
        """
        self.outcome_analyzer = outcome_analyzer
        self.pattern_recognizer = pattern_recognizer

        # Learning state
        self.learning_updates: List[LearningUpdate] = []
        self.agent_feedback_history: List[AgentFeedback] = []

        # Adaptive thresholds (start with defaults)
        # Lower threshold for paper trading to generate more learning data
        self.confidence_threshold = 0.35
        self.min_agent_agreement = 2  # Out of 3 agents
        # Long options strategies for 0DTE SPY
        self.strategy_weights = {
            "LONG_CALL": 1.0,  # Buy calls when bullish
            "LONG_PUT": 1.0,   # Buy puts when bearish
        }

        # Learning rate controls
        self.learning_rate = 0.1  # How quickly to adjust thresholds
        self.min_samples_for_adjustment = 10

    def process_trade_outcome(
        self,
        outcome: TradeOutcome,
        market_data: Dict[str, Any]
    ) -> List[LearningUpdate]:
        """
        Process a completed trade and generate learning updates

        Args:
            outcome: The trade outcome to process
            market_data: Market conditions during the trade

        Returns:
            List of learning updates generated
        """
        updates = []

        # Analyze the outcome
        analysis = self.outcome_analyzer.analyze_outcome(outcome)

        # Update pattern recognition
        updated_patterns = self.pattern_recognizer.record_trade_patterns(
            market_data,
            outcome.pnl,
            outcome.strategy_type
        )

        # Generate outcome-based updates
        outcome_updates = self._generate_outcome_updates(outcome, analysis)
        updates.extend(outcome_updates)

        # Generate pattern-based updates
        pattern_updates = self._generate_pattern_updates(updated_patterns, outcome)
        updates.extend(pattern_updates)

        # Check if thresholds need adjustment
        threshold_updates = self._check_threshold_adjustments()
        updates.extend(threshold_updates)

        # Store updates
        self.learning_updates.extend(updates)

        logger.info(
            f"Processed trade {outcome.trade_id}: "
            f"Generated {len(updates)} learning updates"
        )

        return updates

    def _generate_outcome_updates(
        self,
        outcome: TradeOutcome,
        analysis: OutcomeAnalysis
    ) -> List[LearningUpdate]:
        """Generate learning updates from outcome analysis"""
        updates = []

        # Generate updates from recommendations
        for i, rec in enumerate(analysis.recommendations):
            update = LearningUpdate(
                update_id=f"{outcome.trade_id}_rec_{i}",
                update_type="RECOMMENDATION",
                source="OUTCOME_ANALYSIS",
                description=rec,
                data={
                    "trade_id": outcome.trade_id,
                    "strategy": outcome.strategy_type,
                    "pnl": outcome.pnl,
                    "quality_score": analysis.score,
                }
            )
            updates.append(update)

        # Generate strategy weight update
        if outcome.pnl > 50:  # Significant win
            self._adjust_strategy_weight(outcome.strategy_type, 0.05)
            updates.append(LearningUpdate(
                update_id=f"{outcome.trade_id}_weight_up",
                update_type="STRATEGY_WEIGHT",
                source="OUTCOME_ANALYSIS",
                description=f"Increased {outcome.strategy_type} weight after ${outcome.pnl:.0f} profit",
                data={"strategy": outcome.strategy_type, "adjustment": 0.05}
            ))
        elif outcome.pnl < -100:  # Significant loss
            self._adjust_strategy_weight(outcome.strategy_type, -0.05)
            updates.append(LearningUpdate(
                update_id=f"{outcome.trade_id}_weight_down",
                update_type="STRATEGY_WEIGHT",
                source="OUTCOME_ANALYSIS",
                description=f"Decreased {outcome.strategy_type} weight after ${outcome.pnl:.0f} loss",
                data={"strategy": outcome.strategy_type, "adjustment": -0.05}
            ))

        return updates

    def _generate_pattern_updates(
        self,
        updated_pattern_ids: List[str],
        outcome: TradeOutcome
    ) -> List[LearningUpdate]:
        """Generate learning updates from pattern updates"""
        updates = []

        for pattern_id in updated_pattern_ids:
            pattern = self.pattern_recognizer.patterns.get(pattern_id)
            if pattern and pattern.total_occurrences >= 5:
                # Create update about pattern learning
                update = LearningUpdate(
                    update_id=f"{outcome.trade_id}_pattern_{pattern_id}",
                    update_type="PATTERN_UPDATE",
                    source="PATTERN_RECOGNITION",
                    description=f"Pattern '{pattern.name}' updated: {pattern.win_rate:.0%} win rate",
                    data={
                        "pattern_id": pattern_id,
                        "pattern_name": pattern.name,
                        "win_rate": pattern.win_rate,
                        "total_occurrences": pattern.total_occurrences,
                        "avg_pnl": pattern.avg_pnl,
                    }
                )
                updates.append(update)

        return updates

    def _check_threshold_adjustments(self) -> List[LearningUpdate]:
        """Check if confidence thresholds need adjustment"""
        updates = []

        # Get aggregate performance
        agg = self.outcome_analyzer.get_aggregate_learnings(last_n=20)

        if agg.get("period_trades", 0) < self.min_samples_for_adjustment:
            return updates  # Not enough data

        # Adjust confidence threshold based on win rate
        win_rate = agg.get("win_rate", 0.5)
        target_win_rate = 0.65

        if win_rate < target_win_rate - 0.1:
            # Losing too much, increase threshold
            new_threshold = min(0.85, self.confidence_threshold + self.learning_rate)
            if new_threshold != self.confidence_threshold:
                old_threshold = self.confidence_threshold
                self.confidence_threshold = new_threshold
                updates.append(LearningUpdate(
                    update_id=f"threshold_adj_{datetime.now().timestamp()}",
                    update_type="THRESHOLD_ADJUSTMENT",
                    source="AGGREGATE",
                    description=f"Increased confidence threshold from {old_threshold:.2f} to {new_threshold:.2f}",
                    data={
                        "old_threshold": old_threshold,
                        "new_threshold": new_threshold,
                        "win_rate": win_rate,
                        "reason": "Win rate below target"
                    }
                ))
        elif win_rate > target_win_rate + 0.1 and self.confidence_threshold > 0.25:
            # Winning consistently, can lower threshold to take more trades
            new_threshold = max(0.25, self.confidence_threshold - self.learning_rate / 2)
            if new_threshold != self.confidence_threshold:
                old_threshold = self.confidence_threshold
                self.confidence_threshold = new_threshold
                updates.append(LearningUpdate(
                    update_id=f"threshold_adj_{datetime.now().timestamp()}",
                    update_type="THRESHOLD_ADJUSTMENT",
                    source="AGGREGATE",
                    description=f"Decreased confidence threshold from {old_threshold:.2f} to {new_threshold:.2f}",
                    data={
                        "old_threshold": old_threshold,
                        "new_threshold": new_threshold,
                        "win_rate": win_rate,
                        "reason": "Win rate exceeds target"
                    }
                ))

        return updates

    def _adjust_strategy_weight(self, strategy: str, adjustment: float) -> None:
        """Adjust weight for a strategy"""
        if strategy in self.strategy_weights:
            new_weight = max(0.5, min(1.5, self.strategy_weights[strategy] + adjustment))
            self.strategy_weights[strategy] = new_weight

    def generate_agent_context(self) -> Dict[str, Any]:
        """
        Generate context information to be passed to AI agents

        Returns:
            Context dictionary with learnings and recommendations
        """
        # Get recent learnings
        agg = self.outcome_analyzer.get_aggregate_learnings(last_n=30)
        pattern_stats = self.pattern_recognizer.get_pattern_statistics()

        # Get recent recommendations
        recent_recs = []
        for update in self.learning_updates[-20:]:
            if update.update_type == "RECOMMENDATION":
                recent_recs.append(update.description)

        # Build context
        context = {
            "performance_summary": {
                "recent_win_rate": agg.get("win_rate", 0.5),
                "recent_trades": agg.get("period_trades", 0),
                "total_pnl": agg.get("total_pnl", 0),
                "avg_quality_score": agg.get("avg_quality_score", 50),
            },
            "current_thresholds": {
                "confidence_threshold": self.confidence_threshold,
                "min_agent_agreement": self.min_agent_agreement,
            },
            "strategy_weights": self.strategy_weights.copy(),
            "top_recommendations": list(set(recent_recs))[:5],
            "pattern_insights": {
                "top_patterns": pattern_stats.get("top_performing", []),
                "patterns_to_avoid": pattern_stats.get("worst_performing", []),
            },
            "warnings": agg.get("top_warnings", [])[:3],
            "generated_at": datetime.now().isoformat(),
        }

        return context

    def generate_agent_feedback(
        self,
        outcome: TradeOutcome,
        analysis: OutcomeAnalysis
    ) -> List[AgentFeedback]:
        """
        Generate specific feedback for each AI agent

        Args:
            outcome: The trade outcome
            analysis: Analysis of the outcome

        Returns:
            List of feedback for each agent
        """
        feedback_list = []

        learnings = analysis.learnings

        # Market Agent Feedback
        market_analysis = learnings.get("market", {})
        if market_analysis.get("strategy_aligned") == False:
            feedback_list.append(AgentFeedback(
                agent_name="market_agent",
                feedback_type="NEGATIVE",
                message=f"Market direction assessment was incorrect. Strategy {outcome.strategy_type} conflicted with {outcome.spy_move_percent:+.2f}% SPY move.",
                adjustments={
                    "increase_weight": "trend_indicators",
                    "condition": f"vix={outcome.vix_entry:.1f}"
                }
            ))
        elif outcome.pnl > 0:
            feedback_list.append(AgentFeedback(
                agent_name="market_agent",
                feedback_type="POSITIVE",
                message=f"Correct market assessment for {outcome.market_regime} regime. Trade profitable.",
                adjustments={}
            ))

        # Risk Agent Feedback
        if outcome.pnl < -100 and outcome.risk_agent_score > 0.7:
            feedback_list.append(AgentFeedback(
                agent_name="risk_agent",
                feedback_type="NEGATIVE",
                message=f"Risk was underestimated. Approved trade with {outcome.risk_agent_score:.0%} confidence but resulted in ${outcome.pnl:.0f} loss.",
                adjustments={
                    "increase_sensitivity": True,
                    "review_conditions": {
                        "vix": outcome.vix_entry,
                        "market_regime": outcome.market_regime
                    }
                }
            ))
        elif outcome.pnl > 0 and outcome.risk_agent_score < 0.5:
            feedback_list.append(AgentFeedback(
                agent_name="risk_agent",
                feedback_type="ADJUSTMENT",
                message=f"Risk may have been overestimated. Trade was profitable despite low risk score ({outcome.risk_agent_score:.0%}).",
                adjustments={
                    "consider_loosening": True
                }
            ))

        # Execution Agent Feedback
        exit_analysis = learnings.get("exit", {})
        if exit_analysis.get("exit_quality") == "excellent":
            feedback_list.append(AgentFeedback(
                agent_name="execution_agent",
                feedback_type="POSITIVE",
                message=f"Excellent exit execution. Captured {outcome.profit_captured_percent:.0f}% of max profit.",
                adjustments={}
            ))
        elif exit_analysis.get("exit_quality") == "poor":
            feedback_list.append(AgentFeedback(
                agent_name="execution_agent",
                feedback_type="NEGATIVE",
                message=f"Exit could be improved. {exit_analysis.get('warnings', [''])[0] if exit_analysis.get('warnings') else 'Review exit timing.'}",
                adjustments={
                    "review_exit_rules": True,
                    "outcome_type": outcome.outcome_type.value
                }
            ))

        # Store feedback
        self.agent_feedback_history.extend(feedback_list)

        return feedback_list

    def get_strategy_recommendation(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get strategy recommendation based on learnings

        Args:
            market_data: Current market conditions

        Returns:
            Recommendation with strategy rankings
        """
        recommendations = {}

        for strategy in self.strategy_weights.keys():
            # Get pattern-based recommendation
            pattern_rec = self.pattern_recognizer.get_recommendations_for_strategy(
                strategy, market_data
            )

            # Combine with strategy weight
            base_score = pattern_rec["confidence"]
            weight = self.strategy_weights[strategy]
            adjusted_score = base_score * weight

            recommendations[strategy] = {
                "score": adjusted_score,
                "pattern_confidence": pattern_rec["confidence"],
                "weight": weight,
                "recommendation": pattern_rec["recommendation"],
                "reasoning": pattern_rec["reasoning"],
                "favorable_patterns": pattern_rec["favorable_patterns"],
                "unfavorable_patterns": pattern_rec["unfavorable_patterns"],
            }

        # Sort by score
        ranked = sorted(
            recommendations.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        return {
            "rankings": [{"strategy": k, **v} for k, v in ranked],
            "top_recommendation": ranked[0][0] if ranked else None,
            "confidence_threshold": self.confidence_threshold,
        }

    def should_execute_trade(
        self,
        ai_confidence: float,
        agent_scores: Dict[str, float],
        strategy: str
    ) -> Dict[str, Any]:
        """
        Determine if a trade should be executed based on learnings

        Args:
            ai_confidence: Overall AI confidence
            agent_scores: Individual agent confidence scores
            strategy: Proposed strategy

        Returns:
            Decision with reasoning
        """
        # Check confidence threshold
        meets_threshold = ai_confidence >= self.confidence_threshold

        # Check agent agreement
        agents_above = sum(1 for score in agent_scores.values() if score > 0.6)
        meets_agreement = agents_above >= self.min_agent_agreement

        # Check strategy weight
        strategy_weight = self.strategy_weights.get(strategy, 1.0)
        weighted_confidence = ai_confidence * strategy_weight

        # Make decision
        should_execute = meets_threshold and meets_agreement and strategy_weight >= 0.7

        return {
            "execute": should_execute,
            "confidence": ai_confidence,
            "threshold": self.confidence_threshold,
            "meets_threshold": meets_threshold,
            "agent_agreement": agents_above,
            "required_agreement": self.min_agent_agreement,
            "meets_agreement": meets_agreement,
            "strategy_weight": strategy_weight,
            "weighted_confidence": weighted_confidence,
            "reasoning": self._build_decision_reasoning(
                meets_threshold, meets_agreement, strategy_weight, ai_confidence
            )
        }

    def _build_decision_reasoning(
        self,
        meets_threshold: bool,
        meets_agreement: bool,
        strategy_weight: float,
        confidence: float
    ) -> str:
        """Build reasoning for trade execution decision"""
        reasons = []

        if not meets_threshold:
            reasons.append(
                f"Confidence {confidence:.0%} below threshold {self.confidence_threshold:.0%}"
            )
        else:
            reasons.append(f"Confidence {confidence:.0%} meets threshold")

        if not meets_agreement:
            reasons.append("Insufficient agent agreement")
        else:
            reasons.append("Agent agreement requirement met")

        if strategy_weight < 0.7:
            reasons.append(f"Strategy weight {strategy_weight:.2f} below minimum")
        elif strategy_weight > 1.0:
            reasons.append(f"Strategy has elevated weight {strategy_weight:.2f}")

        return "; ".join(reasons)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learnings"""
        return {
            "total_updates": len(self.learning_updates),
            "recent_updates": [
                {
                    "type": u.update_type,
                    "description": u.description,
                }
                for u in self.learning_updates[-10:]
            ],
            "current_thresholds": {
                "confidence": self.confidence_threshold,
                "min_agreement": self.min_agent_agreement,
            },
            "strategy_weights": self.strategy_weights,
            "feedback_count": len(self.agent_feedback_history),
            "agent_feedback_summary": self._summarize_agent_feedback(),
        }

    def _summarize_agent_feedback(self) -> Dict[str, Any]:
        """Summarize feedback by agent"""
        summary = {}

        for feedback in self.agent_feedback_history[-50:]:
            if feedback.agent_name not in summary:
                summary[feedback.agent_name] = {
                    "positive": 0,
                    "negative": 0,
                    "adjustment": 0,
                }
            summary[feedback.agent_name][feedback.feedback_type.lower()] += 1

        return summary

    def export_learnings(self) -> str:
        """Export learnings to JSON format for persistence"""
        # Export full pattern data (not just statistics)
        patterns_data = {}
        for pattern_id, pattern in self.pattern_recognizer.patterns.items():
            patterns_data[pattern_id] = {
                "total_occurrences": pattern.total_occurrences,
                "winning_occurrences": pattern.winning_occurrences,
                "losing_occurrences": pattern.losing_occurrences,
                "avg_pnl": pattern.avg_pnl,
                "total_pnl": pattern.total_pnl,
                "preferred_strategies": pattern.preferred_strategies,
                "avoid_strategies": pattern.avoid_strategies,
                "confidence_score": pattern.confidence_score,
                "last_updated": pattern.last_updated.isoformat() if pattern.last_updated else None,
            }

        data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "total_trades_learned": sum(p.total_occurrences for p in self.pattern_recognizer.patterns.values()),
            "thresholds": {
                "confidence": self.confidence_threshold,
                "min_agreement": self.min_agent_agreement,
                "learning_rate": self.learning_rate,
            },
            "strategy_weights": self.strategy_weights,
            "patterns": patterns_data,
            "outcome_summary": self.outcome_analyzer.get_aggregate_learnings(last_n=100),
            "recent_updates": [
                {
                    "id": u.update_id,
                    "type": u.update_type,
                    "description": u.description,
                    "data": u.data,
                }
                for u in self.learning_updates[-50:]
            ],
        }
        return json.dumps(data, indent=2, default=str)

    def import_learnings(self, json_data: str) -> bool:
        """Import learnings from JSON format"""
        try:
            data = json.loads(json_data)

            # Restore thresholds
            thresholds = data.get("thresholds", {})
            self.confidence_threshold = thresholds.get("confidence", self.confidence_threshold)
            self.min_agent_agreement = thresholds.get("min_agreement", self.min_agent_agreement)
            self.learning_rate = thresholds.get("learning_rate", self.learning_rate)

            # Restore strategy weights
            self.strategy_weights.update(data.get("strategy_weights", {}))

            # Restore pattern data
            patterns_data = data.get("patterns", {})
            for pattern_id, pattern_state in patterns_data.items():
                if pattern_id in self.pattern_recognizer.patterns:
                    pattern = self.pattern_recognizer.patterns[pattern_id]
                    pattern.total_occurrences = pattern_state.get("total_occurrences", 0)
                    pattern.winning_occurrences = pattern_state.get("winning_occurrences", 0)
                    pattern.losing_occurrences = pattern_state.get("losing_occurrences", 0)
                    pattern.avg_pnl = pattern_state.get("avg_pnl", 0.0)
                    pattern.total_pnl = pattern_state.get("total_pnl", 0.0)
                    pattern.confidence_score = pattern_state.get("confidence_score", 0.0)

                    # Restore strategy preferences learned from trades
                    pattern.preferred_strategies = pattern_state.get("preferred_strategies", pattern.preferred_strategies)
                    pattern.avoid_strategies = pattern_state.get("avoid_strategies", pattern.avoid_strategies)

                    if pattern_state.get("last_updated"):
                        try:
                            pattern.last_updated = datetime.fromisoformat(pattern_state["last_updated"])
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse last_updated for pattern: {e}")
                            pattern.last_updated = None

            total_trades = data.get("total_trades_learned", 0)
            logger.info(f"Successfully imported learnings from {total_trades} trades")
            logger.info(f"  - Confidence threshold: {self.confidence_threshold:.2f}")
            logger.info(f"  - Strategy weights: {self.strategy_weights}")

            # Log pattern insights
            patterns_with_data = sum(1 for p in self.pattern_recognizer.patterns.values() if p.total_occurrences >= 5)
            logger.info(f"  - Patterns with data: {patterns_with_data}")

            return True

        except Exception as e:
            logger.error(f"Failed to import learnings: {e}")
            import traceback
            traceback.print_exc()
            return False
