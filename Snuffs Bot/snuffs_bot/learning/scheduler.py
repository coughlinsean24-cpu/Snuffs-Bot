"""
Learning Scheduler

Schedules and orchestrates the continuous learning process.
Coordinates analysis, pattern updates, and feedback integration.
"""

import asyncio
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from loguru import logger

from .outcome_analyzer import OutcomeAnalyzer, TradeOutcome, OutcomeType
from .pattern_recognition import PatternRecognizer
from .feedback_integrator import FeedbackIntegrator
from .metrics_tracker import LearningMetricsTracker


class LearningEvent(Enum):
    """Types of learning events"""
    TRADE_COMPLETED = "TRADE_COMPLETED"
    PERIOD_END = "PERIOD_END"
    DAILY_SUMMARY = "DAILY_SUMMARY"
    PATTERN_REFRESH = "PATTERN_REFRESH"
    THRESHOLD_CHECK = "THRESHOLD_CHECK"
    EXPORT_LEARNINGS = "EXPORT_LEARNINGS"


class LearningScheduler:
    """
    Schedules and coordinates continuous learning activities.

    Responsibilities:
    - Process trade completions in real-time
    - Run periodic analysis and updates
    - Generate daily learning summaries
    - Coordinate all learning components
    """

    def __init__(self):
        """Initialize the learning scheduler"""
        # Initialize components
        self.outcome_analyzer = OutcomeAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.feedback_integrator = FeedbackIntegrator(
            self.outcome_analyzer,
            self.pattern_recognizer
        )
        self.metrics_tracker = LearningMetricsTracker()

        # Scheduler state
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self._event_handlers: Dict[LearningEvent, List[Callable]] = {
            event: [] for event in LearningEvent
        }

        # Scheduling intervals
        self.period_interval_minutes = 60  # Close period every hour
        self.pattern_refresh_minutes = 30  # Refresh patterns every 30 min
        self.threshold_check_minutes = 120  # Check thresholds every 2 hours

        # Daily summary time
        self.daily_summary_time = dt_time(16, 15)  # 4:15 PM ET

        # State tracking
        self.last_period_close = datetime.now()
        self.last_pattern_refresh = datetime.now()
        self.last_threshold_check = datetime.now()
        self.trades_today = 0

    def register_handler(
        self,
        event: LearningEvent,
        handler: Callable
    ) -> None:
        """
        Register an event handler

        Args:
            event: Event type to handle
            handler: Callback function
        """
        self._event_handlers[event].append(handler)

    async def _emit_event(
        self,
        event: LearningEvent,
        data: Dict[str, Any]
    ) -> None:
        """Emit an event to all registered handlers"""
        for handler in self._event_handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")

    async def process_trade_completion(
        self,
        trade_id: str,
        strategy: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        market_data: Dict[str, Any],
        ai_confidence: float = 0.7,
        agent_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Process a completed trade through the learning system

        Args:
            trade_id: Unique trade identifier
            strategy: Strategy used
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            pnl: Trade P&L
            exit_reason: Reason for exit
            market_data: Market conditions during trade
            ai_confidence: Overall AI confidence
            agent_scores: Individual agent scores

        Returns:
            Processing results including learnings
        """
        agent_scores = agent_scores or {
            "market_agent": 0.7,
            "risk_agent": 0.7,
            "execution_agent": 0.7
        }

        # Map exit reason to outcome type
        outcome_type_map = {
            "PROFIT_TARGET": OutcomeType.PROFIT_TARGET,
            "STOP_LOSS": OutcomeType.STOP_LOSS,
            "TIME_EXIT": OutcomeType.TIME_EXIT,
            "MANUAL": OutcomeType.MANUAL_EXIT,
            "MARKET_CLOSE": OutcomeType.MARKET_CLOSE,
            "RISK_BREACH": OutcomeType.RISK_BREACH,
        }
        outcome_type = outcome_type_map.get(exit_reason, OutcomeType.MANUAL_EXIT)

        # Calculate metrics
        max_profit = market_data.get("max_profit", entry_price * 100)
        hold_duration = int((exit_time - entry_time).total_seconds() / 60)

        pnl_percent = (pnl / (entry_price * 100)) * 100 if entry_price > 0 else 0
        profit_captured = (pnl / max_profit * 100) if max_profit > 0 and pnl > 0 else 0

        # Create outcome object
        outcome = TradeOutcome(
            trade_id=trade_id,
            strategy_type=strategy,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            outcome_type=outcome_type,
            spy_price_entry=market_data.get("spy_price_entry", 0),
            vix_entry=market_data.get("vix_entry", 15),
            market_regime=market_data.get("market_regime", "NEUTRAL"),
            trend_direction=market_data.get("trend_direction", "NEUTRAL"),
            spy_price_exit=market_data.get("spy_price_exit", 0),
            vix_exit=market_data.get("vix_exit", 15),
            spy_move_percent=market_data.get("spy_move_percent", 0),
            ai_confidence=ai_confidence,
            market_agent_score=agent_scores.get("market_agent", 0.7),
            risk_agent_score=agent_scores.get("risk_agent", 0.7),
            execution_agent_score=agent_scores.get("execution_agent", 0.7),
            max_profit_potential=max_profit,
            max_loss_potential=market_data.get("max_loss", 500),
            profit_captured_percent=profit_captured,
            hold_duration_minutes=hold_duration,
        )

        # Process through feedback integrator
        learning_updates = self.feedback_integrator.process_trade_outcome(
            outcome, market_data
        )

        # Generate agent feedback
        analysis = self.outcome_analyzer.analyses[-1]  # Get the just-created analysis
        agent_feedback = self.feedback_integrator.generate_agent_feedback(
            outcome, analysis
        )

        # Record in metrics tracker
        self.metrics_tracker.record_trade(
            trade_id=trade_id,
            pnl=pnl,
            strategy=strategy,
            confidence=ai_confidence,
            quality_score=analysis.score,
            timestamp=exit_time
        )

        self.trades_today += 1

        # Emit event
        await self._emit_event(LearningEvent.TRADE_COMPLETED, {
            "trade_id": trade_id,
            "pnl": pnl,
            "strategy": strategy,
            "learning_updates": len(learning_updates),
            "quality_score": analysis.score,
        })

        return {
            "trade_id": trade_id,
            "analysis": {
                "success": analysis.success,
                "score": analysis.score,
                "key_factors": analysis.key_factors,
                "recommendations": analysis.recommendations,
            },
            "learning_updates": [
                {"type": u.update_type, "description": u.description}
                for u in learning_updates
            ],
            "agent_feedback": [
                {"agent": f.agent_name, "type": f.feedback_type, "message": f.message}
                for f in agent_feedback
            ],
        }

    async def _run_periodic_tasks(self) -> None:
        """Run periodic learning tasks"""
        while self.is_running:
            try:
                now = datetime.now()

                # Period close check
                if (now - self.last_period_close).total_seconds() >= self.period_interval_minutes * 60:
                    await self._close_period()

                # Pattern refresh check
                if (now - self.last_pattern_refresh).total_seconds() >= self.pattern_refresh_minutes * 60:
                    await self._refresh_patterns()

                # Threshold check
                if (now - self.last_threshold_check).total_seconds() >= self.threshold_check_minutes * 60:
                    await self._check_thresholds()

                # Daily summary check
                if now.time() >= self.daily_summary_time and self.trades_today > 0:
                    await self._generate_daily_summary()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic tasks: {e}")
                await asyncio.sleep(60)

    async def _close_period(self) -> None:
        """Close the current performance period"""
        period = self.metrics_tracker.close_period()

        if period:
            # Take a snapshot
            pattern_count = len([
                p for p in self.pattern_recognizer.patterns.values()
                if p.total_occurrences >= 5
            ])

            self.metrics_tracker.take_snapshot(
                confidence_threshold=self.feedback_integrator.confidence_threshold,
                strategy_weights=self.feedback_integrator.strategy_weights,
                pattern_count=pattern_count,
            )

            await self._emit_event(LearningEvent.PERIOD_END, {
                "period": {
                    "trades": period.trades,
                    "win_rate": period.win_rate,
                    "total_pnl": period.total_pnl,
                }
            })

        self.last_period_close = datetime.now()
        logger.info("Performance period closed")

    async def _refresh_patterns(self) -> None:
        """Refresh pattern statistics"""
        stats = self.pattern_recognizer.get_pattern_statistics()

        await self._emit_event(LearningEvent.PATTERN_REFRESH, {
            "patterns_with_data": stats["patterns_with_data"],
            "top_performing": stats["top_performing"],
        })

        self.last_pattern_refresh = datetime.now()
        logger.info(f"Patterns refreshed: {stats['patterns_with_data']} with data")

    async def _check_thresholds(self) -> None:
        """Check and potentially adjust thresholds"""
        # Get current state
        old_threshold = self.feedback_integrator.confidence_threshold

        # The feedback integrator checks thresholds internally
        # We just need to log and emit event
        new_threshold = self.feedback_integrator.confidence_threshold

        await self._emit_event(LearningEvent.THRESHOLD_CHECK, {
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "changed": old_threshold != new_threshold,
        })

        self.last_threshold_check = datetime.now()

    async def _generate_daily_summary(self) -> None:
        """Generate end-of-day learning summary"""
        # Get comprehensive report
        report = self.metrics_tracker.generate_report()

        # Get learning summary
        learning_summary = self.feedback_integrator.get_learning_summary()

        # Get aggregate learnings
        aggregate = self.outcome_analyzer.get_aggregate_learnings(last_n=self.trades_today)

        summary = {
            "date": datetime.now().date().isoformat(),
            "trades_today": self.trades_today,
            "performance": {
                "total_pnl": aggregate.get("total_pnl", 0),
                "win_rate": aggregate.get("win_rate", 0),
                "avg_quality_score": aggregate.get("avg_quality_score", 50),
            },
            "learning_updates": learning_summary.get("total_updates", 0),
            "current_thresholds": {
                "confidence": self.feedback_integrator.confidence_threshold,
            },
            "strategy_weights": self.feedback_integrator.strategy_weights,
            "top_recommendations": aggregate.get("top_recommendations", [])[:3],
            "learning_progress": report.get("learning_progress", {}),
        }

        await self._emit_event(LearningEvent.DAILY_SUMMARY, summary)

        # Reset daily counter
        self.trades_today = 0

        logger.info(
            f"Daily summary generated: {summary['performance']['total_pnl']:+.2f} P&L, "
            f"{summary['performance']['win_rate']:.1%} win rate"
        )

    async def start(self) -> None:
        """Start the learning scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        self.is_running = True

        # Start periodic task
        task = asyncio.create_task(self._run_periodic_tasks())
        self._tasks.append(task)

        logger.info("Learning scheduler started")

    async def stop(self) -> None:
        """Stop the learning scheduler"""
        self.is_running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

        logger.info("Learning scheduler stopped")

    def get_agent_context(self) -> Dict[str, Any]:
        """
        Get current context for AI agents

        Returns:
            Context dictionary with all learnings
        """
        return self.feedback_integrator.generate_agent_context()

    def get_strategy_recommendation(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get strategy recommendation based on current market and learnings

        Args:
            market_data: Current market conditions

        Returns:
            Strategy recommendation with reasoning
        """
        return self.feedback_integrator.get_strategy_recommendation(market_data)

    def should_execute(
        self,
        confidence: float,
        agent_scores: Dict[str, float],
        strategy: str
    ) -> Dict[str, Any]:
        """
        Determine if a trade should be executed

        Args:
            confidence: AI confidence
            agent_scores: Individual agent scores
            strategy: Proposed strategy

        Returns:
            Execution decision with reasoning
        """
        return self.feedback_integrator.should_execute_trade(
            confidence, agent_scores, strategy
        )

    def get_learning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive learning report

        Returns:
            Full learning report
        """
        return {
            "metrics_report": self.metrics_tracker.generate_report(),
            "pattern_statistics": self.pattern_recognizer.get_pattern_statistics(),
            "learning_summary": self.feedback_integrator.get_learning_summary(),
            "current_thresholds": {
                "confidence": self.feedback_integrator.confidence_threshold,
                "min_agreement": self.feedback_integrator.min_agent_agreement,
            },
            "strategy_weights": self.feedback_integrator.strategy_weights,
        }

    def export_learnings(self) -> str:
        """Export all learnings to JSON"""
        return self.feedback_integrator.export_learnings()

    def import_learnings(self, json_data: str) -> bool:
        """Import learnings from JSON"""
        return self.feedback_integrator.import_learnings(json_data)

    def save_learnings_to_file(self, filepath: str = None) -> bool:
        """
        Save all learnings to a file for persistence.

        This is called automatically on shutdown and can be called manually.
        The same file is used for both Paper and Live modes so knowledge
        accumulated in Paper trading transfers to Live trading.

        Args:
            filepath: Custom file path, or uses default

        Returns:
            True if saved successfully
        """
        import os
        from pathlib import Path

        if filepath is None:
            # Default to project root/data/learnings.json
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            filepath = str(data_dir / "learnings.json")

        try:
            learnings_json = self.export_learnings()

            with open(filepath, 'w') as f:
                f.write(learnings_json)

            logger.success(f"Saved learnings to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save learnings to {filepath}: {e}")
            return False

    def load_learnings_from_file(self, filepath: str = None) -> bool:
        """
        Load learnings from a file on startup.

        This restores all learned parameters including:
        - Confidence thresholds
        - Strategy weights
        - Pattern statistics and win rates

        Args:
            filepath: Custom file path, or uses default

        Returns:
            True if loaded successfully
        """
        import os
        from pathlib import Path

        if filepath is None:
            project_root = Path(__file__).parent.parent.parent
            filepath = str(project_root / "data" / "learnings.json")

        if not os.path.exists(filepath):
            logger.info(f"No learnings file found at {filepath} - starting fresh")
            return False

        try:
            with open(filepath, 'r') as f:
                learnings_json = f.read()

            success = self.import_learnings(learnings_json)

            if success:
                logger.success(f"Loaded learnings from {filepath}")
            return success

        except Exception as e:
            logger.error(f"Failed to load learnings from {filepath}: {e}")
            return False
