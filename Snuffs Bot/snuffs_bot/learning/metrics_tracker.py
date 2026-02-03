"""
Learning Metrics Tracker

Tracks and visualizes learning progress over time.
Monitors key performance indicators and learning effectiveness.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import statistics

from loguru import logger


@dataclass
class LearningSnapshot:
    """Snapshot of learning state at a point in time"""
    timestamp: datetime
    trades_analyzed: int
    win_rate: float
    avg_pnl: float
    confidence_threshold: float
    strategy_weights: Dict[str, float]
    pattern_count: int
    avg_quality_score: float


@dataclass
class PerformancePeriod:
    """Performance metrics for a specific period"""
    start_time: datetime
    end_time: datetime
    trades: int
    wins: int
    losses: int
    total_pnl: float
    avg_pnl: float
    best_trade: float
    worst_trade: float
    win_rate: float
    profit_factor: float


class LearningMetricsTracker:
    """
    Tracks learning progress and effectiveness over time.

    Monitors:
    - Performance improvements
    - Learning rate and efficiency
    - Pattern recognition accuracy
    - Threshold adjustment effectiveness
    """

    def __init__(self):
        """Initialize the metrics tracker"""
        self.snapshots: List[LearningSnapshot] = []
        self.performance_periods: List[PerformancePeriod] = []
        self.trade_outcomes: List[Dict[str, Any]] = []

        # Tracking state
        self.period_start = datetime.now()
        self.current_period_trades: List[Dict[str, Any]] = []

        # Benchmarks
        self.baseline_win_rate = 0.50
        self.baseline_avg_pnl = 0.0

    def record_trade(
        self,
        trade_id: str,
        pnl: float,
        strategy: str,
        confidence: float,
        quality_score: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade outcome

        Args:
            trade_id: Unique trade identifier
            pnl: Profit/loss from trade
            strategy: Strategy used
            confidence: AI confidence at entry
            quality_score: Quality score from analysis
            timestamp: Trade timestamp (defaults to now)
        """
        trade = {
            "trade_id": trade_id,
            "pnl": pnl,
            "strategy": strategy,
            "confidence": confidence,
            "quality_score": quality_score,
            "timestamp": timestamp or datetime.now(),
            "is_win": pnl > 0,
        }

        self.trade_outcomes.append(trade)
        self.current_period_trades.append(trade)

        logger.debug(f"Recorded trade {trade_id}: ${pnl:+.2f}")

    def take_snapshot(
        self,
        confidence_threshold: float,
        strategy_weights: Dict[str, float],
        pattern_count: int
    ) -> LearningSnapshot:
        """
        Take a snapshot of current learning state

        Args:
            confidence_threshold: Current confidence threshold
            strategy_weights: Current strategy weights
            pattern_count: Number of patterns with data

        Returns:
            Learning snapshot
        """
        recent_trades = self.trade_outcomes[-50:] if self.trade_outcomes else []

        if recent_trades:
            wins = [t for t in recent_trades if t["is_win"]]
            win_rate = len(wins) / len(recent_trades)
            avg_pnl = statistics.mean([t["pnl"] for t in recent_trades])
            avg_quality = statistics.mean([t["quality_score"] for t in recent_trades])
        else:
            win_rate = 0.5
            avg_pnl = 0.0
            avg_quality = 50.0

        snapshot = LearningSnapshot(
            timestamp=datetime.now(),
            trades_analyzed=len(self.trade_outcomes),
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            confidence_threshold=confidence_threshold,
            strategy_weights=strategy_weights.copy(),
            pattern_count=pattern_count,
            avg_quality_score=avg_quality,
        )

        self.snapshots.append(snapshot)

        logger.info(
            f"Snapshot taken: {len(self.trade_outcomes)} trades, "
            f"{win_rate:.1%} win rate, ${avg_pnl:.2f} avg P&L"
        )

        return snapshot

    def close_period(self) -> Optional[PerformancePeriod]:
        """
        Close the current performance period and start a new one

        Returns:
            The completed performance period, or None if no trades
        """
        if not self.current_period_trades:
            return None

        trades = self.current_period_trades
        pnls = [t["pnl"] for t in trades]
        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]

        win_pnl = sum(t["pnl"] for t in wins) if wins else 0
        loss_pnl = sum(t["pnl"] for t in losses) if losses else 0
        profit_factor = abs(win_pnl / loss_pnl) if loss_pnl != 0 else win_pnl if win_pnl > 0 else 0

        period = PerformancePeriod(
            start_time=self.period_start,
            end_time=datetime.now(),
            trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            total_pnl=sum(pnls),
            avg_pnl=statistics.mean(pnls) if pnls else 0,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            win_rate=len(wins) / len(trades) if trades else 0,
            profit_factor=profit_factor,
        )

        self.performance_periods.append(period)

        # Reset for next period
        self.period_start = datetime.now()
        self.current_period_trades = []

        logger.info(
            f"Period closed: {period.trades} trades, "
            f"${period.total_pnl:+.2f} P&L, {period.win_rate:.1%} win rate"
        )

        return period

    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Calculate learning progress metrics

        Returns:
            Dictionary with progress metrics
        """
        if len(self.snapshots) < 2:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Need more snapshots to calculate progress",
                "snapshots": len(self.snapshots),
            }

        # Compare first and last snapshots
        first = self.snapshots[0]
        last = self.snapshots[-1]

        # Calculate improvements
        win_rate_change = last.win_rate - first.win_rate
        avg_pnl_change = last.avg_pnl - first.avg_pnl
        quality_change = last.avg_quality_score - first.avg_quality_score

        # Calculate trend (using last 5 snapshots)
        recent = self.snapshots[-5:]
        if len(recent) >= 2:
            win_rate_trend = recent[-1].win_rate - recent[0].win_rate
            pnl_trend = recent[-1].avg_pnl - recent[0].avg_pnl
        else:
            win_rate_trend = 0
            pnl_trend = 0

        # Determine learning effectiveness
        is_improving = win_rate_change > 0 and avg_pnl_change > 0
        trend_positive = win_rate_trend > 0 and pnl_trend > 0

        return {
            "status": "IMPROVING" if is_improving else "NEEDS_ATTENTION",
            "total_trades": len(self.trade_outcomes),
            "total_snapshots": len(self.snapshots),
            "improvements": {
                "win_rate": {
                    "initial": first.win_rate,
                    "current": last.win_rate,
                    "change": win_rate_change,
                    "change_percent": (win_rate_change / first.win_rate * 100) if first.win_rate > 0 else 0,
                },
                "avg_pnl": {
                    "initial": first.avg_pnl,
                    "current": last.avg_pnl,
                    "change": avg_pnl_change,
                },
                "quality_score": {
                    "initial": first.avg_quality_score,
                    "current": last.avg_quality_score,
                    "change": quality_change,
                },
            },
            "recent_trend": {
                "win_rate_direction": "UP" if win_rate_trend > 0 else "DOWN" if win_rate_trend < 0 else "FLAT",
                "pnl_direction": "UP" if pnl_trend > 0 else "DOWN" if pnl_trend < 0 else "FLAT",
                "is_positive": trend_positive,
            },
            "vs_baseline": {
                "win_rate_vs_baseline": last.win_rate - self.baseline_win_rate,
                "avg_pnl_vs_baseline": last.avg_pnl - self.baseline_avg_pnl,
            },
        }

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance breakdown by strategy

        Returns:
            Performance metrics for each strategy
        """
        strategy_trades: Dict[str, List[Dict]] = {}

        for trade in self.trade_outcomes:
            strategy = trade["strategy"]
            if strategy not in strategy_trades:
                strategy_trades[strategy] = []
            strategy_trades[strategy].append(trade)

        results = {}
        for strategy, trades in strategy_trades.items():
            pnls = [t["pnl"] for t in trades]
            wins = [t for t in trades if t["is_win"]]

            results[strategy] = {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(trades) - len(wins),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": statistics.mean(pnls) if pnls else 0,
                "best_trade": max(pnls) if pnls else 0,
                "worst_trade": min(pnls) if pnls else 0,
                "avg_confidence": statistics.mean([t["confidence"] for t in trades]) if trades else 0,
                "avg_quality": statistics.mean([t["quality_score"] for t in trades]) if trades else 0,
            }

        return results

    def get_confidence_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of confidence thresholds

        Returns:
            Analysis of confidence vs outcomes
        """
        if not self.trade_outcomes:
            return {"status": "NO_DATA"}

        # Group trades by confidence buckets
        buckets = {
            "low": [],      # < 0.55
            "medium": [],   # 0.55 - 0.70
            "high": [],     # 0.70 - 0.85
            "very_high": [] # > 0.85
        }

        for trade in self.trade_outcomes:
            conf = trade["confidence"]
            if conf < 0.55:
                buckets["low"].append(trade)
            elif conf < 0.70:
                buckets["medium"].append(trade)
            elif conf < 0.85:
                buckets["high"].append(trade)
            else:
                buckets["very_high"].append(trade)

        results = {}
        for bucket_name, trades in buckets.items():
            if trades:
                wins = [t for t in trades if t["is_win"]]
                results[bucket_name] = {
                    "trades": len(trades),
                    "win_rate": len(wins) / len(trades),
                    "avg_pnl": statistics.mean([t["pnl"] for t in trades]),
                }
            else:
                results[bucket_name] = {
                    "trades": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                }

        # Determine optimal threshold
        optimal_threshold = "medium"  # Default
        best_win_rate = 0
        for bucket_name, data in results.items():
            if data["trades"] >= 5 and data["win_rate"] > best_win_rate:
                best_win_rate = data["win_rate"]
                optimal_threshold = bucket_name

        return {
            "buckets": results,
            "optimal_threshold_bucket": optimal_threshold,
            "recommendation": self._get_threshold_recommendation(results),
        }

    def _get_threshold_recommendation(
        self,
        bucket_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate threshold recommendation based on bucket analysis"""
        high_data = bucket_results.get("high", {})
        very_high_data = bucket_results.get("very_high", {})
        medium_data = bucket_results.get("medium", {})

        if very_high_data.get("trades", 0) >= 5:
            if very_high_data["win_rate"] > 0.7:
                if high_data.get("win_rate", 0) > 0.65:
                    return "Consider lowering threshold - high confidence trades performing well"
                return "Current high threshold effective"

        if high_data.get("trades", 0) >= 5 and high_data["win_rate"] < 0.5:
            return "Consider raising threshold - high confidence trades underperforming"

        if medium_data.get("trades", 0) >= 5 and medium_data["win_rate"] > 0.6:
            return "Medium confidence trades performing well - threshold may be optimal"

        return "Continue monitoring - need more data for recommendation"

    def get_time_of_day_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by time of day

        Returns:
            Performance metrics by time period
        """
        periods = {
            "morning_open": [],     # 9:30 - 10:00
            "mid_morning": [],      # 10:00 - 11:30
            "midday": [],           # 11:30 - 13:30
            "afternoon": [],        # 13:30 - 15:00
            "close": [],            # 15:00 - 16:00
        }

        for trade in self.trade_outcomes:
            ts = trade["timestamp"]
            if isinstance(ts, str):
                continue  # Skip if timestamp is not datetime

            hour = ts.hour
            minute = ts.minute

            if hour == 9 and minute >= 30:
                periods["morning_open"].append(trade)
            elif hour == 10 or (hour == 11 and minute < 30):
                periods["mid_morning"].append(trade)
            elif (hour == 11 and minute >= 30) or hour == 12 or (hour == 13 and minute < 30):
                periods["midday"].append(trade)
            elif (hour == 13 and minute >= 30) or hour == 14:
                periods["afternoon"].append(trade)
            elif hour >= 15:
                periods["close"].append(trade)

        results = {}
        for period_name, trades in periods.items():
            if trades:
                wins = [t for t in trades if t["is_win"]]
                results[period_name] = {
                    "trades": len(trades),
                    "win_rate": len(wins) / len(trades),
                    "avg_pnl": statistics.mean([t["pnl"] for t in trades]),
                    "total_pnl": sum([t["pnl"] for t in trades]),
                }
            else:
                results[period_name] = {
                    "trades": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "total_pnl": 0,
                }

        return results

    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """
        Analyze drawdowns from peak

        Returns:
            Drawdown metrics
        """
        if not self.trade_outcomes:
            return {"status": "NO_DATA"}

        # Calculate cumulative P&L
        cumulative = 0
        peak = 0
        max_drawdown = 0
        drawdown_periods = []
        current_drawdown_start = None

        for trade in sorted(self.trade_outcomes, key=lambda t: t["timestamp"]):
            cumulative += trade["pnl"]

            if cumulative > peak:
                peak = cumulative
                if current_drawdown_start:
                    # End of drawdown period
                    drawdown_periods.append({
                        "start": current_drawdown_start,
                        "end": trade["timestamp"],
                        "depth": max_drawdown,
                    })
                    current_drawdown_start = None
            else:
                drawdown = peak - cumulative
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                if not current_drawdown_start and drawdown > 0:
                    current_drawdown_start = trade["timestamp"]

        return {
            "max_drawdown": max_drawdown,
            "current_cumulative": cumulative,
            "peak": peak,
            "current_drawdown": peak - cumulative if peak > cumulative else 0,
            "drawdown_periods": len(drawdown_periods),
            "recovery_needed": max(0, peak - cumulative),
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive learning report

        Returns:
            Full learning metrics report
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_trades": len(self.trade_outcomes),
                "total_snapshots": len(self.snapshots),
                "periods_completed": len(self.performance_periods),
            },
            "learning_progress": self.get_learning_progress(),
            "strategy_performance": self.get_strategy_performance(),
            "confidence_effectiveness": self.get_confidence_effectiveness(),
            "time_of_day_performance": self.get_time_of_day_performance(),
            "drawdown_analysis": self.get_drawdown_analysis(),
            "recent_periods": [
                {
                    "start": p.start_time.isoformat(),
                    "end": p.end_time.isoformat(),
                    "trades": p.trades,
                    "win_rate": p.win_rate,
                    "total_pnl": p.total_pnl,
                }
                for p in self.performance_periods[-5:]
            ],
        }
