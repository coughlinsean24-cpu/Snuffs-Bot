"""
Learning Loop Module

Self-training system that learns from simulated trades using live market data.
Analyzes outcomes, recognizes patterns, and improves AI decision-making.
"""

from .outcome_analyzer import OutcomeAnalyzer, TradeOutcome
from .pattern_recognition import PatternRecognizer, MarketPattern
from .feedback_integrator import FeedbackIntegrator
from .metrics_tracker import LearningMetricsTracker
from .scheduler import LearningScheduler

__all__ = [
    "OutcomeAnalyzer",
    "TradeOutcome",
    "PatternRecognizer",
    "MarketPattern",
    "FeedbackIntegrator",
    "LearningMetricsTracker",
    "LearningScheduler",
]
