"""
AI Trading Agents

Specialized agents that work together to make trading decisions:
- MarketAgent: Market analysis and opportunity identification
- RiskAgent: Risk assessment and guardrail enforcement
- ExecutionAgent: Trade execution parameter optimization
- ExitAgent: Real-time exit decisions for open positions
"""

from .base_agent import BaseAgent
from .market_agent import MarketAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .exit_agent import ExitAgent, QuickExitDecision

__all__ = [
    "BaseAgent",
    "MarketAgent",
    "RiskAgent",
    "ExecutionAgent",
    "ExitAgent",
    "QuickExitDecision",
]
