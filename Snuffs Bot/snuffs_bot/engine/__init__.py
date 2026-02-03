"""
Trading Engine Module

Main trading engine that orchestrates all components:
- AI decision making
- Strategy execution
- Risk management
- Paper trading simulation
- Learning loop
"""

from .trading_engine import TradingEngine, EngineState

__all__ = ["TradingEngine", "EngineState"]
