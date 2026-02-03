"""
Paper Trading Simulator

Simulates trading using live market data from Tastytrade,
but all trades are tracked internally - no orders sent to broker.

Features:
- Virtual portfolio with configurable starting capital
- Dynamic position sizing for any account size ($200 to $100k+)
- Long calls/puts strategy (0DTE SPY options)
- ADAPTIVE EXITS: Trailing stops, momentum exits, reversal detection
- REAL-TIME MONITORING: Sub-second position tracking for 0DTE
- Real-time position updates using live market prices
- Realistic fill simulation with slippage modeling
- P&L tracking and performance metrics
- Parallel execution alongside live trading
"""

from .simulator import PaperTradingSimulator, SimulatorConfig
from .virtual_portfolio import VirtualPortfolio, VirtualPosition
from .order_simulator import OrderSimulator, SimulatedFill
from .execution_coordinator import ExecutionCoordinator
from .position_sizing import PositionSizer, PositionSize, calculate_contracts_for_amount
from .adaptive_exits import AdaptiveExitManager, ExitSignal, ExitRecommendation
from .realtime_monitor import RealTimePositionMonitor, DynamicExitMode, DynamicTargets, MonitoredPosition

__all__ = [
    "PaperTradingSimulator",
    "SimulatorConfig",
    "VirtualPortfolio",
    "VirtualPosition",
    "OrderSimulator",
    "SimulatedFill",
    "ExecutionCoordinator",
    "PositionSizer",
    "PositionSize",
    "calculate_contracts_for_amount",
    "AdaptiveExitManager",
    "ExitSignal",
    "ExitRecommendation",
    "RealTimePositionMonitor",
    "DynamicExitMode",
    "DynamicTargets",
    "MonitoredPosition",
]
