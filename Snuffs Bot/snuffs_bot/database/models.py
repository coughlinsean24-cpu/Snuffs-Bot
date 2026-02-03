"""
SQLAlchemy database models for 0DTE trading bot

Tables:
- trades: Core trade records (live and paper)
- ai_decisions: AI reasoning and decisions
- learning_insights: Post-trade analysis and learnings
- market_snapshots: Historical market data
- performance_metrics: Aggregated performance stats
- risk_limits: Dynamic risk management limits
"""

from sqlalchemy import Column, Integer, String, DECIMAL, TIMESTAMP, Boolean, ForeignKey, Text, BigInteger, Date, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class Trade(Base):
    """
    Core trade records for both live and paper trading

    Tracks complete lifecycle from entry to exit with Greeks,
    P&L, market conditions, and linkage to AI decisions
    """
    __tablename__ = 'trades'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Trade classification
    trade_type = Column(String(20), nullable=False, index=True)  # 'LIVE' or 'PAPER'
    strategy = Column(String(50), nullable=False, index=True)  # 'iron_condor', 'credit_spread', etc.

    # Entry details
    entry_time = Column(TIMESTAMP, nullable=False, index=True)
    entry_price = Column(DECIMAL(10, 4))
    entry_legs = Column(JSONB, nullable=False)  # Array of leg details with symbols, quantities, actions

    # Exit details
    exit_time = Column(TIMESTAMP)
    exit_price = Column(DECIMAL(10, 4))
    exit_reason = Column(String(50))  # 'profit_target', 'stop_loss', 'time_stop', 'manual'

    # Financial metrics
    position_size = Column(Integer, nullable=False)  # Number of contracts
    max_risk = Column(DECIMAL(10, 2), nullable=False)  # Maximum potential loss
    gross_pnl = Column(DECIMAL(10, 2))  # P&L before transaction costs
    pnl = Column(DECIMAL(10, 2))  # Net realized P&L (after all costs)
    pnl_percent = Column(DECIMAL(6, 2))  # P&L as percentage of risk
    fees = Column(DECIMAL(10, 2), default=0)  # Commission and fees
    slippage = Column(DECIMAL(10, 4), default=0)  # Execution slippage (entry + exit)

    # Greeks at entry (for options)
    delta = Column(DECIMAL(6, 4))
    gamma = Column(DECIMAL(6, 4))
    theta = Column(DECIMAL(6, 4))
    vega = Column(DECIMAL(6, 4))

    # Market snapshot at entry
    spy_price = Column(DECIMAL(10, 2))
    vix = Column(DECIMAL(6, 2))
    market_condition = Column(String(20))  # 'bullish', 'bearish', 'neutral'

    # Status tracking
    status = Column(String(20), nullable=False, default='OPEN', index=True)  # 'OPEN', 'CLOSED', 'CANCELLED'

    # External references
    tastytrade_order_id = Column(String(100))  # Tastytrade API order ID

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    ai_decision = relationship("AIDecision", back_populates="trade", uselist=False)
    learning_insights = relationship("LearningInsight", back_populates="trade")

    def __repr__(self):
        return f"<Trade(id={self.id}, strategy={self.strategy}, status={self.status}, pnl={self.pnl})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "trade_type": self.trade_type,
            "strategy": self.strategy,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": float(self.pnl) if self.pnl else None,
            "status": self.status,
            "spy_price": float(self.spy_price) if self.spy_price else None,
            "vix": float(self.vix) if self.vix else None,
        }


# Create indexes for better query performance
Index('idx_trades_status_entry', Trade.status, Trade.entry_time)
Index('idx_trades_type_strategy', Trade.trade_type, Trade.strategy)


class AIDecision(Base):
    """
    AI reasoning and decisions from the 3-agent system

    Stores responses from Market, Risk, and Execution agents
    along with consensus decision and token usage tracking
    """
    __tablename__ = 'ai_decisions'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    decision_time = Column(TIMESTAMP, nullable=False, server_default=func.now(), index=True)

    # Market analysis agent
    market_agent_response = Column(JSONB, nullable=False)
    market_confidence = Column(Integer)  # 0-100

    # Risk analysis agent
    risk_agent_response = Column(JSONB, nullable=False)
    risk_approval = Column(Boolean)

    # Execution planning agent
    execution_agent_response = Column(JSONB, nullable=False)

    # Consensus decision
    consensus_decision = Column(String(20), nullable=False)  # 'EXECUTE', 'REJECT', 'DEFER'
    consensus_reasoning = Column(Text)

    # Outcome tracking
    trade_id = Column(Integer, ForeignKey('trades.id'), index=True)
    was_executed = Column(Boolean, default=False)

    # Claude AI usage metrics
    total_tokens_used = Column(Integer)
    estimated_cost = Column(DECIMAL(6, 4))  # USD

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    trade = relationship("Trade", back_populates="ai_decision")

    def __repr__(self):
        return f"<AIDecision(id={self.id}, consensus={self.consensus_decision}, confidence={self.market_confidence})>"


# Index for AI decisions by trade
Index('idx_ai_decisions_trade', AIDecision.trade_id, AIDecision.decision_time)


class LearningInsight(Base):
    """
    Post-trade analysis and continuous learning insights

    AI analyzes completed trades to identify patterns,
    what worked, what failed, and recommendations
    """
    __tablename__ = 'learning_insights'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False, index=True)

    # Analysis content
    what_worked = Column(Text)
    what_failed = Column(Text)
    market_conditions_analysis = Column(Text)
    strategy_effectiveness_score = Column(Integer)  # 1-10 rating

    # Forward-looking recommendations
    future_recommendations = Column(Text)
    pattern_identified = Column(Text)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    trade = relationship("Trade", back_populates="learning_insights")

    def __repr__(self):
        return f"<LearningInsight(trade_id={self.trade_id}, score={self.strategy_effectiveness_score})>"


class MarketSnapshot(Base):
    """
    Historical market data snapshots

    Captures market conditions at regular intervals for
    backtesting, analysis, and AI training
    """
    __tablename__ = 'market_snapshots'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_time = Column(TIMESTAMP, nullable=False, unique=True, index=True)

    # SPY data
    spy_price = Column(DECIMAL(10, 2))
    spy_volume = Column(BigInteger)
    spy_daily_change_percent = Column(DECIMAL(6, 2))

    # Volatility indicators
    vix = Column(DECIMAL(6, 2))
    vix_trend = Column(String(20))  # 'rising', 'falling', 'stable'

    # Options market data
    atm_call_iv = Column(DECIMAL(6, 2))  # At-the-money implied volatility
    atm_put_iv = Column(DECIMAL(6, 2))
    put_call_ratio = Column(DECIMAL(6, 4))

    # Market breadth
    advance_decline_ratio = Column(DECIMAL(6, 4))

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())

    def __repr__(self):
        return f"<MarketSnapshot(time={self.snapshot_time}, spy={self.spy_price}, vix={self.vix})>"


class PerformanceMetric(Base):
    """
    Aggregated performance metrics by period

    Daily, weekly, and monthly statistics for tracking
    bot performance and strategy effectiveness
    """
    __tablename__ = 'performance_metrics'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Period definition
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    period_type = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly'

    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(DECIMAL(5, 2))  # Percentage

    # Financial metrics
    total_pnl = Column(DECIMAL(10, 2))
    avg_win = Column(DECIMAL(10, 2))
    avg_loss = Column(DECIMAL(10, 2))
    largest_win = Column(DECIMAL(10, 2))
    largest_loss = Column(DECIMAL(10, 2))

    # Risk-adjusted metrics
    sharpe_ratio = Column(DECIMAL(6, 4))
    max_drawdown = Column(DECIMAL(6, 2))  # Percentage
    profit_factor = Column(DECIMAL(6, 4))  # Gross profit / Gross loss

    # Strategy breakdown (JSON)
    strategy_performance = Column(JSONB)  # {"iron_condor": {...}, "credit_spread": {...}}

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())

    def __repr__(self):
        return f"<PerformanceMetric(period={self.period_type}, pnl={self.total_pnl}, win_rate={self.win_rate})>"


# Unique constraint on period
Index('idx_performance_period', PerformanceMetric.period_start, PerformanceMetric.period_end, PerformanceMetric.period_type, unique=True)


class RiskLimit(Base):
    """
    Dynamic risk management limits

    Configurable risk limits that can be updated in real-time
    Tracks current exposure against limits
    """
    __tablename__ = 'risk_limits'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Limit definition
    limit_type = Column(String(50), nullable=False)  # 'max_daily_loss', 'max_position_size', etc.
    limit_value = Column(DECIMAL(10, 2), nullable=False)
    current_exposure = Column(DECIMAL(10, 2), default=0)
    is_active = Column(Boolean, default=True)

    # Timestamps
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<RiskLimit(type={self.limit_type}, value={self.limit_value}, exposure={self.current_exposure})>"

    def is_limit_exceeded(self) -> bool:
        """Check if current exposure exceeds limit"""
        if not self.is_active:
            return False
        return float(self.current_exposure) >= float(self.limit_value)


# Metadata for all tables
metadata = Base.metadata
