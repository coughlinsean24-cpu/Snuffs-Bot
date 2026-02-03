"""
Local AI Trading System

A self-learning trading AI that runs entirely locally without external API calls.
Learns from live market data and trade outcomes to make entry/exit decisions.

Components:
- DataCollector: Records SPY, options, Greeks, indicators continuously
- FeatureEngineer: Builds ML features from raw market data  
- TradingModel: XGBoost-based entry/exit prediction model
- Trainer: Handles model training and retraining
- HybridOrchestrator: Manages local AI with optional Claude fallback
- NewsCollector: Fetches and analyzes news for market context awareness

The model understands:
- Time-of-day patterns (0DTE options behave differently at open vs close)
- Momentum across multiple timeframes (1m, 5m, 15m)
- Volatility (VIX, IV, spreads)
- Greeks (delta, gamma, theta)
- Market structure (liquidity, bid-ask spreads)
- News/context (WHY the market is moving - geopolitics, Fed, earnings)
"""

from .data_collector import DataCollector, MarketSnapshot, TradeRecord
from .feature_engineer import FeatureEngineer, FeatureSet
from .trading_model import LocalTradingModel, TradingDecision
from .trainer import ModelTrainer
from .hybrid_orchestrator import HybridOrchestrator, HybridDecision
from .news_collector import NewsCollector, NewsItem, MarketContext

__all__ = [
    # Data collection
    "DataCollector",
    "MarketSnapshot",
    "TradeRecord",
    
    # Feature engineering
    "FeatureEngineer",
    "FeatureSet",
    
    # Model
    "LocalTradingModel",
    "TradingDecision",
    
    # Training
    "ModelTrainer",
    
    # Orchestration
    "HybridOrchestrator",
    "HybridDecision",
    
    # News/Context
    "NewsCollector",
    "NewsItem",
    "MarketContext",
]
