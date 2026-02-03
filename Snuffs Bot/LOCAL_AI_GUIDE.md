# Local AI Trading System

## Overview

The Local AI is a self-learning XGBoost-based trading model that runs **entirely locally** with:
- **Zero API costs** (no Claude API calls)
- **Instant decisions** (sub-millisecond inference)
- **Continuous learning** from every trade

## Architecture

```
snuffs_bot/local_ai/
├── __init__.py            # Module exports
├── data_collector.py      # Records all market data to SQLite
├── feature_engineer.py    # Transforms raw data → 36 ML features
├── trading_model.py       # XGBoost entry/exit prediction model
├── trainer.py             # Model training and retraining logic
└── hybrid_orchestrator.py # Manages local AI with optional Claude fallback
```

## Features Tracked (36 Total)

### Price Momentum (6 features)
- 1-minute, 5-minute, 15-minute SPY changes
- Today's change, momentum acceleration, trend strength

### Time of Day (6 features)
- Hour (cyclically encoded), time decay factor
- First hour (9:30-10:30), last hour (3:00-4:00), power hour (2:00-3:00)

### Volatility (6 features)
- VIX normalized, VIX high indicator
- Call IV, Put IV, IV skew, mean IV

### Greeks (8 features)
- Delta, gamma, theta, moneyness for both calls and puts

### Market Structure (6 features)
- SPY spread, call spread, put spread
- Bid-ask ratios, liquidity score

### Direction Indicators (4 features)
- Bullish/bearish momentum, momentum strength, trend consistency

## How It Works

### Phase 1: Data Collection (Automatic)
While using Claude, the local AI **silently records**:
- Every market snapshot (SPY price, VIX, option Greeks)
- Every trade entry and exit
- All outcomes for learning

### Phase 2: Training (Automatic after 30+ trades)
Once you have enough trades, the model **automatically trains**:
- Uses XGBoost gradient boosting
- 80/20 train/validation split
- Retrains every 20 new trades

### Phase 3: Prediction
When enabled, the local AI makes decisions using:
- **Rule-based logic** (before training)
- **XGBoost model** (after training)

## Configuration

In `.env`:
```dotenv
# Enable Local AI (replaces Claude)
USE_LOCAL_AI=true

# Data storage location
LOCAL_AI_DATA_DIR=data/local_ai

# Training thresholds
LOCAL_AI_MIN_TRADES=30      # Min trades before training
LOCAL_AI_RETRAIN_INTERVAL=20 # Retrain after N new trades
```

## Rule-Based Decision Logic

When the XGBoost model isn't trained yet, the local AI uses rules:

### Entry Rules:
1. **Wait 10 minutes after open** (too volatile)
2. **No new entries in last 30 minutes** (theta crush)
3. **Need 0.15%+ 5-minute momentum**
4. **Trend consistency** (1m, 5m, 15m same direction)
5. **VIX < 35** (avoid extreme volatility)

### Exit Rules (FAST SCALPING):
- **Profit target**: 15%
- **Stop loss**: -20%
- **Trailing stop**: If up 5%+, exit on 15% drawdown
- **Time decay**: Exit if <40 min left without profit
- **Adverse momentum**: Exit on 0.3%+ move against position

## Data Storage

All data is stored in `data/local_ai/`:
- `market_data.db` - SQLite database with snapshots and trades
- `models/entry_model.pkl` - Trained XGBoost model
- `models/model_meta.json` - Model metadata
- `training_history.json` - Training run history

## Usage

### Check Current Status
```python
from snuffs_bot.local_ai import HybridOrchestrator

local_ai = HybridOrchestrator()
print(local_ai.trainer.get_status_report())
```

### Force Retrain
```python
metrics = local_ai.force_retrain()
print(f"Accuracy: {metrics['val_accuracy']:.1%}")
```

### Get Learning Insights
```python
insights = local_ai.get_stats()
print(f"Best trading hour: {insights['learning_insights']['best_trading_hour']}")
```

## Time-of-Day Learning

The model specifically learns:
- **9:30-10:30**: High volatility, more cautious
- **10:30-12:30**: Best trending conditions
- **14:00-15:00**: Power hour, increased activity
- **15:00-16:00**: Theta decay acceleration

This is critical for 0DTE options where **time decay accelerates exponentially** toward close.

## Migration Path

1. **Current State**: Claude API (costs ~$3-10/hour)
2. **Hybrid Mode**: Claude + data collection (default)
3. **Local AI Mode**: Full local model (set `USE_LOCAL_AI=true`)

The bot is currently collecting data in hybrid mode. After 30+ trades, you can enable local AI mode to eliminate API costs entirely.
