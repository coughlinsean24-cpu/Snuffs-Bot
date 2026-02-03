# Snuffs Bot - 0DTE SPY Options Trading Bot

> **Autonomous paper trading bot for collecting ML training data on SPY 0DTE options**

---

## Project Goals

### Primary Objective
Build a self-improving trading bot that:
1. **Collects 100+ paper trades** to train a robust ML model
2. **Learns from every trade** to improve decision-making over time
3. **Minimizes losses** while maximizing learning opportunities
4. **Provides transparency** into all trading decisions

### Success Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Total Trades | 100+ | 127 |
| Win Rate | >50% | 45.7% |
| LONG_CALL Win Rate | >55% | **55.4%** |
| LONG_PUT Win Rate | >45% | 35.5% |
| Net P&L | Positive | **+$40** |

### Long-Term Vision
- Graduate from paper trading to live trading with small capital
- Achieve consistent profitability through ML-driven decisions
- Scale position sizes as confidence in the model grows

---

## Changelog

### February 3, 2026
- **Minimum Option Price Filter** - Added $0.15 minimum to prevent selecting garbage options with wide spreads
- **Dynamic Max Option Price** - Max price now scales with account size (risk-based calculation)
- **Render Deployment** - Added `render.yaml` for cloud hosting ($7-14/month)
- **Dashboard Support** - Enabled Streamlit dashboard for Render deployment

### February 2, 2026
- **VIX Live Data Fix** - Added Yahoo Finance fallback for VIX (Tastytrade streaming doesn't support index symbols)
- **Dynamic Position Sizing** - PositionSizer now calculates optimal contract count based on option price and account balance
- **TimeSeriesSplit ML Training** - Prevents data leakage by training on past, validating on future
- **XGBoost Regularization** - Added stronger regularization to prevent overfitting on small datasets:
  - `reg_alpha=0.1` (L1), `reg_lambda=1.0` (L2)
  - `max_depth=3` (reduced from 5)
  - `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`
- **Early Stopping** - Stops training when validation loss stops improving
- **Transaction Cost Modeling** - Added `gross_pnl`, `slippage`, `fees` columns to trade records

### February 1, 2026
- **OTM Option Selection** - Target delta changed from 0.45 (ATM) to 0.35 (OTM) for faster moves
- **Single-Contract Exits** - Tighter profit/stop targets for single contracts (+12%/-15%)
- **Dollar-Based Exits** - Take profit at $30, stop loss at $75 regardless of percentage
- **Trailing Stop** - 15% drawdown from peak triggers exit

### January 2026
- **Initial Bot Development** - Core trading engine, Tastytrade API integration
- **Local AI System** - XGBoost model with 68 engineered features
- **Paper Trading Simulator** - Realistic fills, slippage, and fees
- **Learning Scheduler** - Continuous market snapshot collection

---

## Features Implemented

### Trading Engine
- [x] Tastytrade API integration (authentication, streaming, orders)
- [x] Paper trading simulator with realistic execution
- [x] Real-time position monitoring (100ms intervals)
- [x] Background learning (snapshots every minute)
- [x] EOD force-flat for 0DTE options

### AI/ML System
- [x] XGBoost classifier for entry decisions
- [x] 68 engineered features (momentum, Greeks, technicals, time)
- [x] TimeSeriesSplit validation (prevents data leakage)
- [x] Early stopping to prevent overfitting
- [x] Rule-based fallback when model not trained
- [x] Confidence threshold adjustment based on trade history

### Risk Management
- [x] Minimum option price filter ($0.15)
- [x] Maximum option price filter (dynamic based on account)
- [x] Delta range constraints (0.15-0.55)
- [x] VIX-based trade gating (max VIX 35)
- [x] Position sizing based on account risk %
- [x] Max daily loss limit
- [x] Max concurrent positions limit

### Data Collection
- [x] Market snapshots (SPY price, VIX, Greeks, technicals)
- [x] Trade records with full entry/exit details
- [x] Simulated trades for outcome analysis
- [x] News context integration

### Deployment
- [x] Docker containerization
- [x] Render cloud deployment config
- [x] Streamlit dashboard
- [x] Persistent storage for databases

### Pending Features
- [ ] PostgreSQL migration (for production)
- [ ] Multi-timeframe analysis
- [ ] Sentiment analysis from news
- [ ] Discord/Telegram alerts
- [ ] Backtesting framework improvements

---

## Technical Documentation

This section describes exactly how the bot makes trading decisions, including all data points collected, features engineered, and decision rules.

---

## Table of Contents
1. [Trading Overview](#trading-overview)
2. [Raw Data Collection](#raw-data-collection)
3. [Technical Indicators](#technical-indicators)
4. [Engineered Features](#engineered-features)
5. [Decision Logic](#decision-logic)
6. [Exit Logic](#exit-logic)
7. [Model Training](#model-training)

---

## Trading Overview

The bot uses a hybrid decision system:
1. **XGBoost Model** - When trained (15+ trades), predicts probability of profitable trade
2. **Rule-Based Fallback** - When no model exists, uses technical indicator rules

**Trading Target:** SPY 0DTE (Zero Days to Expiration) options
**Actions:** LONG_CALL, LONG_PUT, HOLD, EXIT

---

## Raw Data Collection

### SPY Price Data (8 fields)
| Field | Description | Source |
|-------|-------------|--------|
| `spy_price` | Current SPY mark price (bid-ask midpoint) | Tastytrade API |
| `spy_bid` | Current bid price | Tastytrade API |
| `spy_ask` | Current ask price | Tastytrade API |
| `spy_volume` | Current minute volume | Tastytrade API |
| `spy_high_today` | Intraday high | Calculated |
| `spy_low_today` | Intraday low | Calculated |
| `spy_open_today` | Opening price | Calculated |
| `spy_spread` | Ask minus bid (liquidity measure) | Calculated |

### SPY Momentum (4 fields)
| Field | Description | Calculation |
|-------|-------------|-------------|
| `spy_change_1m` | 1-minute price change % | `(current - price_1m_ago) / price_1m_ago * 100` |
| `spy_change_5m` | 5-minute price change % | `(current - price_5m_ago) / price_5m_ago * 100` |
| `spy_change_15m` | 15-minute price change % | `(current - price_15m_ago) / price_15m_ago * 100` |
| `spy_change_today` | Daily change from open % | `(current - open) / open * 100` |

### VIX Data (2 fields)
| Field | Description | Notes |
|-------|-------------|-------|
| `vix` | CBOE Volatility Index | Market fear gauge; typical range 12-50 |
| `vix_change` | VIX percentage change | Change from prior reading |

### Time-of-Day Features (5 fields) - CRITICAL FOR 0DTE
| Field | Description | Calculation |
|-------|-------------|-------------|
| `hour` | Current hour (0-23) | Eastern Time |
| `minute` | Current minute (0-59) | Eastern Time |
| `minutes_since_open` | Minutes since 9:30 AM | 0-390 range |
| `minutes_until_close` | Minutes until 4:00 PM | 390-0 range |
| `time_decay_factor` | Theta decay progression | `minutes_since_open / 390` (0.0 at open, 1.0 at close) |

### ATM Call Option Data (11 fields)
| Field | Description | Notes |
|-------|-------------|-------|
| `call_strike` | At-the-money strike price | Nearest to SPY price |
| `call_price` | Option mark (mid) price | Entry cost |
| `call_bid` | Option bid price | Exit floor |
| `call_ask` | Option ask price | Entry ceiling |
| `call_delta` | Delta Greek | Price sensitivity, 0-1 for calls |
| `call_gamma` | Gamma Greek | Delta acceleration |
| `call_theta` | Theta Greek | Time decay (negative) |
| `call_vega` | Vega Greek | IV sensitivity |
| `call_iv` | Implied volatility | Option's expected volatility |
| `call_volume` | Trading volume | Activity level |
| `call_open_interest` | Open interest | Existing positions |

### ATM Put Option Data (11 fields)
| Field | Description | Notes |
|-------|-------------|-------|
| `put_strike` | At-the-money strike price | Nearest to SPY price |
| `put_price` | Option mark (mid) price | Entry cost |
| `put_bid` | Option bid price | Exit floor |
| `put_ask` | Option ask price | Entry ceiling |
| `put_delta` | Delta Greek | Price sensitivity, -1 to 0 for puts |
| `put_gamma` | Gamma Greek | Delta acceleration |
| `put_theta` | Theta Greek | Time decay (negative) |
| `put_vega` | Vega Greek | IV sensitivity |
| `put_iv` | Implied volatility | Option's expected volatility |
| `put_volume` | Trading volume | Activity level |
| `put_open_interest` | Open interest | Existing positions |

### IV Skew (1 field)
| Field | Description | Significance |
|-------|-------------|--------------|
| `iv_skew` | `put_iv - call_iv` | Positive = bearish sentiment (puts more expensive) |

### Market Events (6 fields)
| Field | Description | Values |
|-------|-------------|--------|
| `fed_speaking` | Fed official speaking today | 0 or 1 |
| `fomc_day` | FOMC rate decision day | 0 or 1 |
| `rate_decision` | Rate decision announced | 0 or 1 |
| `earnings_major` | Major earnings (AAPL, MSFT, etc.) | 0 or 1 |
| `economic_data` | Major data release (CPI, jobs) | 0 or 1 |
| `event_notes` | Free-form event description | Text |

### News Context (6 fields)
| Field | Description | Values |
|-------|-------------|--------|
| `news_sentiment` | Overall news sentiment | -1.0 (bearish) to +1.0 (bullish) |
| `war_tensions` | Geopolitical conflict news | 0 or 1 |
| `tariff_news` | Trade war/tariff news | 0 or 1 |
| `fed_hawkish` | Fed raising rates stance | 0 or 1 |
| `fed_dovish` | Fed cutting rates stance | 0 or 1 |
| `recession_fears` | Recession fears in news | 0 or 1 |

---

## Technical Indicators

### RSI - Relative Strength Index (14-period)
| Field | Calculation | Signal |
|-------|-------------|--------|
| `rsi_14` | `100 - (100 / (1 + RS))` where RS = avg_gains / avg_losses | 0-100 |
| `rsi_signal` | OVERBOUGHT if >70, OVERSOLD if <30, else NEUTRAL | Text |
| `rsi_oversold` | 1 if RSI < 30 | Binary |
| `rsi_overbought` | 1 if RSI > 70 | Binary |

**Interpretation:** RSI < 30 suggests oversold (bullish reversal), RSI > 70 suggests overbought (bearish reversal)

### MACD - Moving Average Convergence Divergence
| Field | Calculation | Notes |
|-------|-------------|-------|
| `macd_line` | EMA(12) - EMA(26) | Main signal |
| `macd_signal` | 9-period EMA of MACD line | Smoothed signal |
| `macd_histogram` | `macd_line - macd_signal` | Momentum strength |
| `macd_crossover` | BULLISH (hist crosses above 0), BEARISH (below), NONE | Entry signal |

**Interpretation:** Bullish crossover = buy signal, Bearish crossover = sell signal

### VWAP - Volume Weighted Average Price
| Field | Calculation | Notes |
|-------|-------------|-------|
| `vwap` | `cumulative(typical_price × volume) / cumulative(volume)` | Resets daily |
| `price_vs_vwap` | `(price - vwap) / vwap * 100` | % deviation |
| `vwap_signal` | ABOVE (>0.1%), BELOW (<-0.1%), AT | Position |

**Interpretation:** Price above VWAP = bullish (buyers in control), below = bearish

### Bollinger Bands (20-period, 2 std dev)
| Field | Calculation | Notes |
|-------|-------------|-------|
| `bb_upper` | SMA(20) + 2 × StdDev | Upper band |
| `bb_middle` | SMA(20) | Middle band |
| `bb_lower` | SMA(20) - 2 × StdDev | Lower band |
| `bb_width` | `(upper - lower) / middle * 100` | Volatility measure |
| `bb_position` | `(price - lower) / (upper - lower)` | 0=lower, 0.5=middle, 1=upper |

**Interpretation:** Near lower band (position < 0.1) = potential bounce, near upper (> 0.9) = potential pullback

### Moving Averages
| Field | Calculation | Notes |
|-------|-------------|-------|
| `ema_9` | 9-period Exponential MA | Fast MA |
| `ema_21` | 21-period Exponential MA | Slow MA |
| `sma_50` | 50-period Simple MA | Trend reference |
| `ma_trend` | BULLISH if price > EMA9 > EMA21, BEARISH if price < EMA9 < EMA21 | Trend state |

**Interpretation:** Bullish trend = price above both EMAs with EMA9 > EMA21

### ATR - Average True Range (14-period)
| Field | Calculation | Notes |
|-------|-------------|-------|
| `atr_14` | Average of True Ranges | Volatility measure |
| `atr_percent` | `atr / price * 100` | Normalized volatility |

Where True Range = max(high-low, |high-prev_close|, |low-prev_close|)

### Momentum
| Field | Calculation | Notes |
|-------|-------------|-------|
| `momentum_10` | `price - price_10_bars_ago` | Absolute momentum |
| `rate_of_change` | `(price - price_10) / price_10 * 100` | ROC % |

### Composite Signal
| Field | Description | Values |
|-------|-------------|--------|
| `tech_signal` | Combined technical signal | BUY, SELL, HOLD |
| `tech_signal_strength` | Confidence of signal | 0.0 to 1.0 |

**Composite Signal Calculation:**
```
Bullish Points:
+2 if RSI oversold
+3 if MACD bullish crossover
+1 if MACD histogram > 0
+2 if price above VWAP
+2 if MA trend bullish
+1 if near lower Bollinger band

Bearish Points:
+2 if RSI overbought
+3 if MACD bearish crossover
+1 if MACD histogram < 0
+2 if price below VWAP
+2 if MA trend bearish
+1 if near upper Bollinger band

Signal = BUY if bullish% > 60%, SELL if bearish% > 60%, else HOLD
```

---

## Engineered Features

The model uses 68 normalized features (all values scaled to roughly -1 to +1 range):

### Category 1: Price Momentum (6 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 0 | `spy_change_1m` | Clipped to ±2%, divided by 2 |
| 1 | `spy_change_5m` | Clipped to ±2%, divided by 2 |
| 2 | `spy_change_15m` | Clipped to ±2%, divided by 2 |
| 3 | `spy_change_today` | Clipped to ±2%, divided by 2 |
| 4 | `spy_momentum_acceleration` | `(1m_change - 5m_change) / 2` |
| 5 | `spy_trend_strength` | `min(1, abs(15m_change) / 2)` |

### Category 2: Time-of-Day (6 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 6 | `hour_sin` | `sin(2π × hour/24)` - cyclical encoding |
| 7 | `hour_cos` | `cos(2π × hour/24)` - cyclical encoding |
| 8 | `time_decay_factor` | Raw value 0-1 |
| 9 | `is_first_hour` | 1 if 9:30-10:30 AM |
| 10 | `is_last_hour` | 1 if 3:00-4:00 PM |
| 11 | `is_power_hour` | 1 if 2:00-3:00 PM |

### Category 3: Volatility (6 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 12 | `vix_normalized` | `(VIX - 20) / 15` |
| 13 | `vix_high` | 1 if VIX > 25 |
| 14 | `call_iv` | `call_iv / 1.0` |
| 15 | `put_iv` | `put_iv / 1.0` |
| 16 | `iv_skew` | `(put_iv - call_iv) / 1.0` |
| 17 | `iv_mean` | `(call_iv + put_iv) / 2` |

### Category 4: Call Greeks (4 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 18 | `call_delta` | Raw delta (0-1) |
| 19 | `call_gamma_normalized` | `min(1, gamma × 10)` |
| 20 | `call_theta_normalized` | `max(-1, theta / 0.5)` |
| 21 | `call_moneyness` | `(spot - strike) / strike * 100 / 2` |

### Category 5: Put Greeks (4 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 22 | `put_delta` | Raw delta (-1 to 0) |
| 23 | `put_gamma_normalized` | `min(1, gamma × 10)` |
| 24 | `put_theta_normalized` | `max(-1, theta / 0.5)` |
| 25 | `put_moneyness` | `(strike - spot) / strike * 100 / 2` |

### Category 6: Market Structure (6 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 26 | `spy_spread_normalized` | `min(1, spread / 0.50)` |
| 27 | `call_spread_normalized` | `min(1, spread / 0.50)` |
| 28 | `put_spread_normalized` | `min(1, spread / 0.50)` |
| 29 | `call_bid_ask_ratio` | `call_bid / call_ask` |
| 30 | `put_bid_ask_ratio` | `put_bid / put_ask` |
| 31 | `liquidity_score` | `1 - min(1, avg_spread / 0.50)` |

### Category 7: Direction Indicators (4 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 32 | `is_bullish_momentum` | 1 if 5m_change > 0.1% |
| 33 | `is_bearish_momentum` | 1 if 5m_change < -0.1% |
| 34 | `momentum_strength` | `min(1, abs(5m_change) / 0.5)` |
| 35 | `trend_consistency` | 1 if 1m, 5m, 15m all same direction |

### Category 8: News/Context (6 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 36 | `news_sentiment` | Clipped to -1 to +1 |
| 37 | `war_tensions` | Binary 0/1 |
| 38 | `tariff_news` | Binary 0/1 |
| 39 | `fed_hawkish` | Binary 0/1 |
| 40 | `fed_dovish` | Binary 0/1 |
| 41 | `recession_fears` | Binary 0/1 |

### Category 9: Volume Features (6 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 42 | `spy_volume_surge` | 1 if volume > 1.5× avg (130K/min) |
| 43 | `call_volume_ratio` | `call_vol / (call_vol + put_vol)` |
| 44 | `put_volume_ratio` | `put_vol / (call_vol + put_vol)` |
| 45 | `call_oi_ratio` | `call_OI / total_OI` |
| 46 | `volume_momentum` | `(curr_vol - prev_vol) / prev_vol` clipped |
| 47 | `option_flow_imbalance` | `(call_vol - put_vol) / total_vol` clipped |

### Category 10: Technical Indicators (20 features)
| Index | Name | Calculation |
|-------|------|-------------|
| 48 | `rsi_14` | `rsi / 100` (normalized 0-1) |
| 49 | `rsi_oversold` | 1 if RSI < 30 |
| 50 | `rsi_overbought` | 1 if RSI > 70 |
| 51 | `macd_histogram` | `clipped(histogram × 10, -1, 1)` |
| 52 | `macd_bullish_cross` | 1 if bullish crossover |
| 53 | `macd_bearish_cross` | 1 if bearish crossover |
| 54 | `macd_above_signal` | 1 if MACD > signal line |
| 55 | `price_vs_vwap` | Clipped to -1 to +1 |
| 56 | `above_vwap` | 1 if price > VWAP |
| 57 | `vwap_deviation` | `min(1, abs(price_vs_vwap))` |
| 58 | `bb_position` | Raw 0-1 value |
| 59 | `bb_width` | `min(1, width / 5)` |
| 60 | `bb_lower_touch` | 1 if position < 0.1 |
| 61 | `bb_upper_touch` | 1 if position > 0.9 |
| 62 | `ma_trend_bullish` | 1 if trend = BULLISH |
| 63 | `ma_trend_bearish` | 1 if trend = BEARISH |
| 64 | `ema_crossover` | 1 if EMA9 near EMA21 |
| 65 | `atr_percent` | `min(1, atr% / 2)` |
| 66 | `high_volatility` | 1 if ATR% > 0.5% |
| 67 | `rate_of_change` | `clipped(ROC / 2, -1, 1)` |

---

## Decision Logic

### Mode Selection
```
IF learning_mode (paper trading):
    confidence_threshold = 0.45 (45%)
    momentum_threshold = 0.03% (very low)
ELSE (live trading):
    confidence_threshold = 0.65 (65%)
    momentum_threshold = 0.10% (higher)
```

### XGBoost Model Decision Flow
```
1. Extract 68 features from market snapshot
2. Get model probability: profitable_prob = model.predict_proba(features)[1]

3. IF profitable_prob < confidence_threshold:
   a. In learning mode: Look at technical signals
      - Count bullish signals: MACD bullish (+2), RSI < 35 (+2), RSI < 45 (+1),
                              VWAP above (+1), 5m_change > 0.01% (+1)
      - Count bearish signals: MACD bearish (+2), RSI > 65 (+2), RSI > 55 (+1),
                              VWAP below (+1), 5m_change < -0.01% (+1)
      - IF bullish_signals >= 1 AND bullish > bearish: LONG_CALL
      - ELIF bearish_signals >= 1 AND bearish > bullish: LONG_PUT
      - ELIF consecutive_holds >= 10: Force random trade (need learning data)
      - ELSE: HOLD
   b. In live mode: HOLD (wait for better setup)

4. IF profitable_prob >= confidence_threshold:
   - Check momentum direction:
     IF spy_5m > momentum_threshold AND spy_15m > 0: LONG_CALL
     ELIF spy_5m < -momentum_threshold AND spy_15m < 0: LONG_PUT
     ELSE (in learning mode): Use technical signals for direction
     ELSE (in live mode): HOLD
```

### Rule-Based Fallback (No Model)
```
RULE 1: Skip first 5 minutes after open
        IF minutes_since_open < 5: HOLD ("Waiting for price discovery")

RULE 2: Avoid last 30 minutes (theta crush)
        IF time_decay_factor > 0.92: HOLD ("Theta decay too aggressive")

RULE 3: Require momentum
        Paper mode threshold: 0.001% (nearly any movement)
        Live mode threshold: 0.15%
        IF abs(spy_5m) < threshold: HOLD ("Insufficient momentum")

RULE 4: Trend consistency
        Paper mode: Just need 5m direction
        Live mode: Need 1m, 5m, 15m all same direction
        IF not consistent: HOLD ("Mixed signals")

RULE 5: VIX check
        IF vix > 35: HOLD ("Market too volatile")

RULE 6: Calculate confidence
        base_confidence = 0.5 + (momentum_strength × 0.3)
        +0.05 if 10:30 AM - 12:30 PM (trending period)
        +0.05 if 2:00 - 3:00 PM (power hour)
        +0.05 if VIX 15-22 (goldilocks zone)
        Cap at 85%

RULE 7: Determine direction
        IF spy_5m > 0: LONG_CALL
        ELSE: LONG_PUT
```

---

## Exit Logic

### Fixed Thresholds
```
PROFIT TARGET: Exit if P&L >= +15%
  - Confidence: 95%
  - Reason: "PROFIT_TARGET"

STOP LOSS: Exit if P&L <= -20%
  - Confidence: 99%
  - Reason: "STOP_LOSS"
```

### Trailing Stop
```
IF max_profit_ever >= 5%:
    drawdown = max_profit - current_pnl
    IF drawdown >= 15%:
        EXIT
        Reason: "TRAILING_STOP"
        Confidence: 90%
```

### Dollar-Based Exits (NEW - for single contracts)
```
DOLLAR PROFIT TARGET:
    IF dollar_pnl >= $30:
        EXIT
        Reason: "DOLLAR_PROFIT_TARGET"
        Confidence: 92%
        Note: Takes profit regardless of percentage for single contracts

DOLLAR STOP LOSS:
    IF dollar_pnl <= -$75:
        EXIT
        Reason: "DOLLAR_STOP_LOSS"
        Confidence: 98%
        Note: Hard stop to protect capital on single-contract trades
```

### Single-Contract Aggressive Exits (NEW)
```
PROFIT TARGET (single contract):
    IF contracts == 1 AND pnl >= +12%:
        EXIT
        Reason: "SINGLE_CONTRACT_PROFIT"
        Confidence: 90%

STOP LOSS (single contract):
    IF contracts == 1 AND pnl <= -15%:
        EXIT
        Reason: "SINGLE_CONTRACT_STOP"
        Confidence: 95%
```

### Time Decay Exit
```
IF time_decay_factor > 0.9 AND pnl < 5%:
    EXIT
    Reason: "TIME_DECAY"
    Confidence: 85%
```

### Adverse Momentum Exit
```
IF holding LONG_CALL AND spy_5m < -0.3%:
    EXIT
    Reason: "ADVERSE_MARKET"
    Confidence: 75%

IF holding LONG_PUT AND spy_5m > +0.3%:
    EXIT
    Reason: "ADVERSE_MARKET"
    Confidence: 75%
```

---

## Strike Selection

### OTM vs ATM Option Selection (NEW)
```
TARGET DELTA: 0.35 (default - OTM for faster moves)

OTM OPTIONS (delta 0.20-0.35):
    - Cheaper premiums ($1-3 vs $4-6)
    - Faster percentage moves (higher gamma)
    - More leverage, higher risk/reward
    - Better for single-contract trading
    - Quicker exit triggers

ATM OPTIONS (delta 0.45-0.55):
    - More expensive premiums
    - Slower percentage moves
    - Higher win probability
    - Better for multi-contract positions
    - Harder to time exits on single contracts
```

### Delta Range Constraints
```
MIN_DELTA: 0.15 (avoid very far OTM - low probability)
MAX_DELTA: 0.55 (avoid deep ITM - high capital, slow moves)

Options outside this range are filtered out before selection.
```

---

## Model Training

### Training Requirements
- Minimum trades: 15
- Retraining interval: Every 20 new trades

### XGBoost Parameters
```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)
```

### Training Labels
```
y = 1 if trade was profitable (P&L > 0)
y = 0 if trade was unprofitable
```

### Feature Importance
After training, the model outputs feature importances showing which features most influence predictions. This helps identify what market conditions are most predictive.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Raw data fields collected | 78 |
| Technical indicator fields | 24 |
| Engineered ML features | 68 |
| Entry decision rules | 7 |
| Exit decision rules | 8 |

---

## Key Thresholds Reference

| Parameter | Paper Mode | Live Mode |
|-----------|------------|-----------|
| Target Delta | 0.35 (OTM) | 0.35 (OTM) |
| Confidence threshold | 45% | 65% |
| Momentum threshold | 0.03% | 0.10% |
| Profit target (multi) | +15% | +15% |
| Profit target (single) | +12% | +12% |
| Stop loss (multi) | -20% | -20% |
| Stop loss (single) | -15% | -15% |
| Dollar profit target | $30 | $30 |
| Dollar stop loss | $75 | $75 |
| Trailing stop trigger | +5% | +5% |
| Trailing stop activation | -15% from peak | -15% from peak |
| Time decay exit threshold | 90% of day | 90% of day |
| VIX maximum | 35 | 35 |
| First N minutes skip | 5 | 5 |

---

## Configuration Reference (NEW)

These settings can be adjusted in `.env` or `settings.py`:

```
# OTM Option Selection
LONG_OPTION_DELTA=0.35          # Target delta (0.35 = OTM, 0.50 = ATM)
PREFER_OTM_OPTIONS=true         # Prefer OTM for faster moves
MIN_OPTION_DELTA=0.15           # Minimum delta to consider
MAX_OPTION_DELTA=0.55           # Maximum delta to consider

# Dollar-Based Exit Thresholds
ENABLE_DOLLAR_EXITS=true        # Use dollar-based exits
MIN_DOLLAR_PROFIT_TARGET=30.0   # Take profit at $30 gain
MAX_DOLLAR_LOSS=75.0            # Stop at $75 loss

# Single-Contract Aggressive Exits
SINGLE_CONTRACT_AGGRESSIVE_EXITS=true
SINGLE_CONTRACT_PROFIT_TARGET_PCT=0.12  # 12% profit target
SINGLE_CONTRACT_STOP_LOSS_PCT=0.15      # 15% stop loss
```

---

*Document generated from Snuffs Bot source code analysis*
*Last updated: February 3, 2026 - Added project goals, changelog, features list, Render deployment*
