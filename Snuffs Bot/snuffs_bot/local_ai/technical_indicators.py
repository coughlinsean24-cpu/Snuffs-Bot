"""
Technical Indicators Module

Calculates standard trading indicators for AI decision making:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- VWAP (Volume Weighted Average Price)
- Bollinger Bands
- EMA/SMA
- ATR (Average True Range)
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class TechnicalIndicators:
    """Container for all technical indicator values"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # RSI
    rsi_14: float = 50.0  # 14-period RSI
    rsi_signal: str = "NEUTRAL"  # OVERBOUGHT, OVERSOLD, NEUTRAL
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = "NONE"  # BULLISH, BEARISH, NONE
    
    # VWAP
    vwap: float = 0.0
    price_vs_vwap: float = 0.0  # % above/below VWAP
    vwap_signal: str = "NEUTRAL"  # ABOVE, BELOW, AT
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0  # 20-period SMA
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.5  # 0 = at lower, 0.5 = middle, 1 = at upper
    
    # Moving Averages
    ema_9: float = 0.0
    ema_21: float = 0.0
    sma_50: float = 0.0
    ma_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    # ATR (volatility)
    atr_14: float = 0.0
    atr_percent: float = 0.0  # ATR as % of price
    
    # Momentum
    momentum_10: float = 0.0  # 10-bar momentum
    rate_of_change: float = 0.0  # ROC
    
    # Composite signals
    overall_signal: str = "HOLD"  # BUY, SELL, HOLD
    signal_strength: float = 0.0  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "rsi_14": self.rsi_14,
            "rsi_signal": self.rsi_signal,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "macd_crossover": self.macd_crossover,
            "vwap": self.vwap,
            "price_vs_vwap": self.price_vs_vwap,
            "vwap_signal": self.vwap_signal,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
            "bb_width": self.bb_width,
            "bb_position": self.bb_position,
            "ema_9": self.ema_9,
            "ema_21": self.ema_21,
            "sma_50": self.sma_50,
            "ma_trend": self.ma_trend,
            "atr_14": self.atr_14,
            "atr_percent": self.atr_percent,
            "momentum_10": self.momentum_10,
            "rate_of_change": self.rate_of_change,
            "overall_signal": self.overall_signal,
            "signal_strength": self.signal_strength,
        }


class TechnicalAnalyzer:
    """
    Calculates technical indicators from price history.
    
    Maintains rolling windows for efficient calculation.
    """
    
    def __init__(self, max_history: int = 200):
        """
        Initialize the technical analyzer
        
        Args:
            max_history: Maximum price history to maintain
        """
        self.max_history = max_history
        
        # Price history (OHLCV)
        self.prices: deque = deque(maxlen=max_history)
        self.highs: deque = deque(maxlen=max_history)
        self.lows: deque = deque(maxlen=max_history)
        self.volumes: deque = deque(maxlen=max_history)
        self.timestamps: deque = deque(maxlen=max_history)
        
        # VWAP tracking (resets daily)
        self.vwap_cumulative_pv: float = 0.0
        self.vwap_cumulative_v: float = 0.0
        self.vwap_date: Optional[datetime] = None
        
        # EMA state (for efficiency)
        self._ema_9: float = 0.0
        self._ema_21: float = 0.0
        self._ema_12: float = 0.0  # For MACD
        self._ema_26: float = 0.0  # For MACD
        self._macd_signal_ema: float = 0.0
        
        # RSI state
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        
        logger.info("TechnicalAnalyzer initialized")
    
    def update(
        self,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: int = 0,
        timestamp: Optional[datetime] = None
    ) -> TechnicalIndicators:
        """
        Update with new price data and calculate all indicators
        
        Args:
            price: Current price (close)
            high: High price (defaults to price)
            low: Low price (defaults to price)
            volume: Volume
            timestamp: Timestamp
            
        Returns:
            TechnicalIndicators with all calculated values
        """
        timestamp = timestamp or datetime.now()
        high = high or price
        low = low or price
        
        # Add to history
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.timestamps.append(timestamp)
        
        # Reset VWAP at start of new day
        if self.vwap_date is None or timestamp.date() != self.vwap_date.date():
            self.vwap_cumulative_pv = 0.0
            self.vwap_cumulative_v = 0.0
            self.vwap_date = timestamp
        
        # Calculate all indicators
        indicators = TechnicalIndicators(timestamp=timestamp)
        
        prices = list(self.prices)
        n = len(prices)
        
        if n < 2:
            return indicators
        
        # RSI (14-period)
        indicators.rsi_14 = self._calculate_rsi(prices, 14)
        if indicators.rsi_14 > 70:
            indicators.rsi_signal = "OVERBOUGHT"
        elif indicators.rsi_14 < 30:
            indicators.rsi_signal = "OVERSOLD"
        else:
            indicators.rsi_signal = "NEUTRAL"
        
        # MACD (12, 26, 9)
        if n >= 26:
            macd_result = self._calculate_macd(prices)
            indicators.macd_line = macd_result["macd"]
            indicators.macd_signal = macd_result["signal"]
            indicators.macd_histogram = macd_result["histogram"]
            indicators.macd_crossover = macd_result["crossover"]
        
        # VWAP
        typical_price = (high + low + price) / 3
        self.vwap_cumulative_pv += typical_price * volume
        self.vwap_cumulative_v += volume
        
        if self.vwap_cumulative_v > 0:
            indicators.vwap = self.vwap_cumulative_pv / self.vwap_cumulative_v
            indicators.price_vs_vwap = ((price - indicators.vwap) / indicators.vwap) * 100
            
            if indicators.price_vs_vwap > 0.1:
                indicators.vwap_signal = "ABOVE"
            elif indicators.price_vs_vwap < -0.1:
                indicators.vwap_signal = "BELOW"
            else:
                indicators.vwap_signal = "AT"
        
        # Bollinger Bands (20-period, 2 std)
        if n >= 20:
            bb = self._calculate_bollinger(prices, 20, 2)
            indicators.bb_upper = bb["upper"]
            indicators.bb_middle = bb["middle"]
            indicators.bb_lower = bb["lower"]
            indicators.bb_width = bb["width"]
            indicators.bb_position = bb["position"]
        
        # Moving Averages
        if n >= 9:
            indicators.ema_9 = self._calculate_ema(prices, 9)
        if n >= 21:
            indicators.ema_21 = self._calculate_ema(prices, 21)
        if n >= 50:
            indicators.sma_50 = sum(prices[-50:]) / 50
        
        # MA Trend
        if n >= 21:
            if price > indicators.ema_9 > indicators.ema_21:
                indicators.ma_trend = "BULLISH"
            elif price < indicators.ema_9 < indicators.ema_21:
                indicators.ma_trend = "BEARISH"
            else:
                indicators.ma_trend = "NEUTRAL"
        
        # ATR (14-period)
        if n >= 14:
            indicators.atr_14 = self._calculate_atr(14)
            indicators.atr_percent = (indicators.atr_14 / price) * 100
        
        # Momentum
        if n >= 10:
            indicators.momentum_10 = price - prices[-10]
            indicators.rate_of_change = ((price - prices[-10]) / prices[-10]) * 100
        
        # Composite Signal
        indicators.overall_signal, indicators.signal_strength = self._calculate_composite_signal(indicators)
        
        return indicators
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Get recent changes
        recent = changes[-(period):]
        
        gains = [c if c > 0 else 0 for c in recent]
        losses = [-c if c < 0 else 0 for c in recent]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema = sum(prices[:period]) / period
        
        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_macd(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate MACD (12, 26, 9)"""
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-period EMA of MACD)
        # For simplicity, use current MACD value
        # In production, you'd track MACD history
        if not hasattr(self, '_macd_history'):
            self._macd_history = deque(maxlen=9)
        self._macd_history.append(macd_line)
        
        if len(self._macd_history) >= 9:
            signal_line = self._calculate_ema(list(self._macd_history), 9)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        # Detect crossover
        crossover = "NONE"
        if len(self._macd_history) >= 2:
            prev_hist = list(self._macd_history)[-2] - signal_line if len(self._macd_history) > 1 else 0
            if prev_hist < 0 and histogram > 0:
                crossover = "BULLISH"
            elif prev_hist > 0 and histogram < 0:
                crossover = "BEARISH"
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "crossover": crossover,
        }
    
    def _calculate_bollinger(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        recent = prices[-period:]
        
        middle = sum(recent) / len(recent)
        
        # Standard deviation
        variance = sum((p - middle) ** 2 for p in recent) / len(recent)
        std = variance ** 0.5
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        width = (upper - lower) / middle * 100
        
        # Position within bands (0 = lower, 0.5 = middle, 1 = upper)
        current = prices[-1]
        if upper != lower:
            position = (current - lower) / (upper - lower)
        else:
            position = 0.5
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "width": width,
            "position": max(0, min(1, position)),
        }
    
    def _calculate_atr(self, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(self.prices) < period + 1:
            return 0.0
        
        highs = list(self.highs)
        lows = list(self.lows)
        closes = list(self.prices)
        
        true_ranges = []
        for i in range(-period, 0):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges)
    
    def _calculate_composite_signal(self, indicators: TechnicalIndicators) -> Tuple[str, float]:
        """
        Calculate composite trading signal from all indicators
        
        Returns:
            (signal, strength) where signal is BUY/SELL/HOLD and strength is 0-1
        """
        bullish_points = 0
        bearish_points = 0
        total_weight = 0
        
        # RSI (weight: 2)
        if indicators.rsi_signal == "OVERSOLD":
            bullish_points += 2
        elif indicators.rsi_signal == "OVERBOUGHT":
            bearish_points += 2
        total_weight += 2
        
        # MACD Crossover (weight: 3)
        if indicators.macd_crossover == "BULLISH":
            bullish_points += 3
        elif indicators.macd_crossover == "BEARISH":
            bearish_points += 3
        # MACD Histogram direction (weight: 1)
        if indicators.macd_histogram > 0:
            bullish_points += 1
        elif indicators.macd_histogram < 0:
            bearish_points += 1
        total_weight += 4
        
        # VWAP (weight: 2)
        if indicators.vwap_signal == "ABOVE":
            bullish_points += 2
        elif indicators.vwap_signal == "BELOW":
            bearish_points += 2
        total_weight += 2
        
        # MA Trend (weight: 2)
        if indicators.ma_trend == "BULLISH":
            bullish_points += 2
        elif indicators.ma_trend == "BEARISH":
            bearish_points += 2
        total_weight += 2
        
        # Bollinger Band position (weight: 1)
        if indicators.bb_position < 0.2:  # Near lower band
            bullish_points += 1
        elif indicators.bb_position > 0.8:  # Near upper band
            bearish_points += 1
        total_weight += 1
        
        # Calculate signal
        if total_weight == 0:
            return "HOLD", 0.0
        
        bull_pct = bullish_points / total_weight
        bear_pct = bearish_points / total_weight
        
        if bull_pct > 0.6 and bull_pct > bear_pct:
            return "BUY", bull_pct
        elif bear_pct > 0.6 and bear_pct > bull_pct:
            return "SELL", bear_pct
        else:
            return "HOLD", max(bull_pct, bear_pct)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current indicator summary"""
        if not self.prices:
            return {"status": "No data"}
        
        latest = self.update(
            self.prices[-1],
            self.highs[-1] if self.highs else None,
            self.lows[-1] if self.lows else None,
            self.volumes[-1] if self.volumes else 0,
        )
        
        return latest.to_dict()
