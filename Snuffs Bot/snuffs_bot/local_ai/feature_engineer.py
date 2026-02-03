"""
Feature Engineer

Transforms raw market data into ML-ready features.
Handles normalization, time-of-day encoding, and feature creation.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class FeatureSet:
    """A complete set of features for model input"""
    features: np.ndarray
    feature_names: List[str]
    timestamp: datetime
    raw_data: Dict[str, Any]


class FeatureEngineer:
    """
    Transforms market snapshots into ML-ready feature vectors.
    
    Features are organized into categories:
    1. Price momentum features (1m, 5m, 15m changes)
    2. Time-of-day features (critical for 0DTE)
    3. Volatility features (VIX, IV, spreads)
    4. Greeks features (delta, gamma, theta)
    5. Market structure features (volume, spreads)
    """
    
    # Feature normalization constants (will be updated during training)
    PRICE_CHANGE_SCALE = 2.0  # ±2% is typical range
    VIX_MEAN = 20.0
    VIX_SCALE = 15.0  # VIX typically ranges 12-50
    IV_SCALE = 1.0  # IV typically 0.1 to 2.0
    SPREAD_SCALE = 0.50  # Spread in dollars
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.feature_names = self._get_feature_names()
        self.n_features = len(self.feature_names)
        
        logger.info(f"FeatureEngineer initialized with {self.n_features} features")
    
    def _get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return [
            # Price momentum (6 features)
            'spy_change_1m',
            'spy_change_5m', 
            'spy_change_15m',
            'spy_change_today',
            'spy_momentum_acceleration',  # 1m - 5m change
            'spy_trend_strength',  # abs(15m change)
            
            # Time of day (6 features)
            'hour_sin',  # Cyclical encoding
            'hour_cos',
            'time_decay_factor',
            'is_first_hour',  # 9:30-10:30
            'is_last_hour',   # 3:00-4:00
            'is_power_hour',  # 2:00-3:00
            
            # Volatility (6 features)
            'vix_normalized',
            'vix_high',  # VIX > 25
            'call_iv',
            'put_iv',
            'iv_skew',
            'iv_mean',
            
            # Greeks - Call (4 features)
            'call_delta',
            'call_gamma_normalized',
            'call_theta_normalized',
            'call_moneyness',  # How far ITM/OTM
            
            # Greeks - Put (4 features)
            'put_delta',
            'put_gamma_normalized',
            'put_theta_normalized',
            'put_moneyness',
            
            # Market structure (6 features)
            'spy_spread_normalized',
            'call_spread_normalized',
            'put_spread_normalized',
            'call_bid_ask_ratio',
            'put_bid_ask_ratio',
            'liquidity_score',
            
            # Direction indicators (4 features)
            'is_bullish_momentum',
            'is_bearish_momentum',
            'momentum_strength',
            'trend_consistency',  # Are 1m, 5m, 15m all same direction?
            
            # News/context features (6 features) - WHY is market moving
            'news_sentiment',      # -1 bearish to +1 bullish
            'war_tensions',        # 1 if geopolitical conflict news
            'tariff_news',         # 1 if trade war/tariff news
            'fed_hawkish',         # 1 if Fed raising rates
            'fed_dovish',          # 1 if Fed cutting rates
            'recession_fears',     # 1 if recession fears in news
            
            # Volume features (6 features) - HIGH SIGNAL for institutional activity
            'spy_volume_surge',    # 1 if volume > 1.5x average
            'call_volume_ratio',   # call volume / (call + put volume)
            'put_volume_ratio',    # put volume / (call + put volume)  
            'call_oi_ratio',       # open interest ratio for calls
            'volume_momentum',     # is volume increasing?
            'option_flow_imbalance',  # call volume - put volume normalized
            
            # === TECHNICAL INDICATORS (20 features) ===
            # RSI (3 features)
            'rsi_14',              # RSI value 0-100, normalized to 0-1
            'rsi_oversold',        # 1 if RSI < 30
            'rsi_overbought',      # 1 if RSI > 70
            
            # MACD (4 features)
            'macd_histogram',      # MACD histogram normalized
            'macd_bullish_cross',  # 1 if bullish crossover
            'macd_bearish_cross',  # 1 if bearish crossover
            'macd_above_signal',   # 1 if MACD > signal line
            
            # VWAP (3 features)
            'price_vs_vwap',       # % above/below VWAP
            'above_vwap',          # 1 if price > VWAP
            'vwap_deviation',      # Absolute distance from VWAP
            
            # Bollinger Bands (4 features)
            'bb_position',         # 0 = lower band, 0.5 = middle, 1 = upper
            'bb_width',            # Band width (volatility indicator)
            'bb_lower_touch',      # 1 if near lower band
            'bb_upper_touch',      # 1 if near upper band
            
            # Moving Averages (3 features)
            'ma_trend_bullish',    # 1 if price > EMA9 > EMA21
            'ma_trend_bearish',    # 1 if price < EMA9 < EMA21
            'ema_crossover',       # 1 if EMA9 just crossed EMA21
            
            # ATR/Volatility (3 features)
            'atr_percent',         # ATR as % of price
            'high_volatility',     # 1 if ATR% > 0.5%
            'rate_of_change',      # Price rate of change
        ]
    
    def extract_features(self, snapshot_dict: Dict[str, Any]) -> FeatureSet:
        """
        Extract features from a market snapshot
        
        Args:
            snapshot_dict: Dictionary from MarketSnapshot or database row
            
        Returns:
            FeatureSet with normalized features
        """
        features = np.zeros(self.n_features, dtype=np.float32)
        
        # --- Price momentum features ---
        spy_1m = snapshot_dict.get('spy_change_1m', 0)
        spy_5m = snapshot_dict.get('spy_change_5m', 0)
        spy_15m = snapshot_dict.get('spy_change_15m', 0)
        spy_today = snapshot_dict.get('spy_change_today', 0)
        
        features[0] = self._normalize_price_change(spy_1m)
        features[1] = self._normalize_price_change(spy_5m)
        features[2] = self._normalize_price_change(spy_15m)
        features[3] = self._normalize_price_change(spy_today)
        features[4] = self._normalize_price_change(spy_1m - spy_5m)  # Acceleration
        features[5] = min(1.0, abs(spy_15m) / self.PRICE_CHANGE_SCALE)  # Trend strength
        
        # --- Time of day features ---
        hour = snapshot_dict.get('hour', 12)
        minute = snapshot_dict.get('minute', 0)
        minutes_since_open = snapshot_dict.get('minutes_since_open', 0)
        
        # Cyclical encoding for hour
        hour_fraction = (hour + minute / 60) / 24
        features[6] = np.sin(2 * np.pi * hour_fraction)
        features[7] = np.cos(2 * np.pi * hour_fraction)
        features[8] = snapshot_dict.get('time_decay_factor', 0.5)
        features[9] = 1.0 if minutes_since_open <= 60 else 0.0  # First hour
        features[10] = 1.0 if minutes_since_open >= 330 else 0.0  # Last hour
        features[11] = 1.0 if 270 <= minutes_since_open <= 330 else 0.0  # Power hour
        
        # --- Volatility features ---
        vix = snapshot_dict.get('vix', 20)
        call_iv = snapshot_dict.get('call_iv', 0.3)
        put_iv = snapshot_dict.get('put_iv', 0.3)
        
        features[12] = (vix - self.VIX_MEAN) / self.VIX_SCALE
        features[13] = 1.0 if vix > 25 else 0.0
        features[14] = call_iv / self.IV_SCALE
        features[15] = put_iv / self.IV_SCALE
        features[16] = (put_iv - call_iv) / self.IV_SCALE  # IV skew
        features[17] = (call_iv + put_iv) / 2 / self.IV_SCALE  # Mean IV
        
        # --- Call Greeks ---
        call_delta = snapshot_dict.get('call_delta', 0.5)
        call_gamma = snapshot_dict.get('call_gamma', 0.1)
        call_theta = snapshot_dict.get('call_theta', -0.1)
        call_strike = snapshot_dict.get('call_strike', 0)
        spy_price = snapshot_dict.get('spy_price', call_strike)
        
        features[18] = call_delta
        features[19] = min(1.0, call_gamma * 10)  # Gamma normalized
        features[20] = max(-1.0, call_theta / 0.5)  # Theta normalized
        features[21] = self._calculate_moneyness(spy_price, call_strike, 'call')
        
        # --- Put Greeks ---
        put_delta = snapshot_dict.get('put_delta', -0.5)
        put_gamma = snapshot_dict.get('put_gamma', 0.1)
        put_theta = snapshot_dict.get('put_theta', -0.1)
        put_strike = snapshot_dict.get('put_strike', 0)
        
        features[22] = put_delta  # Already negative
        features[23] = min(1.0, put_gamma * 10)
        features[24] = max(-1.0, put_theta / 0.5)
        features[25] = self._calculate_moneyness(spy_price, put_strike, 'put')
        
        # --- Market structure ---
        spy_spread = snapshot_dict.get('spy_spread', 0.01)
        call_spread = snapshot_dict.get('call_spread', 0.05)
        put_spread = snapshot_dict.get('put_spread', 0.05)
        
        call_bid = snapshot_dict.get('call_bid', 1)
        call_ask = snapshot_dict.get('call_ask', 1)
        put_bid = snapshot_dict.get('put_bid', 1)
        put_ask = snapshot_dict.get('put_ask', 1)
        
        features[26] = min(1.0, spy_spread / self.SPREAD_SCALE)
        features[27] = min(1.0, call_spread / self.SPREAD_SCALE)
        features[28] = min(1.0, put_spread / self.SPREAD_SCALE)
        features[29] = call_bid / call_ask if call_ask > 0 else 0.9
        features[30] = put_bid / put_ask if put_ask > 0 else 0.9
        
        # Liquidity score: tighter spreads = higher score
        avg_spread = (spy_spread + call_spread + put_spread) / 3
        features[31] = 1.0 - min(1.0, avg_spread / self.SPREAD_SCALE)
        
        # --- Direction indicators ---
        features[32] = 1.0 if spy_5m > 0.1 else 0.0  # Bullish momentum
        features[33] = 1.0 if spy_5m < -0.1 else 0.0  # Bearish momentum
        features[34] = min(1.0, abs(spy_5m) / 0.5)  # Momentum strength
        
        # Trend consistency: all 3 timeframes same direction
        same_direction = (
            (spy_1m > 0 and spy_5m > 0 and spy_15m > 0) or
            (spy_1m < 0 and spy_5m < 0 and spy_15m < 0)
        )
        features[35] = 1.0 if same_direction else 0.0
        
        # --- News/Context features (WHY is market moving) ---
        features[36] = np.clip(snapshot_dict.get('news_sentiment', 0.0), -1.0, 1.0)
        features[37] = 1.0 if snapshot_dict.get('war_tensions', 0) else 0.0
        features[38] = 1.0 if snapshot_dict.get('tariff_news', 0) else 0.0
        features[39] = 1.0 if snapshot_dict.get('fed_hawkish', 0) else 0.0
        features[40] = 1.0 if snapshot_dict.get('fed_dovish', 0) else 0.0
        features[41] = 1.0 if snapshot_dict.get('recession_fears', 0) else 0.0
        
        # --- Volume features (HIGH SIGNAL for institutional activity) ---
        spy_volume = snapshot_dict.get('spy_volume', 0)
        call_volume = snapshot_dict.get('call_volume', 0)
        put_volume = snapshot_dict.get('put_volume', 0)
        call_oi = snapshot_dict.get('call_open_interest', 0)
        put_oi = snapshot_dict.get('put_open_interest', 0)
        prev_spy_volume = snapshot_dict.get('prev_spy_volume', spy_volume)  # For momentum
        
        # Volume surge: is current volume significantly above average?
        # Use 1M average SPY volume (~50M/day = ~130K/min in first hour)
        avg_minute_volume = 130000  # Approximate average minute volume for SPY
        features[42] = 1.0 if spy_volume > avg_minute_volume * 1.5 else 0.0
        
        # Call/Put volume ratios (sentiment indicator)
        total_option_volume = call_volume + put_volume
        if total_option_volume > 0:
            features[43] = call_volume / total_option_volume  # Call ratio
            features[44] = put_volume / total_option_volume   # Put ratio
        else:
            features[43] = 0.5
            features[44] = 0.5
        
        # Open interest ratio (positioning indicator)
        total_oi = call_oi + put_oi
        features[45] = call_oi / total_oi if total_oi > 0 else 0.5
        
        # Volume momentum (is volume accelerating?)
        if prev_spy_volume > 0:
            vol_change = (spy_volume - prev_spy_volume) / prev_spy_volume
            features[46] = np.clip(vol_change, -1.0, 1.0)
        else:
            features[46] = 0.0
        
        # Option flow imbalance: bullish if more call volume, bearish if more put
        if total_option_volume > 0:
            imbalance = (call_volume - put_volume) / total_option_volume
            features[47] = np.clip(imbalance, -1.0, 1.0)
        else:
            features[47] = 0.0
        
        # === TECHNICAL INDICATORS (features 48-67) ===
        
        # RSI (3 features: 48-50)
        rsi = snapshot_dict.get('rsi_14', 50.0)
        features[48] = rsi / 100.0  # Normalize to 0-1
        features[49] = 1.0 if rsi < 30 else 0.0  # Oversold
        features[50] = 1.0 if rsi > 70 else 0.0  # Overbought
        
        # MACD (4 features: 51-54)
        macd_hist = snapshot_dict.get('macd_histogram', 0.0)
        macd_line = snapshot_dict.get('macd_line', 0.0)
        macd_signal = snapshot_dict.get('macd_signal', 0.0)
        macd_crossover = snapshot_dict.get('macd_crossover', 'NONE')
        
        features[51] = np.clip(macd_hist * 10, -1.0, 1.0)  # Normalize histogram
        features[52] = 1.0 if macd_crossover == 'BULLISH' else 0.0
        features[53] = 1.0 if macd_crossover == 'BEARISH' else 0.0
        features[54] = 1.0 if macd_line > macd_signal else 0.0
        
        # VWAP (3 features: 55-57)
        price_vs_vwap = snapshot_dict.get('price_vs_vwap', 0.0)
        vwap_signal = snapshot_dict.get('vwap_signal', 'NEUTRAL')
        
        features[55] = np.clip(price_vs_vwap / 1.0, -1.0, 1.0)  # % normalized
        features[56] = 1.0 if vwap_signal == 'ABOVE' else 0.0
        features[57] = min(1.0, abs(price_vs_vwap) / 1.0)  # Deviation magnitude
        
        # Bollinger Bands (4 features: 58-61)
        bb_position = snapshot_dict.get('bb_position', 0.5)
        bb_width = snapshot_dict.get('bb_width', 2.0)
        
        features[58] = bb_position  # Already 0-1
        features[59] = min(1.0, bb_width / 5.0)  # Normalize width
        features[60] = 1.0 if bb_position < 0.1 else 0.0  # Near lower band
        features[61] = 1.0 if bb_position > 0.9 else 0.0  # Near upper band
        
        # Moving Averages (3 features: 62-64)
        ma_trend = snapshot_dict.get('ma_trend', 'NEUTRAL')
        ema_9 = snapshot_dict.get('ema_9', 0.0)
        ema_21 = snapshot_dict.get('ema_21', 0.0)
        
        features[62] = 1.0 if ma_trend == 'BULLISH' else 0.0
        features[63] = 1.0 if ma_trend == 'BEARISH' else 0.0
        # EMA crossover detection (approximate - if EMA9 just crossed EMA21)
        features[64] = 1.0 if abs(ema_9 - ema_21) < 0.5 else 0.0
        
        # ATR/Volatility (3 features: 65-67)
        atr_percent = snapshot_dict.get('atr_percent', 0.0)
        rate_of_change = snapshot_dict.get('rate_of_change', 0.0)
        
        features[65] = min(1.0, atr_percent / 2.0)  # Normalize
        features[66] = 1.0 if atr_percent > 0.5 else 0.0  # High volatility flag
        features[67] = np.clip(rate_of_change / 2.0, -1.0, 1.0)  # ROC normalized
        
        # Get timestamp
        ts = snapshot_dict.get('timestamp', datetime.now())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        
        return FeatureSet(
            features=features,
            feature_names=self.feature_names,
            timestamp=ts,
            raw_data=snapshot_dict,
        )
    
    def _normalize_price_change(self, change: float) -> float:
        """Normalize price change to roughly -1 to 1"""
        return np.clip(change / self.PRICE_CHANGE_SCALE, -1.0, 1.0)
    
    def _calculate_moneyness(self, spot: float, strike: float, option_type: str) -> float:
        """
        Calculate moneyness: -1 = deep OTM, 0 = ATM, +1 = deep ITM
        """
        if strike == 0 or spot == 0:
            return 0.0
        
        # Moneyness as percentage from ATM
        if option_type == 'call':
            moneyness = (spot - strike) / strike * 100
        else:
            moneyness = (strike - spot) / strike * 100
        
        # Normalize to -1 to 1 range (±2% is the scale)
        return np.clip(moneyness / 2, -1.0, 1.0)
    
    def extract_features_batch(self, snapshots: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[datetime]]:
        """
        Extract features from multiple snapshots
        
        Args:
            snapshots: List of snapshot dictionaries
            
        Returns:
            Tuple of (feature_matrix, timestamps)
        """
        n_samples = len(snapshots)
        X = np.zeros((n_samples, self.n_features), dtype=np.float32)
        timestamps = []
        
        for i, snapshot in enumerate(snapshots):
            fs = self.extract_features(snapshot)
            X[i] = fs.features
            timestamps.append(fs.timestamp)
        
        return X, timestamps
    
    def create_training_features(self, trade_records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix and labels from trade records
        
        Args:
            trade_records: List of completed trade records
            
        Returns:
            Tuple of (X features, y labels where 1=profitable, 0=not)
        """
        n_samples = len(trade_records)
        X = np.zeros((n_samples, self.n_features), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int32)
        
        for i, trade in enumerate(trade_records):
            # Build snapshot-like dict from trade record
            snapshot_dict = {
                'spy_price': trade.get('entry_spy_price', 0),
                'spy_change_5m': trade.get('entry_spy_change_5m', 0),
                'spy_change_15m': trade.get('entry_spy_change_15m', 0),
                'spy_change_1m': trade.get('entry_spy_change_5m', 0) / 5,  # Approximate
                'spy_change_today': 0,
                'hour': trade.get('entry_hour', 12),
                'minute': 0,
                'minutes_since_open': trade.get('entry_minutes_since_open', 0),
                'time_decay_factor': trade.get('entry_time_decay_factor', 0.5),
                'vix': trade.get('entry_vix', 20),
                'call_iv': trade.get('entry_call_iv', 0.3),
                'put_iv': trade.get('entry_put_iv', 0.3),
                'call_delta': trade.get('entry_call_delta', 0.5),
                'call_strike': trade.get('strike', 0),
                'timestamp': trade.get('entry_time', datetime.now()),
            }
            
            fs = self.extract_features(snapshot_dict)
            X[i] = fs.features
            y[i] = 1 if trade.get('was_profitable', 0) else 0
        
        logger.info(f"Created training data: {n_samples} samples, {self.n_features} features")
        logger.info(f"Class distribution: {np.sum(y)} profitable, {len(y) - np.sum(y)} unprofitable")
        
        return X, y
    
    def get_feature_importance_names(self, importances: np.ndarray) -> List[Tuple[str, float]]:
        """
        Get feature names sorted by importance
        
        Args:
            importances: Array of feature importances from model
            
        Returns:
            List of (feature_name, importance) tuples, sorted descending
        """
        pairs = list(zip(self.feature_names, importances))
        return sorted(pairs, key=lambda x: x[1], reverse=True)
