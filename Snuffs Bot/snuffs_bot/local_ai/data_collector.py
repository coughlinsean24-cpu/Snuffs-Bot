"""
Market Data Collector

Records all market data continuously for training the local AI.
Captures SPY price, option prices, Greeks, indicators, and timestamps.
Understands that 0DTE options behave differently throughout the day.
Includes technical indicators: RSI, MACD, VWAP, Bollinger Bands, EMA/SMA.
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from loguru import logger

# Import technical indicators
try:
    from .technical_indicators import TechnicalAnalyzer, TechnicalIndicators
except ImportError:
    from technical_indicators import TechnicalAnalyzer, TechnicalIndicators


@dataclass
class MarketSnapshot:
    """A single point-in-time capture of all market data"""
    timestamp: datetime
    
    # SPY Data
    spy_price: float
    spy_bid: float
    spy_ask: float
    spy_volume: int = 0
    
    # SPY Momentum/Trend Indicators
    spy_change_1m: float = 0.0   # 1-minute price change %
    spy_change_5m: float = 0.0   # 5-minute price change %
    spy_change_15m: float = 0.0  # 15-minute price change %
    spy_change_today: float = 0.0  # Change from open
    spy_high_today: float = 0.0
    spy_low_today: float = 0.0
    spy_open_today: float = 0.0
    
    # VIX Data
    vix: float = 0.0
    vix_change: float = 0.0
    
    # Time-of-day features (critical for 0DTE)
    hour: int = 0
    minute: int = 0
    minutes_since_open: int = 0  # 9:30 AM = 0
    minutes_until_close: int = 0  # 4:00 PM = 0
    time_decay_factor: float = 0.0  # 0.0 at open, 1.0 at close
    
    # Option Data (ATM Call)
    call_strike: float = 0.0
    call_price: float = 0.0
    call_bid: float = 0.0
    call_ask: float = 0.0
    call_delta: float = 0.0
    call_gamma: float = 0.0
    call_theta: float = 0.0
    call_vega: float = 0.0
    call_iv: float = 0.0
    call_volume: int = 0
    call_open_interest: int = 0
    
    # Option Data (ATM Put)
    put_strike: float = 0.0
    put_price: float = 0.0
    put_bid: float = 0.0
    put_ask: float = 0.0
    put_delta: float = 0.0
    put_gamma: float = 0.0
    put_theta: float = 0.0
    put_vega: float = 0.0
    put_iv: float = 0.0
    put_volume: int = 0
    put_open_interest: int = 0
    
    # Spread/Liquidity indicators
    spy_spread: float = 0.0  # ask - bid
    call_spread: float = 0.0
    put_spread: float = 0.0
    
    # IV Skew
    iv_skew: float = 0.0  # put_iv - call_iv
    
    # ========== TECHNICAL INDICATORS ==========
    # RSI
    rsi_14: float = 50.0        # 14-period RSI
    rsi_signal: str = "NEUTRAL"  # OVERBOUGHT, OVERSOLD, NEUTRAL
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = "NONE"  # BULLISH, BEARISH, NONE
    
    # VWAP
    vwap: float = 0.0
    price_vs_vwap: float = 0.0   # % above/below VWAP
    vwap_signal: str = "NEUTRAL"  # ABOVE, BELOW, AT
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0       # 20-period SMA
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.5     # 0 = at lower, 0.5 = middle, 1 = at upper
    
    # Moving Averages
    ema_9: float = 0.0
    ema_21: float = 0.0
    sma_50: float = 0.0
    ma_trend: str = "NEUTRAL"    # BULLISH, BEARISH, NEUTRAL
    
    # Volatility
    atr_14: float = 0.0          # Average True Range
    atr_percent: float = 0.0     # ATR as % of price
    
    # Momentum
    momentum_10: float = 0.0     # 10-bar momentum
    rate_of_change: float = 0.0  # ROC %
    
    # Composite Signal from Technical Analysis
    tech_signal: str = "HOLD"    # BUY, SELL, HOLD
    tech_signal_strength: float = 0.0  # 0-1
    # ==========================================
    
    # Market Events (critical for AI learning)
    fed_speaking: int = 0       # 1 if Fed chair/officials speaking
    fomc_day: int = 0           # 1 if FOMC rate decision day
    rate_decision: int = 0      # 1 if rate decision announced today
    earnings_major: int = 0     # 1 if major earnings (AAPL, MSFT, etc)
    economic_data: int = 0      # 1 if major economic data (CPI, jobs, etc)
    event_notes: str = ""       # Free-form notes about current events
    
    # News-based context (from NewsCollector)
    news_sentiment: float = 0.0    # -1.0 (bearish) to +1.0 (bullish)
    war_tensions: int = 0          # 1 if war/military conflict news
    tariff_news: int = 0           # 1 if trade war/tariff news
    fed_hawkish: int = 0           # 1 if Fed raising rates/hawkish tone
    fed_dovish: int = 0            # 1 if Fed cutting rates/dovish tone
    recession_fears: int = 0       # 1 if recession fears in news
    context_summary: str = ""      # Brief summary of why market is moving
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketSnapshot':
        """Create from dictionary"""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class TradeRecord:
    """Record of a completed trade for training"""
    trade_id: str
    
    # Entry conditions
    entry_time: datetime
    entry_snapshot: MarketSnapshot
    strategy: str  # LONG_CALL, LONG_PUT
    strike: float
    entry_price: float
    contracts: int
    
    # Exit conditions
    exit_time: Optional[datetime] = None
    exit_snapshot: Optional[MarketSnapshot] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    
    # Outcome
    pnl: float = 0.0
    pnl_percent: float = 0.0
    hold_duration_seconds: int = 0
    max_profit_reached: float = 0.0
    max_loss_reached: float = 0.0
    
    # Labels for training
    was_profitable: bool = False
    optimal_exit_time: Optional[datetime] = None  # When max profit was reached


class DataCollector:
    """
    Collects and stores market data for AI training.
    
    Uses SQLite for persistent storage of:
    - Market snapshots (every few seconds during market hours)
    - Trade records with full entry/exit context
    - Price history for indicator calculation
    - Technical indicators (RSI, MACD, VWAP, Bollinger Bands)
    """
    
    def __init__(self, data_dir: str = "data/local_ai"):
        """
        Initialize the data collector
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "market_data.db"
        self._init_database()
        
        # Price history for calculating momentum
        self.price_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000  # Keep last 1000 data points
        
        # Today's OHLC
        self.today_open: Optional[float] = None
        self.today_high: float = 0.0
        self.today_low: float = float('inf')
        self.last_date: Optional[str] = None
        
        # Technical Analyzer for RSI, MACD, VWAP, Bollinger Bands, etc.
        self.technical_analyzer = TechnicalAnalyzer(max_history=200)
        
        # Prime price history from recent database entries
        self._load_recent_price_history()
        
        logger.info(f"DataCollector initialized with technical indicators, storing data in {self.db_path}")
    
    def _load_recent_price_history(self):
        """Load recent price history from database for momentum calculations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get last 20 minutes of data
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(minutes=20)).isoformat()
            
            cursor.execute("""
                SELECT timestamp, spy_price
                FROM market_snapshots
                WHERE timestamp > ? AND spy_price > 0
                ORDER BY timestamp DESC
                LIMIT 100
            """, (cutoff,))
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                timestamp_str, price = row
                if price and price > 0:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        self.price_history.append({
                            'timestamp': timestamp,
                            'price': float(price)
                        })
                    except:
                        pass
            
            # Reverse to chronological order
            self.price_history.reverse()
            
            # Also initialize the TechnicalAnalyzer with historical data
            for point in self.price_history:
                self.technical_analyzer.update(
                    price=point['price'],
                    high=point['price'],
                    low=point['price'],
                    volume=100000,  # Default volume
                    timestamp=point['timestamp']
                )
            
            if self.price_history:
                logger.info(f"Loaded {len(self.price_history)} recent price points for momentum calculation")
                logger.info(f"Technical indicators initialized with {len(self.price_history)} historical points")
        except Exception as e:
            logger.debug(f"Could not load price history: {e}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL,
                
                -- SPY Data
                spy_price REAL,
                spy_bid REAL,
                spy_ask REAL,
                spy_volume INTEGER,
                spy_change_1m REAL,
                spy_change_5m REAL,
                spy_change_15m REAL,
                spy_change_today REAL,
                spy_high_today REAL,
                spy_low_today REAL,
                spy_open_today REAL,
                
                -- VIX
                vix REAL,
                vix_change REAL,
                
                -- Time features
                minutes_since_open INTEGER,
                minutes_until_close INTEGER,
                time_decay_factor REAL,
                
                -- Call option
                call_strike REAL,
                call_price REAL,
                call_bid REAL,
                call_ask REAL,
                call_delta REAL,
                call_gamma REAL,
                call_theta REAL,
                call_vega REAL,
                call_iv REAL,
                call_volume INTEGER,
                call_open_interest INTEGER,
                
                -- Put option
                put_strike REAL,
                put_price REAL,
                put_bid REAL,
                put_ask REAL,
                put_delta REAL,
                put_gamma REAL,
                put_theta REAL,
                put_vega REAL,
                put_iv REAL,
                put_volume INTEGER,
                put_open_interest INTEGER,
                
                -- Spreads
                spy_spread REAL,
                call_spread REAL,
                put_spread REAL,
                iv_skew REAL,
                
                -- Technical Indicators
                rsi_14 REAL DEFAULT 50,
                rsi_signal TEXT DEFAULT 'NEUTRAL',
                macd_line REAL DEFAULT 0,
                macd_signal REAL DEFAULT 0,
                macd_histogram REAL DEFAULT 0,
                macd_crossover TEXT DEFAULT 'NONE',
                vwap REAL DEFAULT 0,
                price_vs_vwap REAL DEFAULT 0,
                vwap_signal TEXT DEFAULT 'NEUTRAL',
                bb_upper REAL DEFAULT 0,
                bb_middle REAL DEFAULT 0,
                bb_lower REAL DEFAULT 0,
                bb_width REAL DEFAULT 0,
                bb_position REAL DEFAULT 0.5,
                ema_9 REAL DEFAULT 0,
                ema_21 REAL DEFAULT 0,
                sma_50 REAL DEFAULT 0,
                ma_trend TEXT DEFAULT 'NEUTRAL',
                atr_14 REAL DEFAULT 0,
                atr_percent REAL DEFAULT 0,
                momentum_10 REAL DEFAULT 0,
                rate_of_change REAL DEFAULT 0,
                tech_signal TEXT DEFAULT 'HOLD',
                tech_signal_strength REAL DEFAULT 0,
                
                -- News/Context (WHY market is moving)
                news_sentiment REAL DEFAULT 0,
                war_tensions INTEGER DEFAULT 0,
                tariff_news INTEGER DEFAULT 0,
                fed_hawkish INTEGER DEFAULT 0,
                fed_dovish INTEGER DEFAULT 0,
                recession_fears INTEGER DEFAULT 0,
                context_summary TEXT DEFAULT '',
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on timestamp and date for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp 
            ON market_snapshots(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_date 
            ON market_snapshots(date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_hour 
            ON market_snapshots(hour)
        """)
        
        # Trade records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                
                -- Entry
                entry_time TEXT NOT NULL,
                entry_snapshot_id INTEGER,
                strategy TEXT NOT NULL,
                strike REAL,
                entry_price REAL,
                contracts INTEGER,
                
                -- Exit
                exit_time TEXT,
                exit_snapshot_id INTEGER,
                exit_price REAL,
                exit_reason TEXT,
                
                -- Outcome
                pnl REAL,
                pnl_percent REAL,
                hold_duration_seconds INTEGER,
                max_profit_reached REAL,
                max_loss_reached REAL,
                was_profitable INTEGER,
                
                -- Entry conditions (denormalized for easy training)
                entry_spy_price REAL,
                entry_spy_change_5m REAL,
                entry_spy_change_15m REAL,
                entry_vix REAL,
                entry_hour INTEGER,
                entry_minutes_since_open INTEGER,
                entry_call_iv REAL,
                entry_put_iv REAL,
                entry_call_delta REAL,
                entry_time_decay_factor REAL,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (entry_snapshot_id) REFERENCES market_snapshots(id),
                FOREIGN KEY (exit_snapshot_id) REFERENCES market_snapshots(id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_strategy 
            ON trade_records(strategy)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_profitable 
            ON trade_records(was_profitable)
        """)
        
        conn.commit()
        conn.close()
        
        logger.debug("Database initialized successfully")
    
    def record_snapshot(self, snapshot: MarketSnapshot) -> int:
        """
        Record a market snapshot to the database
        
        Args:
            snapshot: The market snapshot to record
            
        Returns:
            The ID of the inserted record
        """
        # Update technical indicators before recording
        self._update_technical_indicators(snapshot)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_str = snapshot.timestamp.strftime("%Y-%m-%d")
        
        cursor.execute("""
            INSERT INTO market_snapshots (
                timestamp, date, hour, minute,
                spy_price, spy_bid, spy_ask, spy_volume,
                spy_change_1m, spy_change_5m, spy_change_15m, spy_change_today,
                spy_high_today, spy_low_today, spy_open_today,
                vix, vix_change,
                minutes_since_open, minutes_until_close, time_decay_factor,
                call_strike, call_price, call_bid, call_ask,
                call_delta, call_gamma, call_theta, call_vega, call_iv,
                call_volume, call_open_interest,
                put_strike, put_price, put_bid, put_ask,
                put_delta, put_gamma, put_theta, put_vega, put_iv,
                put_volume, put_open_interest,
                spy_spread, call_spread, put_spread, iv_skew,
                rsi_14, rsi_signal, macd_line, macd_signal, macd_histogram, macd_crossover,
                vwap, price_vs_vwap, vwap_signal,
                bb_upper, bb_middle, bb_lower, bb_width, bb_position,
                ema_9, ema_21, sma_50, ma_trend,
                atr_14, atr_percent, momentum_10, rate_of_change,
                tech_signal, tech_signal_strength,
                fed_speaking, fomc_day, rate_decision, earnings_major, economic_data, event_notes,
                news_sentiment, war_tensions, tariff_news, fed_hawkish, fed_dovish, recession_fears, context_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.timestamp.isoformat(),
            date_str,
            snapshot.hour,
            snapshot.minute,
            snapshot.spy_price,
            snapshot.spy_bid,
            snapshot.spy_ask,
            snapshot.spy_volume,
            snapshot.spy_change_1m,
            snapshot.spy_change_5m,
            snapshot.spy_change_15m,
            snapshot.spy_change_today,
            snapshot.spy_high_today,
            snapshot.spy_low_today,
            snapshot.spy_open_today,
            snapshot.vix,
            snapshot.vix_change,
            snapshot.minutes_since_open,
            snapshot.minutes_until_close,
            snapshot.time_decay_factor,
            snapshot.call_strike,
            snapshot.call_price,
            snapshot.call_bid,
            snapshot.call_ask,
            snapshot.call_delta,
            snapshot.call_gamma,
            snapshot.call_theta,
            snapshot.call_vega,
            snapshot.call_iv,
            snapshot.call_volume,
            snapshot.call_open_interest,
            snapshot.put_strike,
            snapshot.put_price,
            snapshot.put_bid,
            snapshot.put_ask,
            snapshot.put_delta,
            snapshot.put_gamma,
            snapshot.put_theta,
            snapshot.put_vega,
            snapshot.put_iv,
            snapshot.put_volume,
            snapshot.put_open_interest,
            snapshot.spy_spread,
            snapshot.call_spread,
            snapshot.put_spread,
            snapshot.iv_skew,
            # Technical Indicators
            snapshot.rsi_14,
            snapshot.rsi_signal,
            snapshot.macd_line,
            snapshot.macd_signal,
            snapshot.macd_histogram,
            snapshot.macd_crossover,
            snapshot.vwap,
            snapshot.price_vs_vwap,
            snapshot.vwap_signal,
            snapshot.bb_upper,
            snapshot.bb_middle,
            snapshot.bb_lower,
            snapshot.bb_width,
            snapshot.bb_position,
            snapshot.ema_9,
            snapshot.ema_21,
            snapshot.sma_50,
            snapshot.ma_trend,
            snapshot.atr_14,
            snapshot.atr_percent,
            snapshot.momentum_10,
            snapshot.rate_of_change,
            snapshot.tech_signal,
            snapshot.tech_signal_strength,
            # Event flags
            snapshot.fed_speaking,
            snapshot.fomc_day,
            snapshot.rate_decision,
            snapshot.earnings_major,
            snapshot.economic_data,
            snapshot.event_notes,
            # News context fields
            snapshot.news_sentiment,
            snapshot.war_tensions,
            snapshot.tariff_news,
            snapshot.fed_hawkish,
            snapshot.fed_dovish,
            snapshot.recession_fears,
            snapshot.context_summary[:500] if snapshot.context_summary else "",
        ))
        
        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update price history for momentum calculations
        self._update_price_history(snapshot)
        
        return snapshot_id
    
    def _update_technical_indicators(self, snapshot: MarketSnapshot):
        """Calculate and update technical indicators on the snapshot"""
        if snapshot.spy_price <= 0:
            return
        
        try:
            # Update the technical analyzer with new price data
            # Use SPY high/low from today if available, else just price
            high = snapshot.spy_high_today if snapshot.spy_high_today > 0 else snapshot.spy_price
            low = snapshot.spy_low_today if snapshot.spy_low_today > 0 else snapshot.spy_price
            
            indicators = self.technical_analyzer.update(
                price=snapshot.spy_price,
                high=high,
                low=low,
                volume=snapshot.spy_volume,
                timestamp=snapshot.timestamp
            )
            
            # Copy indicator values to snapshot
            snapshot.rsi_14 = indicators.rsi_14
            snapshot.rsi_signal = indicators.rsi_signal
            snapshot.macd_line = indicators.macd_line
            snapshot.macd_signal = indicators.macd_signal
            snapshot.macd_histogram = indicators.macd_histogram
            snapshot.macd_crossover = indicators.macd_crossover
            snapshot.vwap = indicators.vwap
            snapshot.price_vs_vwap = indicators.price_vs_vwap
            snapshot.vwap_signal = indicators.vwap_signal
            snapshot.bb_upper = indicators.bb_upper
            snapshot.bb_middle = indicators.bb_middle
            snapshot.bb_lower = indicators.bb_lower
            snapshot.bb_width = indicators.bb_width
            snapshot.bb_position = indicators.bb_position
            snapshot.ema_9 = indicators.ema_9
            snapshot.ema_21 = indicators.ema_21
            snapshot.sma_50 = indicators.sma_50
            snapshot.ma_trend = indicators.ma_trend
            snapshot.atr_14 = indicators.atr_14
            snapshot.atr_percent = indicators.atr_percent
            snapshot.momentum_10 = indicators.momentum_10
            snapshot.rate_of_change = indicators.rate_of_change
            snapshot.tech_signal = indicators.overall_signal
            snapshot.tech_signal_strength = indicators.signal_strength
            
            logger.debug(f"Technical indicators updated: RSI={indicators.rsi_14:.1f}, "
                        f"MACD={indicators.macd_histogram:.4f}, VWAP={indicators.vwap:.2f}, "
                        f"Signal={indicators.overall_signal}")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
    
    def _update_price_history(self, snapshot: MarketSnapshot):
        """Update in-memory price history"""
        self.price_history.append({
            'timestamp': snapshot.timestamp,
            'price': snapshot.spy_price,
            'vix': snapshot.vix,
        })
        
        # Trim to max size
        if len(self.price_history) > self.max_history_size:
            self.price_history = self.price_history[-self.max_history_size:]
        
        # Update today's OHLC
        today = snapshot.timestamp.strftime("%Y-%m-%d")
        if self.last_date != today:
            self.today_open = snapshot.spy_price
            self.today_high = snapshot.spy_price
            self.today_low = snapshot.spy_price
            self.last_date = today
        else:
            if snapshot.spy_price > self.today_high:
                self.today_high = snapshot.spy_price
            if snapshot.spy_price < self.today_low:
                self.today_low = snapshot.spy_price
    
    def update_live_price(self, spy_price: float, vix: float = 0.0):
        """Update price history with live price (for momentum calculation without saving to DB)"""
        if spy_price <= 0:
            return
        
        now = datetime.now()
        self.price_history.append({
            'timestamp': now,
            'price': spy_price,
            'vix': vix,
        })
        
        # Trim to max size
        if len(self.price_history) > self.max_history_size:
            self.price_history = self.price_history[-self.max_history_size:]
    
    def calculate_momentum(self, current_price: float, minutes_ago: int) -> float:
        """Calculate price change from N minutes ago"""
        if not self.price_history:
            return 0.0
        
        target_time = datetime.now() - timedelta(minutes=minutes_ago)
        
        # Find closest price to target time
        closest = None
        min_diff = float('inf')
        
        for entry in self.price_history:
            diff = abs((entry['timestamp'] - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest = entry
        
        if closest and closest['price'] > 0:
            return ((current_price / closest['price']) - 1) * 100
        
        return 0.0
    
    def build_snapshot_from_live_data(
        self,
        spy_data: Dict[str, Any],
        vix: float,
        call_option: Dict[str, Any],
        put_option: Dict[str, Any],
    ) -> MarketSnapshot:
        """
        Build a MarketSnapshot from live Tastytrade data
        
        Args:
            spy_data: SPY quote data from Tastytrade
            vix: Current VIX value
            call_option: ATM call option data with Greeks
            put_option: ATM put option data with Greeks
            
        Returns:
            Complete MarketSnapshot
        """
        now = datetime.now()
        
        # Calculate time features
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        minutes_since_open = max(0, int((now - market_open).total_seconds() / 60))
        minutes_until_close = max(0, int((market_close - now).total_seconds() / 60))
        
        # Time decay factor: 0.0 at open, 1.0 at close
        total_minutes = 390  # 6.5 hours
        time_decay_factor = min(1.0, minutes_since_open / total_minutes)
        
        # Extract SPY data
        spy_price = float(spy_data.get('mark', spy_data.get('last', 0)))
        spy_bid = float(spy_data.get('bid', spy_price))
        spy_ask = float(spy_data.get('ask', spy_price))
        
        # Calculate momentum
        spy_change_1m = self.calculate_momentum(spy_price, 1)
        spy_change_5m = self.calculate_momentum(spy_price, 5)
        spy_change_15m = self.calculate_momentum(spy_price, 15)
        
        # Today's change
        if self.today_open and self.today_open > 0:
            spy_change_today = ((spy_price / self.today_open) - 1) * 100
        else:
            spy_change_today = 0.0
        
        # Extract call option data
        call_price = float(call_option.get('mark', call_option.get('last', 0)))
        call_bid = float(call_option.get('bid', 0))
        call_ask = float(call_option.get('ask', 0))
        
        # Greeks (may need adjustment based on Tastytrade response format)
        call_delta = float(call_option.get('delta', 0.5))
        call_gamma = float(call_option.get('gamma', 0))
        call_theta = float(call_option.get('theta', 0))
        call_vega = float(call_option.get('vega', 0))
        call_iv = float(call_option.get('implied_volatility', call_option.get('iv', 0)))
        
        # Extract put option data
        put_price = float(put_option.get('mark', put_option.get('last', 0)))
        put_bid = float(put_option.get('bid', 0))
        put_ask = float(put_option.get('ask', 0))
        
        put_delta = float(put_option.get('delta', -0.5))
        put_gamma = float(put_option.get('gamma', 0))
        put_theta = float(put_option.get('theta', 0))
        put_vega = float(put_option.get('vega', 0))
        put_iv = float(put_option.get('implied_volatility', put_option.get('iv', 0)))
        
        return MarketSnapshot(
            timestamp=now,
            
            # SPY
            spy_price=spy_price,
            spy_bid=spy_bid,
            spy_ask=spy_ask,
            spy_volume=int(spy_data.get('volume', 0)),
            spy_change_1m=spy_change_1m,
            spy_change_5m=spy_change_5m,
            spy_change_15m=spy_change_15m,
            spy_change_today=spy_change_today,
            spy_high_today=self.today_high,
            spy_low_today=self.today_low if self.today_low != float('inf') else spy_price,
            spy_open_today=self.today_open or spy_price,
            
            # VIX
            vix=vix,
            vix_change=0.0,  # Would need VIX history
            
            # Time
            hour=now.hour,
            minute=now.minute,
            minutes_since_open=minutes_since_open,
            minutes_until_close=minutes_until_close,
            time_decay_factor=time_decay_factor,
            
            # Call
            call_strike=float(call_option.get('strike', 0)),
            call_price=call_price,
            call_bid=call_bid,
            call_ask=call_ask,
            call_delta=call_delta,
            call_gamma=call_gamma,
            call_theta=call_theta,
            call_vega=call_vega,
            call_iv=call_iv,
            call_volume=int(call_option.get('volume', 0)),
            call_open_interest=int(call_option.get('open_interest', 0)),
            
            # Put
            put_strike=float(put_option.get('strike', 0)),
            put_price=put_price,
            put_bid=put_bid,
            put_ask=put_ask,
            put_delta=put_delta,
            put_gamma=put_gamma,
            put_theta=put_theta,
            put_vega=put_vega,
            put_iv=put_iv,
            put_volume=int(put_option.get('volume', 0)),
            put_open_interest=int(put_option.get('open_interest', 0)),
            
            # Spreads
            spy_spread=spy_ask - spy_bid,
            call_spread=call_ask - call_bid,
            put_spread=put_ask - put_bid,
            iv_skew=put_iv - call_iv,
        )
        
        # Update the snapshot with current technical indicators
        # This ensures the snapshot has the latest RSI, MACD, etc.
        self._update_technical_indicators(snapshot)
        
        return snapshot
    
    def record_trade_entry(
        self,
        trade_id: str,
        snapshot: MarketSnapshot,
        strategy: str,
        strike: float,
        entry_price: float,
        contracts: int,
        is_local_ai: bool = False
    ) -> None:
        """Record a trade entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First record the snapshot
        snapshot_id = self.record_snapshot(snapshot)
        
        cursor.execute("""
            INSERT INTO trade_records (
                trade_id, entry_time, entry_snapshot_id, strategy,
                strike, entry_price, contracts, is_local_ai,
                entry_spy_price, entry_spy_change_5m, entry_spy_change_15m,
                entry_vix, entry_hour, entry_minutes_since_open,
                entry_call_iv, entry_put_iv, entry_call_delta,
                entry_time_decay_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            snapshot.timestamp.isoformat(),
            snapshot_id,
            strategy,
            strike,
            entry_price,
            contracts,
            1 if is_local_ai else 0,
            snapshot.spy_price,
            snapshot.spy_change_5m,
            snapshot.spy_change_15m,
            snapshot.vix,
            snapshot.hour,
            snapshot.minutes_since_open,
            snapshot.call_iv,
            snapshot.put_iv,
            snapshot.call_delta,
            snapshot.time_decay_factor,
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded trade entry: {trade_id} {strategy} @ ${entry_price}")
    
    def record_trade_exit(
        self,
        trade_id: str,
        snapshot: MarketSnapshot,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_percent: float,
        max_profit_reached: float,
        max_loss_reached: float,
    ) -> None:
        """Record a trade exit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Record exit snapshot
        snapshot_id = self.record_snapshot(snapshot)
        
        # Get entry time to calculate duration
        cursor.execute(
            "SELECT entry_time FROM trade_records WHERE trade_id = ?",
            (trade_id,)
        )
        result = cursor.fetchone()
        
        hold_duration = 0
        if result:
            entry_time = datetime.fromisoformat(result[0])
            hold_duration = int((snapshot.timestamp - entry_time).total_seconds())
        
        cursor.execute("""
            UPDATE trade_records SET
                exit_time = ?,
                exit_snapshot_id = ?,
                exit_price = ?,
                exit_reason = ?,
                pnl = ?,
                pnl_percent = ?,
                hold_duration_seconds = ?,
                max_profit_reached = ?,
                max_loss_reached = ?,
                was_profitable = ?
            WHERE trade_id = ?
        """, (
            snapshot.timestamp.isoformat(),
            snapshot_id,
            exit_price,
            exit_reason,
            pnl,
            pnl_percent,
            hold_duration,
            max_profit_reached,
            max_loss_reached,
            1 if pnl > 0 else 0,
            trade_id,
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded trade exit: {trade_id} P&L=${pnl:.2f} ({exit_reason})")
    
    def get_training_data(self, min_trades: int = 10) -> List[Dict[str, Any]]:
        """
        Get completed trades for training in CHRONOLOGICAL ORDER.

        IMPORTANT: Data is returned oldest-first (ASC) for TimeSeriesSplit.
        This prevents data leakage - we train on past, validate on future.

        Args:
            min_trades: Minimum number of trades required

        Returns:
            List of trade records with all features, sorted oldest to newest
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # CRITICAL: ORDER BY ASC for time-series ML (oldest first)
        cursor.execute("""
            SELECT * FROM trade_records
            WHERE exit_time IS NOT NULL
            ORDER BY entry_time ASC
        """)
        
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"Retrieved {len(trades)} completed trades for training")
        return trades
    
    def get_snapshot_count(self) -> int:
        """Get total number of snapshots recorded"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_snapshots")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_trade_count(self) -> int:
        """Get total number of completed trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trade_records WHERE exit_time IS NOT NULL")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_snapshots_for_hour(self, hour: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get snapshots for a specific hour of day"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM market_snapshots 
            WHERE hour = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (hour, limit))
        
        snapshots = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return snapshots
