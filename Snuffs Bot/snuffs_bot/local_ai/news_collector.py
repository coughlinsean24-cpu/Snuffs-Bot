"""
News and Market Context Collector

Fetches and analyzes news to help the AI understand WHY the market is moving.
Tracks geopolitical events, Fed policy, earnings, and other market-moving news.

Uses free news APIs and simple sentiment analysis to score market context.
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

from loguru import logger

# Try to import requests for API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed - news collection disabled")


@dataclass
class NewsItem:
    """A single news article/headline"""
    headline: str
    source: str
    published_at: datetime
    url: str = ""
    summary: str = ""
    
    # Sentiment analysis results
    sentiment_score: float = 0.0  # -1.0 (bearish) to +1.0 (bullish)
    category: str = ""  # GEOPOLITICAL, FED, EARNINGS, ECONOMIC, OTHER
    relevance: float = 0.0  # 0.0 to 1.0 (how relevant to market)
    
    # Key entities mentioned
    tickers_mentioned: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['published_at'] = self.published_at.isoformat()
        return d


@dataclass 
class MarketContext:
    """Current market context from news analysis"""
    timestamp: datetime
    
    # Overall sentiment
    overall_sentiment: float = 0.0  # -1.0 to +1.0
    sentiment_strength: float = 0.0  # 0.0 to 1.0 (how confident)
    
    # Category-specific sentiment
    geopolitical_sentiment: float = 0.0
    fed_sentiment: float = 0.0
    earnings_sentiment: float = 0.0
    economic_sentiment: float = 0.0
    
    # Event flags (for learning)
    war_tensions: int = 0        # War/military conflict news
    tariff_news: int = 0         # Trade war/tariff news
    fed_hawkish: int = 0         # Fed raising rates/hawkish
    fed_dovish: int = 0          # Fed cutting rates/dovish
    recession_fears: int = 0     # Recession/economic slowdown
    inflation_high: int = 0      # High inflation news
    earnings_beat: int = 0       # Positive earnings surprises
    earnings_miss: int = 0       # Negative earnings surprises
    
    # Top themes currently driving market
    top_themes: List[str] = field(default_factory=list)
    
    # Raw context for AI prompts
    context_summary: str = ""
    
    # News items analyzed
    news_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketContext':
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class NewsCollector:
    """
    Collects and analyzes news to provide market context.
    
    Features:
    - Fetches from free news APIs (NewsAPI, Finnhub, RSS)
    - Simple sentiment analysis using keyword matching
    - Categorizes news (geopolitical, Fed, earnings, etc.)
    - Stores historical context for AI learning
    - Provides real-time context summary for trading decisions
    """
    
    # Bearish keywords with weights
    BEARISH_KEYWORDS = {
        # Geopolitical
        'war': -2.0, 'invasion': -2.0, 'attack': -1.5, 'bomb': -1.5, 'missile': -1.5,
        'conflict': -1.5, 'military': -1.0, 'troops': -1.0, 'sanctions': -1.0,
        'iran': -1.0, 'russia': -1.0, 'china': -0.5, 'nuclear': -1.5, 'tensions': -1.0,
        'escalat': -1.5, 'retaliat': -1.5, 'strike': -1.0,
        
        # Economic
        'recession': -2.0, 'crash': -2.0, 'crisis': -1.5, 'collapse': -2.0,
        'plunge': -1.5, 'tumble': -1.5, 'tank': -1.5, 'sell-off': -1.5, 'selloff': -1.5,
        'bear market': -1.5, 'downturn': -1.5, 'slowdown': -1.0,
        'inflation': -0.5, 'stagflation': -1.5,
        
        # Fed/rates
        'rate hike': -1.0, 'hawkish': -1.0, 'tightening': -1.0, 'higher for longer': -1.0,
        
        # Earnings
        'miss': -1.0, 'disappoints': -1.0, 'weak guidance': -1.5, 'cuts forecast': -1.5,
        'layoffs': -1.0, 'job cuts': -1.0, 'restructuring': -0.5,
        
        # Trade
        'tariff': -1.0, 'trade war': -1.5, 'duties': -0.5,
        
        # General
        'fears': -1.0, 'worried': -0.5, 'concern': -0.5, 'warning': -1.0, 'risk': -0.5,
        'uncertainty': -0.5, 'volatile': -0.5, 'turmoil': -1.0,
    }
    
    # Bullish keywords with weights
    BULLISH_KEYWORDS = {
        # Economic
        'rally': 1.5, 'surge': 1.5, 'soar': 1.5, 'jump': 1.0, 'gain': 0.5,
        'bull market': 1.5, 'recovery': 1.0, 'rebound': 1.0, 'upbeat': 1.0,
        'optimis': 1.0, 'growth': 0.5, 'expansion': 0.5,
        'soft landing': 1.5, 'goldilocks': 1.0,
        
        # Fed/rates
        'rate cut': 1.5, 'dovish': 1.0, 'easing': 1.0, 'stimulus': 1.5, 'pivot': 1.0,
        'pause': 0.5,
        
        # Earnings
        'beat': 1.0, 'beats': 1.0, 'exceeds': 1.0, 'strong earnings': 1.5,
        'raises guidance': 1.5, 'record profit': 1.5, 'blowout': 1.5,
        'hiring': 0.5, 'jobs added': 1.0,
        
        # Trade
        'trade deal': 1.5, 'agreement': 0.5, 'peace': 1.0, 'ceasefire': 1.5,
        'de-escalat': 1.5, 'negotiat': 0.5,
        
        # General
        'confidence': 0.5, 'positive': 0.5, 'strong': 0.5, 'boost': 0.5,
    }
    
    # Category detection keywords
    CATEGORY_KEYWORDS = {
        'GEOPOLITICAL': ['war', 'iran', 'russia', 'china', 'military', 'troops', 
                         'invasion', 'conflict', 'nuclear', 'sanctions', 'tariff',
                         'trade war', 'tensions', 'attack', 'missile', 'bomb'],
        'FED': ['fed', 'fomc', 'powell', 'rate', 'rates', 'interest rate', 
                'federal reserve', 'monetary', 'hawkish', 'dovish', 'inflation'],
        'EARNINGS': ['earnings', 'revenue', 'profit', 'eps', 'guidance', 'quarter',
                     'quarterly', 'beat', 'miss', 'results'],
        'ECONOMIC': ['gdp', 'jobs', 'unemployment', 'cpi', 'ppi', 'retail sales',
                     'economic', 'recession', 'growth', 'housing', 'manufacturing'],
    }
    
    # SPY constituent weights (approximate % of SPY)
    # News about larger companies should have MORE impact on SPY predictions
    # Top 50 holdings = ~60% of SPY
    SPY_WEIGHTS = {
        # Mega-cap Tech (30%+ of SPY combined)
        'AAPL': 7.0, 'apple': 7.0,
        'MSFT': 7.0, 'microsoft': 7.0,
        'NVDA': 5.0, 'nvidia': 5.0,
        'AMZN': 4.0, 'amazon': 4.0,
        'META': 2.5, 'meta': 2.5, 'facebook': 2.5,
        'GOOGL': 2.0, 'GOOG': 1.7, 'google': 3.7, 'alphabet': 3.7,
        'AVGO': 1.5, 'broadcom': 1.5,
        'TSLA': 1.5, 'tesla': 1.5,
        
        # Large-cap Tech & Growth
        'COST': 1.0, 'costco': 1.0,
        'NFLX': 0.8, 'netflix': 0.8,
        'AMD': 0.7, 'amd': 0.7,
        'CRM': 0.7, 'salesforce': 0.7,
        'ADBE': 0.6, 'adobe': 0.6,
        'ORCL': 0.6, 'oracle': 0.6,
        'CSCO': 0.5, 'cisco': 0.5,
        'ACN': 0.5, 'accenture': 0.5,
        'INTC': 0.4, 'intel': 0.4,
        'QCOM': 0.4, 'qualcomm': 0.4,
        
        # Financials (~13% of SPY)
        'BRK': 1.7, 'berkshire': 1.7, 'buffett': 1.7,
        'JPM': 1.3, 'jpmorgan': 1.3, 'jp morgan': 1.3, 'jamie dimon': 1.3,
        'V': 1.0, 'visa': 1.0,
        'MA': 0.9, 'mastercard': 0.9,
        'BAC': 0.7, 'bank of america': 0.7,
        'WFC': 0.5, 'wells fargo': 0.5,
        'GS': 0.4, 'goldman': 0.4, 'goldman sachs': 0.4,
        'MS': 0.4, 'morgan stanley': 0.4,
        'AXP': 0.4, 'amex': 0.4, 'american express': 0.4,
        'BLK': 0.3, 'blackrock': 0.3,
        
        # Healthcare (~12% of SPY)
        'UNH': 1.3, 'unitedhealth': 1.3,
        'LLY': 1.2, 'eli lilly': 1.2, 'lilly': 1.2,
        'JNJ': 0.9, 'johnson': 0.9, 'j&j': 0.9,
        'MRK': 0.7, 'merck': 0.7,
        'ABBV': 0.6, 'abbvie': 0.6,
        'PFE': 0.4, 'pfizer': 0.4,
        'TMO': 0.4, 'thermo fisher': 0.4,
        
        # Consumer
        'WMT': 0.7, 'walmart': 0.7,
        'HD': 0.7, 'home depot': 0.7,
        'PG': 0.7, 'procter': 0.7, 'p&g': 0.7,
        'KO': 0.5, 'coca-cola': 0.5, 'coke': 0.5,
        'PEP': 0.5, 'pepsi': 0.5, 'pepsico': 0.5,
        'MCD': 0.4, 'mcdonalds': 0.4, "mcdonald's": 0.4,
        'NKE': 0.3, 'nike': 0.3,
        'SBUX': 0.3, 'starbucks': 0.3,
        'TGT': 0.2, 'target': 0.2,
        
        # Energy
        'XOM': 1.1, 'exxon': 1.1,
        'CVX': 0.7, 'chevron': 0.7,
        'COP': 0.3, 'conocophillips': 0.3,
        
        # Industrials
        'CAT': 0.4, 'caterpillar': 0.4,
        'BA': 0.3, 'boeing': 0.3,
        'RTX': 0.3, 'raytheon': 0.3,
        'LMT': 0.3, 'lockheed': 0.3,
        'UPS': 0.3,
        'GE': 0.3,
        
        # Others
        'DIS': 0.4, 'disney': 0.4,
        'T': 0.3, 'at&t': 0.3,
        'VZ': 0.3, 'verizon': 0.3,
    }
    
    def __init__(self, data_dir: str = "data/local_ai"):
        """Initialize news collector with storage"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "market_data.db"
        self.context_file = self.data_dir / "market_context.json"
        
        # API keys (loaded from environment)
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.finnhub_key = os.getenv("FINNHUB_KEY", "")
        
        # Cache for rate limiting
        self.last_fetch_time: Optional[datetime] = None
        self.cached_context: Optional[MarketContext] = None
        self.cache_duration_minutes = 5
        
        # Initialize database tables
        self._init_db()
        
        # Load cached context if exists
        self._load_cached_context()
        
        if not self.newsapi_key and not self.finnhub_key:
            logger.warning("No news API keys found. Set NEWSAPI_KEY or FINNHUB_KEY for news collection.")
        
    def _init_db(self) -> None:
        """Initialize database tables for news storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # News items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headline_hash TEXT UNIQUE,
                    headline TEXT,
                    source TEXT,
                    published_at TEXT,
                    url TEXT,
                    summary TEXT,
                    sentiment_score REAL,
                    category TEXT,
                    relevance REAL,
                    tickers TEXT,
                    keywords TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market context history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    overall_sentiment REAL,
                    sentiment_strength REAL,
                    geopolitical_sentiment REAL,
                    fed_sentiment REAL,
                    earnings_sentiment REAL,
                    economic_sentiment REAL,
                    war_tensions INTEGER,
                    tariff_news INTEGER,
                    fed_hawkish INTEGER,
                    fed_dovish INTEGER,
                    recession_fears INTEGER,
                    inflation_high INTEGER,
                    earnings_beat INTEGER,
                    earnings_miss INTEGER,
                    top_themes TEXT,
                    context_summary TEXT,
                    news_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news_items(published_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_timestamp ON market_context(timestamp)")
            
            conn.commit()
            conn.close()
            logger.debug("News database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing news database: {e}")
    
    def _load_cached_context(self) -> None:
        """Load cached context from file"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self.cached_context = MarketContext.from_dict(data)
                    logger.debug(f"Loaded cached context from {self.context_file}")
        except Exception as e:
            logger.debug(f"Could not load cached context: {e}")
    
    def _save_cached_context(self) -> None:
        """Save context to cache file"""
        if self.cached_context:
            try:
                with open(self.context_file, 'w') as f:
                    json.dump(self.cached_context.to_dict(), f, indent=2)
            except Exception as e:
                logger.debug(f"Could not save context cache: {e}")
    
    def get_spy_weight(self, text: str) -> Tuple[float, List[str]]:
        """
        Calculate SPY weight multiplier based on companies mentioned.
        
        News about AAPL (7% of SPY) should have ~7x more impact than
        news about a small cap that's 0.1% of SPY.
        
        Returns:
            (weight_multiplier, tickers_found)
            Weight: 1.0 = baseline, higher = more SPY impact
        """
        import re
        text_lower = text.lower()
        text_upper = text.upper()
        tickers_found = []
        max_weight = 1.0
        
        # Single-letter and short tickers that need word boundary matching
        # to avoid false positives (V in "revenue", T in "the", etc.)
        short_tickers = {'V', 'T', 'GE', 'MA', 'BA', 'GS', 'MS', 'HD', 'KO', 'DIS'}
        
        for keyword, weight in self.SPY_WEIGHTS.items():
            keyword_upper = keyword.upper()
            keyword_lower = keyword.lower()
            
            # For short tickers/words, require word boundaries
            if keyword_upper in short_tickers or len(keyword) <= 2:
                # Match as standalone word only (with word boundaries)
                pattern = r'\b' + re.escape(keyword_upper) + r'\b'
                if re.search(pattern, text_upper):
                    tickers_found.append(keyword_upper)
                    max_weight = max(max_weight, weight)
            else:
                # For longer company names, simple substring match is fine
                if keyword_lower in text_lower:
                    tickers_found.append(keyword_upper)
                    max_weight = max(max_weight, weight)
        
        # Remove duplicates and normalize
        unique_tickers = list(set(tickers_found))
        
        return max_weight, unique_tickers
    
    def analyze_sentiment(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze text sentiment using keyword matching.
        
        Returns:
            (sentiment_score, keywords_found)
            Score: -1.0 (very bearish) to +1.0 (very bullish)
        """
        text_lower = text.lower()
        total_score = 0.0
        keywords_found = []
        
        # Check bearish keywords
        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                total_score += weight
                keywords_found.append(keyword)
        
        # Check bullish keywords
        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                total_score += weight
                keywords_found.append(keyword)
        
        # Normalize score to -1.0 to 1.0 range
        if keywords_found:
            normalized = max(-1.0, min(1.0, total_score / (len(keywords_found) * 1.5)))
        else:
            normalized = 0.0
            
        return normalized, keywords_found
    
    def analyze_sentiment_weighted(self, text: str) -> Tuple[float, float, List[str], List[str]]:
        """
        Analyze sentiment with SPY weight consideration.
        
        Returns:
            (sentiment_score, spy_weight, keywords_found, tickers_found)
        """
        sentiment, keywords = self.analyze_sentiment(text)
        spy_weight, tickers = self.get_spy_weight(text)
        
        # Weight the sentiment by SPY impact
        # AAPL news with sentiment -0.5 -> weighted_sentiment = -0.5 * (1 + 7.0/10) = -0.85
        weight_multiplier = 1.0 + (spy_weight / 10.0)  # 7% weight -> 1.7x multiplier
        weighted_sentiment = max(-1.0, min(1.0, sentiment * weight_multiplier))
        
        return weighted_sentiment, spy_weight, keywords, tickers
    
    def categorize_news(self, text: str) -> str:
        """Categorize news article by topic"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return "OTHER"
    
    def fetch_news_newsapi(self, query: str = "stock market OR SPY OR S&P 500") -> List[NewsItem]:
        """Fetch news from NewsAPI.org"""
        if not REQUESTS_AVAILABLE or not self.newsapi_key:
            return []
            
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.newsapi_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("articles", []):
                try:
                    published = datetime.fromisoformat(
                        article.get("publishedAt", "").replace("Z", "+00:00")
                    )
                except:
                    published = datetime.now()
                
                headline = article.get("title", "")
                summary = article.get("description", "") or ""
                full_text = f"{headline} {summary}"
                
                # Use weighted sentiment based on SPY constituent weight
                weighted_sentiment, spy_weight, keywords, tickers = self.analyze_sentiment_weighted(full_text)
                category = self.categorize_news(full_text)
                
                # Higher relevance for news about bigger SPY components
                base_relevance = 0.8 if category != "OTHER" else 0.3
                relevance = min(1.0, base_relevance + (spy_weight / 20.0))
                
                news_item = NewsItem(
                    headline=headline,
                    source=article.get("source", {}).get("name", "Unknown"),
                    published_at=published,
                    url=article.get("url", ""),
                    summary=summary,
                    sentiment_score=weighted_sentiment,
                    category=category,
                    relevance=relevance,
                    keywords=keywords,
                    tickers_mentioned=tickers,
                )
                articles.append(news_item)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []
    
    def fetch_news_finnhub(self) -> List[NewsItem]:
        """Fetch news from Finnhub (free tier)"""
        if not REQUESTS_AVAILABLE or not self.finnhub_key:
            return []
            
        try:
            # Finnhub general market news
            url = "https://finnhub.io/api/v1/news"
            params = {
                "category": "general",
                "token": self.finnhub_key,
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data[:20]:  # Limit to 20
                try:
                    published = datetime.fromtimestamp(article.get("datetime", 0))
                except:
                    published = datetime.now()
                
                headline = article.get("headline", "")
                summary = article.get("summary", "") or ""
                full_text = f"{headline} {summary}"
                
                # Use weighted sentiment based on SPY constituent weight
                weighted_sentiment, spy_weight, keywords, tickers = self.analyze_sentiment_weighted(full_text)
                category = self.categorize_news(full_text)
                
                # Higher relevance for news about bigger SPY components
                base_relevance = 0.8 if category != "OTHER" else 0.3
                relevance = min(1.0, base_relevance + (spy_weight / 20.0))
                
                news_item = NewsItem(
                    headline=headline,
                    source=article.get("source", "Unknown"),
                    published_at=published,
                    url=article.get("url", ""),
                    summary=summary,
                    sentiment_score=weighted_sentiment,
                    category=category,
                    relevance=relevance,
                    keywords=keywords,
                    tickers_mentioned=tickers,
                )
                articles.append(news_item)
            
            logger.info(f"Fetched {len(articles)} articles from Finnhub")
            return articles
            
        except Exception as e:
            logger.warning(f"Finnhub fetch failed: {e}")
            return []
    
    def add_manual_context(
        self,
        event_type: str,
        description: str,
        sentiment: float = 0.0,
    ) -> None:
        """
        Manually add market context (for events you know about).
        
        Args:
            event_type: GEOPOLITICAL, FED, EARNINGS, ECONOMIC, OTHER
            description: Brief description of the event
            sentiment: -1.0 (bearish) to 1.0 (bullish)
        """
        now = datetime.now()
        
        # Get weighted analysis for manual context too
        weighted_sentiment, spy_weight, keywords, tickers = self.analyze_sentiment_weighted(description)
        
        # Use provided sentiment but apply SPY weight multiplier
        if sentiment != 0.0:
            weight_multiplier = 1.0 + (spy_weight / 10.0)
            final_sentiment = max(-1.0, min(1.0, sentiment * weight_multiplier))
        else:
            final_sentiment = weighted_sentiment
        
        # Create a news item for the manual entry
        news_item = NewsItem(
            headline=description,
            source="MANUAL",
            published_at=now,
            sentiment_score=final_sentiment,
            category=event_type,
            relevance=1.0,  # Manual entries are highly relevant
            keywords=keywords,
            tickers_mentioned=tickers,
        )
        
        self._store_news_item(news_item)
        logger.info(f"Added manual context: {event_type} - {description[:50]}...")
        
        # Invalidate cache to refresh context
        self.cached_context = None
    
    def _store_news_item(self, item: NewsItem) -> None:
        """Store news item in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create hash of headline for deduplication
            headline_hash = hashlib.md5(item.headline.encode()).hexdigest()
            
            cursor.execute("""
                INSERT OR IGNORE INTO news_items 
                (headline_hash, headline, source, published_at, url, summary,
                 sentiment_score, category, relevance, tickers, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                headline_hash,
                item.headline,
                item.source,
                item.published_at.isoformat(),
                item.url,
                item.summary,
                item.sentiment_score,
                item.category,
                item.relevance,
                json.dumps(item.tickers_mentioned),
                json.dumps(item.keywords),
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Error storing news item: {e}")
    
    def get_current_context(self, force_refresh: bool = False) -> MarketContext:
        """
        Get current market context from news analysis.
        
        Uses caching to avoid hitting API rate limits.
        """
        now = datetime.now()
        
        # Check cache
        if not force_refresh and self.cached_context:
            cache_age = (now - self.cached_context.timestamp).total_seconds() / 60
            if cache_age < self.cache_duration_minutes:
                return self.cached_context
        
        # Fetch fresh news
        all_news = []
        
        if self.newsapi_key:
            all_news.extend(self.fetch_news_newsapi())
        
        if self.finnhub_key:
            all_news.extend(self.fetch_news_finnhub())
        
        # Also load recent items from database
        recent_db_news = self._get_recent_news_from_db(hours=4)
        
        # Combine (prefer fresh API news)
        combined_headlines = set()
        combined_news = []
        for item in all_news + recent_db_news:
            if item.headline not in combined_headlines:
                combined_headlines.add(item.headline)
                combined_news.append(item)
                self._store_news_item(item)  # Store for learning
        
        # Build context from news
        context = self._build_context(combined_news)
        
        # Cache it
        self.cached_context = context
        self.last_fetch_time = now
        self._save_cached_context()
        self._store_context(context)
        
        return context
    
    def _get_recent_news_from_db(self, hours: int = 4) -> List[NewsItem]:
        """Get recent news from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            cursor.execute("""
                SELECT headline, source, published_at, url, summary,
                       sentiment_score, category, relevance, keywords
                FROM news_items
                WHERE published_at > ?
                ORDER BY published_at DESC
                LIMIT 50
            """, (cutoff,))
            
            items = []
            for row in cursor.fetchall():
                try:
                    items.append(NewsItem(
                        headline=row[0],
                        source=row[1],
                        published_at=datetime.fromisoformat(row[2]),
                        url=row[3],
                        summary=row[4],
                        sentiment_score=row[5],
                        category=row[6],
                        relevance=row[7],
                        keywords=json.loads(row[8]) if row[8] else [],
                    ))
                except Exception as e:
                    continue
            
            conn.close()
            return items
            
        except Exception as e:
            logger.debug(f"Error loading news from DB: {e}")
            return []
    
    def _build_context(self, news_items: List[NewsItem]) -> MarketContext:
        """Build market context from news items"""
        now = datetime.now()
        
        if not news_items:
            return MarketContext(timestamp=now, context_summary="No recent news available")
        
        # Calculate category-specific sentiments
        category_sentiments = {
            'GEOPOLITICAL': [],
            'FED': [],
            'EARNINGS': [],
            'ECONOMIC': [],
        }
        
        all_sentiments = []
        all_keywords = []
        themes = {}
        
        for item in news_items:
            sentiment = item.sentiment_score * item.relevance
            all_sentiments.append(sentiment)
            all_keywords.extend(item.keywords)
            
            if item.category in category_sentiments:
                category_sentiments[item.category].append(sentiment)
            
            # Track themes
            for kw in item.keywords[:3]:  # Top 3 keywords per article
                themes[kw] = themes.get(kw, 0) + 1
        
        # Calculate averages
        overall = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
        strength = min(1.0, len(news_items) / 20)  # More news = more confidence
        
        def avg(lst): return sum(lst) / len(lst) if lst else 0
        
        # Detect event flags
        keywords_text = ' '.join(all_keywords).lower()
        
        context = MarketContext(
            timestamp=now,
            overall_sentiment=overall,
            sentiment_strength=strength,
            geopolitical_sentiment=avg(category_sentiments['GEOPOLITICAL']),
            fed_sentiment=avg(category_sentiments['FED']),
            earnings_sentiment=avg(category_sentiments['EARNINGS']),
            economic_sentiment=avg(category_sentiments['ECONOMIC']),
            war_tensions=1 if any(kw in keywords_text for kw in ['war', 'invasion', 'attack', 'missile', 'iran']) else 0,
            tariff_news=1 if any(kw in keywords_text for kw in ['tariff', 'trade war', 'duties']) else 0,
            fed_hawkish=1 if any(kw in keywords_text for kw in ['hawkish', 'rate hike', 'tightening']) else 0,
            fed_dovish=1 if any(kw in keywords_text for kw in ['dovish', 'rate cut', 'easing', 'pivot']) else 0,
            recession_fears=1 if 'recession' in keywords_text else 0,
            inflation_high=1 if 'inflation' in keywords_text and overall < 0 else 0,
            earnings_beat=1 if any(kw in keywords_text for kw in ['beat', 'beats', 'blowout']) else 0,
            earnings_miss=1 if any(kw in keywords_text for kw in ['miss', 'disappoints']) else 0,
            top_themes=sorted(themes.keys(), key=themes.get, reverse=True)[:5],
            context_summary=self._generate_summary(news_items, overall),
            news_count=len(news_items),
        )
        
        return context
    
    def _generate_summary(self, news_items: List[NewsItem], overall_sentiment: float) -> str:
        """Generate a human-readable context summary for AI"""
        if not news_items:
            return "No significant news driving the market."
        
        # Group by category
        by_category = {}
        for item in news_items:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)
        
        parts = []
        
        # Overall tone
        if overall_sentiment < -0.5:
            parts.append("BEARISH SENTIMENT: Markets facing significant headwinds.")
        elif overall_sentiment < -0.2:
            parts.append("CAUTIOUS SENTIMENT: Negative news weighing on markets.")
        elif overall_sentiment > 0.5:
            parts.append("BULLISH SENTIMENT: Positive catalysts driving markets.")
        elif overall_sentiment > 0.2:
            parts.append("OPTIMISTIC SENTIMENT: Generally positive news flow.")
        else:
            parts.append("NEUTRAL SENTIMENT: Mixed news with no clear direction.")
        
        # Category summaries
        for cat, items in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
            if len(items) >= 2:  # Only mention if multiple articles
                avg_sent = sum(i.sentiment_score for i in items) / len(items)
                tone = "bearish" if avg_sent < -0.2 else "bullish" if avg_sent > 0.2 else "mixed"
                parts.append(f"{cat}: {len(items)} articles, {tone} tone.")
        
        # Top headlines
        top_headlines = sorted(news_items, key=lambda x: abs(x.sentiment_score), reverse=True)[:3]
        if top_headlines:
            parts.append("KEY HEADLINES:")
            for item in top_headlines:
                tone = "📉" if item.sentiment_score < 0 else "📈" if item.sentiment_score > 0 else "➡️"
                parts.append(f"  {tone} {item.headline[:100]}")
        
        return "\n".join(parts)
    
    def _store_context(self, context: MarketContext) -> None:
        """Store context in database for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO market_context 
                (timestamp, overall_sentiment, sentiment_strength,
                 geopolitical_sentiment, fed_sentiment, earnings_sentiment, economic_sentiment,
                 war_tensions, tariff_news, fed_hawkish, fed_dovish,
                 recession_fears, inflation_high, earnings_beat, earnings_miss,
                 top_themes, context_summary, news_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                context.timestamp.isoformat(),
                context.overall_sentiment,
                context.sentiment_strength,
                context.geopolitical_sentiment,
                context.fed_sentiment,
                context.earnings_sentiment,
                context.economic_sentiment,
                context.war_tensions,
                context.tariff_news,
                context.fed_hawkish,
                context.fed_dovish,
                context.recession_fears,
                context.inflation_high,
                context.earnings_beat,
                context.earnings_miss,
                json.dumps(context.top_themes),
                context.context_summary,
                context.news_count,
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Error storing context: {e}")
    
    def get_context_for_snapshot(self) -> Dict[str, Any]:
        """Get context fields to include in market snapshots"""
        context = self.get_current_context()
        
        return {
            'news_sentiment': context.overall_sentiment,
            'war_tensions': context.war_tensions,
            'tariff_news': context.tariff_news,
            'fed_hawkish': context.fed_hawkish,
            'fed_dovish': context.fed_dovish,
            'recession_fears': context.recession_fears,
            'context_summary': context.context_summary[:500],  # Truncate for storage
        }


# Convenience function
def get_market_context(data_dir: str = "data/local_ai") -> MarketContext:
    """Quick way to get current market context"""
    collector = NewsCollector(data_dir)
    return collector.get_current_context()
