"""
Market data streaming and retrieval functionality for Tastytrade API

Uses WebSocket streaming for real-time quotes. Tastytrade only - no fallbacks.
"""

import threading
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from datetime import datetime


class MarketDataManager:
    """
    Manages market data operations with WebSocket streaming

    This class provides:
    - Real-time streaming quotes via Tastytrade WebSocket
    - Thread-safe quote cache for fast synchronous access
    - Options chains, Greeks, and historical data
    - Tastytrade API only (no yfinance fallback)
    """

    def __init__(self, client):
        """
        Initialize MarketDataManager

        Args:
            client: TastytradeClient instance
        """
        self.client = client
        self._subscriptions: Dict[str, Any] = {}

        # Thread-safe quote cache for streaming data
        self._quote_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._stream_active = False
        self._primary_subscription = None
        self._last_heartbeat = datetime.now()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._base_symbols = ["SPY", "$VIX.X"]  # Core symbols to always stream

    def start_streaming(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Start streaming market data for specified symbols

        Args:
            symbols: List of symbols to stream (default: SPY, VIX)

        Returns:
            True if streaming started successfully
        """
        if self._stream_active:
            logger.debug("Streaming already active")
            return True

        symbols = symbols or self._base_symbols
        self._streaming_symbols = set(symbols)

        try:
            logger.info(f"Starting WebSocket stream for: {symbols}")

            # Create subscription with quote handler
            self._primary_subscription = self.client.session.market_data.subscribe(
                symbols=symbols,
                on_quote=self._handle_streaming_quote,
                aggregation_period=0.5,  # 500ms aggregation for responsiveness
            )

            # Open the WebSocket connection
            self._primary_subscription.open()
            self._stream_active = True
            self._last_heartbeat = datetime.now()
            self._reconnect_attempts = 0

            logger.success(f"WebSocket streaming started for {symbols}")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self._stream_active = False
            return False

    def check_streaming_health(self) -> bool:
        """
        Check if streaming connection is healthy and reconnect if needed.
        
        Returns:
            True if streaming is healthy or was successfully reconnected
        """
        # Check if we've received data recently (within 30 seconds)
        age = (datetime.now() - self._last_heartbeat).total_seconds()
        
        if age > 30 and self._stream_active:
            logger.warning(f"No streaming data for {age:.0f}s - connection may be dead")
            self._stream_active = False
        
        if not self._stream_active and self._reconnect_attempts < self._max_reconnect_attempts:
            logger.info(f"Attempting to reconnect streaming (attempt {self._reconnect_attempts + 1}/{self._max_reconnect_attempts})")
            self._reconnect_attempts += 1
            
            # Close existing subscriptions
            self.stop_streaming()
            
            # Re-add all option symbols we had
            existing_symbols = list(getattr(self, '_streaming_symbols', set()))
            
            # Start fresh
            if self.start_streaming(self._base_symbols):
                # Re-subscribe to option symbols
                for symbol in existing_symbols:
                    if symbol not in self._base_symbols:
                        self.add_streaming_symbol(symbol)
                return True
            return False
        
        return self._stream_active

    def add_streaming_symbol(self, symbol: str) -> bool:
        """
        Add a symbol to the streaming subscription
        
        Args:
            symbol: Symbol to add (e.g., option symbol 'SPY260127C00706000')
            
        Returns:
            True if symbol was added successfully
        """
        if not self._stream_active:
            logger.warning("Cannot add symbol - streaming not active")
            return False
            
        if hasattr(self, '_streaming_symbols') and symbol in self._streaming_symbols:
            return True  # Already streaming
            
        try:
            # Create a new subscription for this option symbol
            # The Tastytrade SDK creates separate subscriptions for different symbols
            option_subscription = self.client.session.market_data.subscribe(
                symbols=[symbol],
                on_quote=self._handle_streaming_quote,
                aggregation_period=0.5,
            )
            option_subscription.open()
            
            # Store the subscription
            sub_id = f"option_{symbol}"
            self._subscriptions[sub_id] = option_subscription
            
            if not hasattr(self, '_streaming_symbols'):
                self._streaming_symbols = set()
            self._streaming_symbols.add(symbol)
            
            logger.info(f"Added {symbol} to streaming (new subscription)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add symbol {symbol} to stream: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop streaming market data"""
        if self._primary_subscription:
            try:
                self._primary_subscription.close()
                logger.info("WebSocket streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")

        self._stream_active = False
        self._primary_subscription = None

    def _handle_streaming_quote(self, event: Dict[str, Any]) -> None:
        """
        Handle incoming streaming quote events

        Updates the thread-safe cache with latest prices.
        """
        try:
            # Update heartbeat on any data received
            self._last_heartbeat = datetime.now()
            
            symbol = event.get("eventSymbol", "")
            # The event also contains the original OCC symbol we subscribed with
            occ_symbol = event.get("symbol", "")

            # Normalize symbol names
            if symbol == "$VIX.X":
                symbol = "VIX"

            bid = event.get("bidPrice", 0.0) or 0.0
            ask = event.get("askPrice", 0.0) or 0.0
            # Use lastPrice if available, otherwise calculate mid-price from bid/ask
            last = event.get("lastPrice", 0.0) or 0.0
            if not last and bid and ask:
                last = (bid + ask) / 2

            quote_data = {
                "symbol": symbol,
                "occ_symbol": occ_symbol,
                "bid": bid,
                "ask": ask,
                "last": last,
                "bid_size": event.get("bidSize", 0),
                "ask_size": event.get("askSize", 0),
                "volume": event.get("dayVolume", 0),
                "timestamp": datetime.now(),
                "source": "TASTYTRADE_STREAM",
            }

            with self._cache_lock:
                # Store by both event symbol and OCC symbol for lookup flexibility
                self._quote_cache[symbol] = quote_data
                if occ_symbol and occ_symbol != symbol:
                    self._quote_cache[occ_symbol] = quote_data
                    # Also store stripped version (no spaces) for matching
                    stripped = occ_symbol.replace(" ", "")
                    self._quote_cache[stripped] = quote_data
                
                # Also store by streamer format (.SPY260129C688) if we can construct it
                # from the OCC symbol (SPY   260129C00688000)
                if occ_symbol and len(occ_symbol) > 15:
                    try:
                        # Parse OCC: "SPY   260129C00688000" -> ".SPY260129C688"
                        parts = occ_symbol.split()
                        if len(parts) >= 2:
                            underlying = parts[0]
                            rest = parts[1]  # "260129C00688000"
                            if len(rest) >= 13:
                                date = rest[:6]  # 260129
                                cp = rest[6]  # C or P
                                strike = rest[7:].lstrip('0')[:3]  # 688
                                streamer_key = f".{underlying}{date}{cp}{strike}"
                                self._quote_cache[streamer_key] = quote_data
                    except Exception:
                        pass  # Don't fail on parsing errors

            # Log price updates for key symbols (throttled to avoid log spam)
            if symbol in ["SPY", "VIX"] and last > 0:
                # Only log every 30 seconds to reduce noise
                now = datetime.now()
                last_log_key = f"_last_log_{symbol}"
                last_log = getattr(self, last_log_key, None)
                if last_log is None or (now - last_log).seconds >= 30:
                    logger.info(f"[TASTYTRADE] {symbol}: ${last:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f})")
                    setattr(self, last_log_key, now)
            # Log option price updates - ALWAYS log these for debugging
            elif occ_symbol and len(occ_symbol) > 10 and (bid > 0 or ask > 0):
                mark = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                logger.info(f"[OPTION STREAM] {occ_symbol}: mark=${mark:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f})")

        except Exception as e:
            logger.error(f"Error handling stream quote: {e}")

    @property
    def is_streaming(self) -> bool:
        """Check if streaming is active"""
        return self._stream_active

    def get_cached_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get quote from streaming cache (fast, no network call)

        Args:
            symbol: Trading symbol

        Returns:
            Cached quote or None if not available
        """
        with self._cache_lock:
            return self._quote_cache.get(symbol)

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol from Tastytrade streaming cache.

        Args:
            symbol: Trading symbol

        Returns:
            Quote data dictionary (empty if not available in cache)
        """
        # Normalize symbol for cache lookup (streaming stores $VIX.X as VIX)
        lookup_symbol = "VIX" if symbol == "$VIX.X" else symbol
        
        # Get from streaming cache
        cached = self.get_cached_quote(lookup_symbol)
        if cached and self._stream_active:
            # Check if cache is fresh (< 30 seconds old for options, 5 seconds for equities)
            is_option = len(symbol) > 10 and any(c in symbol for c in ['C', 'P'])
            max_age = 30 if is_option else 5
            age = (datetime.now() - cached.get("timestamp", datetime.min)).total_seconds()
            if age < max_age:
                last = cached.get("last", 0)
                bid = cached.get("bid", 0)
                ask = cached.get("ask", 0)
                mark = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                logger.debug(f"[CACHE] {symbol}: ${mark:.2f} (age: {age:.1f}s)")
                return {
                    "symbol": symbol,
                    "last": last,
                    "previous-close": cached.get("previous_close", last),
                    "bid": bid,
                    "ask": ask,
                    "mark": mark,
                    "source": "TASTYTRADE_STREAM",
                }

        # Not in cache - try VIX fallback if this is a VIX request
        if symbol == "$VIX.X" or lookup_symbol == "VIX":
            vix_data = self._fetch_vix_fallback()
            if vix_data:
                return vix_data

        logger.debug(f"Symbol {symbol} not in streaming cache")
        return {}

    def get_option_quote_with_wait(self, symbol: str, max_wait: float = 2.0) -> Dict[str, Any]:
        """
        Get option quote, waiting for streaming data if necessary.
        
        Args:
            symbol: Option symbol (OCC format)
            max_wait: Maximum seconds to wait for streaming data
            
        Returns:
            Quote data dictionary or empty dict if unavailable
        """
        import time
        
        # Subscribe if not already
        if hasattr(self, 'add_streaming_symbol'):
            self.add_streaming_symbol(symbol)
        
        # Check cache immediately
        cached = self.get_cached_quote(symbol)
        if cached:
            bid = cached.get("bid", 0)
            ask = cached.get("ask", 0)
            if bid > 0 or ask > 0:
                mark = (bid + ask) / 2 if (bid > 0 and ask > 0) else cached.get("last", 0)
                return {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "last": cached.get("last", 0),
                    "mark": mark,
                    "source": "TASTYTRADE_STREAM",
                }
        
        # Wait for streaming data
        start = time.time()
        while (time.time() - start) < max_wait:
            time.sleep(0.2)
            cached = self.get_cached_quote(symbol)
            if cached:
                bid = cached.get("bid", 0)
                ask = cached.get("ask", 0)
                if bid > 0 or ask > 0:
                    mark = (bid + ask) / 2 if (bid > 0 and ask > 0) else cached.get("last", 0)
                    logger.info(f"[OPTION WAIT] Got {symbol} after {time.time()-start:.1f}s: ${mark:.2f}")
                    return {
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "last": cached.get("last", 0),
                        "mark": mark,
                        "source": "TASTYTRADE_STREAM",
                    }
        
        logger.warning(f"[OPTION WAIT] Timeout waiting for {symbol} after {max_wait}s")
        return {}

    def get_option_quote_rest(self, option_symbol: str) -> Dict[str, Any]:
        """
        Get option quote using REST API (not streaming).
        This is more reliable when streaming is unavailable.
        
        Args:
            option_symbol: Option symbol in OCC format (e.g., 'SPY260127C00697000')
            
        Returns:
            Quote data dictionary with bid, ask, mark, etc.
        """
        try:
            # Parse the option symbol to get underlying
            # OCC format: SYMBOL + DATE(6) + C/P + STRIKE(8)
            underlying = option_symbol[:3]  # SPY
            
            # Get the full option chain
            response = self.client.session.api.get(
                f"/option-chains/{underlying}/compact"
            )
            
            if response and response.get("data"):
                chain = response["data"]
                # Search for our specific option in the chain
                for item in chain.get("items", []):
                    if item.get("symbol") == option_symbol:
                        bid = float(item.get("bid", 0) or 0)
                        ask = float(item.get("ask", 0) or 0)
                        mark = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                        
                        logger.info(f"[REST API] {option_symbol}: mark=${mark:.2f} (bid=${bid:.2f}, ask=${ask:.2f})")
                        return {
                            "symbol": option_symbol,
                            "bid": bid,
                            "ask": ask,
                            "last": float(item.get("last", mark) or mark),
                            "mark": mark,
                            "source": "TASTYTRADE_REST",
                        }
            
            logger.debug(f"Option {option_symbol} not found in chain")
            return {}
            
        except Exception as e:
            logger.debug(f"REST API quote failed for {option_symbol}: {e}")
            return {}

    def subscribe_quotes(
        self,
        symbols: List[str],
        on_quote: Optional[Callable] = None,
        on_candle: Optional[Callable] = None,
        on_greeks: Optional[Callable] = None,
        aggregation_period: float = 1,
        event_fields: Optional[Dict[str, List[str]]] = None
    ) -> Any:
        """
        Subscribe to real-time market data (custom subscription)

        Args:
            symbols: List of symbols to subscribe to
            on_quote: Callback function for quote events
            on_candle: Callback function for candle events
            on_greeks: Callback function for greeks events
            aggregation_period: Event grouping interval in seconds
            event_fields: Dictionary mapping event types to field lists

        Returns:
            Subscription object
        """
        logger.info(f"Creating custom subscription for {len(symbols)} symbol(s)")

        try:
            subscription = self.client.session.market_data.subscribe(
                symbols=symbols,
                on_quote=on_quote or self._default_quote_handler,
                on_candle=on_candle,
                on_greeks=on_greeks,
                aggregation_period=aggregation_period,
                event_fields=event_fields
            )

            sub_id = f"custom_{len(self._subscriptions)}"
            self._subscriptions[sub_id] = subscription

            logger.success(f"Created subscription {sub_id} for {symbols}")
            return subscription

        except Exception as e:
            logger.error(f"Failed to subscribe to quotes: {e}")
            raise

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get options chain for a symbol

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            expiration: Optional expiration date filter (YYYY-MM-DD)

        Returns:
            Options chain data dictionary
        """
        logger.debug(f"Fetching option chain for {symbol}")

        try:
            params = {}
            if expiration:
                params["expiration"] = expiration

            response = self.client.session.api.get(
                f"/option-chains/{symbol}/nested",
                params
            )

            chain_data = response.get("data", {})
            logger.info(f"Retrieved option chain for {symbol}")
            return chain_data

        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {e}")
            raise

    def get_candles(
        self,
        symbol: str,
        interval: str = "1d",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical candle data

        Args:
            symbol: Trading symbol
            interval: Candle interval (1m, 5m, 15m, 1h, 1d, etc.)
            start_time: Start time in ISO format
            end_time: End time in ISO format
            limit: Maximum number of candles to return

        Returns:
            List of candle dictionaries
        """
        logger.debug(f"Fetching candles for {symbol} ({interval})")

        try:
            params = {
                "interval": interval,
                "limit": limit
            }

            if start_time:
                params["start-time"] = start_time
            if end_time:
                params["end-time"] = end_time

            response = self.client.session.api.get(
                f"/market-data/candles/{symbol}",
                params
            )

            candles = response.get("data", {}).get("items", [])
            logger.info(f"Retrieved {len(candles)} candle(s) for {symbol}")
            return candles

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            raise

    def search_symbols(
        self,
        query: str,
        instrument_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for symbols/instruments

        Args:
            query: Search query
            instrument_types: Filter by instrument types

        Returns:
            List of matching instruments
        """
        logger.debug(f"Searching symbols: {query}")

        try:
            params = {"symbol[]": query}

            if instrument_types:
                for inst_type in instrument_types:
                    params["instrument-type[]"] = inst_type

            response = self.client.session.api.get(
                "/instruments",
                params
            )

            instruments = response.get("data", {}).get("items", [])
            logger.info(f"Found {len(instruments)} instrument(s)")
            return instruments

        except Exception as e:
            logger.error(f"Failed to search symbols: {e}")
            raise

    def get_equity_options(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        option_type: Optional[str] = None,
        strike_price: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get equity options for a symbol

        Args:
            symbol: Underlying equity symbol
            expiration_date: Filter by expiration (YYYY-MM-DD)
            option_type: Filter by type ('C' for call, 'P' for put)
            strike_price: Filter by strike price

        Returns:
            List of option contracts
        """
        logger.debug(f"Fetching equity options for {symbol}")

        try:
            params = {}

            if expiration_date:
                params["expiration-date"] = expiration_date
            if option_type:
                params["option-type"] = option_type
            if strike_price:
                params["strike-price"] = str(strike_price)

            response = self.client.session.api.get(
                f"/instruments/equity-options/{symbol}",
                params
            )

            options = response.get("data", {}).get("items", [])
            logger.info(f"Retrieved {len(options)} option(s) for {symbol}")
            return options

        except Exception as e:
            logger.error(f"Failed to fetch equity options for {symbol}: {e}")
            raise

    def get_greeks(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Get Greeks data for options

        Args:
            symbols: List of option symbols

        Returns:
            List of Greeks data dictionaries
        """
        logger.debug(f"Fetching Greeks for {len(symbols)} option(s)")

        try:
            symbol_params = [("symbol[]", s) for s in symbols]

            response = self.client.session.api.get(
                "/greeks",
                symbol_params
            )

            greeks = response.get("data", {}).get("items", [])
            logger.info(f"Retrieved Greeks for {len(greeks)} option(s)")
            return greeks

        except Exception as e:
            logger.error(f"Failed to fetch Greeks: {e}")
            raise

    def close_subscription(self, subscription: Any) -> None:
        """Close a market data subscription"""
        try:
            subscription.close()
            logger.info("Closed market data subscription")
        except Exception as e:
            logger.error(f"Failed to close subscription: {e}")

    def close_all_subscriptions(self) -> None:
        """Close all active market data subscriptions"""
        # Stop primary streaming
        self.stop_streaming()

        # Close custom subscriptions
        logger.info(f"Closing {len(self._subscriptions)} custom subscription(s)")

        for sub_id, subscription in self._subscriptions.items():
            try:
                subscription.close()
                logger.debug(f"Closed subscription {sub_id}")
            except Exception as e:
                logger.error(f"Error closing subscription {sub_id}: {e}")

        self._subscriptions.clear()

    def _fetch_vix_fallback(self) -> Optional[Dict[str, Any]]:
        """
        Fetch VIX data from Yahoo Finance as fallback.
        Tastytrade streaming doesn't support index symbols like $VIX.X.

        Returns:
            VIX quote data or None if unavailable
        """
        import time

        # Check if we have a recent cached VIX value (avoid hammering Yahoo)
        vix_cache_key = "_vix_fallback_cache"
        vix_cache = getattr(self, vix_cache_key, None)
        if vix_cache:
            cache_age = time.time() - vix_cache.get("timestamp", 0)
            if cache_age < 60:  # Cache for 60 seconds
                return vix_cache.get("data")

        try:
            import urllib.request
            import json

            # Yahoo Finance API endpoint for VIX
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1m&range=1d"

            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                current_price = meta.get("regularMarketPrice", 0)
                previous_close = meta.get("previousClose", current_price)

                if current_price > 0:
                    vix_data = {
                        "symbol": "$VIX.X",
                        "last": current_price,
                        "previous-close": previous_close,
                        "bid": current_price * 0.99,  # Approximate
                        "ask": current_price * 1.01,  # Approximate
                        "mark": current_price,
                        "source": "YAHOO_FINANCE_FALLBACK",
                    }

                    # Cache the result
                    setattr(self, vix_cache_key, {
                        "timestamp": time.time(),
                        "data": vix_data
                    })

                    logger.info(f"[VIX FALLBACK] Yahoo Finance: {current_price:.2f}")
                    return vix_data

        except Exception as e:
            logger.debug(f"VIX fallback failed: {e}")

        return None

    @staticmethod
    def _default_quote_handler(event: Dict[str, Any]) -> None:
        """Default quote handler that logs quote events"""
        symbol = event.get("eventSymbol", "UNKNOWN")
        bid = event.get("bidPrice", 0.0) or 0.0
        ask = event.get("askPrice", 0.0) or 0.0
        last = event.get("lastPrice", 0.0) or 0.0

        logger.info(f"Quote: {symbol} - Bid: ${bid:.2f} Ask: ${ask:.2f} Last: ${last:.2f}")

    def __repr__(self) -> str:
        """String representation"""
        status = "streaming" if self._stream_active else "idle"
        return f"MarketDataManager({status}, subscriptions={len(self._subscriptions)})"
