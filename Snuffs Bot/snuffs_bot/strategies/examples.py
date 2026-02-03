"""
Example trading strategies demonstrating the framework
"""

from typing import Dict, Any, List
from collections import deque
from .base import BaseStrategy


class SimpleMovingAverageStrategy(BaseStrategy):
    """
    Simple Moving Average crossover strategy

    Buys when short-term SMA crosses above long-term SMA
    Sells when short-term SMA crosses below long-term SMA
    """

    def __init__(
        self,
        client,
        symbol: str,
        short_period: int = 10,
        long_period: int = 50,
        quantity: int = 100
    ):
        """
        Initialize SMA strategy

        Args:
            client: TastytradeClient instance
            symbol: Symbol to trade
            short_period: Short-term SMA period
            long_period: Long-term SMA period
            quantity: Number of shares to trade
        """
        super().__init__(client, name=f"SMA_{symbol}")

        self.symbol = symbol
        self.short_period = short_period
        self.long_period = long_period
        self.quantity = quantity

        # Price history
        self.prices: deque = deque(maxlen=long_period)

        # Previous SMA values for crossover detection
        self.prev_short_sma = None
        self.prev_long_sma = None

    async def on_start(self) -> None:
        """Initialize strategy"""
        self.log(
            f"Starting with short={self.short_period}, long={self.long_period}"
        )

        # Subscribe to market data
        def handle_quote(quote):
            # Run quote handler in async context
            import asyncio
            asyncio.create_task(self.on_quote(quote))

        subscription = self.client.market_data.subscribe_quotes(
            symbols=[self.symbol],
            on_quote=handle_quote
        )
        subscription.open()

        self.state["subscription"] = subscription
        self.log("Subscribed to market data")

    async def on_quote(self, quote: Dict[str, Any]) -> None:
        """Process new quote"""
        if quote.get("symbol") != self.symbol:
            return

        # Get last price
        last_price = quote.get("last")
        if not last_price:
            return

        # Add to price history
        self.prices.append(last_price)

        # Need enough data for long SMA
        if len(self.prices) < self.long_period:
            return

        # Calculate SMAs
        short_sma = sum(list(self.prices)[-self.short_period:]) / self.short_period
        long_sma = sum(self.prices) / self.long_period

        # Check for crossover
        if self.prev_short_sma and self.prev_long_sma:
            # Bullish crossover - BUY
            if (
                self.prev_short_sma <= self.prev_long_sma
                and short_sma > long_sma
                and not self.has_position(self.symbol)
            ):
                self.log(f"Bullish crossover detected at ${last_price:.2f}")
                await self._enter_long()

            # Bearish crossover - SELL
            elif (
                self.prev_short_sma >= self.prev_long_sma
                and short_sma < long_sma
                and self.has_position(self.symbol)
            ):
                self.log(f"Bearish crossover detected at ${last_price:.2f}")
                await self._exit_long()

        # Update previous values
        self.prev_short_sma = short_sma
        self.prev_long_sma = long_sma

    async def on_candle(self, candle: Dict[str, Any]) -> None:
        """Process new candle (not used in this strategy)"""
        pass

    async def on_stop(self) -> None:
        """Clean up strategy"""
        self.log("Stopping strategy")

        # Close subscription
        subscription = self.state.get("subscription")
        if subscription:
            self.client.market_data.close_subscription(subscription)

        # Close any open positions
        if self.has_position(self.symbol):
            await self._exit_long()

    async def _enter_long(self) -> None:
        """Enter long position"""
        try:
            from ..api.orders import OrderAction, OrderType

            self.log(f"Entering long position: {self.quantity} shares")

            order = self.client.orders.place_order(
                account_number=None,  # Use default
                symbol=self.symbol,
                quantity=self.quantity,
                action=OrderAction.BUY_TO_OPEN,
                order_type=OrderType.MARKET
            )

            self.log(f"Buy order placed: {order.get('id')}")

        except Exception as e:
            self.log(f"Failed to enter long: {e}", level="ERROR")
            await self.on_error(e)

    async def _exit_long(self) -> None:
        """Exit long position"""
        try:
            from ..api.orders import OrderAction, OrderType

            self.log(f"Exiting long position: {self.quantity} shares")

            order = self.client.orders.place_order(
                account_number=None,  # Use default
                symbol=self.symbol,
                quantity=self.quantity,
                action=OrderAction.SELL_TO_CLOSE,
                order_type=OrderType.MARKET
            )

            self.log(f"Sell order placed: {order.get('id')}")

        except Exception as e:
            self.log(f"Failed to exit long: {e}", level="ERROR")
            await self.on_error(e)


class MomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy

    Buys when price increases by a threshold percentage
    Sells when price decreases by a threshold percentage
    """

    def __init__(
        self,
        client,
        symbol: str,
        buy_threshold: float = 0.02,  # 2% increase
        sell_threshold: float = 0.02,  # 2% decrease
        quantity: int = 100
    ):
        """
        Initialize momentum strategy

        Args:
            client: TastytradeClient instance
            symbol: Symbol to trade
            buy_threshold: Percentage increase to trigger buy (0.02 = 2%)
            sell_threshold: Percentage decrease to trigger sell (0.02 = 2%)
            quantity: Number of shares to trade
        """
        super().__init__(client, name=f"Momentum_{symbol}")

        self.symbol = symbol
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.quantity = quantity

        # Reference price for momentum calculation
        self.reference_price = None
        self.entry_price = None

    async def on_start(self) -> None:
        """Initialize strategy"""
        self.log(
            f"Starting with buy_threshold={self.buy_threshold:.1%}, "
            f"sell_threshold={self.sell_threshold:.1%}"
        )

        # Get current quote to set reference price
        quote = self.client.market_data.get_quote(self.symbol)
        self.reference_price = quote.get("last")

        self.log(f"Reference price set to ${self.reference_price:.2f}")

        # Subscribe to market data
        def handle_quote(quote):
            import asyncio
            asyncio.create_task(self.on_quote(quote))

        subscription = self.client.market_data.subscribe_quotes(
            symbols=[self.symbol],
            on_quote=handle_quote
        )
        subscription.open()

        self.state["subscription"] = subscription
        self.log("Subscribed to market data")

    async def on_quote(self, quote: Dict[str, Any]) -> None:
        """Process new quote"""
        if quote.get("symbol") != self.symbol:
            return

        current_price = quote.get("last")
        if not current_price or not self.reference_price:
            return

        # Calculate price change
        price_change = (current_price - self.reference_price) / self.reference_price

        # Check for buy signal
        if (
            price_change >= self.buy_threshold
            and not self.has_position(self.symbol)
        ):
            self.log(
                f"Momentum buy signal: {price_change:.1%} increase to ${current_price:.2f}"
            )
            await self._enter_long(current_price)

        # Check for sell signal (if in position)
        elif self.has_position(self.symbol) and self.entry_price:
            position_change = (current_price - self.entry_price) / self.entry_price

            # Sell on profit target or stop loss
            if position_change <= -self.sell_threshold:
                self.log(
                    f"Stop loss triggered: {position_change:.1%} at ${current_price:.2f}"
                )
                await self._exit_long()

            elif position_change >= (self.buy_threshold * 2):
                self.log(
                    f"Profit target reached: {position_change:.1%} at ${current_price:.2f}"
                )
                await self._exit_long()

        # Update reference price periodically
        if not self.has_position(self.symbol):
            self.reference_price = current_price

    async def on_candle(self, candle: Dict[str, Any]) -> None:
        """Process new candle (not used in this strategy)"""
        pass

    async def on_stop(self) -> None:
        """Clean up strategy"""
        self.log("Stopping strategy")

        # Close subscription
        subscription = self.state.get("subscription")
        if subscription:
            self.client.market_data.close_subscription(subscription)

        # Close any open positions
        if self.has_position(self.symbol):
            await self._exit_long()

    async def _enter_long(self, price: float) -> None:
        """Enter long position"""
        try:
            from ..api.orders import OrderAction, OrderType

            self.log(f"Entering long position: {self.quantity} shares at ~${price:.2f}")

            order = self.client.orders.place_order(
                account_number=None,
                symbol=self.symbol,
                quantity=self.quantity,
                action=OrderAction.BUY_TO_OPEN,
                order_type=OrderType.MARKET
            )

            self.entry_price = price
            self.log(f"Buy order placed: {order.get('id')}")

        except Exception as e:
            self.log(f"Failed to enter long: {e}", level="ERROR")
            await self.on_error(e)

    async def _exit_long(self) -> None:
        """Exit long position"""
        try:
            from ..api.orders import OrderAction, OrderType

            self.log(f"Exiting long position: {self.quantity} shares")

            order = self.client.orders.place_order(
                account_number=None,
                symbol=self.symbol,
                quantity=self.quantity,
                action=OrderAction.SELL_TO_CLOSE,
                order_type=OrderType.MARKET
            )

            self.entry_price = None
            self.log(f"Sell order placed: {order.get('id')}")

        except Exception as e:
            self.log(f"Failed to exit long: {e}", level="ERROR")
            await self.on_error(e)
