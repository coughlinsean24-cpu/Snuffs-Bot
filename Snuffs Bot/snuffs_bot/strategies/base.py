"""
Base strategy framework for building trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger
from datetime import datetime
import asyncio


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies

    All custom strategies should inherit from this class and implement
    the required methods.
    """

    def __init__(self, client, name: str = "Strategy"):
        """
        Initialize strategy

        Args:
            client: TastytradeClient instance
            name: Strategy name
        """
        self.client = client
        self.name = name
        self.is_running = False
        self.positions: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}

        logger.info(f"Initialized strategy: {name}")

    @abstractmethod
    async def on_start(self) -> None:
        """
        Called when strategy starts

        Use this method to initialize indicators, subscribe to market data,
        load historical data, etc.
        """
        pass

    @abstractmethod
    async def on_quote(self, quote: Dict[str, Any]) -> None:
        """
        Called when a new quote is received

        Args:
            quote: Quote data dictionary
        """
        pass

    @abstractmethod
    async def on_candle(self, candle: Dict[str, Any]) -> None:
        """
        Called when a new candle is formed

        Args:
            candle: Candle data dictionary
        """
        pass

    @abstractmethod
    async def on_stop(self) -> None:
        """
        Called when strategy stops

        Use this method to clean up resources, close positions, etc.
        """
        pass

    async def on_order_fill(self, order: Dict[str, Any]) -> None:
        """
        Called when an order is filled

        Args:
            order: Order data dictionary
        """
        logger.info(f"Order filled: {order.get('id')}")

    async def on_position_update(self, position: Dict[str, Any]) -> None:
        """
        Called when a position is updated

        Args:
            position: Position data dictionary
        """
        symbol = position.get("symbol")
        self.positions[symbol] = position
        logger.debug(f"Position updated: {symbol}")

    async def on_error(self, error: Exception) -> None:
        """
        Called when an error occurs

        Args:
            error: Exception that occurred
        """
        logger.error(f"Strategy error: {error}")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Position dictionary or None
        """
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """
        Check if strategy has a position in symbol

        Args:
            symbol: Trading symbol

        Returns:
            True if position exists
        """
        return symbol in self.positions

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with strategy name

        Args:
            message: Message to log
            level: Log level
        """
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        """String representation"""
        status = "running" if self.is_running else "stopped"
        return f"{self.name}({status}, {len(self.positions)} positions)"


class StrategyRunner:
    """
    Runs and manages multiple trading strategies
    """

    def __init__(self, client):
        """
        Initialize strategy runner

        Args:
            client: TastytradeClient instance
        """
        self.client = client
        self.strategies: List[BaseStrategy] = []
        self.is_running = False

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a strategy to the runner

        Args:
            strategy: Strategy instance to add
        """
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")

    def remove_strategy(self, strategy: BaseStrategy) -> None:
        """
        Remove a strategy from the runner

        Args:
            strategy: Strategy instance to remove
        """
        if strategy in self.strategies:
            self.strategies.remove(strategy)
            logger.info(f"Removed strategy: {strategy.name}")

    async def start(self) -> None:
        """Start all strategies"""
        logger.info(f"Starting {len(self.strategies)} strategie(s)")

        self.is_running = True

        # Start all strategies
        for strategy in self.strategies:
            try:
                strategy.is_running = True
                await strategy.on_start()
                logger.success(f"Started strategy: {strategy.name}")

            except Exception as e:
                logger.error(f"Failed to start strategy {strategy.name}: {e}")
                await strategy.on_error(e)

    async def stop(self) -> None:
        """Stop all strategies"""
        logger.info(f"Stopping {len(self.strategies)} strategie(s)")

        self.is_running = False

        # Stop all strategies
        for strategy in self.strategies:
            try:
                strategy.is_running = False
                await strategy.on_stop()
                logger.success(f"Stopped strategy: {strategy.name}")

            except Exception as e:
                logger.error(f"Failed to stop strategy {strategy.name}: {e}")

    async def run_forever(self) -> None:
        """
        Run strategies forever (until stopped)

        This keeps the event loop running while strategies process
        market data through their callbacks.
        """
        await self.start()

        try:
            # Keep running until stopped
            while self.is_running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")

        finally:
            await self.stop()

    def __repr__(self) -> str:
        """String representation"""
        status = "running" if self.is_running else "stopped"
        return f"StrategyRunner({len(self.strategies)} strategies, {status})"
