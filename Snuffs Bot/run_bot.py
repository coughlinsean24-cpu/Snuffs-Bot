#!/usr/bin/env python3
"""
Snuffs Bot Runner - Entry point for the trading bot

This script is called by the dashboard to start/stop the trading bot.
It reads the trading mode from dashboard_config.json and starts the
appropriate engine configuration.
"""

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables FIRST
from dotenv import load_dotenv
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Add project to path
sys.path.insert(0, str(project_root))

from loguru import logger


def setup_logging():
    """Configure logging for the bot"""
    logger.remove()

    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # File logging
    log_file = project_root / "logs" / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
    log_file.parent.mkdir(exist_ok=True)

    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )


def get_trading_mode() -> str:
    """Read trading mode from dashboard config

    Returns:
        'Paper' = Live data + simulated trades
        'Live' = Live data + real trades
    """
    config_path = project_root / "dashboard" / "dashboard_config.json"

    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("trading_mode", "Paper")
    except Exception as e:
        logger.warning(f"Could not read dashboard config: {e}")

    return "Paper"


async def run_trading_bot():
    """Run the main trading bot"""
    from snuffs_bot.engine.trading_engine import TradingEngine, EngineState
    from snuffs_bot.config.settings import get_settings
    from snuffs_bot.paper_trading.execution_coordinator import ExecutionMode

    # Get trading mode from dashboard config
    trading_mode = get_trading_mode()
    is_live_mode = trading_mode == "Live"

    settings = get_settings()
    engine = TradingEngine(settings)

    # Set the trading mode
    engine.paper_only_mode = not is_live_mode

    # Register event handlers
    def on_state_change(data):
        logger.info(f"Engine state changed to: {data['state']}")

    def on_decision(data):
        logger.info(
            f"Decision: {data['decision']} | "
            f"Confidence: {data['confidence']:.1%} | "
            f"Strategy: {data['strategy']}"
        )

    def on_trade(data):
        logger.success(
            f"Trade executed: {data['strategy']} | "
            f"Entry: ${data['entry_credit']:.2f}"
        )

    def on_error(data):
        logger.error(f"Engine error: {data['message']}")

    engine.register_handler("state_change", on_state_change)
    engine.register_handler("decision_made", on_decision)
    engine.register_handler("trade_executed", on_trade)
    engine.register_handler("error", on_error)

    # Log startup info
    logger.info("=" * 60)
    logger.info("SNUFFS BOT - Autonomous 0DTE SPY Options Trading")
    logger.info("=" * 60)

    if is_live_mode:
        logger.warning("MODE: LIVE TRADING - REAL MONEY AT RISK!")
    else:
        logger.info("MODE: PAPER TRADING - Simulated trades only")

    logger.info(f"Starting Capital: ${settings.starting_capital:,.0f}")
    logger.info(f"Max Daily Loss: ${settings.max_daily_loss:,.0f}")
    logger.info(f"Risk Per Trade: {settings.risk_per_trade_percent*100:.1f}%")
    logger.info(f"Trading Hours: {settings.trading_start_time} - {settings.trading_end_time} EST")
    logger.info("=" * 60)

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    # Start the engine
    await engine.start()

    try:
        # Keep running until shutdown signal
        while not shutdown_event.is_set() and engine.state == EngineState.RUNNING:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Stopping trading engine...")
        await engine.stop()
        logger.info("Trading engine stopped")


def main():
    """Main entry point"""
    setup_logging()

    logger.info("Starting Snuffs Bot...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Trading mode: {get_trading_mode()}")

    try:
        asyncio.run(run_trading_bot())
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
