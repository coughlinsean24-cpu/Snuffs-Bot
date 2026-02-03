#!/usr/bin/env python3
"""
Snuffs Bot - Autonomous 0DTE SPY Options Trading Bot

Main entry point for the trading bot.
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
from typing import List, Tuple

from loguru import logger

# Configure logging
def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging"""
    logger.remove()

    # Console logging
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File logging
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
        )


def validate_startup_requirements(live_mode: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate all requirements before starting the trading bot.

    Returns:
        Tuple of (success, list of error messages)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check settings can be loaded
    try:
        from snuffs_bot.config.settings import get_settings
        settings = get_settings()
    except Exception as e:
        errors.append(f"Failed to load settings: {e}")
        return False, errors

    # 2. Validate Tastytrade credentials
    if not settings.tastytrade_username:
        errors.append("TASTYTRADE_USERNAME is not set in environment")
    if not settings.tastytrade_password:
        errors.append("TASTYTRADE_PASSWORD is not set in environment")

    # 3. Validate Anthropic API key (optional - local AI/rule-based fallback available)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        warnings.append("ANTHROPIC_API_KEY not set - using local AI/rule-based trading only")
    elif not api_key.startswith("sk-"):
        warnings.append("ANTHROPIC_API_KEY format looks invalid (should start with 'sk-')")

    # 4. Validate starting capital and risk parameters
    if settings.starting_capital <= 0:
        errors.append(f"Invalid STARTING_CAPITAL: {settings.starting_capital} (must be > 0)")
    if settings.max_daily_loss <= 0:
        errors.append(f"Invalid MAX_DAILY_LOSS: {settings.max_daily_loss} (must be > 0)")
    if not (0 < settings.risk_per_trade_percent <= 1):
        errors.append(f"Invalid RISK_PER_TRADE_PERCENT: {settings.risk_per_trade_percent} (must be between 0 and 1)")

    # 5. Check database connectivity
    try:
        from snuffs_bot.database.connection import health_check
        if not health_check():
            warnings.append("Database health check failed - some features may not work")
    except ImportError:
        warnings.append("Database module not available - running without persistence")
    except Exception as e:
        warnings.append(f"Database connection issue: {e}")

    # 6. Live mode specific checks
    if live_mode:
        if settings.starting_capital < 1000:
            errors.append(f"Starting capital ${settings.starting_capital} is too low for live trading (min $1000)")
        # Verify account ID is set for live trading
        if not os.getenv("DEFAULT_ACCOUNT"):
            errors.append("DEFAULT_ACCOUNT is required for live trading")

    # Log warnings
    for warning in warnings:
        logger.warning(f"⚠️  {warning}")

    # Log errors
    for error in errors:
        logger.error(f"❌ {error}")

    return len(errors) == 0, errors


async def run_trading_bot(paper_only: bool = True):
    """Run the main trading bot"""
    from snuffs_bot.engine.trading_engine import TradingEngine, EngineState
    from snuffs_bot.config.settings import get_settings

    settings = get_settings()
    engine = TradingEngine(settings)

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

    # Start engine
    logger.info("=" * 60)
    logger.info("SNUFFS BOT - Autonomous 0DTE SPY Options Trading")
    logger.info("=" * 60)
    logger.info(f"Mode: {'PAPER ONLY' if paper_only else 'LIVE TRADING'}")
    logger.info(f"Starting Capital: ${settings.starting_capital:,.0f}")
    logger.info(f"Max Daily Loss: ${settings.max_daily_loss:,.0f}")
    logger.info(f"Risk Per Trade: {settings.risk_per_trade_percent*100:.1f}%")
    logger.info("=" * 60)

    await engine.start()

    try:
        # Keep running until interrupted
        while engine.state == EngineState.RUNNING:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await engine.stop()


async def run_dashboard():
    """Run the Streamlit dashboard"""
    import subprocess

    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard",
        "app.py"
    )

    logger.info("Starting dashboard...")
    process = subprocess.Popen(
        ["streamlit", "run", dashboard_path, "--server.port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    logger.info("Dashboard running at http://localhost:8501")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        process.terminate()


async def run_backtest(start_date: str, end_date: str):
    """Run a backtest simulation"""
    logger.info(f"Running backtest from {start_date} to {end_date}")
    logger.warning("Backtest functionality not yet implemented")
    # TODO: Implement backtesting


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Snuffs Bot - Autonomous 0DTE SPY Options Trading"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Start the trading bot")
    trade_parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Run in paper trading mode only (default)"
    )
    trade_parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (USE WITH CAUTION)"
    )
    trade_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    trade_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start the web dashboard")
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Dashboard port"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run system tests")
    test_parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run tests for specific phase"
    )
    test_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    if args.command == "trade":
        setup_logging(args.verbose, args.log_file)
        paper_only = not args.live

        # Validate startup requirements
        logger.info("Validating startup requirements...")
        valid, errors = validate_startup_requirements(live_mode=args.live)
        if not valid:
            logger.error("=" * 60)
            logger.error("STARTUP VALIDATION FAILED")
            logger.error("=" * 60)
            logger.error("Please fix the following errors before starting:")
            for i, error in enumerate(errors, 1):
                logger.error(f"  {i}. {error}")
            logger.error("=" * 60)
            sys.exit(1)
        logger.success("✓ All startup requirements validated")

        if args.live:
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK!")
            logger.warning("=" * 60)
            response = input("Are you sure you want to proceed? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Aborted")
                return
        asyncio.run(run_trading_bot(paper_only))

    elif args.command == "dashboard":
        setup_logging()
        asyncio.run(run_dashboard())

    elif args.command == "test":
        setup_logging()
        run_tests(args.phase, args.all)

    elif args.command == "status":
        setup_logging()
        check_status()

    elif args.command == "backtest":
        setup_logging()
        asyncio.run(run_backtest(args.start, args.end))

    else:
        parser.print_help()


def run_tests(phase: int = None, run_all: bool = False):
    """Run system tests"""
    import subprocess

    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"
    )

    if run_all or phase is None:
        phases = [1, 2, 3, 4, 5, 6, 7]
    else:
        phases = [phase]

    for p in phases:
        script = f"test_phase{p}.py"
        script_path = os.path.join(scripts_dir, script)

        if os.path.exists(script_path):
            logger.info(f"Running Phase {p} tests...")
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=os.path.dirname(scripts_dir)
            )
            if result.returncode != 0:
                logger.error(f"Phase {p} tests failed")
                if not run_all:
                    return
        else:
            logger.warning(f"Test script not found: {script}")


def check_status():
    """Check system status"""
    from snuffs_bot.config.settings import get_settings
    from snuffs_bot.database.connection import health_check

    logger.info("=" * 60)
    logger.info("SYSTEM STATUS CHECK")
    logger.info("=" * 60)

    # Settings
    try:
        settings = get_settings()
        logger.success("Settings: OK")
        logger.info(f"  Starting Capital: ${settings.starting_capital:,.0f}")
        logger.info(f"  Paper Mode: {not settings.live_trading_enabled}")
    except Exception as e:
        logger.error(f"Settings: FAILED - {e}")

    # Database
    try:
        if health_check():
            logger.success("Database: OK")
        else:
            logger.error("Database: FAILED")
    except Exception as e:
        logger.error(f"Database: FAILED - {e}")

    # Components
    components = [
        ("AI Orchestrator", "snuffs_bot.ai.orchestrator", "AIOrchestrator"),
        ("Strategy Selector", "snuffs_bot.strategies.zero_dte.strategy_selector", "StrategySelector"),
        ("Risk Guardrails", "snuffs_bot.risk.guardrails", "RiskGuardrails"),
        ("Execution Coordinator", "snuffs_bot.paper_trading.execution_coordinator", "ExecutionCoordinator"),
        ("Learning Scheduler", "snuffs_bot.learning.scheduler", "LearningScheduler"),
        ("Trading Engine", "snuffs_bot.engine.trading_engine", "TradingEngine"),
    ]

    for name, module, cls in components:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            logger.success(f"{name}: OK")
        except Exception as e:
            logger.error(f"{name}: FAILED - {e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
