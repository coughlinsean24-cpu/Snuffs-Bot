#!/usr/bin/env python3
"""
Database initialization script

Creates all tables and sets up initial data for the trading bot.
Run this script before starting the bot for the first time.

Usage:
    python scripts/init_database.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snuffs_bot.database.connection import init_database, create_all_tables, health_check
from snuffs_bot.database.models import RiskLimit
from snuffs_bot.database import get_db_session
from snuffs_bot.config import get_settings
from snuffs_bot.utils import setup_logger
from loguru import logger


def create_default_risk_limits(session):
    """Create default risk limits in the database"""
    logger.info("Creating default risk limits...")

    settings = get_settings()

    default_limits = [
        RiskLimit(
            limit_type="max_daily_loss",
            limit_value=settings.max_daily_loss,
            current_exposure=0.0,
            is_active=True
        ),
        RiskLimit(
            limit_type="max_position_size",
            limit_value=settings.max_position_size,
            current_exposure=0.0,
            is_active=True
        ),
        RiskLimit(
            limit_type="max_concurrent_positions",
            limit_value=settings.max_concurrent_positions,
            current_exposure=0.0,
            is_active=True
        ),
        RiskLimit(
            limit_type="max_portfolio_delta",
            limit_value=0.5,  # Maximum absolute portfolio delta
            current_exposure=0.0,
            is_active=True
        ),
    ]

    for limit in default_limits:
        # Check if limit already exists
        existing = session.query(RiskLimit).filter_by(limit_type=limit.limit_type).first()
        if not existing:
            session.add(limit)
            logger.info(f"Created risk limit: {limit.limit_type} = {limit.limit_value}")
        else:
            logger.info(f"Risk limit already exists: {limit.limit_type}")

    session.commit()
    logger.success("Default risk limits created")


def main():
    """Main initialization function"""
    print("=" * 70)
    print("Trading Bot Database Initialization")
    print("=" * 70)
    print()

    # Setup logging
    setup_logger(log_level="INFO")

    try:
        # Load settings
        logger.info("Loading configuration...")
        settings = get_settings()

        # Initialize database connection
        logger.info(f"Connecting to database...")
        init_database()

        # Health check
        logger.info("Performing health check...")
        if not health_check():
            logger.error("Database health check failed!")
            sys.exit(1)

        logger.success("Database connection successful")

        # Create all tables
        logger.info("Creating database tables...")
        create_all_tables()
        logger.success("All tables created")

        # Create default data
        session = get_db_session()
        try:
            create_default_risk_limits(session)
        finally:
            session.close()

        print()
        print("=" * 70)
        print("Database initialization completed successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Update your .env file with proper credentials")
        print("2. Run the trading bot: python main.py")
        print("3. Or start the dashboard: streamlit run dashboard/app.py")
        print()

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
