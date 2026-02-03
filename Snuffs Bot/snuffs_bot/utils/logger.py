"""
Logging configuration for Snuffs Bot
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = "snuffs_bot.log",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> None:
    """
    Configure loguru logger with file and console output

    Args:
        log_file: Path to log file
        log_level: Minimum log level
        rotation: When to rotate log files
        retention: How long to keep old log files
    """
    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )

    # Add file logger
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip"
    )

    logger.info(f"Logger configured: level={log_level}, file={log_file}")
