"""
Snuffs Bot - Tastytrade API Integration
A comprehensive trading bot for Tastytrade platform
"""

__version__ = "1.0.0"
__author__ = "Snuffs Bot Development Team"

from .api.client import TastytradeClient
from .config.settings import Settings

__all__ = ["TastytradeClient", "Settings"]
