"""
Helper functions and utilities
"""

import re
from typing import Dict, Optional
from datetime import datetime


def format_currency(amount, decimals: int = 2) -> str:
    """
    Format amount as currency

    Args:
        amount: Dollar amount (can be float, int, or string)
        decimals: Number of decimal places

    Returns:
        Formatted currency string

    Example:
        >>> format_currency(1234.56)
        '$1,234.56'
    """
    try:
        amount = float(amount) if amount else 0.0
    except (ValueError, TypeError):
        amount = 0.0
    return f"${amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage

    Args:
        value: Decimal value (0.05 = 5%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Example:
        >>> format_percentage(0.0525)
        '5.25%'
    """
    return f"{value * 100:.{decimals}f}%"


def parse_option_symbol(symbol: str) -> Optional[Dict[str, any]]:
    """
    Parse OCC option symbol format

    Args:
        symbol: Option symbol (e.g., 'SPY 250117P00600000')

    Returns:
        Dictionary with parsed components or None if invalid

    Example:
        >>> parse_option_symbol('SPY 250117P00600000')
        {
            'underlying': 'SPY',
            'expiration': '2025-01-17',
            'type': 'P',
            'strike': 600.0
        }
    """
    # OCC format: ROOT YYMMDDCP########
    # Example: SPY 250117P00600000
    pattern = r'^([A-Z/]+)\s+(\d{6})([CP])(\d{8})$'
    match = re.match(pattern, symbol)

    if not match:
        return None

    root, exp_date, opt_type, strike_str = match.groups()

    # Parse expiration date
    year = 2000 + int(exp_date[:2])
    month = int(exp_date[2:4])
    day = int(exp_date[4:6])
    expiration = f"{year:04d}-{month:02d}-{day:02d}"

    # Parse strike price (divide by 1000)
    strike = float(strike_str) / 1000.0

    return {
        "underlying": root,
        "expiration": expiration,
        "type": opt_type,
        "strike": strike,
        "full_symbol": symbol
    }


def calculate_pnl(entry_price: float, exit_price: float, quantity: int) -> Dict[str, float]:
    """
    Calculate profit/loss for a trade

    Args:
        entry_price: Entry price per share
        exit_price: Exit price per share
        quantity: Number of shares

    Returns:
        Dictionary with P&L details
    """
    pnl = (exit_price - entry_price) * quantity
    pnl_percent = ((exit_price - entry_price) / entry_price) if entry_price != 0 else 0

    return {
        "pnl": pnl,
        "pnl_percent": pnl_percent,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity
    }


def is_market_hours(timezone: str = "America/New_York") -> bool:
    """
    Check if current time is during market hours

    Args:
        timezone: Timezone to check (default: Eastern)

    Returns:
        True if during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
    """
    from datetime import datetime
    import pytz

    tz = pytz.timezone(timezone)
    now = datetime.now(tz)

    # Check if weekday (Monday=0, Friday=4)
    if now.weekday() > 4:
        return False

    # Check time (9:30 AM - 4:00 PM)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format

    Args:
        symbol: Trading symbol

    Returns:
        True if valid symbol format
    """
    # Equity: alphanumeric
    if re.match(r'^[A-Z]+$', symbol):
        return True

    # Future: /ESZ2, /NQH3
    if re.match(r'^/[A-Z]+[A-Z]\d{1,2}$', symbol):
        return True

    # Option: OCC format
    if parse_option_symbol(symbol):
        return True

    # Crypto: BTC/USD
    if re.match(r'^[A-Z]+/[A-Z]+$', symbol):
        return True

    return False
