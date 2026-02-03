# Snuffs Bot - Tastytrade API Integration

A comprehensive Python trading bot for the Tastytrade platform with full API integration, real-time market data streaming, and automated trading strategies.

## Features

### Core Functionality
- **Account Management**: View balances, positions, transaction history, and manage watchlists
- **Order Execution**: Place market, limit, and stop orders with support for multi-leg strategies
- **Real-time Market Data**: Stream live quotes, Greeks, and candles via WebSocket
- **Trading Strategies**: Framework for building and running automated trading algorithms
- **Options Trading**: Full support for options chains, Greeks, and complex strategies

### Technical Features
- Official Tastytrade SDK integration
- Async/sync support for all operations
- Comprehensive error handling and logging
- Type-safe with Pydantic models
- Configuration management via environment variables
- Sandbox and production environment support

## Installation

### Prerequisites
- Python 3.8+
- Tastytrade account
- OAuth2 API credentials (client ID, client secret, refresh token)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Snuffs Bot"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```bash
TASTYTRADE_USERNAME=your-username
TASTYTRADE_PASSWORD=your-password
TASTYTRADE_ENVIRONMENT=sandbox  # or 'production'

# OAuth2 credentials (required for API access)
TT_CLIENT_ID=your-client-id
TT_CLIENT_SECRET=your-client-secret
TT_REFRESH_TOKEN=your-refresh-token

DEFAULT_ACCOUNT=your-account-number
PAPER_TRADING=true
```

## Quick Start

### Basic Usage

```python
from snuffs_bot import TastytradeClient
from snuffs_bot.utils import setup_logger

# Setup logging
setup_logger(log_level="INFO")

# Create and connect client
client = TastytradeClient.from_env()
client.connect()

# Get accounts
accounts = client.accounts.get_accounts()
print(f"Found {len(accounts)} account(s)")

# Get market data
quotes = client.market_data.get_quotes(["SPY", "AAPL"])
for quote in quotes:
    print(f"{quote['symbol']}: ${quote['last']}")

# Disconnect
client.disconnect()
```

### Streaming Market Data

```python
from snuffs_bot import TastytradeClient

def handle_quote(event):
    print(f"{event['symbol']}: ${event['last']}")

client = TastytradeClient.from_env()
client.connect()

# Subscribe to real-time quotes
subscription = client.market_data.subscribe_quotes(
    symbols=["SPY", "QQQ"],
    on_quote=handle_quote
)
subscription.open()

# Stream runs until stopped...
```

### Place Orders

```python
from snuffs_bot.api.orders import OrderAction, OrderType

# Place a market order
order = client.orders.place_order(
    account_number=None,  # Uses default from settings
    symbol="SPY",
    quantity=10,
    action=OrderAction.BUY_TO_OPEN,
    order_type=OrderType.MARKET
)

print(f"Order placed: {order['id']}")
```

### Trading Strategies

```python
import asyncio
from snuffs_bot.strategies import SimpleMovingAverageStrategy, StrategyRunner

# Create strategy
strategy = SimpleMovingAverageStrategy(
    client=client,
    symbol="SPY",
    short_period=10,
    long_period=50,
    quantity=10
)

# Run strategy
runner = StrategyRunner(client)
runner.add_strategy(strategy)
await runner.run_forever()
```

## Project Structure

```
Snuffs Bot/
├── snuffs_bot/              # Main package
│   ├── __init__.py
│   ├── api/                 # API integration
│   │   ├── client.py        # Main client wrapper
│   │   ├── accounts.py      # Account management
│   │   ├── orders.py        # Order execution
│   │   └── market_data.py   # Market data streaming
│   ├── strategies/          # Trading strategies
│   │   ├── base.py          # Base strategy class
│   │   └── examples.py      # Example strategies
│   ├── config/              # Configuration
│   │   └── settings.py      # Settings management
│   └── utils/               # Utilities
│       ├── logger.py        # Logging setup
│       └── helpers.py       # Helper functions
├── examples/                # Usage examples
│   ├── basic_usage.py       # Basic API usage
│   ├── streaming_data.py    # Market data streaming
│   └── strategy_example.py  # Trading strategies
├── tests/                   # Test suite
├── requirements.txt         # Dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

## API Reference

### TastytradeClient

Main client for API interaction.

```python
client = TastytradeClient.from_env()
client.connect()
```

**Methods:**
- `connect()` - Connect to Tastytrade API
- `disconnect()` - Disconnect and cleanup
- `is_connected` - Check connection status

**Properties:**
- `accounts` - AccountManager instance
- `orders` - OrderManager instance
- `market_data` - MarketDataManager instance

### AccountManager

Manage accounts and positions.

```python
# Get all accounts
accounts = client.accounts.get_accounts()

# Get account balance
balance = client.accounts.get_balance(account_number)

# Get positions
positions = client.accounts.get_positions(account_number)

# Get transactions
transactions = client.accounts.get_transactions(
    account_number,
    start_date="2025-01-01",
    limit=100
)

# Manage watchlists
watchlists = client.accounts.get_watchlists()
watchlist = client.accounts.create_watchlist("My List", ["SPY", "AAPL"])
```

### OrderManager

Execute and manage orders.

```python
# Place single-leg order
order = client.orders.place_order(
    account_number=account_num,
    symbol="SPY",
    quantity=10,
    action=OrderAction.BUY_TO_OPEN,
    order_type=OrderType.LIMIT,
    price=450.00,
    time_in_force=TimeInForce.DAY
)

# Place multi-leg order (spreads)
legs = [
    {"symbol": "SPY 250117P00600000", "quantity": 1, "action": "Sell to Open"},
    {"symbol": "SPY 250117P00595000", "quantity": 1, "action": "Buy to Open"},
]
order = client.orders.place_multi_leg_order(
    account_number=account_num,
    legs=legs,
    order_type=OrderType.LIMIT,
    price=0.50  # Net credit
)

# Get orders
orders = client.orders.get_orders(account_number)

# Cancel order
client.orders.cancel_order(order_id, account_number)

# Modify order
client.orders.replace_order(order_id, account_number, new_price=451.00)
```

### MarketDataManager

Stream and retrieve market data.

```python
# Get snapshot quotes
quotes = client.market_data.get_quotes(["SPY", "AAPL"])

# Stream real-time quotes
subscription = client.market_data.subscribe_quotes(
    symbols=["SPY"],
    on_quote=lambda event: print(event),
    on_greeks=lambda event: print(event)
)
subscription.open()

# Get option chain
chain = client.market_data.get_option_chain("SPY")

# Get historical candles
candles = client.market_data.get_candles(
    symbol="SPY",
    interval="1d",
    limit=100
)

# Search symbols
instruments = client.market_data.search_symbols("SPY")

# Get options for underlying
options = client.market_data.get_equity_options(
    symbol="SPY",
    expiration_date="2025-01-17"
)
```

### Trading Strategies

Build custom strategies by extending `BaseStrategy`.

```python
from snuffs_bot.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    async def on_start(self):
        # Initialize strategy
        pass

    async def on_quote(self, quote):
        # Process quote events
        pass

    async def on_candle(self, candle):
        # Process candle events
        pass

    async def on_stop(self):
        # Cleanup
        pass
```

## Configuration

### Environment Variables

- `TASTYTRADE_USERNAME` - Account username
- `TASTYTRADE_PASSWORD` - Account password
- `TASTYTRADE_ENVIRONMENT` - "sandbox" or "production"
- `TT_CLIENT_ID` - OAuth2 client ID
- `TT_CLIENT_SECRET` - OAuth2 client secret
- `TT_REFRESH_TOKEN` - OAuth2 refresh token
- `DEFAULT_ACCOUNT` - Default account number
- `PAPER_TRADING` - Enable paper trading mode (true/false)
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FILE` - Log file path

### Getting OAuth2 Credentials

1. Log in to your Tastytrade account
2. Navigate to the API settings page
3. Create a new OAuth2 application
4. Save your client ID, client secret, and refresh token
5. Add them to your `.env` file

## Examples

See the `examples/` directory for complete examples:

- **basic_usage.py** - Basic API operations
- **streaming_data.py** - Real-time market data
- **strategy_example.py** - Automated trading strategies

Run examples:
```bash
python examples/basic_usage.py
python examples/streaming_data.py
python examples/strategy_example.py
```

## Safety Features

- **Sandbox Environment**: Test with paper trading before going live
- **Paper Trading Mode**: Additional safety flag
- **Comprehensive Logging**: Track all operations
- **Error Handling**: Robust error recovery
- **Rate Limiting**: Respect API limits

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Strategies

1. Create a new file in `snuffs_bot/strategies/`
2. Extend `BaseStrategy`
3. Implement required methods
4. Add to `__init__.py`

## Resources

- **Tastytrade API Docs**: https://developer.tastytrade.com/
- **Official Python SDK**: https://github.com/tastytrade/tastytrade-sdk-python
- **API Support**: api.support@tastytrade.com

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes. Trading involves substantial risk. Always test thoroughly in sandbox environment before using with real money. The authors are not responsible for any financial losses.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

Built with the official Tastytrade SDK and powered by Python 🐍
