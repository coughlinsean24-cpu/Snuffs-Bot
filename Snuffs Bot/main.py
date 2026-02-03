#!/usr/bin/env python3
"""
Snuffs Bot - Main CLI Interface

A comprehensive trading bot for Tastytrade with account management,
order execution, market data streaming, and automated strategies.
"""

import sys
import argparse
import asyncio
from snuffs_bot import TastytradeClient
from snuffs_bot.config import get_settings
from snuffs_bot.utils import setup_logger, format_currency, format_percentage
from snuffs_bot.strategies import SimpleMovingAverageStrategy, MomentumStrategy, StrategyRunner
from loguru import logger


def cmd_accounts(client: TastytradeClient, args):
    """Show account information"""
    accounts = client.accounts.get_accounts()

    print(f"\n{'='*70}")
    print(f"{'ACCOUNTS':^70}")
    print(f"{'='*70}\n")

    for account in accounts:
        account_num = account.get("account-number")
        account_type = account.get("account-type-name")
        print(f"Account: {account_num}")
        print(f"Type: {account_type}")

        # Get balance
        balance = client.accounts.get_balance(account_num)
        cash = balance.get("cash-balance", 0)
        nlv = balance.get("net-liquidating-value", 0)
        buying_power = balance.get("derivative-buying-power", 0)

        print(f"Cash Balance: {format_currency(cash)}")
        print(f"Net Liquidating Value: {format_currency(nlv)}")
        print(f"Buying Power: {format_currency(buying_power)}")
        print(f"{'-'*70}")


def cmd_positions(client: TastytradeClient, args):
    """Show current positions"""
    account_num = args.account or client.settings.default_account
    if not account_num:
        print("Error: No account specified")
        return

    positions = client.accounts.get_positions(account_num)

    print(f"\n{'='*70}")
    print(f"{'POSITIONS':^70}")
    print(f"{'='*70}\n")

    if not positions:
        print("No positions")
        return

    print(f"{'Symbol':<15} {'Quantity':>10} {'Avg Price':>12} {'Current':>12} {'P&L':>15}")
    print(f"{'-'*70}")

    for position in positions:
        symbol = position.get("symbol", "")
        quantity = position.get("quantity", 0)
        avg_price = position.get("average-open-price", 0)
        current_price = position.get("close-price", avg_price)
        pnl = (current_price - avg_price) * quantity

        print(
            f"{symbol:<15} {quantity:>10} "
            f"{format_currency(avg_price):>12} "
            f"{format_currency(current_price):>12} "
            f"{format_currency(pnl):>15}"
        )


def cmd_orders(client: TastytradeClient, args):
    """Show recent orders"""
    account_num = args.account or client.settings.default_account
    if not account_num:
        print("Error: No account specified")
        return

    orders = client.orders.get_orders(account_num)

    print(f"\n{'='*70}")
    print(f"{'ORDERS':^70}")
    print(f"{'='*70}\n")

    if not orders:
        print("No orders")
        return

    for order in orders[:20]:  # Show last 20
        order_id = order.get("id", "")[:8]
        status = order.get("status", "")
        order_type = order.get("order-type", "")
        time_placed = order.get("received-at", "")[:19]

        legs = order.get("legs", [])
        leg_desc = ", ".join([
            f"{leg.get('action')} {leg.get('quantity')} {leg.get('symbol')}"
            for leg in legs
        ])

        print(f"ID: {order_id} | Status: {status:10} | Type: {order_type:10}")
        print(f"Time: {time_placed}")
        print(f"Legs: {leg_desc}")
        print(f"{'-'*70}")


def cmd_quote(client: TastytradeClient, args):
    """Get quote for symbols"""
    if not args.symbols:
        print("Error: No symbols specified")
        return

    quotes = client.market_data.get_quotes(args.symbols)

    print(f"\n{'='*70}")
    print(f"{'QUOTES':^70}")
    print(f"{'='*70}\n")

    print(f"{'Symbol':<10} {'Bid':>10} {'Ask':>10} {'Last':>10} {'Volume':>15}")
    print(f"{'-'*70}")

    for quote in quotes:
        symbol = quote.get("symbol", "")
        bid = quote.get("bid", 0)
        ask = quote.get("ask", 0)
        last = quote.get("last", 0)
        volume = quote.get("volume", 0)

        print(
            f"{symbol:<10} "
            f"{format_currency(bid):>10} "
            f"{format_currency(ask):>10} "
            f"{format_currency(last):>10} "
            f"{volume:>15,}"
        )


async def cmd_stream(client: TastytradeClient, args):
    """Stream real-time quotes"""
    if not args.symbols:
        print("Error: No symbols specified")
        return

    def handle_quote(event):
        symbol = event.get("symbol", "")
        bid = event.get("bid", 0)
        ask = event.get("ask", 0)
        last = event.get("last", 0)
        print(
            f"{symbol:6} | Bid: {format_currency(bid):>10} | "
            f"Ask: {format_currency(ask):>10} | Last: {format_currency(last):>10}"
        )

    print(f"\n{'='*70}")
    print(f"{'STREAMING QUOTES':^70}")
    print(f"{'='*70}\n")
    print(f"Streaming {', '.join(args.symbols)}... Press Ctrl+C to stop\n")

    subscription = client.market_data.subscribe_quotes(
        symbols=args.symbols,
        on_quote=handle_quote
    )
    subscription.open()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        subscription.close()


def cmd_chain(client: TastytradeClient, args):
    """Get option chain"""
    if not args.symbol:
        print("Error: No symbol specified")
        return

    chain = client.market_data.get_option_chain(args.symbol)
    expirations = chain.get("expirations", [])

    print(f"\n{'='*70}")
    print(f"{'OPTION CHAIN':^70}")
    print(f"{'='*70}\n")

    print(f"Underlying: {args.symbol}")
    print(f"Expirations available: {len(expirations)}\n")

    for exp in expirations[:10]:  # Show first 10
        exp_date = exp.get("expiration-date")
        dte = exp.get("days-to-expiration")
        print(f"  {exp_date} ({dte} DTE)")


async def cmd_strategy(client: TastytradeClient, args):
    """Run trading strategy"""
    print(f"\n{'='*70}")
    print(f"{'TRADING STRATEGY':^70}")
    print(f"{'='*70}\n")

    if not client.settings.is_sandbox or not client.settings.paper_trading:
        print("ERROR: Strategies can only run in sandbox with paper_trading=true")
        return

    runner = StrategyRunner(client)

    # Add strategies based on arguments
    if args.strategy_type == "sma":
        strategy = SimpleMovingAverageStrategy(
            client=client,
            symbol=args.symbol or "SPY",
            short_period=args.short_period or 10,
            long_period=args.long_period or 50,
            quantity=args.quantity or 10
        )
        runner.add_strategy(strategy)

    elif args.strategy_type == "momentum":
        strategy = MomentumStrategy(
            client=client,
            symbol=args.symbol or "SPY",
            buy_threshold=args.buy_threshold or 0.02,
            sell_threshold=args.sell_threshold or 0.02,
            quantity=args.quantity or 10
        )
        runner.add_strategy(strategy)

    else:
        print(f"Unknown strategy type: {args.strategy_type}")
        return

    print(f"Starting {args.strategy_type} strategy for {args.symbol or 'SPY'}")
    print("Press Ctrl+C to stop\n")

    try:
        await runner.run_forever()
    except KeyboardInterrupt:
        print("\nStopping strategy...")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Snuffs Bot - Tastytrade Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Accounts command
    subparsers.add_parser("accounts", help="Show account information")

    # Positions command
    pos_parser = subparsers.add_parser("positions", help="Show current positions")
    pos_parser.add_argument("--account", help="Account number")

    # Orders command
    ord_parser = subparsers.add_parser("orders", help="Show recent orders")
    ord_parser.add_argument("--account", help="Account number")

    # Quote command
    quote_parser = subparsers.add_parser("quote", help="Get quote for symbols")
    quote_parser.add_argument("symbols", nargs="+", help="Symbols to quote")

    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream real-time quotes")
    stream_parser.add_argument("symbols", nargs="+", help="Symbols to stream")

    # Chain command
    chain_parser = subparsers.add_parser("chain", help="Get option chain")
    chain_parser.add_argument("symbol", help="Underlying symbol")

    # Strategy command
    strat_parser = subparsers.add_parser("strategy", help="Run trading strategy")
    strat_parser.add_argument("strategy_type", choices=["sma", "momentum"])
    strat_parser.add_argument("--symbol", help="Symbol to trade")
    strat_parser.add_argument("--quantity", type=int, help="Quantity to trade")
    strat_parser.add_argument("--short-period", type=int, help="Short SMA period")
    strat_parser.add_argument("--long-period", type=int, help="Long SMA period")
    strat_parser.add_argument("--buy-threshold", type=float, help="Buy threshold")
    strat_parser.add_argument("--sell-threshold", type=float, help="Sell threshold")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logger(log_level=args.log_level)

    # Create client
    try:
        client = TastytradeClient.from_env()
        client.connect()

        # Execute command
        if args.command == "accounts":
            cmd_accounts(client, args)
        elif args.command == "positions":
            cmd_positions(client, args)
        elif args.command == "orders":
            cmd_orders(client, args)
        elif args.command == "quote":
            cmd_quote(client, args)
        elif args.command == "stream":
            asyncio.run(cmd_stream(client, args))
        elif args.command == "chain":
            cmd_chain(client, args)
        elif args.command == "strategy":
            asyncio.run(cmd_strategy(client, args))

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
