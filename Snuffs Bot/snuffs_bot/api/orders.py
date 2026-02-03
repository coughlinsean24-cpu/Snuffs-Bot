"""
Order execution and management functionality for Tastytrade API
"""

from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from loguru import logger


class OrderType(str, Enum):
    """Order types supported by Tastytrade"""
    LIMIT = "Limit"
    MARKET = "Market"
    STOP = "Stop"
    STOP_LIMIT = "Stop Limit"
    NOTIONAL_MARKET = "Notional Market"


class OrderAction(str, Enum):
    """Order actions"""
    BUY_TO_OPEN = "Buy to Open"
    BUY_TO_CLOSE = "Buy to Close"
    SELL_TO_OPEN = "Sell to Open"
    SELL_TO_CLOSE = "Sell to Close"


class TimeInForce(str, Enum):
    """Time in force values"""
    DAY = "Day"
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class OrderStatus(str, Enum):
    """Order status values"""
    RECEIVED = "Received"
    LIVE = "Live"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    EXPIRED = "Expired"
    REJECTED = "Rejected"
    PARTIAL = "PartialFill"


class OrderManager:
    """
    Manages order execution and management operations

    This class provides methods for:
    - Placing orders (market, limit, stop, etc.)
    - Managing existing orders (cancel, modify)
    - Viewing order history
    - Complex multi-leg orders
    """

    def __init__(self, client):
        """
        Initialize OrderManager

        Args:
            client: TastytradeClient instance
        """
        self.client = client

    def place_order(
        self,
        account_number: Optional[str],
        symbol: str,
        quantity: float,
        action: OrderAction,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place a single-leg order

        Args:
            account_number: Account number (uses default if not provided)
            symbol: Trading symbol (e.g., 'AAPL', 'SPY', '/ES')
            quantity: Number of shares/contracts
            action: Order action (BUY_TO_OPEN, SELL_TO_CLOSE, etc.)
            order_type: Type of order (MARKET, LIMIT, etc.)
            price: Limit price (required for LIMIT and STOP_LIMIT orders)
            stop_price: Stop price (required for STOP and STOP_LIMIT orders)
            time_in_force: How long the order remains active
            **kwargs: Additional order parameters

        Returns:
            Order confirmation dictionary

        Raises:
            ValueError: If required parameters are missing
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        # Validate price parameters
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"{order_type} orders require a price")

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError(f"{order_type} orders require a stop_price")

        logger.info(
            f"Placing {order_type.value} order: {action.value} {quantity} {symbol} @ "
            f"{price or 'market'}"
        )

        try:
            # Build order legs
            leg = {
                "instrument-type": self._get_instrument_type(symbol),
                "symbol": symbol,
                "quantity": str(abs(quantity)),
                "action": action.value
            }

            # Build order data
            order_data = {
                "time-in-force": time_in_force.value,
                "order-type": order_type.value,
                "legs": [leg]
            }

            # Add price if specified
            if price is not None:
                order_data["price"] = str(price)

            # Add stop price if specified
            if stop_price is not None:
                order_data["stop-trigger"] = str(stop_price)

            # Add any additional parameters
            order_data.update(kwargs)

            # Submit order
            response = self.client.session.api.post(
                f"/accounts/{account_num}/orders",
                order_data
            )

            order = response.get("data", {})
            order_id = order.get("id")

            logger.success(f"Order placed successfully. Order ID: {order_id}")
            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def place_multi_leg_order(
        self,
        account_number: Optional[str],
        legs: List[Dict[str, Any]],
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place a multi-leg order (spreads, combos, etc.)

        Args:
            account_number: Account number (uses default if not provided)
            legs: List of order leg dictionaries, each containing:
                - symbol: Trading symbol
                - quantity: Number of contracts
                - action: Order action
            order_type: Type of order
            price: Net credit/debit for the entire order
            time_in_force: How long the order remains active
            **kwargs: Additional order parameters

        Returns:
            Order confirmation dictionary

        Example:
            # Iron Condor
            legs = [
                {"symbol": "SPY 250117P00600000", "quantity": 1, "action": "Sell to Open"},
                {"symbol": "SPY 250117P00595000", "quantity": 1, "action": "Buy to Open"},
                {"symbol": "SPY 250117C00615000", "quantity": 1, "action": "Sell to Open"},
                {"symbol": "SPY 250117C00620000", "quantity": 1, "action": "Buy to Open"},
            ]
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        if not legs or len(legs) > 4:
            raise ValueError("Multi-leg orders must have 1-4 legs")

        logger.info(f"Placing {len(legs)}-leg order")

        try:
            # Build order legs
            order_legs = []
            for leg in legs:
                order_leg = {
                    "instrument-type": self._get_instrument_type(leg["symbol"]),
                    "symbol": leg["symbol"],
                    "quantity": str(abs(leg["quantity"])),
                    "action": leg["action"]
                }
                order_legs.append(order_leg)

            # Build order data
            order_data = {
                "time-in-force": time_in_force.value,
                "order-type": order_type.value,
                "legs": order_legs
            }

            # Add price if specified
            if price is not None:
                order_data["price"] = str(price)

            # Add any additional parameters
            order_data.update(kwargs)

            # Submit order
            response = self.client.session.api.post(
                f"/accounts/{account_num}/orders",
                order_data
            )

            order = response.get("data", {})
            order_id = order.get("id")

            logger.success(f"Multi-leg order placed successfully. Order ID: {order_id}")
            return order

        except Exception as e:
            logger.error(f"Failed to place multi-leg order: {e}")
            raise

    def get_orders(
        self,
        account_number: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders for an account

        Args:
            account_number: Account number (uses default if not provided)
            status: Filter by order status
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of order dictionaries
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.debug(f"Fetching orders for account {account_num}")

        try:
            params = {}
            if start_date:
                params["start-date"] = start_date
            if end_date:
                params["end-date"] = end_date

            response = self.client.session.api.get(
                f"/accounts/{account_num}/orders",
                params
            )
            orders = response.get("data", {}).get("items", [])

            # Filter by status if provided
            if status:
                orders = [o for o in orders if o.get("status") == status.value]

            logger.info(f"Retrieved {len(orders)} order(s) for {account_num}")
            return orders

        except Exception as e:
            logger.error(f"Failed to fetch orders for {account_num}: {e}")
            raise

    def get_order(
        self,
        order_id: str,
        account_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get details for a specific order

        Args:
            order_id: Order ID
            account_number: Account number (uses default if not provided)

        Returns:
            Order details dictionary
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.debug(f"Fetching order {order_id}")

        try:
            response = self.client.session.api.get(
                f"/accounts/{account_num}/orders/{order_id}"
            )
            order = response.get("data", {})

            logger.info(f"Retrieved order {order_id}")
            return order

        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            raise

    def cancel_order(
        self,
        order_id: str,
        account_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an order

        Args:
            order_id: Order ID to cancel
            account_number: Account number (uses default if not provided)

        Returns:
            Cancellation confirmation dictionary
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.info(f"Cancelling order {order_id}")

        try:
            response = self.client.session.api.delete(
                f"/accounts/{account_num}/orders/{order_id}"
            )

            logger.success(f"Order {order_id} cancelled successfully")
            return response.get("data", {})

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def replace_order(
        self,
        order_id: str,
        account_number: Optional[str] = None,
        new_price: Optional[float] = None,
        new_quantity: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Replace (modify) an existing order

        Args:
            order_id: Order ID to replace
            account_number: Account number (uses default if not provided)
            new_price: New limit price
            new_quantity: New quantity
            **kwargs: Other order parameters to update

        Returns:
            Updated order dictionary
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.info(f"Replacing order {order_id}")

        try:
            # Get existing order
            existing_order = self.get_order(order_id, account_num)

            # Build update data
            update_data = {}
            if new_price is not None:
                update_data["price"] = str(new_price)
            if new_quantity is not None:
                # Update quantity on all legs
                legs = existing_order.get("legs", [])
                for leg in legs:
                    leg["quantity"] = str(abs(new_quantity))
                update_data["legs"] = legs

            update_data.update(kwargs)

            # Submit replacement
            response = self.client.session.api.patch(
                f"/accounts/{account_num}/orders/{order_id}",
                update_data
            )

            logger.success(f"Order {order_id} replaced successfully")
            return response.get("data", {})

        except Exception as e:
            logger.error(f"Failed to replace order {order_id}: {e}")
            raise

    @staticmethod
    def _get_instrument_type(symbol: str) -> str:
        """
        Determine instrument type from symbol

        Args:
            symbol: Trading symbol

        Returns:
            Instrument type string
        """
        if symbol.startswith("/"):
            return "Future"
        elif " " in symbol and any(c in symbol for c in ["C", "P"]):
            return "Equity Option"
        elif "/" in symbol and not symbol.startswith("/"):
            return "Cryptocurrency"
        else:
            return "Equity"

    def __repr__(self) -> str:
        """String representation"""
        return f"OrderManager(client={self.client})"
