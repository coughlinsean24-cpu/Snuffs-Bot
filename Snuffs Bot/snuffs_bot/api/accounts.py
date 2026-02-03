"""
Account management functionality for Tastytrade API
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class AccountManager:
    """
    Manages account-related operations

    This class provides methods for:
    - Retrieving account information
    - Viewing balances and positions
    - Getting transaction history
    - Managing watchlists
    """

    def __init__(self, client):
        """
        Initialize AccountManager

        Args:
            client: TastytradeClient instance
        """
        self.client = client

    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all accounts associated with the user

        Returns:
            List of account dictionaries with account details

        Raises:
            RuntimeError: If client is not connected
        """
        logger.debug("Fetching user accounts")

        try:
            # Use the SDK's API interface to get accounts
            response = self.client.session.api.get("/customers/me/accounts")
            items = response.get("data", {}).get("items", [])

            # Flatten the nested account structure
            # API returns: {"account": {...}, "authority-level": "..."}
            # We return: {...account data..., "authority-level": "..."}
            accounts = []
            for item in items:
                account_data = item.get("account", {})
                account_data["authority-level"] = item.get("authority-level")
                accounts.append(account_data)

            logger.info(f"Retrieved {len(accounts)} account(s)")
            return accounts

        except Exception as e:
            logger.error(f"Failed to fetch accounts: {e}")
            raise

    def get_account(self, account_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Get details for a specific account

        Args:
            account_number: Account number (uses default if not provided)

        Returns:
            Account details dictionary

        Raises:
            ValueError: If account_number is not provided and no default is set
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.debug(f"Fetching account details for {account_num}")

        try:
            response = self.client.session.api.get(f"/accounts/{account_num}")
            account = response.get("data", {})

            logger.info(f"Retrieved account details for {account_num}")
            return account

        except Exception as e:
            logger.error(f"Failed to fetch account {account_num}: {e}")
            raise

    def get_balance(self, account_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account balance and buying power

        Args:
            account_number: Account number (uses default if not provided)

        Returns:
            Balance information dictionary including:
            - cash-balance: Available cash
            - net-liquidating-value: Total account value
            - buying-power: Available buying power
            - maintenance-requirement: Margin requirements
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.debug(f"Fetching balance for account {account_num}")

        try:
            response = self.client.session.api.get(f"/accounts/{account_num}/balances")
            balances = response.get("data", {})

            logger.info(f"Retrieved balance for {account_num}")
            return balances

        except Exception as e:
            logger.error(f"Failed to fetch balance for {account_num}: {e}")
            raise

    def get_positions(
        self,
        account_number: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current positions for an account

        Args:
            account_number: Account number (uses default if not provided)
            symbol: Optional symbol filter

        Returns:
            List of position dictionaries
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.debug(f"Fetching positions for account {account_num}")

        try:
            response = self.client.session.api.get(f"/accounts/{account_num}/positions")
            positions = response.get("data", {}).get("items", [])

            # Filter by symbol if provided
            if symbol:
                positions = [p for p in positions if p.get("symbol") == symbol]

            logger.info(f"Retrieved {len(positions)} position(s) for {account_num}")
            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions for {account_num}: {e}")
            raise

    def get_transactions(
        self,
        account_number: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history for an account

        Args:
            account_number: Account number (uses default if not provided)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of transactions to return

        Returns:
            List of transaction dictionaries
        """
        account_num = account_number or self.client.settings.default_account

        if not account_num:
            raise ValueError(
                "Account number must be provided or set in settings.default_account"
            )

        logger.debug(f"Fetching transactions for account {account_num}")

        try:
            params = {"per-page": limit}
            if start_date:
                params["start-date"] = start_date
            if end_date:
                params["end-date"] = end_date

            response = self.client.session.api.get(
                f"/accounts/{account_num}/transactions",
                params
            )
            transactions = response.get("data", {}).get("items", [])

            logger.info(f"Retrieved {len(transactions)} transaction(s) for {account_num}")
            return transactions

        except Exception as e:
            logger.error(f"Failed to fetch transactions for {account_num}: {e}")
            raise

    def get_watchlists(self) -> List[Dict[str, Any]]:
        """
        Get all watchlists for the user

        Returns:
            List of watchlist dictionaries
        """
        logger.debug("Fetching watchlists")

        try:
            response = self.client.session.api.get("/watchlists")
            watchlists = response.get("data", {}).get("items", [])

            logger.info(f"Retrieved {len(watchlists)} watchlist(s)")
            return watchlists

        except Exception as e:
            logger.error(f"Failed to fetch watchlists: {e}")
            raise

    def create_watchlist(
        self,
        name: str,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Create a new watchlist

        Args:
            name: Watchlist name
            symbols: List of symbols to add

        Returns:
            Created watchlist dictionary
        """
        logger.debug(f"Creating watchlist '{name}' with {len(symbols)} symbol(s)")

        try:
            data = {
                "name": name,
                "watchlist-entries": [{"symbol": symbol} for symbol in symbols]
            }

            response = self.client.session.api.post("/watchlists", data)
            watchlist = response.get("data", {})

            logger.info(f"Created watchlist '{name}'")
            return watchlist

        except Exception as e:
            logger.error(f"Failed to create watchlist '{name}': {e}")
            raise

    def __repr__(self) -> str:
        """String representation"""
        return f"AccountManager(client={self.client})"
