"""
Main Tastytrade API Client wrapper
Provides a high-level interface to the Tastytrade API
"""

from typing import Optional, Dict, Any
from loguru import logger
from tastytrade_sdk import Tastytrade

from ..config.settings import Settings, get_settings
from .accounts import AccountManager
from .orders import OrderManager
from .market_data import MarketDataManager


class TastytradeClient:
    """
    Main client for interacting with the Tastytrade API

    This class wraps the official tastytrade-sdk and provides a cleaner,
    more organized interface for trading operations.

    Attributes:
        session: The underlying Tastytrade SDK session
        accounts: Account management interface
        orders: Order execution and management interface
        market_data: Market data streaming interface
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ):
        """
        Initialize the Tastytrade client

        Args:
            settings: Application settings (uses default if not provided)
            client_id: OAuth2 client ID (overrides settings)
            client_secret: OAuth2 client secret (overrides settings)
            refresh_token: OAuth2 refresh token (overrides settings)
        """
        self.settings = settings or get_settings()
        self._session: Optional[Tastytrade] = None
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

        # Initialize managers (lazy-loaded)
        self._accounts: Optional[AccountManager] = None
        self._orders: Optional[OrderManager] = None
        self._market_data: Optional[MarketDataManager] = None

        logger.info(
            f"TastytradeClient initialized for {self.settings.tastytrade_environment} environment"
        )

    @classmethod
    def from_env(cls, settings: Optional[Settings] = None) -> "TastytradeClient":
        """
        Create client from environment variables

        Environment variables required:
            - TT_CLIENT_ID: OAuth2 client ID
            - TT_CLIENT_SECRET: OAuth2 client secret
            - TT_REFRESH_TOKEN: OAuth2 refresh token
            - API_BASE_URL: (optional) Override API endpoint

        Args:
            settings: Application settings

        Returns:
            TastytradeClient instance
        """
        return cls(settings=settings)

    def connect(self) -> None:
        """
        Establish connection to Tastytrade API

        This method initializes the underlying SDK session using username/password.

        NOTE: This method guards against multiple connection attempts. If already
        connected, it will return early without creating a new session. This
        prevents IP blocking from Tastytrade due to rapid repeated login attempts.

        Raises:
            ValueError: If credentials are not configured
            ConnectionError: If unable to connect to API
        """
        # Guard against multiple connection attempts - Tastytrade may block IP
        # for rapid repeated login attempts
        if self.is_connected:
            logger.debug("Already connected to Tastytrade API, skipping reconnection")
            return

        try:
            logger.info("Connecting to Tastytrade API...")

            # Determine API base URL based on environment (SDK expects host without https://)
            api_base = self.settings.api_url.replace("https://", "")
            logger.debug(f"Using API base: {api_base}")

            # Create session and login with username/password
            session = Tastytrade(api_base_url=api_base)
            session.login(
                login=self.settings.tastytrade_username,
                password=self.settings.tastytrade_password
            )

            # Only set session after successful login
            self._session = session
            logger.success("Successfully connected to Tastytrade API")

        except Exception as e:
            self._session = None  # Ensure session is None on failure
            logger.error(f"Failed to connect to Tastytrade API: {e}")
            raise ConnectionError(f"Unable to connect to Tastytrade: {e}") from e

    def disconnect(self) -> None:
        """
        Disconnect from Tastytrade API and clean up resources
        """
        try:
            if self._session:
                # Close any active market data subscriptions
                if self._market_data:
                    self._market_data.close_all_subscriptions()

                # Logout from the session
                try:
                    self._session.logout()
                except Exception:
                    pass  # Ignore logout errors

                logger.info("Disconnected from Tastytrade API")
                self._session = None

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to API"""
        return self._session is not None

    @property
    def session(self) -> Tastytrade:
        """
        Get the underlying Tastytrade session

        Returns:
            Tastytrade session instance

        Raises:
            RuntimeError: If not connected to API
        """
        if not self._session:
            raise RuntimeError(
                "Not connected to Tastytrade API. Call connect() first."
            )
        return self._session

    @property
    def accounts(self) -> AccountManager:
        """
        Get the account manager interface

        Returns:
            AccountManager instance
        """
        if not self._accounts:
            self._accounts = AccountManager(self)
        return self._accounts

    @property
    def orders(self) -> OrderManager:
        """
        Get the order manager interface

        Returns:
            OrderManager instance
        """
        if not self._orders:
            self._orders = OrderManager(self)
        return self._orders

    @property
    def market_data(self) -> MarketDataManager:
        """
        Get the market data manager interface

        Returns:
            MarketDataManager instance
        """
        if not self._market_data:
            self._market_data = MarketDataManager(self)
        return self._market_data

    def __enter__(self) -> "TastytradeClient":
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.disconnect()

    def __repr__(self) -> str:
        """String representation"""
        status = "connected" if self.is_connected else "disconnected"
        return f"TastytradeClient({self.settings.tastytrade_environment}, {status})"
