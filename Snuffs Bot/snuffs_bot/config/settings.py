"""
Configuration settings for Snuffs Bot using Pydantic
"""

from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Tastytrade Credentials
    tastytrade_username: str = Field(
        ...,
        description="Tastytrade account username/email"
    )
    tastytrade_password: str = Field(
        ...,
        description="Tastytrade account password"
    )

    # Environment Configuration
    tastytrade_environment: Literal["sandbox", "production"] = Field(
        default="sandbox",
        description="API environment to use"
    )

    # API URLs
    tastytrade_cert_url: str = Field(
        default="https://api.cert.tastyworks.com",
        description="Sandbox/Certification API URL"
    )
    tastytrade_prod_url: str = Field(
        default="https://api.tastyworks.com",
        description="Production API URL"
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default="snuffs_bot.log",
        description="Log file path"
    )

    # Trading Configuration
    default_account: Optional[str] = Field(
        default=None,
        description="Default account number to use"
    )
    paper_trading: bool = Field(
        default=True,
        description="Enable paper trading mode"
    )

    # Session Configuration
    session_timeout: int = Field(
        default=14 * 60,  # 14 minutes (tokens expire at 15)
        description="Session timeout in seconds"
    )
    auto_refresh_token: bool = Field(
        default=True,
        description="Automatically refresh access tokens"
    )

    # Rate Limiting
    max_requests_per_second: int = Field(
        default=10,
        description="Maximum API requests per second"
    )

    # Anthropic Claude Configuration
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude AI decision engine"
    )
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model to use (claude-sonnet-4-20250514, claude-opus-4-20250514)"
    )
    claude_max_tokens: int = Field(
        default=2000,
        description="Maximum response tokens from Claude"
    )
    claude_temperature: float = Field(
        default=0.7,
        description="Temperature for Claude responses (0.0-1.0)"
    )

    # Data Directory (for cloud deployments like Render)
    data_dir: str = Field(
        default="data",
        description="Base directory for all data storage (databases, learnings, etc.)"
    )

    # Local AI Configuration
    use_local_ai: bool = Field(
        default=False,
        description="Use local XGBoost AI instead of Claude API"
    )
    local_ai_data_dir: str = Field(
        default="",  # Will be computed from data_dir
        description="Directory to store local AI training data"
    )

    @field_validator("local_ai_data_dir", mode="before")
    @classmethod
    def set_local_ai_data_dir(cls, v, info):
        """Set local_ai_data_dir based on data_dir if not explicitly set"""
        if not v:
            data_dir = info.data.get("data_dir", "data")
            return f"{data_dir}/local_ai"
        return v
    local_ai_min_trades: int = Field(
        default=100,
        description="Minimum trades before training local AI model (100+ required for 68 features)"
    )
    local_ai_retrain_interval: int = Field(
        default=20,
        description="Number of new trades before retraining model"
    )

    # Database Configuration
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL (postgresql://user:pass@host:port/db)"
    )
    database_pool_size: int = Field(
        default=5,
        description="Database connection pool size"
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for real-time updates"
    )

    # Capital Configuration
    starting_capital: float = Field(
        default=100000.0,
        description="Starting capital for paper trading and position sizing"
    )
    live_trading_enabled: bool = Field(
        default=False,
        description="Enable live trading (USE WITH CAUTION)"
    )

    # 0DTE Trading Configuration
    enable_paper_trading_simulator: bool = Field(
        default=True,
        description="Enable parallel paper trading simulation"
    )
    paper_trade_percentage: float = Field(
        default=0.5,
        description="Percentage of trades to run as paper (0.0-1.0)"
    )

    # Risk Management Configuration
    max_daily_loss: float = Field(
        default=500.0,
        description="Maximum daily loss in dollars"
    )
    max_position_size: float = Field(
        default=5000.0,
        description="Maximum position size (risk) in dollars"
    )
    max_concurrent_positions: int = Field(
        default=4,
        description="Maximum number of concurrent open positions"
    )
    risk_per_trade_percent: float = Field(
        default=0.04,
        description="Risk percentage per trade (0.04 = 4%)"
    )

    # 0DTE Strategy Parameters
    min_profit_target: float = Field(
        default=0.15,
        description="Minimum profit target as % of max profit (0.15 = 15%)"
    )
    max_stop_loss: float = Field(
        default=0.25,
        description="Maximum stop loss as % of max loss (0.25 = 25%)"
    )
    trading_start_time: str = Field(
        default="09:35",
        description="Earliest entry time (HH:MM EST)"
    )
    trading_end_time: str = Field(
        default="12:00",
        description="Latest entry time / force exit (HH:MM EST)"
    )
    wing_width: int = Field(
        default=5,
        description="Spread wing width in dollars for credit spreads"
    )
    delta_target: float = Field(
        default=0.10,
        description="Target delta for short strikes (0.10 = 10 delta)"
    )

    # OTM Option Selection Configuration
    long_option_delta: float = Field(
        default=0.35,
        description="Target delta for long options (0.35 = OTM, 0.50 = ATM). Lower = cheaper, faster moves but lower probability"
    )
    prefer_otm_options: bool = Field(
        default=True,
        description="Prefer OTM options for faster percentage moves and lower capital requirement"
    )
    min_option_delta: float = Field(
        default=0.15,
        description="Minimum delta to consider (avoid very far OTM with low probability)"
    )
    max_option_delta: float = Field(
        default=0.55,
        description="Maximum delta to consider (avoid deep ITM with high capital requirement)"
    )

    # Dollar-Based Exit Thresholds (for single-contract protection)
    enable_dollar_exits: bool = Field(
        default=True,
        description="Enable dollar-based exit thresholds in addition to percentage"
    )
    min_dollar_profit_target: float = Field(
        default=30.0,
        description="Minimum dollar profit to take (regardless of percentage)"
    )
    max_dollar_loss: float = Field(
        default=75.0,
        description="Maximum dollar loss per trade (hard stop)"
    )
    single_contract_aggressive_exits: bool = Field(
        default=True,
        description="Use tighter exits for single-contract positions"
    )
    single_contract_profit_target_pct: float = Field(
        default=0.12,
        description="Profit target % for single contracts (12% = faster exits)"
    )
    single_contract_stop_loss_pct: float = Field(
        default=0.15,
        description="Stop loss % for single contracts (15% = quicker protection)"
    )

    # Dashboard Configuration
    dashboard_port: int = Field(
        default=8501,
        description="Streamlit dashboard port"
    )
    dashboard_auto_refresh_seconds: int = Field(
        default=5,
        description="Dashboard auto-refresh interval in seconds"
    )

    # AI Learning Configuration
    enable_continuous_learning: bool = Field(
        default=True,
        description="Enable AI continuous learning from trade results"
    )
    learning_context_window: int = Field(
        default=10,
        description="Number of recent trades to include in AI context"
    )

    @field_validator("tastytrade_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is valid"""
        if v not in ["sandbox", "production"]:
            raise ValueError("Environment must be 'sandbox' or 'production'")
        return v

    @property
    def api_url(self) -> str:
        """Get the appropriate API URL based on environment"""
        return (
            self.tastytrade_cert_url
            if self.tastytrade_environment == "sandbox"
            else self.tastytrade_prod_url
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.tastytrade_environment == "production"

    @property
    def is_sandbox(self) -> bool:
        """Check if running in sandbox environment"""
        return self.tastytrade_environment == "sandbox"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get or create the global settings instance

    Args:
        reload: Force reload settings from environment

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None or reload:
        _settings = Settings()

    return _settings
