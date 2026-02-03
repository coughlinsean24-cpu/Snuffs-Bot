"""
Risk Guardrails System

Non-negotiable risk limits that override AI decisions.
These are hard stops that protect capital regardless of what
the AI agents recommend.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional
import pytz

from loguru import logger

from ..config.settings import get_settings
from ..database.models import RiskLimit
from ..database.connection import db_session_scope


class RiskViolationType(Enum):
    """Types of risk violations"""
    DAILY_LOSS_EXCEEDED = "DAILY_LOSS_EXCEEDED"
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    MAX_POSITIONS_EXCEEDED = "MAX_POSITIONS_EXCEEDED"
    PORTFOLIO_DELTA_EXCEEDED = "PORTFOLIO_DELTA_EXCEEDED"
    OUTSIDE_TRADING_HOURS = "OUTSIDE_TRADING_HOURS"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"
    RISK_PER_TRADE_EXCEEDED = "RISK_PER_TRADE_EXCEEDED"


@dataclass
class RiskViolation:
    """Details of a risk violation"""
    violation_type: RiskViolationType
    limit_value: float
    current_value: float
    message: str
    severity: str = "HIGH"  # HIGH, MEDIUM, LOW
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        return f"{self.violation_type.value}: {self.message}"


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    passed: bool
    violations: List[RiskViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_violation(self, violation: RiskViolation) -> None:
        """Add a violation and mark as failed"""
        self.violations.append(violation)
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't fail the check)"""
        self.warnings.append(message)

    @property
    def is_critical(self) -> bool:
        """Check if any violations are critical"""
        return any(v.severity == "HIGH" for v in self.violations)


class RiskGuardrails:
    """
    Risk management guardrails

    These limits cannot be overridden by AI decisions.
    They protect capital and ensure responsible trading.
    """

    # Trading hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    TRADING_CUTOFF = time(15, 45)  # Stop new trades before close

    def __init__(self):
        """Initialize guardrails from settings and database"""
        self.settings = get_settings()
        self._load_limits()

        logger.info("Risk guardrails initialized")

    def _load_limits(self) -> None:
        """Load limits from settings (can be overridden by database)"""
        self.max_daily_loss = self.settings.max_daily_loss
        self.max_position_size = self.settings.max_position_size
        self.max_concurrent_positions = self.settings.max_concurrent_positions
        self.risk_per_trade_percent = self.settings.risk_per_trade_percent
        self.max_portfolio_delta = 0.5  # Default

        # Try to load from database
        try:
            self._load_from_database()
        except Exception as e:
            logger.warning(f"Could not load limits from database: {e}")

    def _load_from_database(self) -> None:
        """Load risk limits from database"""
        with db_session_scope() as session:
            limits = session.query(RiskLimit).filter(RiskLimit.is_active == True).all()

            for limit in limits:
                if limit.limit_type == "max_daily_loss":
                    self.max_daily_loss = float(limit.limit_value)
                elif limit.limit_type == "max_position_size":
                    self.max_position_size = float(limit.limit_value)
                elif limit.limit_type == "max_concurrent_positions":
                    self.max_concurrent_positions = int(limit.limit_value)
                elif limit.limit_type == "max_portfolio_delta":
                    self.max_portfolio_delta = float(limit.limit_value)

            logger.debug(f"Loaded {len(limits)} risk limits from database")

    def check_all(
        self,
        proposed_trade: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> RiskCheckResult:
        """
        Run all risk checks against a proposed trade

        Args:
            proposed_trade: Trade details including max_loss, contracts
            portfolio_state: Current portfolio state including daily_pnl, open_positions

        Returns:
            RiskCheckResult with pass/fail and any violations
        """
        result = RiskCheckResult(passed=True)

        # Check each guardrail
        self._check_daily_loss(result, portfolio_state)
        self._check_position_size(result, proposed_trade, portfolio_state)
        self._check_concurrent_positions(result, portfolio_state)
        self._check_trading_hours(result)
        self._check_buying_power(result, proposed_trade, portfolio_state)
        self._check_risk_per_trade(result, proposed_trade, portfolio_state)
        self._check_portfolio_delta(result, proposed_trade, portfolio_state)

        # Log result
        if result.passed:
            logger.debug("All risk checks passed")
        else:
            for violation in result.violations:
                logger.warning(f"Risk violation: {violation}")

        return result

    def _check_daily_loss(
        self,
        result: RiskCheckResult,
        portfolio_state: Dict[str, Any]
    ) -> None:
        """Check if daily loss limit has been exceeded"""
        # Skip daily loss check for paper trading - we want to learn from all trades
        is_paper_trading = self.settings.paper_trading
        if is_paper_trading:
            result.metadata["daily_loss_remaining"] = float('inf')
            result.add_warning("Daily loss limit bypassed for paper trading learning")
            return
            
        daily_pnl = portfolio_state.get("daily_pnl", 0)

        if daily_pnl <= -self.max_daily_loss:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.DAILY_LOSS_EXCEEDED,
                limit_value=self.max_daily_loss,
                current_value=abs(daily_pnl),
                message=f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.max_daily_loss:.2f}",
                severity="HIGH"
            ))
        elif daily_pnl <= -self.max_daily_loss * 0.8:
            result.add_warning(
                f"Daily loss ${abs(daily_pnl):.2f} approaching limit ${self.max_daily_loss:.2f}"
            )

        result.metadata["daily_loss_remaining"] = self.max_daily_loss + daily_pnl

    def _check_position_size(
        self,
        result: RiskCheckResult,
        proposed_trade: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> None:
        """Check if position size is within limits"""
        max_loss = proposed_trade.get("max_loss", 0)

        if max_loss > self.max_position_size:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE_EXCEEDED,
                limit_value=self.max_position_size,
                current_value=max_loss,
                message=f"Position max loss ${max_loss:.2f} exceeds limit ${self.max_position_size:.2f}",
                severity="HIGH"
            ))

    def _check_concurrent_positions(
        self,
        result: RiskCheckResult,
        portfolio_state: Dict[str, Any]
    ) -> None:
        """Check if max concurrent positions would be exceeded"""
        open_positions = portfolio_state.get("open_positions", 0)

        if open_positions >= self.max_concurrent_positions:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.MAX_POSITIONS_EXCEEDED,
                limit_value=self.max_concurrent_positions,
                current_value=open_positions,
                message=f"Already have {open_positions} open positions (max: {self.max_concurrent_positions})",
                severity="MEDIUM"
            ))

        result.metadata["positions_available"] = self.max_concurrent_positions - open_positions

    def _check_trading_hours(self, result: RiskCheckResult) -> None:
        """Check if within trading hours"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        current_time = now.time()

        # Check if market is open
        if current_time < self.MARKET_OPEN:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.OUTSIDE_TRADING_HOURS,
                limit_value=0,
                current_value=0,
                message=f"Market not open yet (opens at {self.MARKET_OPEN})",
                severity="HIGH"
            ))
        elif current_time >= self.TRADING_CUTOFF:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.OUTSIDE_TRADING_HOURS,
                limit_value=0,
                current_value=0,
                message=f"Past trading cutoff time ({self.TRADING_CUTOFF})",
                severity="MEDIUM"
            ))

        # Check if weekday
        if now.weekday() >= 5:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.OUTSIDE_TRADING_HOURS,
                limit_value=0,
                current_value=0,
                message="Market closed on weekends",
                severity="HIGH"
            ))

    def _check_buying_power(
        self,
        result: RiskCheckResult,
        proposed_trade: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> None:
        """Check if sufficient buying power exists"""
        buying_power = portfolio_state.get("buying_power", 0)
        required_bp = proposed_trade.get("buying_power_required", 0)

        # For credit spreads, BP required is typically max loss
        if required_bp == 0:
            required_bp = proposed_trade.get("max_loss", 0)

        if required_bp > buying_power:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.INSUFFICIENT_BUYING_POWER,
                limit_value=buying_power,
                current_value=required_bp,
                message=f"Need ${required_bp:.2f} BP, only ${buying_power:.2f} available",
                severity="HIGH"
            ))

    def _check_risk_per_trade(
        self,
        result: RiskCheckResult,
        proposed_trade: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> None:
        """Check if trade risk is within percentage limit"""
        account_value = portfolio_state.get("account_value", 100000)
        max_loss = proposed_trade.get("max_loss", 0)

        max_allowed_risk = account_value * self.risk_per_trade_percent
        risk_percent = (max_loss / account_value) * 100 if account_value else 0

        if max_loss > max_allowed_risk:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.RISK_PER_TRADE_EXCEEDED,
                limit_value=max_allowed_risk,
                current_value=max_loss,
                message=f"Trade risk {risk_percent:.1f}% exceeds limit {self.risk_per_trade_percent*100:.1f}%",
                severity="MEDIUM"
            ))

        result.metadata["risk_per_trade_percent"] = risk_percent

    def _check_portfolio_delta(
        self,
        result: RiskCheckResult,
        proposed_trade: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> None:
        """Check if portfolio delta would exceed limit"""
        current_delta = portfolio_state.get("total_delta", 0)
        trade_delta = proposed_trade.get("position_delta", 0)
        new_delta = current_delta + trade_delta

        if abs(new_delta) > self.max_portfolio_delta:
            result.add_violation(RiskViolation(
                violation_type=RiskViolationType.PORTFOLIO_DELTA_EXCEEDED,
                limit_value=self.max_portfolio_delta,
                current_value=abs(new_delta),
                message=f"Portfolio delta {new_delta:.2f} would exceed limit {self.max_portfolio_delta}",
                severity="MEDIUM"
            ))

    def check_trading_hours(self) -> RiskCheckResult:
        """
        Public method to check if within trading hours

        Returns:
            RiskCheckResult with pass/fail for trading hours
        """
        result = RiskCheckResult(passed=True)
        self._check_trading_hours(result)
        return result

    def get_limits_summary(self) -> Dict[str, Any]:
        """Get summary of all limits"""
        return {
            "max_daily_loss": self.max_daily_loss,
            "max_position_size": self.max_position_size,
            "max_concurrent_positions": self.max_concurrent_positions,
            "risk_per_trade_percent": self.risk_per_trade_percent * 100,
            "max_portfolio_delta": self.max_portfolio_delta,
            "trading_cutoff": str(self.TRADING_CUTOFF),
        }

    def update_limit(
        self,
        limit_type: str,
        new_value: float
    ) -> bool:
        """
        Update a risk limit in the database

        Args:
            limit_type: Type of limit to update
            new_value: New limit value

        Returns:
            True if updated successfully
        """
        try:
            with db_session_scope() as session:
                limit = session.query(RiskLimit).filter(
                    RiskLimit.limit_type == limit_type
                ).first()

                if limit:
                    limit.limit_value = new_value
                    limit.updated_at = datetime.utcnow()
                    logger.info(f"Updated {limit_type} to {new_value}")
                else:
                    # Create new limit
                    new_limit = RiskLimit(
                        limit_type=limit_type,
                        limit_value=new_value,
                        is_active=True
                    )
                    session.add(new_limit)
                    logger.info(f"Created new limit {limit_type} = {new_value}")

            # Reload limits
            self._load_limits()
            return True

        except Exception as e:
            logger.error(f"Failed to update limit: {e}")
            return False

    def emergency_stop(self) -> None:
        """
        Emergency stop - set all limits to prevent trading

        Call this when a critical situation requires immediate stop
        """
        logger.critical("EMERGENCY STOP ACTIVATED")

        self.max_daily_loss = 0
        self.max_concurrent_positions = 0
        self.max_position_size = 0

        # Update database
        try:
            self.update_limit("max_daily_loss", 0)
            self.update_limit("max_concurrent_positions", 0)
        except Exception:
            pass

        logger.critical("All trading disabled - manual intervention required")
