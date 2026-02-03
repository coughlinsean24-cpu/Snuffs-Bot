"""
Long Put Strategy for 0DTE SPY Options

Buy SPY puts when bearish - profit from downward price movement.
Simple directional strategy with defined risk (premium paid).

Supports both ATM and OTM strike selection:
- OTM options: Cheaper premiums, faster % moves, lower probability
- ATM options: More expensive, slower % moves, higher probability
"""

from datetime import datetime, time
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from .base_zero_dte import (
    ZeroDTEStrategy,
    SpreadPosition,
    OptionLeg,
    PositionStatus,
    ExitReason
)


class LongPutStrategy(ZeroDTEStrategy):
    """
    Long Put Strategy for 0DTE SPY Options

    Strategy: Buy a put option when bearish on SPY

    Characteristics:
    - High profit potential (as SPY falls)
    - Max loss = premium paid
    - Theta works against you (time decay hurts)
    - Need strong downward move to profit
    - Best when VIX is low (cheaper premiums)
    
    Strike Selection (configurable):
    - OTM (delta 0.20-0.35): Cheaper, faster % moves, more leverage
    - ATM (delta 0.45-0.55): More expensive, slower % moves, higher win rate
    """

    # Default parameters - now favor OTM for faster moves
    DEFAULT_DELTA = 0.35          # OTM for faster percentage moves (was 0.45)
    DEFAULT_PROFIT_TARGET = 12.0  # Tighter for single contracts (was 15)
    DEFAULT_STOP_LOSS = 15.0      # Tighter stop (was 20)
    
    # Delta bounds for strike validation
    MIN_DELTA = 0.15  # Avoid very far OTM (low probability)
    MAX_DELTA = 0.55  # Avoid deep ITM (high capital, slow moves)

    # Price bounds for option selection
    MIN_OPTION_PRICE = 0.15  # $0.15 = $15/contract min (avoid garbage options)
    MAX_OPTION_PRICE = 5.00  # $5.00 = $500/contract max (adjustable based on account)

    def __init__(
        self,
        target_delta: float = DEFAULT_DELTA,
        profit_target_percent: float = DEFAULT_PROFIT_TARGET,
        stop_loss_percent: float = DEFAULT_STOP_LOSS,
        prefer_otm: bool = True,
        min_delta: float = MIN_DELTA,
        max_delta: float = MAX_DELTA,
        min_option_price: float = MIN_OPTION_PRICE,
        max_option_price: float = MAX_OPTION_PRICE
    ):
        """
        Initialize Long Put Strategy

        Args:
            target_delta: Target delta for strike selection (0.20-0.50 typical)
            profit_target_percent: Profit target as % gain
            stop_loss_percent: Stop loss as % loss
            prefer_otm: If True, prefer OTM options for faster moves
            min_delta: Minimum acceptable delta
            max_delta: Maximum acceptable delta
            min_option_price: Minimum option price per share ($0.15 = $15/contract)
            max_option_price: Maximum option price per share ($5.00 = $500/contract)
        """
        super().__init__("Long Put")
        self.target_delta = target_delta
        self.profit_target_percent = profit_target_percent
        self.stop_loss_percent = stop_loss_percent
        self.prefer_otm = prefer_otm
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.min_option_price = min_option_price
        self.max_option_price = max_option_price

        logger.info(
            f"LongPutStrategy initialized: target_delta={target_delta}, "
            f"profit_target={profit_target_percent}%, stop_loss={stop_loss_percent}%, "
            f"prefer_otm={prefer_otm}, price_range=${min_option_price:.2f}-${max_option_price:.2f}"
        )

    @property
    def strategy_type(self) -> str:
        return "LONG_PUT"

    def calculate_position(
        self,
        spy_price: float,
        option_chain: Dict[str, Any],
        contracts: int = 1
    ) -> Optional[SpreadPosition]:
        """
        Calculate Long Put position

        Args:
            spy_price: Current SPY price
            option_chain: Available options data with puts
            contracts: Number of contracts to trade

        Returns:
            SpreadPosition with single put leg, or None
        """
        puts = option_chain.get("puts", [])
        if not puts:
            logger.warning("No puts available in option chain")
            return None

        # Filter puts to acceptable delta range
        filtered_puts = self._filter_by_delta_range(puts)
        if not filtered_puts:
            logger.warning(f"No puts in acceptable delta range ({self.min_delta}-{self.max_delta})")
            # Fall back to all puts if filtering was too strict
            filtered_puts = puts

        # Find put at target delta (puts have negative delta)
        put_option = self.find_strike_by_delta(filtered_puts, self.target_delta, "PUT")
        if not put_option:
            logger.warning(f"Could not find put at delta {self.target_delta}")
            return None

        strike = put_option.get("strike")
        premium = put_option.get("ask", put_option.get("mid", 0))
        expiration = put_option.get("expiration", datetime.now().strftime("%Y-%m-%d"))
        actual_delta = put_option.get("delta", -self.target_delta)

        if strike is None or strike <= 0:
            logger.warning("Invalid strike for put option")
            return None

        if premium <= 0:
            logger.warning("Invalid premium for put option")
            return None

        # Log strike selection details
        moneyness = "OTM" if strike < spy_price else ("ITM" if strike > spy_price else "ATM")
        logger.info(
            f"Selected {moneyness} put: Strike ${strike} vs SPY ${spy_price:.2f}, "
            f"delta={actual_delta:.2f}, premium=${premium:.2f}"
        )

        # Build the option symbol
        symbol = self._build_option_symbol(spy_price, float(strike), "P", expiration)

        # Create the put leg
        put_leg = OptionLeg(
            symbol=symbol,
            option_type="PUT",
            strike=strike,
            expiration=expiration,
            action="BUY",
            quantity=contracts,
            fill_price=premium,
            delta=actual_delta,
            gamma=put_option.get("gamma", 0),
            theta=put_option.get("theta", 0),
            vega=put_option.get("vega", 0),
            iv=put_option.get("iv", 0)
        )

        # Calculate position parameters
        total_debit = premium * contracts  # Cost per share, not per contract
        max_loss = total_debit * 100  # Per contract in dollars
        # Max profit for puts is strike - premium (if SPY goes to 0)
        max_profit = (strike - premium) * 100 * contracts

        position = SpreadPosition(
            position_id=self.generate_position_id(),
            strategy_type=self.strategy_type,
            legs=[put_leg],
            entry_debit=total_debit,
            entry_credit=0.0,
            max_profit=max_profit,
            max_loss=max_loss,
            contracts=contracts,
            profit_target_percent=self.profit_target_percent,
            stop_loss_percent=self.stop_loss_percent,
            time_stop=self.force_exit_time
        )

        logger.info(
            f"Calculated LONG_PUT: Strike ${strike} @ ${premium:.2f} "
            f"(delta: {actual_delta:.2f}, {moneyness})"
        )

        return position

    def _filter_by_delta_range(self, options: List[Dict]) -> List[Dict]:
        """
        Filter options by delta range AND price range.

        Filters out:
        - Options with delta outside acceptable range
        - Options that are too cheap (garbage with wide spreads)
        - Options that are too expensive (ties up too much capital)
        """
        filtered = []
        for opt in options:
            delta = abs(opt.get("delta", 0))  # Use absolute value for puts
            # Get price - prefer ask (what we pay), fallback to mid
            price = opt.get("ask", opt.get("mid", 0))

            # Skip options outside delta range
            if not (self.min_delta <= delta <= self.max_delta):
                continue

            # Skip options that are too cheap (wide spreads, low probability)
            if price < self.min_option_price:
                continue

            # Skip options that are too expensive
            if price > self.max_option_price:
                continue

            filtered.append(opt)

        if not filtered and options:
            # Log why filtering was strict
            logger.debug(
                f"Option filter removed all {len(options)} options. "
                f"Criteria: delta {self.min_delta}-{self.max_delta}, "
                f"price ${self.min_option_price:.2f}-${self.max_option_price:.2f}"
            )

        return filtered

    def _build_option_symbol(
        self,
        underlying_price: float,
        strike: float,
        option_type: str,
        expiration: str
    ) -> str:
        """Build option symbol in OCC format with 6-char padding"""
        # Format: SPY   YYMMDD C/P STRIKE (6 chars for underlying)
        exp_date = expiration.replace("-", "")
        if len(exp_date) == 8:
            exp_date = exp_date[2:]  # Remove century

        strike_str = f"{int(strike * 1000):08d}"
        # Pad underlying to 6 chars for Tastytrade compatibility
        underlying = "SPY".ljust(6)
        return f"{underlying}{exp_date}{option_type}{strike_str}"

    def get_entry_order(self, position: SpreadPosition) -> Dict[str, Any]:
        """Generate entry order for Long Put"""
        if not position.legs:
            return {}

        leg = position.legs[0]

        return {
            "order_type": "LIMIT",
            "time_in_force": "DAY",
            "price_effect": "DEBIT",
            "price": leg.fill_price,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "action": "BUY_TO_OPEN",
                    "quantity": leg.quantity
                }
            ]
        }

    def get_exit_order(self, position: SpreadPosition) -> Dict[str, Any]:
        """Generate exit order for Long Put"""
        if not position.legs:
            return {}

        leg = position.legs[0]
        exit_price = leg.current_price or leg.fill_price

        return {
            "order_type": "LIMIT",
            "time_in_force": "DAY",
            "price_effect": "CREDIT",
            "price": exit_price,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "action": "SELL_TO_CLOSE",
                    "quantity": leg.quantity
                }
            ]
        }

    def check_exit_conditions(
        self,
        position: SpreadPosition,
        current_price: float
    ) -> tuple[bool, Optional[ExitReason]]:
        """
        Check exit conditions for long put

        For LONG options:
        - Profit target: price increased above entry * (1 + profit_target%)
        - Stop loss: price decreased below entry * (1 - stop_loss%)
        """
        if not position.legs:
            return False, None

        entry_price = position.legs[0].fill_price
        if entry_price is None or entry_price <= 0:
            return False, None

        entry = float(entry_price)
        # Profit target (price went UP - put value increased)
        profit_target_price = entry * (1 + self.profit_target_percent / 100)
        if current_price >= profit_target_price:
            return True, ExitReason.PROFIT_TARGET

        # Stop loss (price went DOWN - put value decreased)
        stop_loss_price = entry * (1 - self.stop_loss_percent / 100)
        if current_price <= stop_loss_price:
            return True, ExitReason.STOP_LOSS

        return False, None
