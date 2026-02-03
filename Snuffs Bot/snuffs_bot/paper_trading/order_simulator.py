"""
Order Simulation Engine

Simulates order fills with realistic slippage and
fill probability based on market conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import random

from loguru import logger


class OrderType(Enum):
    """Types of orders"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class SimulatedFill:
    """Result of a simulated order fill"""
    order_id: str
    status: OrderStatus
    fill_price: float
    requested_price: float
    slippage: float
    fill_time: datetime
    contracts_filled: int
    contracts_requested: int

    # Metadata
    fill_quality: str = "NORMAL"  # GOOD, NORMAL, POOR
    notes: str = ""

    @property
    def was_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def slippage_percent(self) -> float:
        if self.requested_price == 0:
            return 0
        return (self.slippage / self.requested_price) * 100


class OrderSimulator:
    """
    Simulates order execution with realistic fill modeling

    Features:
    - Slippage modeling based on spread width and VIX
    - Fill probability based on limit price vs market
    - Partial fills for large orders
    - Time-based fill delays
    """

    # Slippage parameters
    BASE_SLIPPAGE_PERCENT = 0.02    # 2% base slippage
    VIX_SLIPPAGE_MULTIPLIER = 0.005  # Additional slippage per VIX point above 15
    SPREAD_SLIPPAGE_FACTOR = 0.1    # 10% of bid-ask spread as slippage

    # Fill probability parameters
    BASE_FILL_PROBABILITY = 0.95    # 95% chance for market orders
    LIMIT_FILL_SENSITIVITY = 0.1    # How limit price affects fill chance

    def __init__(self, realistic_mode: bool = True):
        """
        Initialize order simulator

        Args:
            realistic_mode: If True, add realistic slippage and fill delays
        """
        self.realistic_mode = realistic_mode
        self._order_count = 0

        logger.info(f"Order simulator initialized (realistic={realistic_mode})")

    def simulate_entry_fill(
        self,
        order: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> SimulatedFill:
        """
        Simulate filling an entry order

        Args:
            order: Order specification
            market_data: Current market data (bid, ask, vix, etc.)

        Returns:
            SimulatedFill with result
        """
        self._order_count += 1
        order_id = f"sim_order_{self._order_count}"

        requested_price = order.get("price", 0)
        contracts = order.get("contracts", 1)
        order_type = OrderType(order.get("order_type", "LIMIT"))

        bid = market_data.get("spread_bid", requested_price * 0.98)
        ask = market_data.get("spread_ask", requested_price * 1.02)
        vix = market_data.get("vix", 18)

        # Calculate fill price with slippage
        fill_price, slippage = self._calculate_fill_price(
            requested_price=requested_price,
            bid=bid,
            ask=ask,
            vix=vix,
            is_credit=order.get("price_type") == "CREDIT",
            order_type=order_type
        )

        # Determine if order fills
        fill_probability = self._calculate_fill_probability(
            requested_price=requested_price,
            bid=bid,
            ask=ask,
            order_type=order_type,
            is_credit=order.get("price_type") == "CREDIT"
        )

        # Simulate fill decision
        if self.realistic_mode:
            fills = random.random() < fill_probability
        else:
            fills = True

        if not fills:
            return SimulatedFill(
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                fill_price=0,
                requested_price=requested_price,
                slippage=0,
                fill_time=datetime.now(),
                contracts_filled=0,
                contracts_requested=contracts,
                fill_quality="NONE",
                notes="Order did not fill at requested price"
            )

        # Determine fill quality
        fill_quality = self._assess_fill_quality(slippage, requested_price)

        return SimulatedFill(
            order_id=order_id,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            requested_price=requested_price,
            slippage=slippage,
            fill_time=datetime.now(),
            contracts_filled=contracts,
            contracts_requested=contracts,
            fill_quality=fill_quality,
            notes=f"Filled with {slippage:.4f} slippage"
        )

    def simulate_exit_fill(
        self,
        order: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> SimulatedFill:
        """
        Simulate filling an exit order

        Exit orders typically have worse fills due to urgency
        """
        self._order_count += 1
        order_id = f"sim_exit_{self._order_count}"

        requested_price = order.get("price", 0)
        contracts = order.get("contracts", 1)

        bid = market_data.get("spread_bid", requested_price * 0.98)
        ask = market_data.get("spread_ask", requested_price * 1.02)
        vix = market_data.get("vix", 18)

        # Exit fills are typically worse (buying back spread)
        # For credit spreads, we're paying to close, so we pay more
        exit_slippage_multiplier = 1.5  # 50% worse slippage on exits

        fill_price, slippage = self._calculate_fill_price(
            requested_price=requested_price,
            bid=bid,
            ask=ask,
            vix=vix,
            is_credit=False,  # Exits are debits for credit spreads
            order_type=OrderType.LIMIT,
            slippage_multiplier=exit_slippage_multiplier
        )

        fill_quality = self._assess_fill_quality(slippage, requested_price)

        return SimulatedFill(
            order_id=order_id,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            requested_price=requested_price,
            slippage=slippage,
            fill_time=datetime.now(),
            contracts_filled=contracts,
            contracts_requested=contracts,
            fill_quality=fill_quality,
            notes=f"Exit filled with {slippage:.4f} slippage"
        )

    def _calculate_fill_price(
        self,
        requested_price: float,
        bid: float,
        ask: float,
        vix: float,
        is_credit: bool,
        order_type: OrderType,
        slippage_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate realistic fill price with slippage

        Returns:
            Tuple of (fill_price, slippage_amount)
        """
        if not self.realistic_mode:
            return requested_price, 0

        spread = ask - bid
        mid = (bid + ask) / 2

        # Base slippage
        base_slippage = requested_price * self.BASE_SLIPPAGE_PERCENT

        # VIX-based slippage (higher VIX = more slippage)
        vix_slippage = max(0, (vix - 15) * self.VIX_SLIPPAGE_MULTIPLIER * requested_price)

        # Spread-based slippage
        spread_slippage = spread * self.SPREAD_SLIPPAGE_FACTOR

        total_slippage = (base_slippage + vix_slippage + spread_slippage) * slippage_multiplier

        # Add randomness (±50% of calculated slippage)
        slippage_variance = total_slippage * 0.5 * (random.random() - 0.5)
        final_slippage = max(0, total_slippage + slippage_variance)

        # Apply slippage direction
        if is_credit:
            # Selling: we get less than requested
            fill_price = max(0, requested_price - final_slippage)
        else:
            # Buying: we pay more than requested
            fill_price = requested_price + final_slippage

        return round(fill_price, 2), round(final_slippage, 4)

    def _calculate_fill_probability(
        self,
        requested_price: float,
        bid: float,
        ask: float,
        order_type: OrderType,
        is_credit: bool
    ) -> float:
        """
        Calculate probability of order filling

        Returns:
            Probability between 0 and 1
        """
        if order_type == OrderType.MARKET:
            return self.BASE_FILL_PROBABILITY

        mid = (bid + ask) / 2

        if is_credit:
            # For credit orders (selling), higher requested price = lower fill chance
            price_diff = requested_price - mid
            if price_diff <= 0:
                return self.BASE_FILL_PROBABILITY
            # Reduce probability based on how much above mid we're asking
            reduction = price_diff / mid * self.LIMIT_FILL_SENSITIVITY
            return max(0.5, self.BASE_FILL_PROBABILITY - reduction)
        else:
            # For debit orders (buying), lower requested price = lower fill chance
            price_diff = mid - requested_price
            if price_diff <= 0:
                return self.BASE_FILL_PROBABILITY
            reduction = price_diff / mid * self.LIMIT_FILL_SENSITIVITY
            return max(0.5, self.BASE_FILL_PROBABILITY - reduction)

    def _assess_fill_quality(self, slippage: float, requested_price: float) -> str:
        """Assess the quality of a fill"""
        if requested_price == 0:
            return "NORMAL"

        slippage_pct = abs(slippage / requested_price)

        if slippage_pct < 0.01:
            return "EXCELLENT"
        elif slippage_pct < 0.03:
            return "GOOD"
        elif slippage_pct < 0.05:
            return "NORMAL"
        else:
            return "POOR"

    def estimate_slippage(
        self,
        price: float,
        vix: float,
        is_entry: bool = True
    ) -> float:
        """
        Estimate expected slippage for planning

        Args:
            price: Expected price
            vix: Current VIX
            is_entry: True for entry, False for exit

        Returns:
            Estimated slippage in dollars
        """
        base = price * self.BASE_SLIPPAGE_PERCENT
        vix_component = max(0, (vix - 15) * self.VIX_SLIPPAGE_MULTIPLIER * price)

        if not is_entry:
            return (base + vix_component) * 1.5

        return base + vix_component
