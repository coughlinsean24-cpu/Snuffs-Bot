"""
Position Sizing Calculator for Long Options

Calculates optimal position size based on:
- Account balance
- Option premium
- Risk per trade percentage
- Commission costs
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class PositionSize:
    """Result of position sizing calculation"""
    contracts: int
    total_cost: float
    commission: float
    risk_amount: float
    percent_of_account: float
    can_afford: bool
    reason: str


class PositionSizer:
    """
    Calculates position sizes for long options trades

    Designed to work with any account size, from $200 to $100,000+
    """

    def __init__(
        self,
        default_risk_percent: float = 5.0,
        max_risk_percent: float = 10.0,
        min_risk_percent: float = 2.0,
        commission_per_contract: float = 1.00,
        commission_cap_per_leg: float = 10.00,
    ):
        """
        Initialize position sizer

        Args:
            default_risk_percent: Default % of account to risk per trade
            max_risk_percent: Maximum % of account per trade
            min_risk_percent: Minimum % of account per trade
            commission_per_contract: Commission per contract (Tastytrade: $1)
            commission_cap_per_leg: Max commission per leg (Tastytrade: $10)
        """
        self.default_risk_percent = default_risk_percent
        self.max_risk_percent = max_risk_percent
        self.min_risk_percent = min_risk_percent
        self.commission_per_contract = commission_per_contract
        self.commission_cap_per_leg = commission_cap_per_leg

    def calculate_position_size(
        self,
        account_balance: float,
        option_price: float,
        risk_percent: Optional[float] = None,
        min_contracts: int = 1,
        max_contracts: int = 100,
    ) -> PositionSize:
        """
        Calculate optimal position size for a long option trade

        Args:
            account_balance: Current account balance
            option_price: Option premium per share (e.g., 1.50 = $1.50/share = $150/contract)
            risk_percent: Override default risk percentage
            min_contracts: Minimum contracts (default 1)
            max_contracts: Maximum contracts to consider

        Returns:
            PositionSize with recommended contracts and details
        """
        # Use default or provided risk percent
        risk_pct = risk_percent or self.default_risk_percent
        risk_pct = max(self.min_risk_percent, min(self.max_risk_percent, risk_pct))

        # Calculate risk amount (how much we're willing to spend)
        risk_amount = account_balance * (risk_pct / 100)

        # Cost per contract = option price * 100 shares
        cost_per_contract = option_price * 100

        if cost_per_contract <= 0:
            return PositionSize(
                contracts=0,
                total_cost=0,
                commission=0,
                risk_amount=risk_amount,
                percent_of_account=0,
                can_afford=False,
                reason="Invalid option price"
            )

        # Calculate max contracts we can afford
        # Account for commission in the calculation
        max_affordable = int(risk_amount / cost_per_contract)

        # Ensure we can at least buy minimum contracts
        if max_affordable < min_contracts:
            # Check if we can afford at least 1 contract with full account
            single_contract_cost = cost_per_contract + self._calculate_commission(1)

            if account_balance >= single_contract_cost:
                # Can afford 1 contract but it exceeds our risk %
                return PositionSize(
                    contracts=1,
                    total_cost=single_contract_cost,
                    commission=self._calculate_commission(1),
                    risk_amount=single_contract_cost,
                    percent_of_account=(single_contract_cost / account_balance) * 100,
                    can_afford=True,
                    reason=f"1 contract exceeds {risk_pct}% risk but affordable"
                )
            else:
                return PositionSize(
                    contracts=0,
                    total_cost=0,
                    commission=0,
                    risk_amount=risk_amount,
                    percent_of_account=0,
                    can_afford=False,
                    reason=f"Option too expensive (${cost_per_contract:.2f}/contract vs ${account_balance:.2f} account)"
                )

        # Cap at max contracts
        contracts = min(max_affordable, max_contracts)

        # Calculate totals
        commission = self._calculate_commission(contracts)
        total_cost = (contracts * cost_per_contract) + commission

        # Final affordability check
        if total_cost > account_balance:
            contracts -= 1
            if contracts < min_contracts:
                return PositionSize(
                    contracts=0,
                    total_cost=0,
                    commission=0,
                    risk_amount=risk_amount,
                    percent_of_account=0,
                    can_afford=False,
                    reason="Cannot afford with commission included"
                )
            commission = self._calculate_commission(contracts)
            total_cost = (contracts * cost_per_contract) + commission

        percent_of_account = (total_cost / account_balance) * 100

        return PositionSize(
            contracts=contracts,
            total_cost=total_cost,
            commission=commission,
            risk_amount=risk_amount,
            percent_of_account=percent_of_account,
            can_afford=True,
            reason=f"{contracts} contracts @ ${option_price:.2f} = ${total_cost:.2f} ({percent_of_account:.1f}% of account)"
        )

    def _calculate_commission(self, contracts: int) -> float:
        """Calculate commission for given number of contracts (1 leg for long options)"""
        return min(contracts * self.commission_per_contract, self.commission_cap_per_leg)

    def suggest_option_price_range(
        self,
        account_balance: float,
        risk_percent: Optional[float] = None,
        target_contracts: int = 1,
    ) -> dict:
        """
        Suggest optimal option price range based on account size

        Args:
            account_balance: Current account balance
            risk_percent: Risk percentage to use
            target_contracts: Target number of contracts

        Returns:
            Dict with min/max/optimal option price suggestions
        """
        risk_pct = risk_percent or self.default_risk_percent
        risk_amount = account_balance * (risk_pct / 100)

        # Max option price we can afford for target contracts
        commission = self._calculate_commission(target_contracts)
        available_for_options = risk_amount - commission
        max_price_per_share = available_for_options / (target_contracts * 100)

        # Minimum practical option price (avoid super cheap options with wide spreads)
        min_price = 0.10  # $0.10 = $10 per contract minimum

        # Optimal range (25-75% of max affordable)
        optimal_min = max(min_price, max_price_per_share * 0.25)
        optimal_max = max_price_per_share * 0.75

        return {
            "account_balance": account_balance,
            "risk_amount": risk_amount,
            "target_contracts": target_contracts,
            "min_option_price": min_price,
            "max_option_price": max(min_price, max_price_per_share),
            "optimal_range": {
                "min": optimal_min,
                "max": optimal_max if optimal_max > optimal_min else optimal_min * 2
            },
            "guidance": self._get_price_guidance(account_balance, max_price_per_share)
        }

    def _get_price_guidance(self, account_balance: float, max_price: float) -> str:
        """Generate guidance based on account size and max affordable price"""
        if account_balance < 500:
            return "Small account: Focus on options under $0.50 to manage risk"
        elif account_balance < 2000:
            return f"Moderate account: Options up to ${max_price:.2f} fit your risk profile"
        elif account_balance < 10000:
            return f"Growing account: Can consider options up to ${max_price:.2f}"
        else:
            return f"Large account: Wide option selection available up to ${max_price:.2f}"


def calculate_contracts_for_amount(
    amount_to_invest: float,
    option_price: float,
    include_commission: bool = True
) -> tuple[int, float, float]:
    """
    Simple utility to calculate contracts for a given investment amount

    Args:
        amount_to_invest: Dollar amount to invest
        option_price: Option price per share
        include_commission: Whether to account for commission

    Returns:
        Tuple of (contracts, total_cost, remaining_cash)
    """
    cost_per_contract = option_price * 100

    if cost_per_contract <= 0:
        return 0, 0.0, amount_to_invest

    if include_commission:
        # Iteratively find max contracts accounting for commission
        contracts = int(amount_to_invest / cost_per_contract)
        while contracts > 0:
            commission = min(contracts * 1.00, 10.00)  # Tastytrade rates
            total = (contracts * cost_per_contract) + commission
            if total <= amount_to_invest:
                break
            contracts -= 1
    else:
        contracts = int(amount_to_invest / cost_per_contract)

    if contracts == 0:
        return 0, 0.0, amount_to_invest

    commission = min(contracts * 1.00, 10.00) if include_commission else 0
    total_cost = (contracts * cost_per_contract) + commission
    remaining = amount_to_invest - total_cost

    return contracts, total_cost, remaining
