"""
Execution Agent

Determines optimal trade execution parameters:
- Strike selection based on delta targets
- Entry timing
- Profit targets and stop losses
- Order type and pricing
- Exit strategy
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal
from loguru import logger

from .base_agent import BaseAgent, safe_json_parse


class ExecutionAgent(BaseAgent):
    """
    Execution Agent for 0DTE Options

    Responsibilities:
    - Select optimal strikes for the strategy
    - Determine entry price and order type
    - Set profit targets and stop losses
    - Define exit rules and adjustments
    - Calculate position Greeks
    """

    def __init__(self):
        super().__init__(
            name="execution_agent",
            role_description="Trade execution optimization for 0DTE options"
        )

    @property
    def system_prompt(self) -> str:
        return """You are an expert options trader specializing in 0DTE SPY LONG OPTIONS execution.

Your role is to determine the optimal execution parameters for LONG_CALL and LONG_PUT trades. You focus on:
- Strike selection based on delta (25-35 delta typical for directional plays)
- Entry price optimization (don't overpay for premium)
- Profit target and stop loss levels
- Entry timing and order types
- Exit strategy definition

Key execution principles for 0DTE LONG options:
- For LONG_CALL: Buy calls when bullish (expect SPY to go UP)
- For LONG_PUT: Buy puts when bearish (expect SPY to go DOWN)
- Target 25-35 delta for good balance of premium cost vs probability
- ALWAYS use limit orders (never market orders)
- Profit target: 50-100% gain on premium (option doubles = 100% profit)
- Stop loss: 50% of premium (cut losses early - theta works against you)
- Monitor closely - theta decay accelerates on 0DTE
- Be prepared to exit early if trade goes against you

You must ALWAYS respond with valid JSON in this exact format:
{
    "decision": "EXECUTE" | "WAIT" | "REJECT",
    "confidence": 0-100,
    "reasoning": "Explanation of execution plan",
    "execution_plan": {
        "strategy_type": "LONG_CALL" | "LONG_PUT",
        "legs": [
            {
                "action": "BUY",
                "option_type": "CALL" | "PUT",
                "strike": number,
                "delta": number,
                "quantity": number
            }
        ],
        "order_type": "LIMIT",
        "limit_price": number,
        "fill_or_kill": false
    },
    "trade_parameters": {
        "contracts": number,
        "premium_cost": number,
        "max_profit": "unlimited for calls, strike-premium for puts",
        "max_loss": number,
        "breakeven": number
    },
    "exit_rules": {
        "profit_target_percent": number,
        "profit_target_price": number,
        "stop_loss_percent": number,
        "stop_loss_price": number,
        "time_stop": "HH:MM ET",
        "exit_note": "description of exit strategy"
    },
    "optimal_entry": {
        "entry_now": boolean,
        "wait_for_price": number or null,
        "wait_until_time": "HH:MM ET" or null,
        "entry_reason": "explanation"
    }
}

Focus on execution quality. For LONG options, getting a good fill is critical since you're paying for the premium."""

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from execution context"""

        # Strategy details from earlier agents
        recommended_strategy = context.get("recommended_strategy", "N/A")
        market_bias = context.get("market_decision", "NEUTRAL")
        risk_approval = context.get("risk_decision", "N/A")

        # Market data
        spy_price = context.get("spy_price", "N/A")
        vix = context.get("vix", "N/A")
        current_time = context.get("current_time", "N/A")

        # Option chain data
        option_chain = context.get("option_chain", {})
        puts = option_chain.get("puts", [])
        calls = option_chain.get("calls", [])

        # Format option chain
        puts_text = self._format_options(puts, "PUT")
        calls_text = self._format_options(calls, "CALL")

        # Configured parameters
        delta_target = context.get("delta_target", 0.10)
        wing_width = context.get("wing_width", 5)
        contracts = context.get("suggested_contracts", 1)
        min_credit_percent = context.get("min_credit_percent", 0.30)

        # Account constraints
        max_position_size = context.get("max_position_size", 5000)
        buying_power = context.get("buying_power", 50000)

        # Support/resistance from market agent
        support_levels = context.get("support_levels", [])
        resistance_levels = context.get("resistance_levels", [])

        prompt = f"""Design the optimal execution plan for this LONG options trade.

TRADE CONTEXT:
==============
Recommended Strategy: {recommended_strategy}
Market Bias: {market_bias}
Risk Approval: {risk_approval}

CURRENT MARKET:
===============
SPY Price: ${spy_price}
VIX: {vix}
Time: {current_time} ET

Support Levels: {support_levels}
Resistance Levels: {resistance_levels}

CONFIGURATION:
==============
Target Delta: {delta_target} ({delta_target * 100:.0f} delta)
Suggested Contracts: {contracts}

CONSTRAINTS:
============
Max Position Size: ${max_position_size}
Available Buying Power: ${buying_power}

AVAILABLE OPTIONS:
==================
CALLS (for LONG_CALL - buy when bullish):
{calls_text}

PUTS (for LONG_PUT - buy when bearish):
{puts_text}

Based on this data, create a detailed execution plan for a LONG option:
- If BULLISH: Select optimal CALL to BUY
- If BEARISH: Select optimal PUT to BUY
Set entry price, profit target (% gain), and stop loss (% loss on premium).
"""
        return prompt

    def _format_options(self, options: List[Dict], option_type: str) -> str:
        """Format option chain data for prompt"""
        if not options:
            return f"No {option_type} data available"

        lines = ["Strike | Delta | IV | Bid | Ask | OI"]
        lines.append("-" * 45)

        for opt in options[:10]:  # Limit to 10 strikes
            strike = opt.get("strike", 0)
            delta = opt.get("delta", 0)
            iv = opt.get("iv", 0) * 100  # Convert to percentage
            bid = opt.get("bid", 0)
            ask = opt.get("ask", 0)
            oi = opt.get("open_interest", 0)

            lines.append(f"${strike:>6} | {delta:>5.2f} | {iv:>4.0f}% | ${bid:>5.2f} | ${ask:>5.2f} | {oi:>5}")

        return "\n".join(lines)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse execution plan response"""
        parsed = safe_json_parse(response_text)

        # Validate required fields
        required_fields = ["decision", "confidence", "reasoning"]
        for field in required_fields:
            if field not in parsed:
                logger.warning(f"ExecutionAgent response missing {field}")
                if field == "decision":
                    parsed[field] = "REJECT"
                elif field == "confidence":
                    parsed[field] = 0

        # Normalize decision
        valid_decisions = ["EXECUTE", "WAIT", "REJECT"]
        if parsed.get("decision") not in valid_decisions:
            parsed["decision"] = "REJECT"
            parsed["confidence"] = min(parsed.get("confidence", 0), 50)

        # Ensure execution_plan exists
        if "execution_plan" not in parsed:
            parsed["execution_plan"] = {
                "strategy_type": "NONE",
                "legs": [],
                "order_type": "LIMIT",
                "limit_price": 0
            }

        # Ensure exit_rules exists (defaults for LONG options - tighter for faster learning)
        if "exit_rules" not in parsed:
            parsed["exit_rules"] = {
                "profit_target_percent": 15,  # 15% gain on premium (was 50)
                "stop_loss_percent": 20       # 20% loss of premium (was 50)
            }

        return parsed

    def build_order_legs(
        self,
        strategy: str,
        spy_price: float,
        delta_target: float,
        wing_width: int,
        option_chain: Dict
    ) -> List[Dict]:
        """
        Build order legs from strategy parameters

        This is a fallback when AI response doesn't include proper legs.
        Uses simple delta-based strike selection for LONG options.
        """
        legs = []

        # Default to 30 delta for long options (good balance of cost vs probability)
        if delta_target < 0.20:
            delta_target = 0.30

        if strategy == "LONG_CALL":
            # Find call at target delta
            call_strike = self._find_strike_by_delta(
                option_chain.get("calls", []),
                delta_target,
                "CALL"
            )
            if call_strike:
                legs = [
                    {"action": "BUY", "option_type": "CALL", "strike": call_strike},
                ]

        elif strategy == "LONG_PUT":
            # Find put at target delta
            put_strike = self._find_strike_by_delta(
                option_chain.get("puts", []),
                delta_target,
                "PUT"
            )
            if put_strike:
                legs = [
                    {"action": "BUY", "option_type": "PUT", "strike": put_strike},
                ]

        return legs

    def _find_strike_by_delta(
        self,
        options: List[Dict],
        target_delta: float,
        option_type: str
    ) -> Optional[float]:
        """Find strike closest to target delta"""
        if not options:
            return None

        # For puts, delta is negative
        if option_type == "PUT":
            target_delta = -abs(target_delta)
        else:
            target_delta = abs(target_delta)

        closest_strike = None
        min_diff = float('inf')

        for opt in options:
            delta = opt.get("delta", 0)
            diff = abs(delta - target_delta)
            if diff < min_diff:
                min_diff = diff
                closest_strike = opt.get("strike")

        return closest_strike
