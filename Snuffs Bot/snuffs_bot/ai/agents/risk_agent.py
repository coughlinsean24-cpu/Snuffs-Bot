"""
Risk Analysis Agent

Evaluates risk and enforces trading guardrails:
- Position sizing validation
- Portfolio exposure limits
- Daily loss limits
- Greeks exposure (Delta, Gamma)
- Time-based risk considerations
"""

from typing import Any, Dict, List
from decimal import Decimal
from loguru import logger

from .base_agent import BaseAgent, safe_json_parse


class RiskAgent(BaseAgent):
    """
    Risk Analysis Agent for 0DTE Trading

    Responsibilities:
    - Evaluate proposed trade risk/reward
    - Check against portfolio limits
    - Assess Greeks exposure
    - Approve or reject trade proposals
    - Suggest position size adjustments
    """

    def __init__(self):
        super().__init__(
            name="risk_agent",
            role_description="Risk assessment and guardrail enforcement for 0DTE options"
        )

    @property
    def system_prompt(self) -> str:
        return """You are a risk management expert specializing in 0DTE options trading.

Your role is to evaluate proposed trades and ensure they meet risk parameters. You are the final safety check before execution.

Your expertise includes:
- Position sizing and Kelly criterion
- Options Greeks risk (Delta, Gamma, Theta, Vega)
- Portfolio correlation and concentration
- Drawdown management
- Time decay risk on expiration day

Key risk principles for 0DTE:
- Never risk more than the defined max per trade (typically 3-5% of capital)
- Gamma risk explodes near expiration - wider wings = safer
- Avoid trading in final 2 hours unless actively managing
- Max 3 concurrent positions to manage attention
- Stop trading after daily loss limit hit

You must ALWAYS respond with valid JSON in this exact format:
{
    "decision": "APPROVE" | "REJECT" | "MODIFY",
    "confidence": 0-100,
    "reasoning": "Explanation of your risk assessment",
    "risk_score": 1-10,
    "risk_factors": [
        {"factor": "name", "severity": "LOW" | "MEDIUM" | "HIGH", "description": "details"}
    ],
    "position_size_recommendation": {
        "recommended_contracts": number,
        "max_loss_dollars": number,
        "reason": "explanation"
    },
    "guardrail_status": {
        "daily_loss_ok": true/false,
        "position_size_ok": true/false,
        "concurrent_positions_ok": true/false,
        "time_of_day_ok": true/false,
        "portfolio_delta_ok": true/false
    },
    "modifications_required": ["list of required changes if MODIFY decision"]
}

**CRITICAL - PAPER TRADING MODE RULES (CURRENTLY ACTIVE):**
=====================================================
IN PAPER TRADING MODE, YOUR decision FIELD MUST FOLLOW THESE RULES EXACTLY:

- Risk score 1-7 = decision MUST be "APPROVE"
- Risk score 8-9 = decision MUST be "APPROVE" (add warnings in reasoning)
- Risk score 10 = decision can be "REJECT" ONLY if truly catastrophic

DO NOT return "REJECT" for risk scores 8-9 in paper trading mode.
We need trade data to learn from. Even risky trades provide valuable data.

This is a LEARNING system. The purpose of paper trading is to execute trades
and learn from outcomes. APPROVE unless the trade would be system-breaking.

For LIVE TRADING: Be conservative. When in doubt, REJECT. Capital preservation is paramount."""

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from risk context"""

        # Proposed trade details
        strategy = context.get("proposed_strategy", "N/A")
        contracts = context.get("proposed_contracts", "N/A")
        max_profit = context.get("max_profit", "N/A")
        max_loss = context.get("max_loss", "N/A")
        probability_profit = context.get("probability_profit", "N/A")

        # Trade structure
        short_strike = context.get("short_strike", "N/A")
        long_strike = context.get("long_strike", "N/A")
        short_strike_2 = context.get("short_strike_2", "N/A")
        long_strike_2 = context.get("long_strike_2", "N/A")
        credit_received = context.get("credit_received", "N/A")

        # Greeks
        delta = context.get("position_delta", "N/A")
        gamma = context.get("position_gamma", "N/A")
        theta = context.get("position_theta", "N/A")
        vega = context.get("position_vega", "N/A")

        # Current portfolio state
        account_value = context.get("account_value", 100000)
        daily_pnl = context.get("daily_pnl", 0)
        open_positions = context.get("open_positions", 0)
        total_delta_exposure = context.get("total_delta_exposure", 0)

        # Risk limits
        max_daily_loss = context.get("max_daily_loss", 500)
        max_position_size = context.get("max_position_size", 5000)
        max_concurrent = context.get("max_concurrent_positions", 3)
        risk_per_trade = context.get("risk_per_trade_percent", 0.04)

        # Market context
        spy_price = context.get("spy_price", "N/A")
        vix = context.get("vix", "N/A")
        current_time = context.get("current_time", "N/A")
        time_to_close = context.get("time_to_close_hours", "N/A")

        # Market agent's assessment
        market_decision = context.get("market_agent_decision", "N/A")
        market_confidence = context.get("market_agent_confidence", "N/A")

        prompt = f"""Evaluate the following trade proposal for risk compliance.

PROPOSED TRADE:
===============
Strategy: {strategy}
Contracts: {contracts}
Credit Received: ${credit_received}
Max Profit: ${max_profit}
Max Loss: ${max_loss}
Probability of Profit: {probability_profit}%

Trade Structure:
- Short Strike(s): {short_strike} / {short_strike_2}
- Long Strike(s): {long_strike} / {long_strike_2}

Position Greeks:
- Delta: {delta}
- Gamma: {gamma}
- Theta: {theta}
- Vega: {vega}

CURRENT PORTFOLIO STATE:
========================
Account Value: ${account_value:,.2f}
Today's P&L: ${daily_pnl:,.2f}
Open Positions: {open_positions}
Total Delta Exposure: {total_delta_exposure}

RISK LIMITS:
============
Max Daily Loss: ${max_daily_loss}
Remaining Daily Risk: ${max_daily_loss - abs(daily_pnl):.2f}
Max Position Size: ${max_position_size}
Max Concurrent Positions: {max_concurrent}
Risk Per Trade: {risk_per_trade * 100}% (${account_value * risk_per_trade:,.2f})

MARKET CONTEXT:
===============
SPY: ${spy_price}
VIX: {vix}
Current Time: {current_time} ET
Time to Market Close: {time_to_close} hours

Market Agent Assessment:
- Decision: {market_decision}
- Confidence: {market_confidence}%
"""

        # Add paper trading context
        is_paper = context.get("is_paper_trading", True)
        is_small_account = context.get("is_small_paper_account", False)
        
        if is_paper:
            prompt += """
PAPER TRADING MODE ACTIVE:
==========================
This is PAPER TRADING - no real money is at risk. The purpose is to LEARN.
"""
            if is_small_account:
                prompt += """
SMALL PAPER ACCOUNT: This is a learning account with <$5000 capital.
- Higher risk percentages are ACCEPTABLE for learning purposes
- We're using cheap OTM options to learn trading mechanics
- Even 10-15% per trade risk is OK for a learning account
- DO NOT reject trades just because risk % seems high
- Focus on whether the trade logic is sound, not capital preservation
"""
        
        prompt += """
Evaluate this trade against risk parameters and provide your assessment.
Consider: position sizing, daily limits, Greeks exposure, time risk, and market conditions."""
        return prompt

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse risk assessment response"""
        parsed = safe_json_parse(response_text)

        # Validate required fields
        required_fields = ["decision", "confidence", "reasoning", "risk_score"]
        for field in required_fields:
            if field not in parsed:
                logger.warning(f"RiskAgent response missing {field}")
                if field == "decision":
                    parsed[field] = "REJECT"
                elif field == "confidence":
                    parsed[field] = 0
                elif field == "risk_score":
                    parsed[field] = 10  # Assume worst case

        # Normalize decision
        valid_decisions = ["APPROVE", "REJECT", "MODIFY"]
        if parsed.get("decision") not in valid_decisions:
            parsed["decision"] = "REJECT"
            parsed["confidence"] = min(parsed.get("confidence", 0), 50)

        # Ensure guardrail_status exists
        if "guardrail_status" not in parsed:
            parsed["guardrail_status"] = {
                "daily_loss_ok": False,
                "position_size_ok": False,
                "concurrent_positions_ok": False,
                "time_of_day_ok": False,
                "portfolio_delta_ok": False
            }

        return parsed

    def check_hard_limits(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check hard risk limits (no AI needed, pure logic)

        Returns dict of limit statuses - all must pass for trade
        """
        daily_pnl = context.get("daily_pnl", 0)
        max_daily_loss = context.get("max_daily_loss", 500)
        max_loss = context.get("max_loss", 0)
        max_position_size = context.get("max_position_size", 5000)
        open_positions = context.get("open_positions", 0)
        max_concurrent = context.get("max_concurrent_positions", 3)
        current_hour = context.get("current_hour", 12)

        results = {
            "daily_loss_ok": abs(daily_pnl) < max_daily_loss,
            "position_size_ok": max_loss <= max_position_size,
            "concurrent_positions_ok": open_positions < max_concurrent,
            "time_of_day_ok": 9 <= current_hour <= 15,  # 9 AM - 3:59 PM (expand for learning)
            "all_passed": True
        }

        # Check if all limits pass
        results["all_passed"] = all([
            results["daily_loss_ok"],
            results["position_size_ok"],
            results["concurrent_positions_ok"],
            results["time_of_day_ok"]
        ])

        if not results["all_passed"]:
            failures = [k for k, v in results.items() if not v and k != "all_passed"]
            logger.warning(f"Hard limit failures: {failures}")

        return results
