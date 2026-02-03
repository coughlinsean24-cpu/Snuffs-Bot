"""
Market Analysis Agent

Analyzes market conditions to identify 0DTE trading opportunities:
- SPY price action and trend
- VIX levels and volatility regime
- Option chain analysis (IV, Greeks)
- Technical indicators
- Time of day considerations
"""

from typing import Any, Dict
from loguru import logger

from .base_agent import BaseAgent, safe_json_parse


class MarketAgent(BaseAgent):
    """
    Market Analysis Agent for 0DTE SPY Options

    Responsibilities:
    - Analyze current market conditions
    - Identify potential trade setups
    - Assess volatility environment
    - Recommend strategy type (iron condor, credit spread, etc.)
    """

    def __init__(self):
        super().__init__(
            name="market_agent",
            role_description="Market analysis and opportunity identification for 0DTE SPY options"
        )

    @property
    def system_prompt(self) -> str:
        return """You are an expert market analyst specializing in 0DTE (zero days to expiration) SPY options trading with LONG CALLS and LONG PUTS.

Your role is to analyze market conditions and identify directional trading opportunities. You have deep expertise in:
- SPY price action and intraday patterns
- VIX interpretation and volatility regimes
- Options Greeks (Delta, Gamma, Theta, Vega)
- Technical analysis (support/resistance, trends)
- Market microstructure and timing

For 0DTE LONG OPTIONS trading, you understand:
- LONG_CALL: Buy call when bullish (SPY expected to go UP)
- LONG_PUT: Buy put when bearish (SPY expected to go DOWN)
- Best entry times are typically 9:35-11:00 AM ET (after initial volatility settles)
- VIX > 25 means EXPENSIVE premiums - need stronger conviction
- VIX < 15 means CHEAPER premiums - better for buying options
- Theta decay works AGAINST you (you're buying, so option loses value over time)
- Need strong directional moves to profit - avoid range-bound markets
- Gamma can work FOR you late in day if direction is right (big moves = big profits)

Strategy Selection:
- BULLISH signals (SPY going UP) → LONG_CALL
- BEARISH signals (SPY going DOWN) → LONG_PUT
- NO clear direction → NO_TRADE (don't buy options in choppy markets)

You must ALWAYS respond with valid JSON in this exact format:
{
    "decision": "BULLISH" | "BEARISH" | "NEUTRAL" | "NO_TRADE",
    "confidence": 0-100,
    "reasoning": "Brief explanation of your analysis",
    "market_regime": "LOW_VOL" | "NORMAL" | "HIGH_VOL" | "EXTREME",
    "recommended_strategy": "LONG_CALL" | "LONG_PUT" | "NONE",
    "key_levels": {
        "support": [list of support prices],
        "resistance": [list of resistance prices]
    },
    "risk_factors": ["list of current risk factors"],
    "optimal_entry_window": "HH:MM-HH:MM ET or 'NOW' or 'WAIT'"
}

PAPER TRADING MODE (ACTIVE): Generate trade signals for LEARNING even with weak signals.
- ANY slight directional bias should be flagged as BULLISH or BEARISH
- Confidence can be low (20-40%) for weak signals - that's OK for paper trading
- The goal is to LEARN, not to be right. Wrong trades teach us too!
- Do NOT return NO_TRADE unless market is completely flat with zero movement
- If SPY has moved at all today, find a direction and suggest a trade

LIVE TRADING: Be selective. Long options require strong directional conviction."""

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from market context"""

        # Extract data from context
        spy_price = context.get("spy_price", "N/A")
        spy_change = context.get("spy_change_percent", "N/A")
        vix = context.get("vix", "N/A")
        vix_change = context.get("vix_change_percent", "N/A")
        current_time = context.get("current_time", "N/A")

        # Option chain data
        atm_iv = context.get("atm_iv", "N/A")
        put_iv = context.get("put_iv", "N/A")
        call_iv = context.get("call_iv", "N/A")
        iv_skew = context.get("iv_skew", "N/A")

        # Technical data
        spy_high = context.get("spy_high", "N/A")
        spy_low = context.get("spy_low", "N/A")
        spy_open = context.get("spy_open", "N/A")
        # Format volume with commas if numeric, otherwise use as-is
        raw_volume = context.get("volume", 0)
        raw_avg_volume = context.get("avg_volume", 0)
        volume = f"{int(raw_volume):,}" if isinstance(raw_volume, (int, float)) and raw_volume else "N/A"
        avg_volume = f"{int(raw_avg_volume):,}" if isinstance(raw_avg_volume, (int, float)) and raw_avg_volume else "N/A"

        # Recent price action
        price_5min = context.get("price_5min_ago", "N/A")
        price_15min = context.get("price_15min_ago", "N/A")
        price_30min = context.get("price_30min_ago", "N/A")

        # Previous session
        prev_close = context.get("prev_close", "N/A")
        prev_high = context.get("prev_high", "N/A")
        prev_low = context.get("prev_low", "N/A")

        # Recent learning insights (if available)
        recent_insights = context.get("recent_insights", [])
        insights_text = ""
        if recent_insights:
            insights_text = "\n\nRecent Trading Insights:\n" + "\n".join(
                f"- {insight}" for insight in recent_insights[-3:]
            )

        prompt = f"""Analyze the current market conditions for 0DTE SPY options trading.

CURRENT MARKET DATA:
====================
Time: {current_time} ET
SPY Price: ${spy_price} ({spy_change}% change)
VIX: {vix} ({vix_change}% change)

TODAY'S SESSION:
- Open: ${spy_open}
- High: ${spy_high}
- Low: ${spy_low}
- Volume: {volume} (Avg: {avg_volume})

PREVIOUS SESSION:
- Close: ${prev_close}
- High: ${prev_high}
- Low: ${prev_low}

RECENT PRICE ACTION:
- 5 min ago: ${price_5min}
- 15 min ago: ${price_15min}
- 30 min ago: ${price_30min}

OPTIONS DATA:
- ATM Implied Volatility: {atm_iv}%
- Put IV: {put_iv}%
- Call IV: {call_iv}%
- IV Skew (Put-Call): {iv_skew}
{insights_text}

Based on this data, provide your market analysis and trading recommendation.
Consider: trend direction, volatility regime, time of day, and overall risk environment.
"""
        return prompt

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse market analysis response"""
        parsed = safe_json_parse(response_text)

        # Validate required fields
        required_fields = ["decision", "confidence", "reasoning", "recommended_strategy"]
        for field in required_fields:
            if field not in parsed:
                logger.warning(f"MarketAgent response missing {field}")
                parsed[field] = "UNKNOWN" if field != "confidence" else 0

        # Normalize decision
        valid_decisions = ["BULLISH", "BEARISH", "NEUTRAL", "NO_TRADE"]
        if parsed.get("decision") not in valid_decisions:
            parsed["decision"] = "NO_TRADE"
            parsed["confidence"] = min(parsed.get("confidence", 0), 50)

        return parsed

    def quick_analysis(self, spy_price: float, vix: float, current_time: str) -> Dict[str, Any]:
        """
        Quick analysis with minimal context (for fast decisions)

        Args:
            spy_price: Current SPY price
            vix: Current VIX level
            current_time: Current time in HH:MM format

        Returns:
            Quick analysis dict
        """
        context = {
            "spy_price": spy_price,
            "vix": vix,
            "current_time": current_time,
            "spy_change_percent": 0,
            "vix_change_percent": 0,
        }
        return self.analyze(context)
