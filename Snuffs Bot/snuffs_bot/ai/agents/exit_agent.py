"""
AI Exit Agent

Specialized agent for making intelligent exit decisions in real-time.
Uses Claude to analyze position dynamics and recommend optimal exit timing.

This agent is designed for speed - uses concise prompts and fast responses.
"""

import json
from typing import Any, Dict, Optional
from datetime import datetime

import anthropic
from loguru import logger

from ...config.settings import get_settings


class ExitAgent:
    """
    Fast AI agent for exit decisions
    
    Optimized for low-latency exit analysis:
    - Concise prompts
    - Structured outputs
    - Caches recent analyses
    """
    
    # Use a faster model for exit decisions
    EXIT_MODEL = "claude-sonnet-4-20250514"  # Fast and capable
    MAX_TOKENS = 500  # Keep responses concise
    
    def __init__(self):
        """Initialize exit agent"""
        self.settings = get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        
        # Cache recent analyses to avoid repeated calls
        self._analysis_cache: Dict[str, Dict] = {}
        self._cache_ttl_seconds = 5  # Cache for 5 seconds
        
        logger.info("Exit Agent initialized")
    
    @property
    def system_prompt(self) -> str:
        return """You are a 0DTE options exit specialist. Your job is to decide whether to:
1. HOLD - Continue holding the position
2. TAKE_PROFIT - Exit now to lock in gains
3. ADJUST_TARGET - Change the profit target up or down
4. CUT_LOSS - Exit now to limit losses

Consider:
- Current profit/loss percentage
- Momentum (positive = price rising, negative = falling)
- Time until market close
- Drawdown from high (how far from peak)

Respond with ONLY valid JSON:
{
    "decision": "HOLD|TAKE_PROFIT|ADJUST_TARGET|CUT_LOSS",
    "confidence": 0-100,
    "reason": "Brief explanation",
    "adjusted_target": null or new target percentage,
    "urgency": "LOW|MEDIUM|HIGH|IMMEDIATE"
}"""

    async def analyze_exit(
        self,
        position_id: str,
        entry_price: float,
        current_price: float,
        profit_percent: float,
        drawdown_from_high: float,
        momentum: float,
        time_held_seconds: float,
        current_profit_target: float = 50.0,
        spy_price: float = 0.0,
        vix: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Analyze whether to exit a position
        
        Args:
            position_id: Position identifier
            entry_price: Entry price
            current_price: Current mid price
            profit_percent: Current P&L as percentage
            drawdown_from_high: Drawdown from high water mark
            momentum: Recent price momentum (% change in last 10s)
            time_held_seconds: How long position held
            current_profit_target: Current target profit %
            spy_price: Current SPY price
            vix: Current VIX level
        
        Returns:
            Analysis with decision and reasoning
        """
        # Check cache first
        cache_key = f"{position_id}_{int(datetime.now().timestamp() / self._cache_ttl_seconds)}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Build concise prompt
        import pytz
        et = pytz.timezone('US/Eastern')
        current_time = datetime.now(et)
        time_str = current_time.strftime("%H:%M")
        
        prompt = f"""Position: {position_id}
Entry: ${entry_price:.2f} → Current: ${current_price:.2f}
P&L: {profit_percent:+.1f}% | Target: {current_profit_target}%
Drawdown from high: {drawdown_from_high:.1f}%
Momentum (10s): {momentum:+.2f}%
Time held: {int(time_held_seconds)}s
Market: SPY ${spy_price:.2f}, VIX {vix:.1f}
Time (ET): {time_str}

Should we hold, take profit, adjust target, or cut loss?"""

        try:
            response = self.client.messages.create(
                model=self.EXIT_MODEL,
                max_tokens=self.MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON response
            try:
                # Handle markdown code blocks if present
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    response_text = "\n".join(lines[1:-1])
                
                result = json.loads(response_text)
                
                # Normalize response
                result = {
                    "decision": result.get("decision", "HOLD").upper(),
                    "confidence": int(result.get("confidence", 70)),
                    "reason": result.get("reason", "No reason provided"),
                    "adjusted_target": result.get("adjusted_target"),
                    "urgency": result.get("urgency", "MEDIUM").upper(),
                    "recommend_hold": result.get("decision", "").upper() == "HOLD",
                    "position_id": position_id,
                    "analyzed_at": datetime.now().isoformat(),
                }
                
                # Cache result
                self._analysis_cache[cache_key] = result
                
                logger.debug(
                    f"Exit analysis for {position_id}: {result['decision']} "
                    f"({result['confidence']}%) - {result['reason']}"
                )
                
                return result
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse exit analysis response: {response_text[:200]}")
                return self._default_response(position_id, profit_percent)
        
        except Exception as e:
            logger.error(f"Exit analysis failed: {e}")
            return self._default_response(position_id, profit_percent)
    
    def _default_response(self, position_id: str, profit_percent: float) -> Dict[str, Any]:
        """Generate default response when AI fails"""
        # Simple rule-based fallback
        if profit_percent >= 40:
            return {
                "decision": "TAKE_PROFIT",
                "confidence": 60,
                "reason": "Default: Take profit at 40%+",
                "adjusted_target": None,
                "urgency": "MEDIUM",
                "recommend_hold": False,
                "position_id": position_id,
            }
        elif profit_percent <= -35:
            return {
                "decision": "CUT_LOSS",
                "confidence": 60,
                "reason": "Default: Cut loss at -35%",
                "adjusted_target": None,
                "urgency": "HIGH",
                "recommend_hold": False,
                "position_id": position_id,
            }
        else:
            return {
                "decision": "HOLD",
                "confidence": 50,
                "reason": "Default: Continue holding",
                "adjusted_target": None,
                "urgency": "LOW",
                "recommend_hold": True,
                "position_id": position_id,
            }
    
    def analyze_exit_sync(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_exit"""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop for sync call
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.analyze_exit(**kwargs)
                )
                return future.result(timeout=5)
        else:
            return asyncio.run(self.analyze_exit(**kwargs))
    
    def clear_cache(self) -> None:
        """Clear the analysis cache"""
        self._analysis_cache.clear()


class QuickExitDecision:
    """
    Ultra-fast exit decision without AI call
    
    For when every millisecond counts - uses pure rule-based logic
    """
    
    @staticmethod
    def should_exit(
        profit_percent: float,
        drawdown_from_high: float,
        momentum: float,
        time_held_seconds: float,
        profit_target: float = 50.0,
        stop_loss: float = 40.0,
    ) -> tuple[bool, str, str]:
        """
        Ultra-fast exit decision
        
        Returns:
            (should_exit, reason, urgency)
        """
        import pytz
        from datetime import time
        et = pytz.timezone('US/Eastern')
        current_time = datetime.now(et).time()
        
        # Forced exits
        if current_time >= time(15, 55):
            return (True, "Market closing in 5 minutes", "IMMEDIATE")
        
        # Profit target
        if profit_percent >= profit_target:
            return (True, f"Hit {profit_target}% target", "IMMEDIATE")
        
        # Stop loss
        if profit_percent <= -stop_loss:
            return (True, f"Stop loss at -{stop_loss}%", "IMMEDIATE")
        
        # Trailing stop (if we've been profitable)
        if drawdown_from_high >= 20 and profit_percent >= 15:
            return (True, f"Trailing stop - {drawdown_from_high:.0f}% from high", "IMMEDIATE")
        
        # Momentum reversal while profitable
        if profit_percent > 25 and momentum < -8:
            return (True, "Strong reversal in progress", "HIGH")
        
        # Late day profit taking
        if current_time >= time(15, 30) and profit_percent > 15:
            return (True, "Late day profit taking", "HIGH")
        
        return (False, "Continue holding", "LOW")
