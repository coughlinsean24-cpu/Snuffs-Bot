"""
Strategy Selector for 0DTE Trading

Selects the appropriate strategy based on:
- AI agent recommendations
- Market conditions (VIX, trend)
- Risk constraints
- Time of day

Strategies:
- LONG_CALL: Buy calls when bullish (expect SPY to go UP)
- LONG_PUT: Buy puts when bearish (expect SPY to go DOWN)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from loguru import logger

from .base_zero_dte import ZeroDTEStrategy, SpreadPosition
from .long_call import LongCallStrategy
from .long_put import LongPutStrategy
from ...config.settings import get_settings


class MarketRegime(Enum):
    """Market volatility regime based on VIX"""
    LOW_VOL = "LOW_VOL"       # VIX < 15 - cheap premiums, good for buying
    NORMAL = "NORMAL"         # VIX 15-20
    ELEVATED = "ELEVATED"     # VIX 20-25
    HIGH_VOL = "HIGH_VOL"     # VIX 25-30 - expensive premiums
    EXTREME = "EXTREME"       # VIX > 30 - very expensive, avoid buying


class StrategySelector:
    """
    Selects and manages 0DTE long options strategies

    Uses AI recommendations and market conditions to choose
    between LONG_CALL and LONG_PUT based on directional bias.
    """

    def __init__(self):
        """Initialize strategy selector with long options strategies"""
        self.settings = get_settings()

        # Initialize long options strategies only
        self.strategies: Dict[str, ZeroDTEStrategy] = {
            "LONG_CALL": LongCallStrategy(),
            "LONG_PUT": LongPutStrategy(),
        }

        logger.info(f"Strategy Selector initialized with {len(self.strategies)} strategies (LONG_CALL, LONG_PUT)")

    def get_market_regime(self, vix: float) -> MarketRegime:
        """
        Determine market regime from VIX level

        Args:
            vix: Current VIX value

        Returns:
            MarketRegime enum value
        """
        if vix < 15:
            return MarketRegime.LOW_VOL
        elif vix < 20:
            return MarketRegime.NORMAL
        elif vix < 25:
            return MarketRegime.ELEVATED
        elif vix < 30:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.EXTREME

    def select_strategy(
        self,
        ai_recommendation: str,
        market_bias: str,
        vix: float,
        confidence: int
    ) -> Tuple[Optional[ZeroDTEStrategy], str]:
        """
        Select the best strategy based on AI and market conditions

        Args:
            ai_recommendation: AI's recommended strategy type
            market_bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            vix: Current VIX level
            confidence: AI confidence (0-100)

        Returns:
            Tuple of (strategy instance or None, selection reason)
        """
        regime = self.get_market_regime(vix)

        # If AI has a specific recommendation, validate it
        if ai_recommendation in self.strategies:
            strategy = self.strategies[ai_recommendation]

            # Validate recommendation against market conditions
            is_valid, reason = self._validate_strategy_for_conditions(
                ai_recommendation, market_bias, regime, confidence
            )

            if is_valid:
                logger.info(f"Selected AI recommendation: {ai_recommendation}")
                return strategy, f"AI recommendation validated: {reason}"
            else:
                logger.warning(f"AI recommendation rejected: {reason}")

        # If no valid AI recommendation, select based on conditions
        selected, reason = self._select_by_conditions(market_bias, regime, confidence)

        if selected:
            logger.info(f"Selected by conditions: {selected}")
            return self.strategies.get(selected), reason

        return None, "No suitable strategy for current conditions"

    def _validate_strategy_for_conditions(
        self,
        strategy_type: str,
        market_bias: str,
        regime: MarketRegime,
        confidence: int
    ) -> Tuple[bool, str]:
        """
        Validate if a strategy is appropriate for current conditions

        For LONG options:
        - Need directional conviction (BULLISH for calls, BEARISH for puts)
        - Lower VIX = cheaper premiums = better for buying
        - Higher VIX = expensive premiums = need stronger conviction

        Returns:
            Tuple of (is_valid, reason)
        """
        # LONG_CALL validations (bullish play) - relaxed for paper trading
        if strategy_type == "LONG_CALL":
            if market_bias == "BEARISH" and confidence > 70:
                return False, "Strong bearish bias conflicts with long call"
            if market_bias == "NEUTRAL" and confidence < 40:
                return False, "Need some bullish conviction for long call"
            # Allow extreme VIX for learning
            if regime == MarketRegime.HIGH_VOL and confidence < 50:
                return False, "High VIX requires confidence > 50"
            if market_bias == "BULLISH":
                return True, "Bullish conditions favor long call"
            return True, "Conditions acceptable for long call"

        # LONG_PUT validations (bearish play) - relaxed for paper trading
        elif strategy_type == "LONG_PUT":
            if market_bias == "BULLISH" and confidence > 70:
                return False, "Strong bullish bias conflicts with long put"
            if market_bias == "NEUTRAL" and confidence < 40:
                return False, "Need some bearish conviction for long put"
            # Allow extreme VIX for learning
            if regime == MarketRegime.HIGH_VOL and confidence < 50:
                return False, "High VIX requires confidence > 50"
            if market_bias == "BEARISH":
                return True, "Bearish conditions favor long put"
            return True, "Conditions acceptable for long put"

        return False, f"Unknown strategy type: {strategy_type}"

    def _select_by_conditions(
        self,
        market_bias: str,
        regime: MarketRegime,
        confidence: int
    ) -> Tuple[Optional[str], str]:
        """
        Select strategy purely based on market conditions

        For LONG options, we need:
        - Clear directional bias (BULLISH or BEARISH)
        - Reasonable VIX (not extreme - premiums too expensive)
        - Good confidence level

        Returns:
            Tuple of (strategy_type or None, reason)
        """
        # Extreme volatility - in paper mode, still try to trade for learning
        if regime == MarketRegime.EXTREME:
            if market_bias == "BULLISH":
                return "LONG_CALL", "PAPER LEARNING: Trading extreme VIX bullish"
            elif market_bias == "BEARISH":
                return "LONG_PUT", "PAPER LEARNING: Trading extreme VIX bearish"
            return None, "VIX extreme and no clear direction"

        # Low confidence - lower threshold for paper trading learning
        if confidence < 40:
            return None, "Need at least 40% confidence for directional long options"

        # Neutral bias - still try if any confidence for learning
        if market_bias == "NEUTRAL" and confidence >= 50:
            # Slight lean based on recent momentum could be extracted, but for now skip
            return None, "Neutral bias - prefer clear direction"

        # High volatility - lower threshold for paper trading
        if regime == MarketRegime.HIGH_VOL:
            if confidence < 50:
                return None, "High VIX requires confidence > 50"
            if market_bias == "BULLISH":
                return "LONG_CALL", "Bullish in elevated VIX"
            elif market_bias == "BEARISH":
                return "LONG_PUT", "Bearish in elevated VIX"

        # Normal/Low VIX with directional bias - trade with lower confidence for learning
        if market_bias == "BULLISH" and confidence >= 45:
            reason = "Bullish bias"
            if regime == MarketRegime.LOW_VOL:
                reason += " (cheap premiums)"
            return "LONG_CALL", reason

        if market_bias == "BEARISH" and confidence >= 45:
            reason = "Bearish bias"
            if regime == MarketRegime.LOW_VOL:
                reason += " (cheap premiums)"
            return "LONG_PUT", reason

        return None, "No directional signal"

    def calculate_position(
        self,
        strategy_type: str,
        spy_price: float,
        option_chain: Dict[str, Any],
        contracts: int = 1
    ) -> Optional[SpreadPosition]:
        """
        Calculate position using selected strategy

        Args:
            strategy_type: Type of strategy to use
            spy_price: Current SPY price
            option_chain: Available options data
            contracts: Number of contracts

        Returns:
            SpreadPosition or None
        """
        strategy = self.strategies.get(strategy_type)
        if not strategy:
            logger.error(f"Unknown strategy type: {strategy_type}")
            return None

        return strategy.calculate_position(spy_price, option_chain, contracts)

    def get_strategy(self, strategy_type: str) -> Optional[ZeroDTEStrategy]:
        """Get a specific strategy instance"""
        return self.strategies.get(strategy_type)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get daily stats from all strategies"""
        stats = {}
        for name, strategy in self.strategies.items():
            stats[name] = strategy.get_daily_stats()
        return stats

    def get_recommendation_matrix(
        self,
        vix: float,
        market_bias: str,
        confidence: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get suitability scores for all strategies given conditions

        Args:
            vix: Current VIX
            market_bias: Market direction
            confidence: AI confidence

        Returns:
            Dict with strategy scores and reasons
        """
        regime = self.get_market_regime(vix)
        matrix = {}

        for name in self.strategies:
            is_valid, reason = self._validate_strategy_for_conditions(
                name, market_bias, regime, confidence
            )

            # Calculate suitability score for LONG options
            score = 0
            if is_valid:
                score = 50  # Base score if valid

                # LONG_CALL scoring
                if name == "LONG_CALL":
                    if market_bias == "BULLISH":
                        score += 30  # Strong bias bonus
                    # Lower VIX = cheaper premiums = better for buying
                    if regime == MarketRegime.LOW_VOL:
                        score += 20
                    elif regime == MarketRegime.NORMAL:
                        score += 10

                # LONG_PUT scoring
                elif name == "LONG_PUT":
                    if market_bias == "BEARISH":
                        score += 30  # Strong bias bonus
                    # Lower VIX = cheaper premiums = better for buying
                    if regime == MarketRegime.LOW_VOL:
                        score += 20
                    elif regime == MarketRegime.NORMAL:
                        score += 10

                # Confidence adjustment
                score = int(score * (confidence / 100))

            matrix[name] = {
                "valid": is_valid,
                "score": score,
                "reason": reason,
                "regime": regime.value
            }

        return matrix
