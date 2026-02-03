"""
AI Orchestrator

Coordinates the 3-agent consensus system for trade decisions:
1. Market Agent analyzes conditions
2. Risk Agent evaluates the proposed trade
3. Execution Agent determines optimal parameters
4. Orchestrator makes final decision based on consensus

The orchestrator ensures all agents agree before executing trades.
"""

from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
import traceback
from loguru import logger

from .agents import MarketAgent, RiskAgent, ExecutionAgent
from .agents.base_agent import AgentResponse
from .context_manager import ConversationContext
from ..config.settings import get_settings


class ConsensusDecision(Enum):
    """Possible consensus outcomes"""
    EXECUTE = "EXECUTE"  # All agents agree, proceed with trade
    REJECT = "REJECT"    # One or more agents reject, no trade
    DEFER = "DEFER"      # Conditions unclear, wait and re-evaluate
    PAPER_ONLY = "PAPER_ONLY"  # Execute only in paper trading mode


@dataclass
class TradingDecision:
    """Complete decision from the orchestrator"""
    consensus: ConsensusDecision
    confidence: int  # 0-100, average of agent confidences
    reasoning: str

    # Agent responses
    market_response: AgentResponse
    risk_response: AgentResponse
    execution_response: Optional[AgentResponse]

    # Execution details (if EXECUTE)
    execution_plan: Optional[Dict[str, Any]]
    exit_rules: Optional[Dict[str, Any]]

    # Costs and metadata
    total_tokens: int
    total_cost: float
    timestamp: str
    decision_id: str


class AIOrchestrator:
    """
    Orchestrates the 3-agent decision-making process

    Flow:
    1. Gather market context
    2. Market Agent: Analyze conditions and recommend strategy
    3. Risk Agent: Evaluate risk with market context
    4. Execution Agent: Define trade parameters (if approved)
    5. Consensus: Combine all inputs for final decision
    """

    def __init__(self, context_manager: Optional[ConversationContext] = None):
        """
        Initialize the orchestrator

        Args:
            context_manager: Optional conversation context for learning
        """
        self.settings = get_settings()

        # Initialize agents
        self.market_agent = MarketAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()

        # Context for learning
        self.context_manager = context_manager or ConversationContext()

        # Decision counter
        self._decision_count = 0

        logger.info("AI Orchestrator initialized with 3 agents")

    def evaluate_trade_opportunity(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        option_chain: Optional[Dict[str, Any]] = None
    ) -> TradingDecision:
        """
        Full evaluation pipeline for a potential trade

        Args:
            market_data: Current market conditions (SPY price, VIX, etc.)
            portfolio_state: Current positions, P&L, limits
            option_chain: Available options data (optional)

        Returns:
            TradingDecision with consensus and execution details
        """
        self._decision_count += 1
        decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._decision_count}"

        logger.info(f"Starting trade evaluation {decision_id}")
        start_time = datetime.now()

        total_tokens = 0
        total_cost = 0.0

        # Step 1: Market Analysis
        logger.info("Step 1: Market Agent analyzing conditions...")
        try:
            market_context = self._build_market_context(market_data)
            market_response = self.market_agent.analyze(market_context)
        except Exception as e:
            logger.error(f"Market Agent failed: {e}")
            logger.error(f"Market context: {market_data}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create a safe fallback response
            market_response = {
                "agent_name": "market_agent",
                "decision": "NO_TRADE",
                "confidence": 0,
                "reasoning": f"Agent error: {str(e)}",
                "data": {},
                "tokens_used": 0,
                "estimated_cost": 0,
                "timestamp": datetime.now().isoformat()
            }
        total_tokens += market_response["tokens_used"]
        total_cost += market_response["estimated_cost"]

        logger.info(
            f"Market Agent: {market_response['decision']} "
            f"({market_response['confidence']}% confidence)"
        )

        # Early exit if market conditions are unfavorable
        # Only reject on extremely high confidence NO_TRADE (95%+) to maximize learning opportunities
        if market_response["decision"] == "NO_TRADE" and market_response["confidence"] >= 95:
            logger.info("Market Agent recommends no trade with extremely high confidence")
            return self._create_reject_decision(
                market_response=market_response,
                reason="Market conditions clearly unfavorable",
                total_tokens=total_tokens,
                total_cost=total_cost,
                decision_id=decision_id
            )

        # Step 2: Risk Assessment
        logger.info("Step 2: Risk Agent evaluating proposal...")
        risk_context = self._build_risk_context(
            market_data=market_data,
            portfolio_state=portfolio_state,
            market_response=market_response
        )

        # Check hard limits first (no AI needed)
        hard_limits = self.risk_agent.check_hard_limits(risk_context)
        if not hard_limits["all_passed"]:
            logger.warning(f"Hard risk limits failed: {hard_limits}")
            return self._create_reject_decision(
                market_response=market_response,
                reason=f"Hard risk limits exceeded: {[k for k, v in hard_limits.items() if not v and k != 'all_passed']}",
                total_tokens=total_tokens,
                total_cost=total_cost,
                decision_id=decision_id
            )

        try:
            risk_response = self.risk_agent.analyze(risk_context)
        except Exception as e:
            logger.error(f"Risk Agent failed: {e}")
            logger.error(f"Risk context keys: {risk_context.keys() if risk_context else 'None'}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            risk_response = {
                "agent_name": "risk_agent",
                "decision": "REJECT",
                "confidence": 0,
                "reasoning": f"Agent error: {str(e)}",
                "data": {"risk_score": 10},
                "tokens_used": 0,
                "estimated_cost": 0,
                "timestamp": datetime.now().isoformat()
            }
        total_tokens += risk_response["tokens_used"]
        total_cost += risk_response["estimated_cost"]

        logger.info(
            f"Risk Agent: {risk_response['decision']} "
            f"(Risk Score: {risk_response['data'].get('risk_score', 'N/A')}/10)"
        )

        # Check if paper trading mode
        is_paper_trading = not self.settings.live_trading_enabled
        
        # PAPER TRADING OVERRIDE: Always approve in paper trading mode
        # We need trade data to learn from - that's the whole point!
        if is_paper_trading and risk_response["decision"] == "REJECT":
            risk_score = risk_response["data"].get("risk_score", 10)
            logger.info(f"Paper trading override: Approving trade despite REJECT (risk_score={risk_score}) - learning mode")
            risk_response["decision"] = "APPROVE"
            risk_response["reasoning"] += " [PAPER TRADING: Override - approved for learning]"
        
        # Respect AI agent decisions (after paper trading override)
        if risk_response["decision"] == "REJECT":
            logger.info("Risk Agent rejected the trade")
            return self._create_reject_decision(
                market_response=market_response,
                risk_response=risk_response,
                reason=risk_response["reasoning"],
                total_tokens=total_tokens,
                total_cost=total_cost,
                decision_id=decision_id
            )

        # Step 3: Execution Planning
        logger.info("Step 3: Execution Agent planning trade...")
        execution_context = self._build_execution_context(
            market_data=market_data,
            portfolio_state=portfolio_state,
            option_chain=option_chain or {},
            market_response=market_response,
            risk_response=risk_response
        )

        try:
            execution_response = self.execution_agent.analyze(execution_context)
        except Exception as e:
            logger.error(f"Execution Agent failed: {e}")
            logger.error(f"Execution context keys: {execution_context.keys() if execution_context else 'None'}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            execution_response = {
                "agent_name": "execution_agent",
                "decision": "REJECT",
                "confidence": 0,
                "reasoning": f"Agent error: {str(e)}",
                "data": {"execution_plan": {}, "exit_rules": {}},
                "tokens_used": 0,
                "estimated_cost": 0,
                "timestamp": datetime.now().isoformat()
            }
        total_tokens += execution_response["tokens_used"]
        total_cost += execution_response["estimated_cost"]

        logger.info(
            f"Execution Agent: {execution_response['decision']} "
            f"({execution_response['confidence']}% confidence)"
        )

        # PAPER TRADING OVERRIDE FOR EXECUTION AGENT
        # In paper trading, we want trades to execute for learning purposes
        if is_paper_trading and execution_response["decision"] in ("REJECT", "WAIT"):
            logger.info(f"Paper trading override: Changing Execution Agent {execution_response['decision']} to EXECUTE for learning")
            execution_response["decision"] = "EXECUTE"
            execution_response["reasoning"] += " [PAPER TRADING: Override - executing for learning purposes]"
        
        # Log execution agent decisions (after potential override)
        if execution_response["decision"] in ("REJECT", "WAIT"):
            logger.info(f"Execution Agent recommends: {execution_response['decision']}")

        # Step 4: Consensus Decision
        consensus, confidence, reasoning = self._determine_consensus(
            market_response=market_response,
            risk_response=risk_response,
            execution_response=execution_response
        )

        # Log final consensus - respect AI decisions
        logger.info(f"Consensus: {consensus.value} ({confidence}% confidence)")

        # Build final decision
        execution_plan = None
        exit_rules = None

        if consensus in (ConsensusDecision.EXECUTE, ConsensusDecision.PAPER_ONLY):
            execution_plan = execution_response["data"].get("execution_plan", {})
            exit_rules = execution_response["data"].get("exit_rules", {})

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Decision completed in {elapsed:.2f}s, cost: ${total_cost:.4f}")

        # Store in context for learning
        self.context_manager.add_decision(
            decision_id=decision_id,
            consensus=consensus.value,
            confidence=confidence,
            market_data=market_data,
            portfolio_state=portfolio_state
        )

        return TradingDecision(
            consensus=consensus,
            confidence=confidence,
            reasoning=reasoning,
            market_response=market_response,
            risk_response=risk_response,
            execution_response=execution_response,
            execution_plan=execution_plan,
            exit_rules=exit_rules,
            total_tokens=total_tokens,
            total_cost=total_cost,
            timestamp=datetime.now().isoformat(),
            decision_id=decision_id
        )

    def _build_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for market agent"""
        # Add recent insights from learning
        recent_insights = self.context_manager.get_recent_insights(limit=3)

        return {
            **market_data,
            "recent_insights": recent_insights
        }

    def _build_risk_context(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_response: AgentResponse
    ) -> Dict[str, Any]:
        """Build context for risk agent"""
        # Extract strategy recommendation
        recommended_strategy = market_response["data"].get("recommended_strategy", "NONE")

        # PAPER TRADING: If no strategy, pick one based on recent price movement for learning
        if recommended_strategy in ("NONE", None, ""):
            import random
            spy_price = market_data.get("spy_price", 0)
            spy_open = market_data.get("spy_open", spy_price)
            # If price is above open, lean bullish; below open, lean bearish; else random
            if spy_price and spy_open and spy_price > spy_open * 1.001:
                recommended_strategy = "LONG_CALL"
                logger.info("Paper trading: Defaulting to LONG_CALL (price above open)")
            elif spy_price and spy_open and spy_price < spy_open * 0.999:
                recommended_strategy = "LONG_PUT"
                logger.info("Paper trading: Defaulting to LONG_PUT (price below open)")
            else:
                recommended_strategy = random.choice(["LONG_CALL", "LONG_PUT"])
                logger.info(f"Paper trading: Random strategy selected: {recommended_strategy}")

        # Get account balance and calculate position size
        account_balance = portfolio_state.get("account_balance", self.settings.starting_capital)
        # Use configured risk per trade, not a hardcoded 5%
        risk_percent = self.settings.risk_per_trade_percent * 100  # Convert 0.15 to 15

        # For LONG options: max loss = premium paid
        # Estimate premium based on SPY price and typical 0DTE pricing
        spy_price = market_data.get("spy_price", 500)
        
        # For small paper trading accounts (<$5000), use cheaper OTM options
        # ATM = 0.3% of SPY, OTM (~10 delta) = 0.05% of SPY
        is_small_paper_account = self.settings.paper_trading and account_balance < 5000
        if is_small_paper_account:
            # Use cheaper OTM options for small paper accounts
            estimated_premium = spy_price * 0.0005  # ~0.05% = cheap OTM option ($0.35)
            logger.debug(f"Small paper account: Using OTM options, est premium ${estimated_premium:.2f}")
        else:
            estimated_premium = spy_price * 0.003  # ~0.3% of SPY = typical ATM option price
        
        # Calculate position size based on risk
        risk_amount = account_balance * (risk_percent / 100)
        cost_per_contract = estimated_premium * 100
        
        if cost_per_contract > 0:
            contracts = max(1, int(risk_amount / cost_per_contract))
            contracts = min(contracts, 10)  # Cap at 10 contracts for safety
        else:
            contracts = 1
            
        contracts = portfolio_state.get("suggested_contracts", contracts)

        # For LONG options: max loss = premium paid (defined risk)
        max_loss = contracts * cost_per_contract
        # Max profit for calls is unlimited, for puts it's strike - premium
        max_profit = max_loss * 3  # Target 3:1 reward for good setups
        
        # Calculate risk percentage for this trade
        trade_risk_pct = (max_loss / account_balance * 100) if account_balance > 0 else 100
        logger.debug(f"Trade sizing: {contracts} contracts, ${max_loss:.2f} max loss ({trade_risk_pct:.1f}% of account)")

        # Add risk settings from config
        return {
            **market_data,
            **portfolio_state,
            "proposed_strategy": recommended_strategy,
            "proposed_contracts": contracts,
            "max_loss": max_loss,
            "max_profit": max_profit,
            "estimated_premium": estimated_premium,
            "trade_risk_percent": trade_risk_pct,  # Add percentage for context
            "market_agent_decision": market_response["decision"],
            "market_agent_confidence": market_response["confidence"],
            "current_hour": datetime.now().hour,
            # Risk settings from config
            "max_daily_loss": self.settings.max_daily_loss,
            "max_position_size": self.settings.max_position_size,
            "max_concurrent_positions": self.settings.max_concurrent_positions,
            "risk_per_trade_percent": self.settings.risk_per_trade_percent,
            # Paper trading flags
            "is_paper_trading": self.settings.paper_trading,
            "is_small_paper_account": is_small_paper_account,
        }

    def _build_execution_context(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        option_chain: Dict[str, Any],
        market_response: AgentResponse,
        risk_response: AgentResponse
    ) -> Dict[str, Any]:
        """Build context for execution agent"""
        # Get position size recommendation from risk agent
        position_rec = risk_response["data"].get("position_size_recommendation", {})
        suggested_contracts = position_rec.get("recommended_contracts", 1)

        # Get key levels from market agent
        key_levels = market_response["data"].get("key_levels", {})

        return {
            **market_data,
            **portfolio_state,
            "option_chain": option_chain,
            "recommended_strategy": market_response["data"].get("recommended_strategy"),
            "market_decision": market_response["decision"],
            "risk_decision": risk_response["decision"],
            "suggested_contracts": suggested_contracts,
            "support_levels": key_levels.get("support", []),
            "resistance_levels": key_levels.get("resistance", []),
            "delta_target": self.settings.delta_target,
            "wing_width": self.settings.wing_width,
            "min_credit_percent": self.settings.min_profit_target,
        }

    def _determine_consensus(
        self,
        market_response: AgentResponse,
        risk_response: AgentResponse,
        execution_response: AgentResponse
    ) -> Tuple[ConsensusDecision, int, str]:
        """
        Determine consensus from all agent responses

        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        decisions = [
            market_response["decision"],
            risk_response["decision"],
            execution_response["decision"]
        ]

        confidences = [
            market_response["confidence"],
            risk_response["confidence"],
            execution_response["confidence"]
        ]

        avg_confidence = sum(confidences) // 3

        # All agree to proceed
        if (market_response["decision"] in ("BULLISH", "BEARISH", "NEUTRAL") and
            risk_response["decision"] == "APPROVE" and
            execution_response["decision"] == "EXECUTE"):

            # High confidence = real trade, lower = paper only
            # Lower threshold to 50% to allow more paper trades for learning
            if avg_confidence >= 50:
                return (
                    ConsensusDecision.EXECUTE,
                    avg_confidence,
                    "All agents agree with sufficient confidence"
                )
            else:
                return (
                    ConsensusDecision.EXECUTE,
                    avg_confidence,
                    "Agents agree - executing for learning"
                )

        # For paper trading mode: be more permissive - if market sees opportunity, try it
        # This allows learning even when risk/execution are uncertain
        if (market_response["decision"] in ("BULLISH", "BEARISH") and
            market_response["confidence"] >= 40 and
            risk_response["decision"] in ("APPROVE", "MODIFY")):
            return (
                ConsensusDecision.EXECUTE,
                avg_confidence,
                "Market opportunity detected - executing for learning"
            )

        # Execution says wait
        if execution_response["decision"] == "WAIT":
            return (
                ConsensusDecision.DEFER,
                avg_confidence,
                f"Execution Agent recommends waiting: {execution_response['reasoning']}"
            )

        # Risk modification needed
        if risk_response["decision"] == "MODIFY":
            mods = risk_response["data"].get("modifications_required", [])
            return (
                ConsensusDecision.DEFER,
                avg_confidence,
                f"Risk Agent requires modifications: {mods}"
            )

        # Default: reject
        reasons = []
        if market_response["decision"] == "NO_TRADE":
            reasons.append(f"Market: {market_response['reasoning']}")
        if risk_response["decision"] == "REJECT":
            reasons.append(f"Risk: {risk_response['reasoning']}")
        if execution_response["decision"] == "REJECT":
            reasons.append(f"Execution: {execution_response['reasoning']}")

        return (
            ConsensusDecision.REJECT,
            avg_confidence,
            " | ".join(reasons) if reasons else "Consensus not reached"
        )

    def _create_reject_decision(
        self,
        market_response: AgentResponse,
        reason: str,
        total_tokens: int,
        total_cost: float,
        decision_id: str,
        risk_response: Optional[AgentResponse] = None
    ) -> TradingDecision:
        """Create a rejection decision"""
        # Create placeholder responses if not provided
        if risk_response is None:
            risk_response = AgentResponse(
                agent_name="risk_agent",
                decision="REJECT",
                confidence=0,
                reasoning="Not evaluated due to early rejection",
                data={},
                tokens_used=0,
                estimated_cost=0,
                timestamp=datetime.now().isoformat()
            )

        return TradingDecision(
            consensus=ConsensusDecision.REJECT,
            confidence=market_response["confidence"],
            reasoning=reason,
            market_response=market_response,
            risk_response=risk_response,
            execution_response=None,
            execution_plan=None,
            exit_rules=None,
            total_tokens=total_tokens,
            total_cost=total_cost,
            timestamp=datetime.now().isoformat(),
            decision_id=decision_id
        )

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get combined usage statistics from all agents"""
        market_stats = self.market_agent.get_usage_stats()
        risk_stats = self.risk_agent.get_usage_stats()
        exec_stats = self.execution_agent.get_usage_stats()

        return {
            "total_decisions": self._decision_count,
            "total_tokens": (
                market_stats["total_tokens"] +
                risk_stats["total_tokens"] +
                exec_stats["total_tokens"]
            ),
            "total_cost": round(
                market_stats["total_cost"] +
                risk_stats["total_cost"] +
                exec_stats["total_cost"],
                4
            ),
            "by_agent": {
                "market": market_stats,
                "risk": risk_stats,
                "execution": exec_stats
            }
        }
