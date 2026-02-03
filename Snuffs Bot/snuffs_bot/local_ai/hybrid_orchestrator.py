"""
Hybrid AI Orchestrator

Manages AI decision-making using either:
1. Local AI (XGBoost model) - instant, no cost, learns from trades
2. Claude API - as fallback or for complex decisions

The local AI is used when:
- Model is trained and has sufficient confidence
- For real-time exit decisions (speed critical)

Claude is used when:
- Local model not trained yet
- For initial learning bootstrapping
- When local confidence is too low
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..local_ai import (
    DataCollector,
    LocalTradingModel,
    MarketSnapshot,
    ModelTrainer,
    TradingDecision,
)


@dataclass
class HybridDecision:
    """Decision from the hybrid AI system"""
    action: str  # LONG_CALL, LONG_PUT, HOLD, EXIT
    confidence: float
    reasoning: str
    source: str  # "local_ai" or "claude"
    
    # For entries
    suggested_strike: Optional[float] = None
    contracts: int = 1
    
    # For exits
    exit_reason: Optional[str] = None
    
    # Performance
    inference_time_ms: float = 0.0


class HybridOrchestrator:
    """
    Manages the hybrid AI decision system.
    
    UNIFIED LEARNING SYSTEM:
    - Snapshots capture market state continuously
    - Model trades aggressively in paper mode to learn
    - Every trade outcome improves the model
    - The system becomes smarter over time
    
    Uses local AI for:
    - Fast exit decisions (trailing stops, stop losses)
    - Entry decisions when model is trained
    - All decisions during market hours for speed
    
    Uses Claude for:
    - Initial bootstrapping when no model exists (optional)
    - Complex market analysis (optional)
    """
    
    # Confidence threshold - lowered for learning mode
    MIN_LOCAL_CONFIDENCE = 0.40  # Lower threshold to trade more during learning
    
    # Data collection interval (seconds)
    DATA_COLLECTION_INTERVAL = 5
    
    def __init__(
        self,
        use_local_only: bool = False,
        data_dir: str = "data/local_ai",
        learning_mode: bool = True,  # Paper trading = learning mode
    ):
        """
        Initialize the hybrid orchestrator
        
        Args:
            use_local_only: If True, never use Claude API
            data_dir: Directory for local AI data storage
            learning_mode: If True, trade aggressively to gather experience
        """
        self.use_local_only = use_local_only
        self.learning_mode = learning_mode
        
        # Initialize local AI components
        self.data_collector = DataCollector(data_dir=data_dir)
        self.model = LocalTradingModel(learning_mode=learning_mode)
        self.trainer = ModelTrainer(self.data_collector, self.model)
        
        # State tracking
        self.last_snapshot: Optional[MarketSnapshot] = None
        self.last_snapshot_time: Optional[datetime] = None
        self.decisions_made = 0
        self.local_decisions = 0
        self.claude_decisions = 0
        
        # Data collection background task
        self._collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        
        mode_str = "LEARNING (aggressive trading)" if learning_mode else "PRODUCTION (conservative)"
        logger.info(f"HybridOrchestrator initialized. Local only: {use_local_only}, Mode: {mode_str}")
        logger.info(self.trainer.get_status_report())
    
    async def get_entry_decision(
        self,
        market_data: Dict[str, Any],
        vix: float = 20.0,
        call_option: Optional[Dict[str, Any]] = None,
        put_option: Optional[Dict[str, Any]] = None,
    ) -> HybridDecision:
        """
        Get entry decision using local AI or Claude
        
        Args:
            market_data: Current market data including SPY quote
            vix: Current VIX level
            call_option: ATM call option data with Greeks
            put_option: ATM put option data with Greeks
            
        Returns:
            HybridDecision with action and reasoning
        """
        start = datetime.now()
        
        # Build snapshot from market data
        snapshot = self._build_snapshot(market_data, vix, call_option, put_option)
        
        # Only record snapshot during market hours (9:30 AM - 4:15 PM on weekdays)
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_cutoff = now.replace(hour=16, minute=15, second=0, microsecond=0)
        is_weekday = now.weekday() < 5
        is_market_hours = is_weekday and market_open <= now <= market_cutoff
        
        if is_market_hours:
            self.data_collector.record_snapshot(snapshot)
        
        self.last_snapshot = snapshot
        self.last_snapshot_time = datetime.now()
        
        # Convert snapshot to dict for model
        snapshot_dict = snapshot.to_dict()
        snapshot_dict['paper_mode'] = True  # Always paper mode for now - enables aggressive learning
        
        # Check if we've learned to avoid this time/setup
        current_hour = datetime.now().hour
        should_avoid, avoid_reason = self.trainer.should_avoid_trade(current_hour, "")
        if should_avoid:
            logger.info(f"Self-improvement override: {avoid_reason}")
            return HybridDecision(
                action="HOLD",
                confidence=0.8,
                reasoning=f"[Self-Improvement] {avoid_reason}",
                source="local_ai",
                inference_time_ms=(datetime.now() - start).total_seconds() * 1000,
            )
        
        # Get local AI decision
        local_decision = self.model.predict_entry(snapshot_dict)
        
        # Apply confidence adjustment from self-improvement
        conf_adj = self.trainer.get_confidence_adjustment()
        adjusted_confidence = local_decision.confidence - conf_adj  # Higher adj = more selective
        
        # Log the decision
        self.trainer.log_prediction(local_decision, snapshot_dict)
        
        # In learning mode, respect learning trades even with low confidence
        is_learning_trade = "Learning mode" in local_decision.reasoning or "LEARNING" in local_decision.reasoning or "learning" in local_decision.reasoning
        
        # Check if we should use local or fallback
        if self._should_use_local(local_decision):
            self.local_decisions += 1
            
            # In learning mode, always respect the action (don't override to HOLD based on confidence)
            if is_learning_trade and local_decision.action != "HOLD":
                final_action = local_decision.action
                logger.info(f"📚 Learning mode trade: {final_action} (confidence override)")
            else:
                final_action = local_decision.action if adjusted_confidence >= self.MIN_LOCAL_CONFIDENCE else "HOLD"
            
            decision = HybridDecision(
                action=final_action,
                confidence=local_decision.confidence,
                reasoning=local_decision.reasoning + (f" [Adj: {conf_adj:+.2f}]" if conf_adj != 0 else ""),
                source="local_ai",
                suggested_strike=local_decision.suggested_strike,
                contracts=local_decision.suggested_contracts,
                inference_time_ms=(datetime.now() - start).total_seconds() * 1000,
            )
        else:
            # For now, if local AI is unsure and we're not local-only, still use local
            # (Claude integration would go here if desired)
            self.local_decisions += 1
            decision = HybridDecision(
                action=local_decision.action,
                confidence=local_decision.confidence,
                reasoning=f"[Local AI - Low Confidence] {local_decision.reasoning}",
                source="local_ai",
                suggested_strike=local_decision.suggested_strike,
                inference_time_ms=(datetime.now() - start).total_seconds() * 1000,
            )
        
        self.decisions_made += 1
        logger.info(f"Entry decision: {decision.action} (conf: {decision.confidence:.1%}, source: {decision.source})")
        
        return decision
    
    async def get_exit_decision(
        self,
        position_data: Dict[str, Any],
        market_data: Dict[str, Any],
        vix: float = 20.0,
    ) -> HybridDecision:
        """
        Get exit decision - always uses local AI for speed
        
        Args:
            position_data: Current position info
            market_data: Current market data
            vix: Current VIX level
            
        Returns:
            HybridDecision with EXIT or HOLD
        """
        start = datetime.now()
        
        # Build minimal snapshot for exit decision
        snapshot_dict = {
            'spy_price': market_data.get('spy_price', market_data.get('mark', 0)),
            'spy_change_5m': self.data_collector.calculate_momentum(
                market_data.get('spy_price', market_data.get('mark', 0)), 5
            ),
            'vix': vix,
            'time_decay_factor': self._calculate_time_decay(),
            'minutes_since_open': self._calculate_minutes_since_open(),
        }
        
        # Get exit decision from local model
        local_decision = self.model.predict_exit(position_data, snapshot_dict)
        
        decision = HybridDecision(
            action=local_decision.action,
            confidence=local_decision.confidence,
            reasoning=local_decision.reasoning,
            source="local_ai",
            exit_reason=local_decision.exit_reason,
            inference_time_ms=(datetime.now() - start).total_seconds() * 1000,
        )
        
        if decision.action == "EXIT":
            logger.info(f"Exit decision: {decision.exit_reason} - {decision.reasoning}")
        
        return decision
    
    def record_trade_entry(
        self,
        trade_id: str,
        strategy: str,
        strike: float,
        entry_price: float,
        contracts: int,
        is_local_ai: bool = False,
    ) -> None:
        """Record a trade entry for learning"""
        if self.last_snapshot:
            self.data_collector.record_trade_entry(
                trade_id=trade_id,
                snapshot=self.last_snapshot,
                strategy=strategy,
                strike=strike,
                entry_price=entry_price,
                contracts=contracts,
                is_local_ai=is_local_ai,
            )
    
    def record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_percent: float,
        max_profit: float = 0,
        max_loss: float = 0,
    ) -> None:
        """Record a trade exit for learning"""
        if self.last_snapshot:
            self.data_collector.record_trade_exit(
                trade_id=trade_id,
                snapshot=self.last_snapshot,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                pnl_percent=pnl_percent,
                max_profit_reached=max_profit,
                max_loss_reached=max_loss,
            )
        
        # Update self-improvement analysis after each trade
        improvement_result = self.trainer.update_self_improvement()
        if improvement_result.get('actions'):
            for action in improvement_result['actions']:
                logger.info(f"Self-improvement: {action}")
        
        # Check if we should retrain after this trade
        if self.trainer.should_retrain():
            logger.info("Triggering model retraining...")
            metrics = self.trainer.train_model()
            if metrics:
                logger.info(f"Retraining complete: {metrics}")
    
    def _should_use_local(self, decision: TradingDecision) -> bool:
        """Determine if local AI decision should be used"""
        if self.use_local_only:
            return True
        
        # Use local if confidence is above threshold
        if decision.confidence >= self.MIN_LOCAL_CONFIDENCE:
            return True
        
        # Use local for HOLD decisions (conservative)
        if decision.action == "HOLD":
            return True
        
        return True  # Default to local for now
    
    def _build_snapshot(
        self,
        market_data: Dict[str, Any],
        vix: float,
        call_option: Optional[Dict[str, Any]],
        put_option: Optional[Dict[str, Any]],
    ) -> MarketSnapshot:
        """Build a MarketSnapshot from raw data"""
        # Handle different key names in market_data
        spy_price = market_data.get('spy_price') or market_data.get('mark') or market_data.get('last', 0)
        spy_bid = market_data.get('bid', spy_price)
        spy_ask = market_data.get('ask', spy_price)
        
        return self.data_collector.build_snapshot_from_live_data(
            spy_data={
                'mark': spy_price,
                'bid': spy_bid,
                'ask': spy_ask,
                'volume': market_data.get('volume', 0),
            },
            vix=vix,
            call_option=call_option or {},
            put_option=put_option or {},
        )
    
    def _calculate_time_decay(self) -> float:
        """Calculate current time decay factor"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            return 0.0
        if now > market_close:
            return 1.0
        
        total_minutes = 390
        minutes_since_open = (now - market_open).total_seconds() / 60
        return min(1.0, minutes_since_open / total_minutes)
    
    def _calculate_minutes_since_open(self) -> int:
        """Calculate minutes since market open"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return max(0, int((now - market_open).total_seconds() / 60))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            'decisions_made': self.decisions_made,
            'local_decisions': self.local_decisions,
            'claude_decisions': self.claude_decisions,
            'local_ratio': self.local_decisions / max(1, self.decisions_made),
            'model_stats': self.model.get_model_stats(),
            'data_stats': {
                'snapshots': self.data_collector.get_snapshot_count(),
                'trades': self.data_collector.get_trade_count(),
            },
            'learning_insights': self.trainer.get_learning_insights(),
        }
    
    def force_retrain(self) -> Optional[Dict[str, Any]]:
        """Force model retraining"""
        return self.trainer.train_model(force=True)
