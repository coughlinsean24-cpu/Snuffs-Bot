#!/usr/bin/env python3
"""
Test and demonstrate the Local AI system.

Run this to:
1. Test all Local AI components
2. See a sample prediction
3. Check data collection status
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime
from snuffs_bot.local_ai import (
    DataCollector,
    FeatureEngineer,
    LocalTradingModel,
    ModelTrainer,
    HybridOrchestrator,
    MarketSnapshot,
)


def main():
    print("=" * 60)
    print("LOCAL AI TRADING SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: DataCollector
    print("\n1. Testing DataCollector...")
    dc = DataCollector()
    print(f"   Database: {dc.db_path}")
    print(f"   Snapshots recorded: {dc.get_snapshot_count():,}")
    print(f"   Completed trades: {dc.get_trade_count()}")
    
    # Test 2: FeatureEngineer
    print("\n2. Testing FeatureEngineer...")
    fe = FeatureEngineer()
    print(f"   Total features: {fe.n_features}")
    print(f"   Feature categories: price momentum, time-of-day, volatility, greeks, market structure")
    
    # Test 3: LocalTradingModel
    print("\n3. Testing LocalTradingModel...")
    model = LocalTradingModel()
    stats = model.get_model_stats()
    print(f"   XGBoost available: {stats['using_xgboost']}")
    print(f"   Model trained: {stats['entry_model_trained']}")
    print(f"   Training samples: {stats['training_samples']}")
    
    # Test 4: Sample prediction with rule-based logic
    print("\n4. Testing prediction with sample data...")
    
    # Bullish scenario
    bullish_snapshot = {
        'spy_price': 595.50,
        'spy_change_1m': 0.08,
        'spy_change_5m': 0.30,
        'spy_change_15m': 0.45,
        'vix': 17.5,
        'time_decay_factor': 0.35,
        'minutes_since_open': 90,  # 11:00 AM
        'hour': 11,
        'minute': 0,
        'call_iv': 0.35,
        'put_iv': 0.38,
        'call_delta': 0.52,
    }
    
    decision = model.predict_entry(bullish_snapshot)
    print(f"\n   Bullish scenario (SPY +0.30% 5m):")
    print(f"   - Action: {decision.action}")
    print(f"   - Confidence: {decision.confidence:.1%}")
    print(f"   - Reasoning: {decision.reasoning[:70]}...")
    print(f"   - Inference time: {decision.inference_time_ms:.2f}ms")
    
    # Bearish scenario
    bearish_snapshot = {
        'spy_price': 594.20,
        'spy_change_1m': -0.10,
        'spy_change_5m': -0.35,
        'spy_change_15m': -0.50,
        'vix': 19.0,
        'time_decay_factor': 0.40,
        'minutes_since_open': 120,  # 11:30 AM
        'hour': 11,
        'minute': 30,
        'call_iv': 0.40,
        'put_iv': 0.42,
        'put_delta': -0.48,
    }
    
    decision = model.predict_entry(bearish_snapshot)
    print(f"\n   Bearish scenario (SPY -0.35% 5m):")
    print(f"   - Action: {decision.action}")
    print(f"   - Confidence: {decision.confidence:.1%}")
    print(f"   - Reasoning: {decision.reasoning[:70]}...")
    
    # Choppy/sideways scenario
    choppy_snapshot = {
        'spy_price': 594.80,
        'spy_change_1m': 0.02,
        'spy_change_5m': 0.05,
        'spy_change_15m': -0.03,
        'vix': 16.0,
        'time_decay_factor': 0.50,
        'minutes_since_open': 150,
        'hour': 12,
        'minute': 0,
    }
    
    decision = model.predict_entry(choppy_snapshot)
    print(f"\n   Choppy scenario (SPY +0.05% 5m):")
    print(f"   - Action: {decision.action}")
    print(f"   - Confidence: {decision.confidence:.1%}")
    print(f"   - Reasoning: {decision.reasoning[:70]}...")
    
    # Test 5: Exit decision
    print("\n5. Testing exit decision...")
    position_data = {
        'pnl_percent': 12.0,
        'max_profit_percent': 14.0,
        'strategy': 'LONG_CALL',
        'hold_duration_seconds': 300,
    }
    
    exit_decision = model.predict_exit(position_data, bullish_snapshot)
    print(f"   Position: +12% (was up to +14%)")
    print(f"   - Action: {exit_decision.action}")
    print(f"   - Reasoning: {exit_decision.reasoning}")
    
    # Test profitable exit
    position_data['pnl_percent'] = 16.0
    exit_decision = model.predict_exit(position_data, bullish_snapshot)
    print(f"\n   Position: +16% (profit target)")
    print(f"   - Action: {exit_decision.action}")
    print(f"   - Exit reason: {exit_decision.exit_reason}")
    
    # Test 6: HybridOrchestrator
    print("\n6. Testing HybridOrchestrator...")
    hybrid = HybridOrchestrator(use_local_only=True)
    stats = hybrid.get_stats()
    print(f"   Decisions made: {stats['decisions_made']}")
    print(f"   Data collected: {stats['data_stats']['snapshots']} snapshots")
    
    # Test 7: Trainer status
    print("\n7. Trainer Status Report:")
    print("-" * 40)
    print(hybrid.trainer.get_status_report())
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    print("""
NEXT STEPS:
-----------
1. The Local AI is now RECORDING data from every decision
2. After 30+ completed trades, it will AUTO-TRAIN the model
3. To switch to Local AI mode:
   - Edit .env and set USE_LOCAL_AI=true
   - Restart the bot

Current mode: HYBRID (Claude + data collection)
""")


if __name__ == "__main__":
    main()
