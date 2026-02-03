"""
Dashboard Data Provider

Connects the Streamlit dashboard to actual data sources:
- PostgreSQL database for historical data
- Paper trading simulator for live positions
- Tastytrade API for real account data
- Redis for real-time updates
"""

import sys
import os

# IMPORTANT: Load .env BEFORE any other imports that depend on env vars
from dotenv import load_dotenv
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_project_root, ".env")
load_dotenv(_env_path)

# Now add parent to path for imports
sys.path.insert(0, _project_root)

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd

from loguru import logger

try:
    from snuffs_bot.database.connection import db_session_scope, init_database
    from snuffs_bot.database.models import Trade, AIDecision, PerformanceMetric, RiskLimit
    # Initialize database connection for dashboard
    init_database()
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database modules not available")
except Exception as e:
    DB_AVAILABLE = False
    logger.warning(f"Database initialization failed: {e}")

try:
    from snuffs_bot.paper_trading.execution_coordinator import ExecutionCoordinator, ExecutionMode
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    logger.warning("Paper trading modules not available")

try:
    from snuffs_bot.api.client import TastytradeClient
    from snuffs_bot.api.accounts import AccountManager
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False
    logger.warning("Tastytrade API modules not available")

# Dashboard config path
DASHBOARD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "dashboard_config.json")

def get_trading_mode() -> str:
    """Get current trading mode from dashboard config

    Returns:
        'Paper' = Live data + simulated trades
        'Live' = Live data + real trades
    """
    try:
        if os.path.exists(DASHBOARD_CONFIG_PATH):
            with open(DASHBOARD_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config.get("trading_mode", "Paper")
    except Exception:
        pass
    return "Paper"


class DashboardDataProvider:
    """
    Provides data to the Streamlit dashboard from various sources
    """

    def __init__(self, execution_coordinator: Optional['ExecutionCoordinator'] = None):
        """
        Initialize data provider

        Args:
            execution_coordinator: Optional coordinator for live position data
        """
        self.coordinator = execution_coordinator
        self._cache = {}
        self._cache_expiry = {}
        self.cache_duration = timedelta(seconds=1)  # 1 second cache for real-time
        self._tastytrade_client = None
        self._account_manager = None

    def _get_tastytrade_client(self):
        """Get or create Tastytrade client for live data"""
        if not TASTYTRADE_AVAILABLE:
            return None

        if self._tastytrade_client is None:
            try:
                self._tastytrade_client = TastytradeClient()
                self._tastytrade_client.connect()
                self._account_manager = AccountManager(self._tastytrade_client)
                logger.info("Tastytrade client connected for dashboard")
            except Exception as e:
                logger.error(f"Failed to connect Tastytrade client: {e}")
                return None

        return self._tastytrade_client

    def _get_account_manager(self):
        """Get account manager for live data"""
        if self._account_manager is None:
            self._get_tastytrade_client()
        return self._account_manager

    def get_tastytrade_status(self) -> Dict[str, Any]:
        """Get Tastytrade connection status with last checked time in EST"""
        from zoneinfo import ZoneInfo
        est = ZoneInfo("America/New_York")
        now_est = datetime.now(est)
        
        status = {
            "connected": False,
            "last_checked": now_est.strftime("%I:%M:%S %p EST"),
            "account_number": None,
            "error": None
        }
        
        if not TASTYTRADE_AVAILABLE:
            status["error"] = "Tastytrade module not available"
            return status
        
        try:
            client = self._get_tastytrade_client()
            if client and client.is_connected:
                status["connected"] = True
                # Try to get account number
                account_mgr = self._get_account_manager()
                if account_mgr:
                    accounts = account_mgr.get_accounts()
                    if accounts:
                        # API returns account-number with hyphen
                        status["account_number"] = accounts[0].get("account-number", 
                                                   accounts[0].get("account_number", "Connected"))
            else:
                status["error"] = "Not connected"
        except Exception as e:
            status["error"] = str(e)
        
        return status

    def get_learning_mode_status(self) -> Dict[str, Any]:
        """Get current learning/trading mode status"""
        from pathlib import Path
        import json
        import sqlite3

        status = {
            "mode": "unknown",
            "is_learning_mode": False,
            "confidence_threshold": 0.65,
            "momentum_threshold": 0.10,
            "model_trained": False,
            "model_accuracy": None,
            "training_samples": 0,
            "total_trades_today": 0,
            "total_trades_collected": 0,  # Total trades for ML training
            "min_trades_required": 100,   # Minimum trades before ML model trains
            "description": "Unknown mode",
        }

        try:
            # Check if paper trading mode
            trading_mode = get_trading_mode()
            is_paper = trading_mode == "Paper"
            status["is_learning_mode"] = is_paper

            if is_paper:
                status["mode"] = "LEARNING"
                status["confidence_threshold"] = 0.45
                status["momentum_threshold"] = 0.03
                status["description"] = "Aggressive learning - gathering experience"
            else:
                status["mode"] = "LIVE"
                status["confidence_threshold"] = 0.65
                status["momentum_threshold"] = 0.10
                status["description"] = "Conservative trading - protecting capital"

            # Get model status from training history
            project_root = Path(__file__).parent.parent
            history_path = project_root / "data" / "local_ai" / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    if history:
                        latest = history[-1]
                        status["model_trained"] = True
                        status["training_samples"] = latest.get("n_trades", 0)
                        metrics = latest.get("metrics", {})
                        status["model_accuracy"] = metrics.get("val_accuracy")

            # Get today's trade count from PostgreSQL
            if DB_AVAILABLE:
                try:
                    from datetime import date
                    with db_session_scope() as session:
                        today_trades = session.query(Trade).filter(
                            Trade.trade_type == "PAPER",
                            Trade.entry_time >= datetime.combine(date.today(), datetime.min.time())
                        ).count()
                        status["total_trades_today"] = today_trades

                        # Get TOTAL closed trades for ML training (all time)
                        total_trades = session.query(Trade).filter(
                            Trade.trade_type == "PAPER",
                            Trade.status == "CLOSED"
                        ).count()
                        status["total_trades_collected"] = total_trades
                except Exception:
                    pass

            # Also check local_ai SQLite database for trade_records count
            try:
                local_ai_db = project_root / "data" / "local_ai" / "market_data.db"
                if local_ai_db.exists():
                    conn = sqlite3.connect(local_ai_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM trade_records WHERE exit_time IS NOT NULL")
                    local_trades = cursor.fetchone()[0]
                    conn.close()
                    # Use the higher of the two counts (in case they differ)
                    status["total_trades_collected"] = max(status["total_trades_collected"], local_trades)
            except Exception:
                pass

        except Exception as e:
            status["error"] = str(e)

        return status

    def get_background_learner_status(self) -> Dict[str, Any]:
        """Get background learner status from status file"""
        from zoneinfo import ZoneInfo
        import json
        
        est = ZoneInfo("America/New_York")
        now_est = datetime.now(est)
        
        status = {
            "running": False,
            "last_update": None,
            "snapshots_collected": 0,
            "simulations_run": 0,
            "simulation_win_rate": 0,
            "total_snapshots": 0,
            "is_market_hours": False,
            "status_age_seconds": None,
        }
        
        try:
            status_path = Path("data/local_ai/learner_status.json")
            if status_path.exists():
                with open(status_path, 'r') as f:
                    data = json.load(f)
                
                status.update(data)
                
                # Calculate age of status
                if data.get('last_update'):
                    last_update = datetime.fromisoformat(data['last_update'])
                    # Make last_update timezone-aware if it isn't
                    if last_update.tzinfo is None:
                        last_update = last_update.replace(tzinfo=est)
                    age = (now_est - last_update).total_seconds()
                    status['status_age_seconds'] = age
                    status['running'] = age < 120  # Consider running if updated within 2 min
                    status['last_update'] = last_update.strftime("%I:%M:%S %p EST")
        except Exception as e:
            status['error'] = str(e)
        
        return status

    def get_ai_learning_details(self) -> Dict[str, Any]:
        """
        Get detailed AI learning statistics for developer dashboard.
        Includes simulation history, confidence tracking, and learned patterns.
        """
        import sqlite3
        import json
        from pathlib import Path
        from zoneinfo import ZoneInfo
        
        est = ZoneInfo("America/New_York")
        now_est = datetime.now(est)
        
        details = {
            "simulations": {
                "total": 0,
                "profitable": 0,
                "win_rate": 0,
                "by_action": {},
                "recent": [],
            },
            "model": {
                "trained": False,
                "training_samples": 0,
                "accuracy": None,
                "last_training": None,
            },
            "self_improvement": {
                "recent_win_rate": 0,
                "target_win_rate": 0.55,
                "confidence_adjustment": 0,
                "learned_patterns": {},
            },
            "data": {
                "total_snapshots": 0,
                "total_trades": 0,
                "simulated_trades": 0,
            },
            "hourly_performance": {},
            "strategy_performance": {},
        }
        
        try:
            # Get simulation data from database
            # Use absolute path based on project root
            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "local_ai" / "market_data.db"
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get snapshot count
                cursor.execute("SELECT COUNT(*) FROM market_snapshots")
                details["data"]["total_snapshots"] = cursor.fetchone()[0]
                
                # Get trade count
                cursor.execute("SELECT COUNT(*) FROM trade_records")
                details["data"]["total_trades"] = cursor.fetchone()[0]
                
                # Check if simulated_trades table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='simulated_trades'
                """)
                if cursor.fetchone():
                    # Get simulation stats
                    cursor.execute("SELECT COUNT(*) FROM simulated_trades")
                    total = cursor.fetchone()[0]
                    details["simulations"]["total"] = total
                    details["data"]["simulated_trades"] = total
                    
                    cursor.execute("SELECT COUNT(*) FROM simulated_trades WHERE was_profitable = 1")
                    profitable = cursor.fetchone()[0]
                    details["simulations"]["profitable"] = profitable
                    details["simulations"]["win_rate"] = profitable / total if total > 0 else 0
                    
                    # By action type
                    cursor.execute("""
                        SELECT action, COUNT(*), SUM(was_profitable)
                        FROM simulated_trades
                        GROUP BY action
                    """)
                    for row in cursor.fetchall():
                        action, count, wins = row
                        details["simulations"]["by_action"][action] = {
                            "count": count,
                            "wins": wins or 0,
                            "win_rate": (wins or 0) / count if count > 0 else 0
                        }
                    
                    # Recent simulations (last 10)
                    cursor.execute("""
                        SELECT prediction_time, action, confidence, spy_entry, spy_exit,
                               price_change_pct, was_profitable
                        FROM simulated_trades
                        ORDER BY prediction_time DESC
                        LIMIT 10
                    """)
                    for row in cursor.fetchall():
                        # Convert bytes to proper types if needed (SQLite issue)
                        def safe_float(val):
                            if val is None:
                                return None
                            if isinstance(val, bytes):
                                return float(val.decode())
                            return float(val)
                        
                        def safe_str(val):
                            if val is None:
                                return None
                            if isinstance(val, bytes):
                                return val.decode()
                            return str(val)
                        
                        details["simulations"]["recent"].append({
                            "time": safe_str(row[0]),
                            "action": safe_str(row[1]),
                            "confidence": safe_float(row[2]),
                            "entry": safe_float(row[3]),
                            "exit": safe_float(row[4]),
                            "change_pct": safe_float(row[5]),
                            "profitable": bool(row[6]),
                        })
                
                conn.close()
            
            # Get performance state (self-improvement data)
            perf_path = project_root / "data" / "local_ai" / "performance_state.json"
            if perf_path.exists():
                with open(perf_path, 'r') as f:
                    perf_data = json.load(f)
                details["self_improvement"]["recent_win_rate"] = perf_data.get("recent_win_rate", 0)
                details["self_improvement"]["confidence_adjustment"] = perf_data.get("confidence_adjustment", 0)
                details["self_improvement"]["learned_patterns"] = perf_data.get("learned_patterns", {})
            
            # Get model stats if local AI is initialized
            try:
                from snuffs_bot.local_ai import HybridOrchestrator
                orchestrator = HybridOrchestrator(use_local_only=True)
                
                model_stats = orchestrator.model.get_model_stats()
                details["model"]["trained"] = model_stats.get("entry_model_trained", False)
                details["model"]["training_samples"] = model_stats.get("training_samples", 0)
                details["model"]["accuracy"] = model_stats.get("model_accuracy")
                details["model"]["last_training"] = model_stats.get("last_training_time")
                
                # Get trainer insights
                trainer = orchestrator.trainer
                
                # Hourly analysis
                hourly = trainer.analyze_by_time_of_day()
                details["hourly_performance"] = hourly
                
                # Strategy analysis
                strategy = trainer.analyze_by_strategy()
                details["strategy_performance"] = strategy
                
            except Exception as e:
                details["model"]["error"] = str(e)
        
        except Exception as e:
            details["error"] = str(e)
        
        return details

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]

    def _set_cache(self, key: str, data: Any) -> None:
        """Set cached data with expiry"""
        self._cache[key] = data
        self._cache_expiry[key] = datetime.now() + self.cache_duration

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        trading_mode = get_trading_mode()
        # Don't cache portfolio summary - we need real-time updates

        # Get starting capital from settings
        try:
            from snuffs_bot.config.settings import get_settings
            settings = get_settings()
            starting_capital = settings.starting_capital
        except Exception:
            starting_capital = 100000.0

        # Default paper trading values
        summary = {
            "account_value": starting_capital,
            "starting_capital": starting_capital,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "open_positions": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_exposure": 0.0,
            "buying_power": starting_capital,
            "win_rate": 0.0,
            "data_source": "paper",
            "trading_mode": trading_mode,
        }

        # LIVE MODE: Fetch real account data from Tastytrade
        if trading_mode == "Live" and TASTYTRADE_AVAILABLE:
            try:
                account_mgr = self._get_account_manager()
                if account_mgr:
                    # Get account balance
                    balance = account_mgr.get_balance()
                    positions = account_mgr.get_positions()

                    # Parse balance data
                    net_liq = float(balance.get("net-liquidating-value", 0))
                    cash = float(balance.get("cash-balance", 0))
                    buying_power = float(balance.get("derivative-buying-power", balance.get("buying-power", 0)))

                    # Calculate exposure from positions
                    total_exposure = 0
                    for pos in positions:
                        qty = abs(float(pos.get("quantity", 0)))
                        mark = float(pos.get("mark", 0)) * 100  # Options are 100 shares
                        total_exposure += qty * mark

                    summary.update({
                        "account_value": net_liq,
                        "starting_capital": net_liq,
                        "buying_power": buying_power,
                        "open_positions": len(positions),
                        "total_exposure": total_exposure,
                        "data_source": "tastytrade",
                    })

                    logger.debug(f"Fetched Tastytrade portfolio: ${net_liq:,.2f}")

            except Exception as e:
                logger.error(f"Failed to fetch Tastytrade portfolio: {e}")
                summary["data_source"] = "default"

        # PAPER MODE: Use paper trading simulator data if available
        if trading_mode == "Paper" and self.coordinator:
            try:
                paper_perf = self.coordinator.get_paper_performance()
                portfolio = paper_perf.get("portfolio", {})

                summary.update({
                    "account_value": portfolio.get("total_value", 100000.0),
                    "starting_capital": portfolio.get("starting_capital", 100000.0),
                    "daily_pnl": portfolio.get("daily_pnl", 0.0),
                    "total_pnl": portfolio.get("total_pnl", 0.0),
                    "open_positions": portfolio.get("open_positions", 0),
                    "total_trades": portfolio.get("total_trades", 0),
                    "winning_trades": portfolio.get("winning_trades", 0),
                    "losing_trades": portfolio.get("losing_trades", 0),
                    "total_exposure": portfolio.get("total_exposure", 0.0),
                    "buying_power": portfolio.get("buying_power", 100000.0),
                })

                if summary["total_trades"] > 0:
                    summary["win_rate"] = (summary["winning_trades"] / summary["total_trades"]) * 100

            except Exception as e:
                logger.debug(f"Could not get coordinator data: {e}")

        # Try to supplement from database (filter by trade type for correct mode)
        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    # Get today's trades filtered by trading mode
                    today = datetime.now().date()
                    query = session.query(Trade).filter(
                        Trade.entry_time >= datetime.combine(today, datetime.min.time())
                    )
                    
                    # In Paper mode, show paper trades; in Live mode, show live trades
                    if trading_mode == "Paper":
                        query = query.filter(Trade.trade_type == "PAPER")
                    else:
                        query = query.filter(Trade.trade_type != "PAPER")
                    
                    today_trades = query.all()

                    if today_trades:
                        # Count open positions from DB
                        open_count = sum(1 for t in today_trades if t.status == "OPEN")
                        summary["open_positions"] = max(summary["open_positions"], open_count)
                        
                        # Get closed trades for win rate calculation
                        closed_trades = [t for t in today_trades if t.status == "CLOSED"]
                        
                        summary["total_trades"] = len(closed_trades)
                        summary["winning_trades"] = sum(1 for t in closed_trades if t.pnl and float(t.pnl) > 0)
                        summary["losing_trades"] = sum(1 for t in closed_trades if t.pnl and float(t.pnl) < 0)
                        summary["daily_pnl"] = sum(float(t.pnl or 0) for t in closed_trades)

                        if summary["total_trades"] > 0:
                            summary["win_rate"] = (summary["winning_trades"] / summary["total_trades"]) * 100

            except Exception as e:
                logger.debug(f"Could not get database data: {e}")

        return summary

    def force_close_position(self, position_id: str) -> Dict[str, Any]:
        """
        Force close a position at market price (emergency override)
        
        Args:
            position_id: Position ID to close
            
        Returns:
            Dict with success status and details
        """
        import asyncio
        from datetime import datetime
        
        trading_mode = get_trading_mode()
        
        # Paper trading - close via coordinator
        if trading_mode == "Paper" and self.coordinator:
            try:
                # Get current market data
                market_data = {"spy_price": 0.0}
                if hasattr(self.coordinator, 'market_data_provider') and self.coordinator.market_data_provider:
                    spy_quote = self.coordinator.market_data_provider.get_quote("SPY")
                    if spy_quote:
                        market_data["spy_price"] = spy_quote.get("mark", spy_quote.get("last", 0))
                
                # Close the position with FORCE_SELL reason
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.coordinator.close_position(position_id, market_data, "FORCE_SELL")
                    )
                finally:
                    loop.close()
                
                if result.get("success"):
                    logger.info(f"Force closed position {position_id}: P&L=${result.get('pnl', 0):.2f}")
                    return {
                        "success": True,
                        "position_id": position_id,
                        "pnl": result.get("pnl", 0),
                        "reason": "FORCE_SELL"
                    }
                else:
                    return {"success": False, "error": result.get("error", "Unknown error")}
                    
            except Exception as e:
                logger.error(f"Failed to force close position {position_id}: {e}")
                return {"success": False, "error": str(e)}
        
        # Database fallback - update the trade directly
        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    trade = session.query(Trade).filter(Trade.id == int(position_id)).first()
                    if trade and trade.status == "OPEN":
                        # Use current exit_price as the closing price
                        exit_price = float(trade.exit_price or trade.entry_price or 0)
                        entry_price = float(trade.entry_price or 0)
                        position_size = trade.position_size or 1
                        
                        # Calculate P&L
                        pnl = (exit_price - entry_price) * 100 * position_size
                        
                        # Update trade
                        trade.status = "CLOSED"
                        trade.exit_time = datetime.now()
                        trade.exit_reason = "FORCE_SELL"
                        trade.pnl = pnl
                        if entry_price > 0:
                            trade.pnl_percent = ((exit_price / entry_price) - 1) * 100
                        
                        session.commit()
                        logger.info(f"Force closed trade {position_id} via DB: P&L=${pnl:.2f}")
                        return {
                            "success": True,
                            "position_id": position_id,
                            "pnl": pnl,
                            "reason": "FORCE_SELL"
                        }
                    else:
                        return {"success": False, "error": "Position not found or already closed"}
                        
            except Exception as e:
                logger.error(f"Failed to force close via DB: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "No method available to close position"}

    def get_open_positions(self) -> pd.DataFrame:
        """Get currently open positions"""
        trading_mode = get_trading_mode()
        # Don't cache positions - we need real-time updates
        positions_data = []

        # LIVE MODE: Fetch real positions from Tastytrade
        if trading_mode == "Live" and TASTYTRADE_AVAILABLE:
            try:
                account_mgr = self._get_account_manager()
                if account_mgr:
                    positions = account_mgr.get_positions()

                    for pos in positions:
                        symbol = pos.get("symbol", "")
                        quantity = float(pos.get("quantity", 0))
                        avg_price = float(pos.get("average-open-price", 0))
                        mark = float(pos.get("mark", 0))
                        cost = float(pos.get("cost-effect", 0))
                        pnl = float(pos.get("realized-day-gain", 0)) + (mark * quantity * 100 - abs(cost))

                        positions_data.append({
                            "Position ID": symbol,
                            "Strategy": pos.get("instrument-type", "Option"),
                            "Entry Time": pos.get("created-at", ""),
                            "Entry Price": avg_price,
                            "Current Price": mark,
                            "Unrealized P&L": pnl,
"Qty": int(quantity),
                            "Source": "LIVE",
                        })

                    logger.debug(f"Fetched {len(positions_data)} Tastytrade positions")

            except Exception as e:
                logger.error(f"Failed to fetch Tastytrade positions: {e}")

        # PAPER MODE: Show paper trading positions (simulated trades)
        if trading_mode == "Paper" and self.coordinator:
            try:
                paper_perf = self.coordinator.get_paper_performance()
                open_positions = paper_perf.get("open_positions", [])

                for pos in open_positions:
                    # Extract strike price from legs
                    strike = 0
                    legs = pos.get("legs", [])
                    if legs and len(legs) > 0:
                        strike = legs[0].get("strike", 0)
                    
                    entry_price = float(pos.get("entry_price", 0))
                    current_price = float(pos.get("current_price", 0))
                    contracts = int(pos.get("contracts", 1))
                    
                    # unrealized_pnl from portfolio is already in dollars (price diff × 100 × contracts)
                    pnl = float(pos.get("unrealized_pnl", 0))
                    pnl_pct = round(((current_price / entry_price) - 1) * 100, 2) if entry_price > 0 else 0
                    
                    positions_data.append({
                        "Position ID": pos.get("position_id", ""),
                        "Strategy": pos.get("strategy_type", ""),
                        "Strike": round(float(strike), 0),
                        "SPY": round(float(pos.get("spy_price_at_entry", 0)), 2),
                        "Qty": contracts,
                        "Entry Time": pos.get("entry_time", ""),
                        "Entry Price": round(entry_price, 2),
                        "Current Price": round(current_price, 2),
                        "Unrealized P&L": round(pnl, 2),  # Already in dollars from portfolio
                        "P&L %": pnl_pct,
                        "Max Profit": round(float(pos.get("max_profit", 0)), 2),
                        "Max Loss": round(float(pos.get("max_loss", 0)), 2),
                        "Source": "PAPER",
                    })

            except Exception as e:
                logger.debug(f"Could not get coordinator positions: {e}")

        # Also check database for any positions (always check DB for persistence)
        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    # Get open positions, filtered by trade type if in Paper mode
                    query = session.query(Trade).filter(
                        Trade.status == "OPEN"
                    )
                    
                    # In Paper mode, show paper trades; in Live mode, show live trades
                    if trading_mode == "Paper":
                        query = query.filter(Trade.trade_type == "PAPER")
                    else:
                        query = query.filter(Trade.trade_type != "PAPER")
                    
                    open_trades = query.order_by(Trade.entry_time.desc()).all()

                    for trade in open_trades:
                        # Avoid duplicates if we already got from coordinator
                        existing_ids = [p.get("Position ID") for p in positions_data]
                        trade_id = str(trade.id)
                        if trade_id not in existing_ids:
                            entry = round(float(trade.entry_price or 0), 2)
                            # For open trades, exit_price holds current market price (updated by bot)
                            current = round(float(trade.exit_price or trade.entry_price or 0), 2)
                            # Use stored P&L if available, otherwise calculate
                            unrealized_pnl = round(float(trade.pnl) if trade.pnl else (current - entry) * 100, 2)
                            pnl_pct = round(((current / entry) - 1) * 100, 1) if entry > 0 else 0
                            
                            # Extract strike from entry_legs JSON
                            strike = 0
                            spy_price = float(trade.spy_price or 0)
                            if trade.entry_legs:
                                try:
                                    legs = trade.entry_legs if isinstance(trade.entry_legs, list) else []
                                    if legs and len(legs) > 0:
                                        strike = legs[0].get("strike", 0)
                                except:
                                    pass
                            
                            positions_data.append({
                                "Position ID": trade_id,
                                "Strategy": trade.strategy or "",
                                "Strike": strike,
                                "SPY": round(spy_price, 2),
                                "Qty": trade.position_size or 1,
                                "Entry Time": trade.entry_time.strftime("%H:%M:%S") if trade.entry_time else "",
                                "Entry Price": entry,
                                "Current Price": current,
                                "Unrealized P&L": unrealized_pnl,
                                "P&L %": pnl_pct,
                                "Max Loss": round(entry * 100, 2),  # Max loss is premium paid
                                "Source": "DATABASE",
                            })

            except Exception as e:
                logger.debug(f"Could not get database positions: {e}")

        df = pd.DataFrame(positions_data)
        return df

    def get_recent_trades(self, limit: int = 50, full_details: bool = False) -> pd.DataFrame:
        """Get recent closed trades
        
        Args:
            limit: Maximum number of trades to return
            full_details: If True, include all available fields for export
        """
        # Don't cache trades - we need real-time updates
        trades_data = []

        # Get from paper trading coordinator
        if self.coordinator:
            try:
                paper_perf = self.coordinator.get_paper_performance()
                recent_closed = paper_perf.get("recent_closed", [])

                for trade in recent_closed[:limit]:
                    entry_price = float(trade.get("entry_price", 0))
                    exit_price = float(trade.get("exit_price", 0))
                    pnl = float(trade.get("realized_pnl", 0))
                    # Calculate P&L % from prices if not provided
                    pnl_pct = float(trade.get("pnl_percent", 0))
                    if pnl_pct == 0 and entry_price > 0:
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    
                    trade_row = {
                        "Time": trade.get("exit_time", ""),
                        "Entry Time": trade.get("entry_time", ""),
                        "Strategy": trade.get("strategy_type", ""),
                        "Entry": round(entry_price, 2),
                        "Exit": round(exit_price, 2),
                        "P&L": round(pnl, 2),
                        "P&L %": round(pnl_pct, 1),
                        "Exit Reason": trade.get("exit_reason", ""),
                        "Type": "PAPER",
                    }
                    if full_details:
                        trade_row.update({
                            "Quantity": trade.get("quantity", 1),
                        })
                    trades_data.append(trade_row)

            except Exception as e:
                logger.debug(f"Could not get coordinator trades: {e}")

        # Get from database (always check for persistence)
        if DB_AVAILABLE:
            try:
                trading_mode = get_trading_mode()
                with db_session_scope() as session:
                    query = session.query(Trade).filter(
                        Trade.status == "CLOSED"
                    )
                    
                    # In Paper mode, show paper trades; in Live mode, show live trades
                    if trading_mode == "Paper":
                        query = query.filter(Trade.trade_type == "PAPER")
                    else:
                        query = query.filter(Trade.trade_type != "PAPER")
                    
                    closed_trades = query.order_by(Trade.exit_time.desc()).limit(limit).all()

                    for trade in closed_trades:
                        # Avoid duplicates
                        existing_times = [t.get("Time") for t in trades_data]
                        trade_time = trade.exit_time.strftime("%H:%M:%S") if trade.exit_time else ""
                        
                        entry_price = float(trade.entry_price or 0)
                        exit_price = float(trade.exit_price or 0)
                        pnl = float(trade.pnl or 0)
                        # Calculate P&L % from prices if not available
                        pnl_pct = float(trade.pnl_percent or 0)
                        if pnl_pct == 0 and entry_price > 0:
                            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                        
                        trade_row = {
                            "Time": trade_time,
                            "Strategy": trade.strategy or "",
                            "Entry": round(entry_price, 2),
                            "Exit": round(exit_price, 2),
                            "P&L": round(pnl, 2),
                            "P&L %": round(pnl_pct, 1),
                            "Exit Reason": trade.exit_reason or "",
                            "Type": trade.trade_type or "LIVE",
                            "Entry Time": trade.entry_time.strftime("%Y-%m-%d %H:%M:%S") if trade.entry_time else "",
                        }
                        
                        if full_details:
                            # Parse entry_legs JSON for option details
                            legs_info = ""
                            if trade.entry_legs:
                                try:
                                    legs = trade.entry_legs if isinstance(trade.entry_legs, list) else []
                                    if legs:
                                        leg = legs[0]
                                        legs_info = f"{leg.get('symbol', '')} {leg.get('type', '')} ${leg.get('strike', '')}"
                                except:
                                    pass
                            
                            # Get transaction cost fields
                            gross_pnl = float(trade.gross_pnl or pnl)  # Fallback to net if gross not available
                            fees = float(trade.fees or 0)
                            slippage = float(trade.slippage or 0) if hasattr(trade, 'slippage') else 0

                            trade_row.update({
                                "ID": trade.id,
                                "Entry Time": trade.entry_time.strftime("%Y-%m-%d %H:%M:%S") if trade.entry_time else "",
                                "Exit Time": trade.exit_time.strftime("%Y-%m-%d %H:%M:%S") if trade.exit_time else "",
                                "P&L %": float(trade.pnl_percent or 0),
                                "Quantity": trade.position_size or 1,
                                "Max Risk": float(trade.max_risk or 0),
                                "SPY Price": float(trade.spy_price or 0),
                                "VIX": float(trade.vix or 0),
                                "Market": trade.market_condition or "",
                                "Delta": float(trade.delta or 0),
                                "Theta": float(trade.theta or 0),
                                "Option": legs_info,
                                "Gross P&L": round(gross_pnl, 2),
                                "Fees": round(fees, 2),
                                "Slippage": round(slippage, 4),
                                "Net P&L": round(pnl, 2),
                            })
                        
                        trades_data.append(trade_row)

            except Exception as e:
                logger.debug(f"Could not get database trades: {e}")

        # Also fetch from local_ai database (trade_records) which has VIX/Delta data
        try:
            import sqlite3
            from pathlib import Path
            local_ai_db = Path("data/local_ai/market_data.db")
            if local_ai_db.exists():
                conn = sqlite3.connect(local_ai_db)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        trade_id, strategy, entry_time, exit_time,
                        entry_price, exit_price, pnl, pnl_percent,
                        exit_reason, contracts, entry_spy_price,
                        entry_vix, entry_call_delta
                    FROM trade_records 
                    WHERE exit_time IS NOT NULL 
                    ORDER BY exit_time DESC 
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    trade_id, strategy, entry_time, exit_time, entry_price, exit_price, pnl, pnl_pct, exit_reason, contracts, spy_price, vix, delta = row
                    
                    # Avoid duplicates by checking trade_id
                    existing_ids = [t.get("ID", "") for t in trades_data]
                    if trade_id in existing_ids:
                        continue
                    
                    trade_row = {
                        "ID": trade_id,
                        "Strategy": strategy or "",
                        "Entry Time": entry_time or "",
                        "Exit Time": exit_time or "",
                        "Entry": round(float(entry_price or 0), 2),
                        "Exit": round(float(exit_price or 0), 2),
                        "P&L": round(float(pnl or 0), 2),
                        "P&L %": round(float(pnl_pct or 0), 1),
                        "Exit Reason": exit_reason or "",
                        "Quantity": contracts or 1,
                        "SPY Price": round(float(spy_price or 0), 2),
                        "VIX": round(float(vix or 0), 1),
                        "Delta": round(float(delta or 0), 2),
                        "Type": "PAPER",
                    }
                    trades_data.append(trade_row)
                
                conn.close()
        except Exception as e:
            logger.debug(f"Could not get local_ai trades: {e}")

        df = pd.DataFrame(trades_data)
        return df

    def get_ai_decisions(self, limit: int = 50, today_only: bool = True) -> pd.DataFrame:
        """Get recent AI decisions
        
        Args:
            limit: Maximum number of decisions to return
            today_only: If True, only return today's decisions
        """
        # Don't cache AI decisions - we need real-time updates
        decisions_data = []

        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    query = session.query(AIDecision)
                    
                    # Filter to today only by default
                    if today_only:
                        today = datetime.now().date()
                        start_of_day = datetime.combine(today, datetime.min.time())
                        query = query.filter(AIDecision.decision_time >= start_of_day)
                    
                    decisions = query.order_by(
                        AIDecision.decision_time.desc()
                    ).limit(limit).all()

                    for decision in decisions:
                        # Extract strategy from execution response if available
                        strategy = "N/A"
                        strike = ""
                        if decision.execution_agent_response:
                            strategy = decision.execution_agent_response.get("strategy_type", 
                                       decision.execution_agent_response.get("strategy", "N/A"))
                            # Try to get strike from execution plan
                            exec_plan = decision.execution_agent_response.get("data", {}).get("execution_plan", {})
                            if exec_plan:
                                strike = exec_plan.get("strike", "")
                            # Also try from legs
                            legs = decision.execution_agent_response.get("data", {}).get("legs", [])
                            if legs and len(legs) > 0:
                                strike = legs[0].get("strike", strike)

                        # Extract market regime from market response - try multiple paths
                        market = "N/A"
                        if decision.market_agent_response:
                            # Try direct field first
                            market = decision.market_agent_response.get("market_regime", 
                                     decision.market_agent_response.get("data", {}).get("market_regime", "N/A"))

                        # Get reasoning - for rejections, show why; for executes, show market reasoning
                        reason = ""
                        if decision.consensus_decision == "REJECT":
                            # For rejections, show consensus reasoning (why it was rejected)
                            reason = decision.consensus_reasoning or ""
                            if not reason and decision.risk_agent_response:
                                # Try risk agent for rejection reason
                                reason = decision.risk_agent_response.get("reasoning", "")
                            if not reason and decision.market_agent_response:
                                reason = decision.market_agent_response.get("reasoning", "")
                        else:
                            # For executes, show market reasoning
                            if decision.market_agent_response:
                                reason = decision.market_agent_response.get("reasoning", "")
                        
                        # Truncate reason for display
                        reason_display = reason[:80] + "..." if len(reason) > 80 else reason

                        # Get precise confidence (not rounded)
                        confidence = float(decision.market_confidence or 0)

                        decisions_data.append({
                            "Time": decision.decision_time.strftime("%H:%M:%S") if decision.decision_time else "",
                            "Decision": decision.consensus_decision or "",
                            "Confidence": confidence,
                            "Strategy": strategy,
                            "Strike": f"${strike}" if strike else "",
                            "Market": market,
                            "Risk": "APPROVE" if decision.risk_approval else "REJECT",
                            "Reason": reason_display,
                        })

            except Exception as e:
                logger.debug(f"Could not get AI decisions: {e}")

        df = pd.DataFrame(decisions_data)
        return df

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for the specified period"""
        cache_key = f"performance_{days}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        metrics = {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0,
            "daily_pnl_series": [],
        }

        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    start_date = datetime.now() - timedelta(days=days)
                    trades = session.query(Trade).filter(
                        Trade.exit_time >= start_date,
                        Trade.status == "CLOSED"
                    ).all()

                    if trades:
                        pnls = [t.pnl or 0 for t in trades]
                        wins = [p for p in pnls if p > 0]
                        losses = [p for p in pnls if p < 0]

                        metrics["total_pnl"] = sum(pnls)
                        metrics["total_trades"] = len(trades)
                        metrics["win_rate"] = (len(wins) / len(trades)) * 100 if trades else 0
                        metrics["avg_win"] = sum(wins) / len(wins) if wins else 0
                        metrics["avg_loss"] = sum(losses) / len(losses) if losses else 0

                        if losses:
                            metrics["profit_factor"] = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else 0

                        # Calculate daily P&L series
                        daily_pnl = {}
                        for trade in trades:
                            if trade.exit_time:
                                date_key = trade.exit_time.date()
                                daily_pnl[date_key] = daily_pnl.get(date_key, 0) + float(trade.pnl or 0)

                        metrics["daily_pnl_series"] = [
                            {"date": str(d), "pnl": p}
                            for d, p in sorted(daily_pnl.items())
                        ]

            except Exception as e:
                logger.debug(f"Could not get performance metrics: {e}")

        self._set_cache(cache_key, metrics)
        return metrics

    def get_risk_limits(self) -> List[Dict[str, Any]]:
        """Get current risk limit configurations"""
        cache_key = "risk_limits"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        limits = []

        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    risk_limits = session.query(RiskLimit).filter(
                        RiskLimit.is_active == True
                    ).all()

                    for limit in risk_limits:
                        limits.append({
                            "name": limit.limit_name,
                            "limit_value": limit.limit_value,
                            "current_exposure": limit.current_exposure or 0,
                            "description": limit.description or "",
                        })

            except Exception as e:
                logger.debug(f"Could not get risk limits: {e}")

        # Default limits if none found
        if not limits:
            limits = [
                {"name": "Max Daily Loss", "limit_value": 500, "current_exposure": 0, "description": "Maximum daily loss allowed"},
                {"name": "Max Position Size", "limit_value": 5000, "current_exposure": 0, "description": "Maximum single position size"},
                {"name": "Max Concurrent Positions", "limit_value": 3, "current_exposure": 0, "description": "Maximum open positions"},
            ]

        self._set_cache(cache_key, limits)
        return limits

    def get_intraday_pnl(self) -> pd.DataFrame:
        """Get intraday P&L data points"""
        trading_mode = get_trading_mode()
        # Don't cache - we need real-time updates

        # Generate time series for today
        today = datetime.now().date()
        times = pd.date_range(
            start=datetime.combine(today, datetime.strptime("09:30", "%H:%M").time()),
            end=datetime.combine(today, datetime.strptime("16:00", "%H:%M").time()),
            freq="15min"
        )

        # Start with zeros, would be filled from real data
        pnl_values = [0.0] * len(times)

        if DB_AVAILABLE:
            try:
                with db_session_scope() as session:
                    today_start = datetime.combine(today, datetime.min.time())
                    query = session.query(Trade).filter(
                        Trade.exit_time >= today_start,
                        Trade.status == "CLOSED"
                    )
                    
                    # Filter by trading mode
                    if trading_mode == "Paper":
                        query = query.filter(Trade.trade_type == "PAPER")
                    else:
                        query = query.filter(Trade.trade_type != "PAPER")
                    
                    trades = query.order_by(Trade.exit_time).all()

                    cumulative = 0.0
                    trade_idx = 0

                    for i, t in enumerate(times):
                        # Add P&L from trades that closed before this time
                        while trade_idx < len(trades) and trades[trade_idx].exit_time <= t:
                            cumulative += float(trades[trade_idx].pnl or 0)
                            trade_idx += 1
                        pnl_values[i] = cumulative

            except Exception as e:
                logger.debug(f"Could not get intraday P&L: {e}")

        df = pd.DataFrame({
            "time": times,
            "pnl": pnl_values
        })

        return df

    def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        status = {
            "is_running": False,
            "mode": "PAPER_ONLY",
            "last_decision_time": None,
            "open_positions": 0,
            "today_trades": 0,
        }

        if self.coordinator:
            try:
                summary = self.coordinator.get_summary()
                status["is_running"] = self.coordinator.paper_simulator.is_running
                status["mode"] = summary.get("mode", "PAPER_ONLY")
                status["open_positions"] = len(
                    self.coordinator.paper_simulator.portfolio.positions
                )
            except Exception as e:
                logger.debug(f"Could not get bot status: {e}")

        return status

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        self._cache_expiry.clear()

    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data (SPY, VIX prices) from Tastytrade"""
        cache_key = "market_data"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        market_data = {
            "spy_price": 596.89,  # Default fallback
            "spy_change": 0.0,
            "spy_change_pct": 0.0,
            "vix": 15.5,
            "vix_change": 0.0,
            "data_source": "DEFAULT",
        }

        # Get from Tastytrade API
        try:
            client = self._get_tastytrade_client()
            if client and client.is_connected:
                # Get SPY quote from Tastytrade
                spy_quote = client.market_data.get_quote("SPY")
                if spy_quote and spy_quote.get("last", 0) > 0:
                    spy_last = spy_quote.get("last", 0) or spy_quote.get("mark", 0)
                    spy_prev = spy_quote.get("previous-close", spy_last)
                    market_data["spy_price"] = spy_last
                    market_data["spy_change"] = spy_last - spy_prev
                    market_data["spy_change_pct"] = ((spy_last - spy_prev) / spy_prev * 100) if spy_prev else 0
                    market_data["data_source"] = "TASTYTRADE"

                # Get VIX quote from Tastytrade (symbol is $VIX.X)
                vix_quote = client.market_data.get_quote("$VIX.X")
                if vix_quote and vix_quote.get("last", 0) > 0:
                    vix_last = vix_quote.get("last", 0) or vix_quote.get("mark", 0)
                    vix_prev = vix_quote.get("previous-close", vix_last)
                    market_data["vix"] = vix_last
                    market_data["vix_change"] = vix_last - vix_prev

        except Exception as e:
            logger.debug(f"Could not fetch market data from Tastytrade: {e}")

        self._set_cache(cache_key, market_data)
        return market_data

    def get_trades_for_date(self, target_date: datetime.date, include_all_types: bool = True) -> pd.DataFrame:
        """
        Get all trades for a specific date (for historical reports)
        
        Args:
            target_date: The date to fetch trades for
            include_all_types: If True, include both Paper and Live trades
            
        Returns:
            DataFrame with all trade details
        """
        trades_data = []
        
        if not DB_AVAILABLE:
            return pd.DataFrame(trades_data)
        
        try:
            with db_session_scope() as session:
                # Get all trades for the target date
                start_of_day = datetime.combine(target_date, datetime.min.time())
                end_of_day = datetime.combine(target_date, datetime.max.time())
                
                query = session.query(Trade).filter(
                    Trade.entry_time >= start_of_day,
                    Trade.entry_time <= end_of_day
                )
                
                if not include_all_types:
                    trading_mode = get_trading_mode()
                    if trading_mode == "Paper":
                        query = query.filter(Trade.trade_type == "PAPER")
                    else:
                        query = query.filter(Trade.trade_type != "PAPER")
                
                trades = query.order_by(Trade.entry_time.asc()).all()
                
                for trade in trades:
                    # Parse entry_legs for option details
                    option_info = ""
                    strike = ""
                    if trade.entry_legs:
                        try:
                            legs = trade.entry_legs if isinstance(trade.entry_legs, list) else []
                            if legs:
                                leg = legs[0]
                                strike = leg.get("strike", "")
                                option_info = f"{leg.get('symbol', '')} {leg.get('type', '')} ${strike}"
                        except:
                            pass
                    
                    trades_data.append({
                        "ID": trade.id,
                        "Entry Time": trade.entry_time.strftime("%Y-%m-%d %H:%M:%S") if trade.entry_time else "",
                        "Exit Time": trade.exit_time.strftime("%Y-%m-%d %H:%M:%S") if trade.exit_time else "",
                        "Status": trade.status or "",
                        "Strategy": trade.strategy or "",
                        "Strike": strike,
                        "Entry Price": round(float(trade.entry_price or 0), 2),
                        "Exit Price": round(float(trade.exit_price or 0), 2),
                        "P&L": round(float(trade.pnl or 0), 2),
                        "P&L %": round(float(trade.pnl_percent or 0), 2),
                        "Exit Reason": trade.exit_reason or "",
                        "Quantity": trade.position_size or 1,
                        "Max Risk": round(float(trade.max_risk or 0), 2),
                        "SPY Price": round(float(trade.spy_price or 0), 2),
                        "VIX": round(float(trade.vix or 0), 2),
                        "Market Condition": trade.market_condition or "",
                        "Delta": round(float(trade.delta or 0), 3),
                        "Theta": round(float(trade.theta or 0), 3),
                        "Option": option_info,
                        "Fees": round(float(trade.fees or 0), 2),
                        "Trade Type": trade.trade_type or "LIVE",
                    })
        
        except Exception as e:
            logger.error(f"Error fetching trades for date {target_date}: {e}")
        
        return pd.DataFrame(trades_data)

    def get_ai_decisions_for_date(self, target_date: datetime.date) -> pd.DataFrame:
        """
        Get all AI decisions for a specific date (for historical reports)
        
        Args:
            target_date: The date to fetch decisions for
            
        Returns:
            DataFrame with all AI decision details
        """
        decisions_data = []
        
        if not DB_AVAILABLE:
            return pd.DataFrame(decisions_data)
        
        try:
            with db_session_scope() as session:
                start_of_day = datetime.combine(target_date, datetime.min.time())
                end_of_day = datetime.combine(target_date, datetime.max.time())
                
                decisions = session.query(AIDecision).filter(
                    AIDecision.decision_time >= start_of_day,
                    AIDecision.decision_time <= end_of_day
                ).order_by(AIDecision.decision_time.asc()).all()
                
                for decision in decisions:
                    # Extract detailed information
                    strategy = "N/A"
                    strike = ""
                    entry_price = 0
                    
                    if decision.execution_agent_response:
                        strategy = decision.execution_agent_response.get("strategy_type", 
                                   decision.execution_agent_response.get("strategy", "N/A"))
                        exec_plan = decision.execution_agent_response.get("data", {}).get("execution_plan", {})
                        if exec_plan:
                            strike = exec_plan.get("strike", "")
                            entry_price = exec_plan.get("entry_price", 0)
                        legs = decision.execution_agent_response.get("data", {}).get("legs", [])
                        if legs and len(legs) > 0:
                            strike = legs[0].get("strike", strike)
                    
                    market_regime = "N/A"
                    spy_price = 0
                    vix = 0
                    if decision.market_agent_response:
                        market_regime = decision.market_agent_response.get("market_regime",
                                        decision.market_agent_response.get("data", {}).get("market_regime", "N/A"))
                        spy_price = decision.market_agent_response.get("data", {}).get("spy_price", 0)
                        vix = decision.market_agent_response.get("data", {}).get("vix", 0)
                    
                    # Get full reasoning
                    market_reasoning = ""
                    risk_reasoning = ""
                    if decision.market_agent_response:
                        market_reasoning = decision.market_agent_response.get("reasoning", "")
                    if decision.risk_agent_response:
                        risk_reasoning = decision.risk_agent_response.get("reasoning", "")
                    
                    decisions_data.append({
                        "ID": decision.id,
                        "Time": decision.decision_time.strftime("%Y-%m-%d %H:%M:%S") if decision.decision_time else "",
                        "Decision": decision.consensus_decision or "",
                        "Confidence": round(float(decision.market_confidence or 0), 3),
                        "Strategy": strategy,
                        "Strike": f"${strike}" if strike else "",
                        "Entry Price": round(float(entry_price), 2) if entry_price else "",
                        "Market Regime": market_regime,
                        "SPY Price": round(float(spy_price), 2) if spy_price else "",
                        "VIX": round(float(vix), 2) if vix else "",
                        "Risk Approval": "APPROVE" if decision.risk_approval else "REJECT",
                        "Consensus Reasoning": decision.consensus_reasoning or "",
                        "Market Reasoning": market_reasoning,
                        "Risk Reasoning": risk_reasoning,
                        "Tokens Used": decision.total_tokens_used or 0,
                        "Cost": round(float(decision.estimated_cost or 0), 4),
                    })
        
        except Exception as e:
            logger.error(f"Error fetching AI decisions for date {target_date}: {e}")
        
        return pd.DataFrame(decisions_data)

    def get_daily_summary(self, target_date: datetime.date) -> Dict[str, Any]:
        """
        Get a summary of trading activity for a specific date
        
        Args:
            target_date: The date to summarize
            
        Returns:
            Dict with summary metrics
        """
        summary = {
            "date": target_date.strftime("%Y-%m-%d"),
            "total_trades": 0,
            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "total_ai_decisions": 0,
            "execute_decisions": 0,
            "reject_decisions": 0,
            "hold_decisions": 0,
            "strategies_used": [],
        }
        
        trades_df = self.get_trades_for_date(target_date)
        if not trades_df.empty:
            summary["total_trades"] = len(trades_df)
            closed = trades_df[trades_df["Status"] == "CLOSED"]
            summary["closed_trades"] = len(closed)
            
            if not closed.empty and "P&L" in closed.columns:
                summary["winning_trades"] = len(closed[closed["P&L"] > 0])
                summary["losing_trades"] = len(closed[closed["P&L"] < 0])
                summary["total_pnl"] = round(closed["P&L"].sum(), 2)
                summary["win_rate"] = round(summary["winning_trades"] / len(closed) * 100, 1) if len(closed) > 0 else 0
                
                wins = closed[closed["P&L"] > 0]["P&L"]
                losses = closed[closed["P&L"] < 0]["P&L"]
                summary["avg_win"] = round(wins.mean(), 2) if len(wins) > 0 else 0
                summary["avg_loss"] = round(losses.mean(), 2) if len(losses) > 0 else 0
                summary["best_trade"] = round(closed["P&L"].max(), 2)
                summary["worst_trade"] = round(closed["P&L"].min(), 2)
            
            if "Strategy" in trades_df.columns:
                summary["strategies_used"] = trades_df["Strategy"].unique().tolist()
        
        decisions_df = self.get_ai_decisions_for_date(target_date)
        if not decisions_df.empty:
            summary["total_ai_decisions"] = len(decisions_df)
            if "Decision" in decisions_df.columns:
                summary["execute_decisions"] = len(decisions_df[decisions_df["Decision"] == "EXECUTE"])
                summary["reject_decisions"] = len(decisions_df[decisions_df["Decision"] == "REJECT"])
                summary["hold_decisions"] = len(decisions_df[decisions_df["Decision"] == "HOLD"])
        
        return summary

    def get_available_dates(self, days_back: int = 30) -> List[datetime.date]:
        """
        Get list of dates that have trading data
        
        Args:
            days_back: How many days back to check
            
        Returns:
            List of dates with data
        """
        dates = []
        
        if not DB_AVAILABLE:
            return dates
        
        try:
            with db_session_scope() as session:
                cutoff = datetime.now() - timedelta(days=days_back)
                
                # Get distinct dates from trades
                from sqlalchemy import func
                trade_dates = session.query(
                    func.date(Trade.entry_time)
                ).filter(
                    Trade.entry_time >= cutoff
                ).distinct().all()
                
                for (d,) in trade_dates:
                    if d and d not in dates:
                        dates.append(d)
                
                # Also check AI decisions
                decision_dates = session.query(
                    func.date(AIDecision.decision_time)
                ).filter(
                    AIDecision.decision_time >= cutoff
                ).distinct().all()
                
                for (d,) in decision_dates:
                    if d and d not in dates:
                        dates.append(d)
        
        except Exception as e:
            logger.error(f"Error fetching available dates: {e}")
        
        return sorted(dates, reverse=True)


# Singleton instance for easy access
_data_provider: Optional[DashboardDataProvider] = None


def get_data_provider(coordinator: Optional['ExecutionCoordinator'] = None) -> DashboardDataProvider:
    """Get or create the data provider singleton"""
    global _data_provider
    if _data_provider is None:
        _data_provider = DashboardDataProvider(coordinator)
    elif coordinator is not None:
        _data_provider.coordinator = coordinator
    return _data_provider
