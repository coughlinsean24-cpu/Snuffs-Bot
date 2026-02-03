"""
Spywave AI Trading Dashboard

Professional single-page dashboard for autonomous 0DTE trading.
Clean design with top navigation menu.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import time
import subprocess
import json
from streamlit_autorefresh import st_autorefresh

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from snuffs_bot.config.settings import get_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

from data_provider import get_data_provider


def format_strategy_name(strategy: str) -> str:
    """Convert strategy names to cleaner display format"""
    if not strategy:
        return "Unknown"
    
    # Map internal names to display names
    mappings = {
        "LONG_CALL": "Call",
        "LONG_PUT": "Put",
        "long_call": "Call",
        "long_put": "Put",
        "CALL_CREDIT_SPREAD": "Call Credit",
        "PUT_CREDIT_SPREAD": "Put Credit",
        "IRON_CONDOR": "Iron Condor",
    }
    
    return mappings.get(strategy, strategy.replace("_", " ").title())


# Dashboard config file path
DASHBOARD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "dashboard_config.json")

def load_dashboard_config():
    """Load dashboard configuration from file"""
    default_config = {
        "trading_mode": "Paper",
        "auto_refresh": True,
        "refresh_rate": 2,  # Fast 2-second refresh
        "current_page": "Dashboard"
    }
    try:
        if os.path.exists(DASHBOARD_CONFIG_PATH):
            with open(DASHBOARD_CONFIG_PATH, 'r') as f:
                saved = json.load(f)
                default_config.update(saved)
    except Exception:
        pass
    return default_config

def save_dashboard_config(config):
    """Save dashboard configuration to file"""
    try:
        with open(DASHBOARD_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False

# Page configuration
st.set_page_config(
    page_title="Spywave AI",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

# Clean, professional CSS - Minimal and consistent
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu, header, footer {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}
    .block-container {padding: 1rem 2rem !important; max-width: 100%;}

    /* Base - Clean dark theme */
    .stApp {
        background: #0f172a;
        color: #e2e8f0;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    h4 {
        font-size: 0.9rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem !important;
    }
    
    /* Stat cards - Clean, no gradients */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        transition: border-color 0.15s ease;
    }
    
    .metric-card:hover {
        border-color: #475569;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .metric-value-lg {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Status colors */
    .positive { color: #22c55e !important; }
    .negative { color: #ef4444 !important; }
    .neutral { color: #64748b !important; }
    
    /* Navigation buttons */
    .stButton button {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    
    .stButton button:hover {
        border-color: #22c55e !important;
        color: #22c55e !important;
        background: rgba(34, 197, 94, 0.05) !important;
    }
    
    .stButton button:focus {
        box-shadow: none !important;
    }
    
    /* Primary action buttons - Active nav state */
    .stButton button[kind="primary"] {
        border: 1px solid #22c55e !important;
        color: #22c55e !important;
        background: rgba(34, 197, 94, 0.1) !important;
        font-weight: 600 !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background: rgba(34, 197, 94, 0.15) !important;
        border-color: #22c55e !important;
    }
    
    /* Data tables */
    .stDataFrame {
        background: #1e293b !important;
        border-radius: 6px !important;
    }
    
    .stDataFrame td, .stDataFrame th {
        color: #e2e8f0 !important;
        font-size: 0.85rem !important;
        border-color: #334155 !important;
    }
    
    .stDataFrame th {
        background: #0f172a !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    
    /* Metrics override */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
    
    /* Dividers */
    hr {
        border-color: #334155 !important;
        margin: 1rem 0 !important;
    }
    
    .stDivider {
        background-color: #334155 !important;
    }
    
    /* Alerts/info boxes */
    .stAlert {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    
    /* Selectbox/inputs */
    .stSelectbox label, .stTextInput label {
        color: #94a3b8 !important;
    }
    
    .stSelectbox > div > div {
        background: #1e293b !important;
        border-color: #334155 !important;
    }
    
    /* Radio/checkbox */
    .stRadio label, .stCheckbox label {
        color: #e2e8f0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        border-radius: 6px !important;
        color: #94a3b8 !important;
    }
    
    .streamlit-expanderContent {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
    }
    
    /* Column gaps */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }
    
    /* Status badge styles */
    .status-running {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        color: #22c55e;
    }
    
    .status-stopped {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #ef4444;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    /* Slider */
    .stSlider label {
        color: #e2e8f0 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #22c55e !important;
""", unsafe_allow_html=True)


def is_running_on_render():
    """Check if running on Render cloud platform"""
    return os.environ.get("RENDER") == "true"


def is_bot_running():
    """Check if the bot process is running - cached to prevent flickering"""
    # On Render, bot status is managed by the separate worker service
    # We assume it's running if we're on Render (check Render dashboard for actual status)
    if is_running_on_render():
        return True  # Bot runs as separate Render Background Worker

    # Use session state to cache the result for stability
    if 'bot_status_cache' not in st.session_state:
        st.session_state.bot_status_cache = False
        st.session_state.bot_status_time = 0

    # Only recheck every 2 seconds to prevent flickering
    current_time = time.time()
    if current_time - st.session_state.bot_status_time > 2:
        try:
            result = subprocess.run(["pgrep", "-f", "run_bot.py|snuffs_bot.main"], capture_output=True, text=True, timeout=1)
            st.session_state.bot_status_cache = result.returncode == 0
            st.session_state.bot_status_time = current_time
        except Exception:
            pass  # Keep previous status on error

    return st.session_state.bot_status_cache


def start_bot():
    """Start the trading bot"""
    # On Render, bot is managed as separate Background Worker service
    if is_running_on_render():
        st.info("Bot runs as a separate Render service. Check Render dashboard to manage.")
        return True

    if is_bot_running():
        return True
    try:
        # Use relative paths that work in any environment
        import sys
        bot_dir = Path(__file__).parent.parent
        bot_script = bot_dir / "run_bot.py"

        # Try run_bot.py first, fall back to module
        if bot_script.exists():
            subprocess.Popen(
                [sys.executable, str(bot_script)],
                stdout=open("/tmp/snuffs_bot.log", "a"),
                stderr=subprocess.STDOUT,
                cwd=str(bot_dir),
                start_new_session=True
            )
        else:
            # Use module invocation
            subprocess.Popen(
                [sys.executable, "-m", "snuffs_bot.main", "trade", "--paper"],
                stdout=open("/tmp/snuffs_bot.log", "a"),
                stderr=subprocess.STDOUT,
                cwd=str(bot_dir),
                start_new_session=True
            )
        time.sleep(2)
        return is_bot_running()
    except Exception as e:
        st.error(f"Failed to start: {e}")
        return False


def stop_bot():
    """Stop the trading bot"""
    # On Render, bot is managed as separate Background Worker service
    if is_running_on_render():
        st.info("Bot runs as a separate Render service. Check Render dashboard to manage.")
        return False

    try:
        subprocess.run(["pkill", "-f", "run_bot.py|snuffs_bot.main"], capture_output=True)
        time.sleep(1)
        return not is_bot_running()
    except Exception:
        return False


def render_top_nav():
    """Render top navigation bar - Clean, professional design"""
    pages = ["Dashboard", "Positions", "Trades", "AI Decisions", "Reports", "Settings"]
    
    config = load_dashboard_config()
    current_page = st.session_state.get("current_page", "Dashboard")
    trading_mode = config.get("trading_mode", "Paper")
    bot_running = is_bot_running()
    
    # Clean header with minimal styling
    mode_class = "positive" if trading_mode == "Paper" else "negative"
    mode_color = "#22c55e" if trading_mode == "Paper" else "#ef4444"
    bot_color = "#22c55e" if bot_running else "#64748b"
    bot_text = "Running" if bot_running else "Stopped"
    
    st.markdown(f'''
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #334155;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">Spywave</span>
                <span style="color: {mode_color}; font-size: 0.7rem; font-weight: 500; padding: 3px 10px; border: 1px solid {mode_color}; border-radius: 4px; text-transform: uppercase;">{trading_mode}</span>
                <span style="color: {bot_color}; font-size: 0.75rem; font-weight: 500;">• {bot_text}</span>
            </div>
            <div style="color: #64748b; font-size: 0.8rem;">
                {datetime.now().strftime("%b %d")} <span style="color: #94a3b8; font-weight: 500;">{datetime.now().strftime("%H:%M")}</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Use streamlit columns for navigation (6 pages now)
    cols = st.columns([1, 1, 1, 1, 1, 1])
    selected_page = current_page
    
    # Map display names back to internal names
    page_map = {
        "Positions": "Open Positions",
        "Trades": "Trade History",
    }
    
    # Reverse map for checking active state
    reverse_page_map = {v: k for k, v in page_map.items()}
    
    for i, page in enumerate(pages):
        with cols[i]:
            # Check if this page is the current/active one
            internal_name = page_map.get(page, page)
            is_active = (internal_name == current_page) or (page == current_page)
            
            if st.button(page, key=f"nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
                # Map display name to internal name
                selected_page = internal_name
    
    # Also map current_page for comparison
    return selected_page


def render_dashboard_page():
    """Main dashboard page with overview - Clean design"""
    provider = get_data_provider()
    portfolio = provider.get_portfolio_summary()
    market_data = provider.get_market_data()
    bot_running = is_bot_running()
    
    # Key Metrics Row - Clean stat cards
    cols = st.columns(6, gap="small")
    
    with cols[0]:
        pnl = float(portfolio.get("daily_pnl", 0))
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        accent_style = f"border-left: 3px solid {pnl_color};"
        st.markdown(f'''
        <div class="metric-card" style="{accent_style}">
            <div class="metric-value-lg" style="color: {pnl_color};">${pnl:+,.2f}</div>
            <div class="metric-label">Today's P&L</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        account = float(portfolio.get("account_value", 1000))
        starting = float(portfolio.get("starting_capital", 1000))
        change = ((account - starting) / starting * 100) if starting > 0 else 0
        change_color = "#22c55e" if change >= 0 else "#ef4444"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${account:,.0f}</div>
            <div class="metric-label">Account</div>
            <div class="metric-delta" style="color: {change_color};">{change:+.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[2]:
        positions = portfolio.get("open_positions", 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{positions}</div>
            <div class="metric-label">Positions</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[3]:
        total = portfolio.get("total_trades", 0)
        wins = portfolio.get("winning_trades", 0)
        win_rate = (wins / total * 100) if total > 0 else 0
        wr_color = "#22c55e" if win_rate >= 55 else "#f59e0b" if win_rate >= 45 else "#64748b"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: {wr_color};">{win_rate:.0f}%</div>
            <div class="metric-label">Win Rate</div>
            <div class="metric-delta" style="color: #64748b;">{wins}W · {total-wins}L</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[4]:
        buying_power = float(portfolio.get("buying_power", 0))
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${buying_power:,.0f}</div>
            <div class="metric-label">Buying Power</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[5]:
        spy = market_data.get("spy_price", 0)
        spy_chg = market_data.get("spy_change_pct", 0)
        spy_color = "#22c55e" if spy_chg >= 0 else "#ef4444"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${spy:.2f}</div>
            <div class="metric-label">SPY</div>
            <div class="metric-delta" style="color: {spy_color};">{spy_chg:+.2f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Thin separator line
    st.markdown('<hr style="border: none; border-top: 1px solid #475569; margin: 16px 0;">', unsafe_allow_html=True)
    
    # Learning Mode Banner (if in learning mode)
    learning_status = provider.get_learning_mode_status()
    if learning_status.get("is_learning_mode"):
        accuracy_text = f"{learning_status['model_accuracy']*100:.0f}%" if learning_status.get("model_accuracy") else "Training..."

        # ML Training Progress
        trades_collected = learning_status.get("total_trades_collected", 0)
        min_trades = learning_status.get("min_trades_required", 100)
        progress_pct = min(100, (trades_collected / min_trades) * 100)

        # Progress bar color: yellow when collecting, green when ready
        progress_color = "#22c55e" if trades_collected >= min_trades else "#fbbf24"
        progress_text = f"{trades_collected}/{min_trades}" if trades_collected < min_trades else "Ready!"

        st.markdown(f'''
        <div style="background: linear-gradient(90deg, #1e1b4b 0%, #312e81 100%); border: 1px solid #6366f1; border-radius: 8px; padding: 12px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #6366f1; padding: 6px 12px; border-radius: 4px; font-weight: 600; color: white; font-size: 0.8rem;">LEARNING MODE</div>
                <div style="color: #c7d2fe; font-size: 0.85rem;">{learning_status['description']}</div>
            </div>
            <div style="display: flex; gap: 24px; align-items: center;">
                <div style="text-align: center;">
                    <div style="color: #a5b4fc; font-size: 0.7rem;">ML TRAINING PROGRESS</div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 80px; height: 8px; background: #374151; border-radius: 4px; overflow: hidden;">
                            <div style="width: {progress_pct}%; height: 100%; background: {progress_color};"></div>
                        </div>
                        <div style="color: {progress_color}; font-weight: 600; font-size: 0.85rem;">{progress_text}</div>
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #a5b4fc; font-size: 0.7rem;">TODAY</div>
                    <div style="color: white; font-weight: 600;">{learning_status['total_trades_today']}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #a5b4fc; font-size: 0.7rem;">MODEL</div>
                    <div style="color: white; font-weight: 600;">{accuracy_text}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Main content: Bot Control + Positions + Recent Trades
    col_bot, col_positions, col_trades = st.columns([1, 2, 2])
    
    # Bot Control - Clean, minimal design
    with col_bot:
        st.markdown("#### Bot Control")
        
        if bot_running:
            if st.button("Stop Bot", use_container_width=True, key="stop", type="secondary"):
                if stop_bot():
                    st.rerun()
            
            st.markdown('''
            <div style="background: #1e293b; border: 1px solid #22c55e; border-radius: 6px; padding: 12px; text-align: center; margin-top: 8px;">
                <div style="color: #22c55e; font-size: 0.9rem; font-weight: 600;">Running</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">Actively trading</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            if st.button("Start Bot", use_container_width=True, key="start", type="primary"):
                if start_bot():
                    st.rerun()
            
            st.markdown('''
            <div style="background: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 12px; text-align: center; margin-top: 8px;">
                <div style="color: #64748b; font-size: 0.9rem; font-weight: 600;">Stopped</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">Click to start</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Tastytrade connection status (only once)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        tt_status = provider.get_tastytrade_status()
        tt_color = "#22c55e" if tt_status["connected"] else "#64748b"
        tt_text = tt_status.get("account_number", "Connected") if tt_status["connected"] else tt_status.get("error", "Disconnected")
        st.markdown(f'''
        <div style="background: #1e293b; border: 1px solid {tt_color}; border-radius: 6px; padding: 10px; text-align: center;">
            <div style="color: {tt_color}; font-size: 0.8rem; font-weight: 500;">Tastytrade</div>
            <div style="color: #94a3b8; font-size: 0.7rem; margin-top: 2px;">{tt_text}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # AI Learning Status
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        learner_status = provider.get_background_learner_status()
        if learner_status.get("running"):
            sims = learner_status.get("simulations_run", 0)
            snapshots = learner_status.get("total_snapshots", 0)
            st.markdown(f'''
            <div style="background: #1e293b; border: 1px solid #6366f1; border-radius: 6px; padding: 10px; text-align: center;">
                <div style="color: #6366f1; font-size: 0.8rem; font-weight: 500;">AI Learning</div>
                <div style="color: #94a3b8; font-size: 0.7rem; margin-top: 2px;">{snapshots:,} snapshots</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 10px; text-align: center;">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">AI Learning</div>
                <div style="color: #64748b; font-size: 0.7rem; margin-top: 2px;">Waiting</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Positions column
    with col_positions:
        st.markdown("#### Open Positions")
        positions_df = provider.get_open_positions()
        
        if positions_df.empty or (len(positions_df) == 1 and "No open positions" in str(positions_df.iloc[0].get("Position ID", ""))):
            st.info("No open positions")
        else:
            for idx, row in positions_df.iterrows():
                strategy = format_strategy_name(row.get('Strategy', 'N/A'))
                pnl = float(row.get('Unrealized P&L', 0))
                pnl_pct = float(row.get('P&L %', 0)) if 'P&L %' in row else 0
                position_id = str(row.get('Position ID', ''))
                entry_price = float(row.get('Entry Price', 0))
                current_price = float(row.get('Current Price', 0))
                strike = float(row.get('Strike', 0))
                spy_price = float(row.get('SPY', 0))
                entry_time = str(row.get('Entry Time', ''))
                # Extract just time from entry_time
                if ' ' in entry_time:
                    entry_time = entry_time.split(' ')[1][:8]
                elif 'T' in entry_time:
                    entry_time = entry_time.split('T')[1][:8]
                
                pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                direction_emoji = "📈" if "CALL" in strategy.upper() else "📉"
                
                st.markdown(f'''
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #f8fafc; font-size: 1rem;">{direction_emoji} {strategy}</span>
                        <span style="color: {pnl_color}; font-weight: 700; font-size: 1.1rem;">${pnl:+.2f} ({pnl_pct:+.1f}%)</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; font-size: 0.8rem; color: #94a3b8;">
                        <div><span style="color: #64748b;">Strike:</span> <span style="color: #e2e8f0;">${strike:.0f}</span></div>
                        <div><span style="color: #64748b;">Entry:</span> <span style="color: #e2e8f0;">${entry_price:.2f}</span></div>
                        <div><span style="color: #64748b;">Current:</span> <span style="color: #e2e8f0;">${current_price:.2f}</span></div>
                        <div><span style="color: #64748b;">SPY@Entry:</span> <span style="color: #e2e8f0;">${spy_price:.2f}</span></div>
                        <div><span style="color: #64748b;">Time:</span> <span style="color: #e2e8f0;">{entry_time}</span></div>
                        <div></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                if st.button("Close Position", key=f"close_{position_id}_{idx}", type="secondary", use_container_width=True):
                    result = provider.force_close_position(position_id)
                    if result.get("success"):
                        st.rerun()
    
    # Trades column - Today's trades only
    with col_trades:
        st.markdown("#### Today's Trades")
        trades_df = provider.get_recent_trades(limit=200)  # Get enough to cover full day
        
        # Filter to today only
        today = datetime.now().strftime("%Y-%m-%d")
        if not trades_df.empty and 'Entry Time' in trades_df.columns:
            trades_df = trades_df[trades_df['Entry Time'].astype(str).str.startswith(today)]
        elif not trades_df.empty and 'Time' in trades_df.columns:
            pass
        
        if trades_df.empty:
            st.info("No trades today")
        else:
            # Show up to 5 most recent
            for idx, row in trades_df.head(5).iterrows():
                strategy = format_strategy_name(row.get('Strategy', 'N/A'))
                pnl = float(row.get('P&L', 0))
                entry = float(row.get('Entry', 0))
                exit_price = float(row.get('Exit', 0))
                exit_reason = row.get('Exit Reason', '')
                exit_time = row.get('Time', '')
                pnl_color = "#22c55e" if pnl > 0 else "#ef4444"
                result_indicator = "+" if pnl > 0 else "−"
                
                # Format exit reason to be shorter
                reason_short = exit_reason[:15] + ".." if len(str(exit_reason)) > 15 else exit_reason
                
                st.markdown(f'''
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 600; color: #f8fafc;">{strategy}</span>
                            <span style="color: #64748b; font-size: 0.75rem; margin-left: 8px;">{exit_time}</span>
                        </div>
                        <span style="color: {pnl_color}; font-weight: 600;">${pnl:+.2f}</span>
                    </div>
                    <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">
                        ${entry:.2f} → ${exit_price:.2f}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Show today's summary
            total_pnl = trades_df['P&L'].sum()
            wins = len(trades_df[trades_df['P&L'] > 0])
            losses = len(trades_df[trades_df['P&L'] <= 0])
            pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
            st.markdown(f'''
            <div style="text-align: center; padding: 8px; margin-top: 8px; border-top: 1px solid #334155;">
                <span style="color: {pnl_color}; font-weight: 600;">${total_pnl:+.2f}</span>
                <span style="color: #64748b; font-size: 0.8rem;"> ({wins}W / {losses}L)</span>
            </div>
            ''', unsafe_allow_html=True)
    
    st.divider()
    
    # Chart columns: P/L Chart + SPY TradingView Chart
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### P&L Performance")
        
        # Timeframe selector
        timeframe = st.selectbox(
            "Timeframe",
            ["Today", "This Week", "This Month", "All Time"],
            key="pnl_timeframe",
            label_visibility="collapsed"
        )
        
        # Get trade data for P/L chart
        all_trades = provider.get_recent_trades(limit=500)
        
        if not all_trades.empty and 'P&L' in all_trades.columns and 'Exit Time' in all_trades.columns:
            # Filter by timeframe
            try:
                all_trades['Exit Time'] = pd.to_datetime(all_trades['Exit Time'], errors='coerce')
                now = datetime.now()
                
                if timeframe == "Today":
                    mask = all_trades['Exit Time'].dt.date == now.date()
                elif timeframe == "This Week":
                    week_start = now - timedelta(days=now.weekday())
                    mask = all_trades['Exit Time'] >= week_start.replace(hour=0, minute=0, second=0)
                elif timeframe == "This Month":
                    mask = all_trades['Exit Time'].dt.month == now.month
                else:
                    mask = pd.Series([True] * len(all_trades))
                
                filtered_trades = all_trades[mask].copy()
                
                if not filtered_trades.empty:
                    # Sort by exit time and calculate cumulative P/L
                    filtered_trades = filtered_trades.sort_values('Exit Time')
                    filtered_trades['Cumulative P&L'] = filtered_trades['P&L'].cumsum()
                    
                    # Create the chart
                    fig = go.Figure()
                    
                    # Add cumulative P/L line
                    fig.add_trace(go.Scatter(
                        x=filtered_trades['Exit Time'],
                        y=filtered_trades['Cumulative P&L'],
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='#22c55e', width=2),
                        marker=dict(size=6),
                        fill='tozeroy',
                        fillcolor='rgba(34, 197, 94, 0.1)'
                    ))
                    
                    # Color markers based on individual trade P/L
                    colors = ['#22c55e' if p > 0 else '#ef4444' for p in filtered_trades['P&L']]
                    fig.add_trace(go.Scatter(
                        x=filtered_trades['Exit Time'],
                        y=filtered_trades['Cumulative P&L'],
                        mode='markers',
                        marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
                        showlegend=False,
                        hovertemplate='<b>Trade P&L:</b> $%{customdata:.2f}<br><b>Cumulative:</b> $%{y:.2f}<extra></extra>',
                        customdata=filtered_trades['P&L']
                    ))
                    
                    # Styling
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15, 23, 42, 0.5)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=300,
                        xaxis=dict(showgrid=False, color='#94a3b8'),
                        yaxis=dict(showgrid=True, gridcolor='#334155', color='#94a3b8', tickprefix='$'),
                        showlegend=False,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary stats - clean inline display
                    total_pnl = filtered_trades['P&L'].sum()
                    wins = len(filtered_trades[filtered_trades['P&L'] > 0])
                    losses = len(filtered_trades[filtered_trades['P&L'] <= 0])
                    pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
                    
                    st.markdown(f'''
                    <div style="display: flex; justify-content: space-around; text-align: center; padding: 12px; background: #1e293b; border: 1px solid #334155; border-radius: 6px;">
                        <div><span style="color: {pnl_color}; font-weight: 600; font-size: 1.1rem;">${total_pnl:+,.2f}</span><br><span style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Total</span></div>
                        <div><span style="color: #22c55e; font-weight: 500;">{wins}</span><br><span style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Wins</span></div>
                        <div><span style="color: #ef4444; font-weight: 500;">{losses}</span><br><span style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;">Losses</span></div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.info(f"No trades for {timeframe.lower()}")
            except Exception as e:
                st.info("No P&L data available yet")
        else:
            st.info("No trade data available yet")
    
    with chart_col2:
        st.markdown("#### SPY Chart")
        
        # Timeframe selector for TradingView
        tv_timeframe = st.selectbox(
            "Chart Interval",
            ["1", "5", "15", "60", "D", "W"],
            index=1,
            format_func=lambda x: {"1": "1 Min", "5": "5 Min", "15": "15 Min", "60": "1 Hour", "D": "Daily", "W": "Weekly"}[x],
            key="tv_timeframe",
            label_visibility="collapsed"
        )
        
        # TradingView Widget
        tradingview_html = f'''
        <div class="tradingview-widget-container" style="height:350px;width:100%">
            <iframe 
                scrolling="no" 
                allowtransparency="true" 
                frameborder="0" 
                src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol=SPY&interval={tv_timeframe}&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=1e293b&studies=[]&theme=dark&style=1&timezone=America%2FNew_York&studies_overrides={{}}&overrides={{}}&enabled_features=[]&disabled_features=[]&locale=en&utm_source=&utm_medium=widget_new&utm_campaign=chart&utm_term=SPY" 
                style="width: 100%; height: 350px; border: 1px solid #334155; border-radius: 6px;">
            </iframe>
        </div>
        '''
        st.components.v1.html(tradingview_html, height=360)
    
    # Developer Section - AI Learning Stats (collapsible)
    st.divider()
    
    with st.expander("Developer Stats", expanded=False):
        render_ai_developer_stats()


def render_ai_developer_stats():
    """Render detailed AI learning statistics for developer monitoring"""
    provider = get_data_provider()
    details = provider.get_ai_learning_details()
    
    # Top row: Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Simulations",
            details["simulations"]["total"],
            delta=f"{details['simulations']['win_rate']*100:.1f}% win rate" if details["simulations"]["total"] > 0 else None
        )
    
    with col2:
        st.metric(
            "Model Trained",
            "Yes" if details["model"]["trained"] else "No",
            delta=f"{details['model']['training_samples']} samples" if details["model"]["training_samples"] > 0 else None
        )
    
    with col3:
        conf_adj = details["self_improvement"]["confidence_adjustment"]
        st.metric(
            "Confidence Adj",
            f"{conf_adj:+.2f}",
            delta="More selective" if conf_adj > 0 else ("More aggressive" if conf_adj < 0 else "Neutral")
        )
    
    with col4:
        st.metric(
            "Data Points",
            f"{details['data']['total_snapshots']:,}",
            delta=f"{details['data']['total_trades']} real trades"
        )
    
    # Simulation results by action
    st.markdown("##### Simulation Results by Action")
    
    by_action = details["simulations"]["by_action"]
    if by_action:
        action_cols = st.columns(len(by_action))
        for idx, (action, stats) in enumerate(by_action.items()):
            with action_cols[idx]:
                win_rate = stats["win_rate"] * 100
                color = "#22c55e" if win_rate >= 55 else ("#f59e0b" if win_rate >= 45 else "#ef4444")
                display_action = format_strategy_name(action)
                st.markdown(f'''
                <div style="background: rgba(30, 41, 59, 0.8); border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="color: #e2e8f0; font-weight: 600; font-size: 0.9rem;">{display_action}</div>
                    <div style="color: {color}; font-size: 1.5rem; font-weight: 700;">{win_rate:.1f}%</div>
                    <div style="color: #64748b; font-size: 0.75rem;">{stats['wins']}/{stats['count']} wins</div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No simulation data yet - waiting for market hours")
    
    # Recent simulations table
    st.markdown("##### Recent Simulations")
    
    recent = details["simulations"]["recent"]
    if recent:
        for sim in recent[:5]:  # Show last 5
            result_emoji = "✅" if sim["profitable"] else "❌"
            change_color = "#22c55e" if sim["change_pct"] and sim["change_pct"] > 0 else "#ef4444"
            display_action = format_strategy_name(sim['action'])
            
            c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
            with c1:
                st.markdown(result_emoji)
            with c2:
                st.markdown(f"**{display_action}**")
            with c3:
                st.markdown(f"Conf: {sim['confidence']*100:.0f}%")
            with c4:
                if sim['entry'] and sim['exit']:
                    st.markdown(f"${sim['entry']:.2f} → ${sim['exit']:.2f}")
                else:
                    st.markdown("Pending...")
            with c5:
                if sim['change_pct']:
                    st.markdown(f'<span style="color: {change_color};">{sim["change_pct"]:+.2f}%</span>', unsafe_allow_html=True)
    else:
        st.info("No recent simulations")
    
    # Learned patterns
    st.markdown("##### Learned Patterns")
    
    patterns = details["self_improvement"]["learned_patterns"]
    if patterns:
        pat_col1, pat_col2 = st.columns(2)
        
        with pat_col1:
            avoid_hours = patterns.get("avoid_hours", [])
            if avoid_hours:
                st.markdown(f"**Avoid Hours:** {', '.join(str(h) + ':00' for h in avoid_hours)}")
            
            preferred_hours = patterns.get("preferred_hours", [])
            if preferred_hours:
                st.markdown(f"**Preferred Hours:** {', '.join(str(h) + ':00' for h in preferred_hours)}")
        
        with pat_col2:
            avoid_strats = patterns.get("avoid_strategies", [])
            if avoid_strats:
                st.markdown(f"**Avoid Strategies:** {', '.join(avoid_strats)}")
            
            preferred_strats = patterns.get("preferred_strategies", [])
            if preferred_strats:
                st.markdown(f"**Preferred Strategies:** {', '.join(preferred_strats)}")
        
        if not any([avoid_hours, preferred_hours, avoid_strats, preferred_strats]):
            st.info("No patterns learned yet - needs more data")
    else:
        st.info("No patterns learned yet - needs more data")
    
    # Hourly performance heatmap
    hourly = details.get("hourly_performance", {})
    if hourly:
        st.markdown("##### Performance by Hour")
        
        hours_data = []
        for hour, stats in sorted(hourly.items()):
            win_rate = stats.get("win_rate", 0) * 100
            total = stats.get("total_trades", 0)
            hours_data.append({"Hour": f"{hour}:00", "Win Rate %": win_rate, "Trades": total})
        
        if hours_data:
            import pandas as pd
            df = pd.DataFrame(hours_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Model accuracy
    if details["model"]["accuracy"]:
        st.markdown(f"##### Model Accuracy: **{details['model']['accuracy']*100:.1f}%**")
        if details["model"]["last_training"]:
            st.caption(f"Last trained: {details['model']['last_training']}")


def render_positions_page():
    """Full positions page"""
    provider = get_data_provider()
    positions_df = provider.get_open_positions()
    
    st.markdown("### Open Positions")
    
    if positions_df.empty or (len(positions_df) == 1 and "No open positions" in str(positions_df.iloc[0].get("Position ID", ""))):
        st.info("No open positions currently")
        return
    
    # Header row with more columns
    h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns([2, 1, 1, 1, 1, 1.2, 1, 1, 1])
    h1.markdown("**Strategy**")
    h2.markdown("**Strike**")
    h3.markdown("**SPY@Entry**")
    h4.markdown("**Entry**")
    h5.markdown("**Current**")
    h6.markdown("**P&L**")
    h7.markdown("**Time**")
    h8.markdown("**Qty**")
    h9.markdown("**Action**")
    
    st.divider()
    
    for idx, row in positions_df.iterrows():
        cols = st.columns([2, 1, 1, 1, 1, 1.2, 1, 1, 1])
        
        strategy = format_strategy_name(row.get('Strategy', 'N/A'))
        strike = float(row.get('Strike', 0))
        spy_price = float(row.get('SPY', 0))
        entry = float(row.get('Entry Price', 0))
        current = float(row.get('Current Price', entry))
        pnl = float(row.get('Unrealized P&L', 0))
        pnl_pct = float(row.get('P&L %', 0)) if 'P&L %' in row else 0
        position_id = str(row.get('Position ID', ''))
        qty = int(row.get('Qty', 1))
        entry_time = str(row.get('Entry Time', ''))
        # Extract just time from entry_time
        if ' ' in entry_time:
            entry_time = entry_time.split(' ')[1][:8]
        elif 'T' in entry_time:
            entry_time = entry_time.split('T')[1][:8]
        
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        direction_emoji = "📈" if "CALL" in strategy.upper() else "📉"
        
        cols[0].markdown(f"**{direction_emoji} {strategy}**")
        cols[1].markdown(f"${strike:.0f}")
        cols[2].markdown(f"${spy_price:.2f}")
        cols[3].markdown(f"${entry:.2f}")
        cols[4].markdown(f"${current:.2f}")
        cols[5].markdown(f'<span style="color: {pnl_color}; font-weight: 600;">${pnl:+.2f} ({pnl_pct:+.1f}%)</span>', unsafe_allow_html=True)
        cols[6].markdown(f"{entry_time}")
        cols[7].markdown(f"{qty}")
        
        with cols[8]:
            if st.button("SELL", key=f"sell_{position_id}_{idx}", type="primary"):
                result = provider.force_close_position(position_id)
                if result.get("success"):
                    st.success(f"Closed for ${result.get('pnl', 0):.2f}")
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed"))


def render_trades_page():
    """Trades history page - Shows trades grouped by day"""
    provider = get_data_provider()
    trades_df = provider.get_recent_trades(limit=500, full_details=True)

    st.markdown("### Trade History")

    # Timeframe selector
    col1, col2 = st.columns([1, 3])
    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time"],
            index=0,
            label_visibility="collapsed"
        )

    # Filter by timeframe
    if not trades_df.empty and 'Entry Time' in trades_df.columns:
        trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'], errors='coerce')
        now = datetime.now()

        if timeframe == "Today":
            mask = trades_df['Entry Time'].dt.date == now.date()
        elif timeframe == "Last 7 Days":
            week_ago = now - timedelta(days=7)
            mask = trades_df['Entry Time'] >= week_ago
        elif timeframe == "Last 30 Days":
            month_ago = now - timedelta(days=30)
            mask = trades_df['Entry Time'] >= month_ago
        else:  # All Time
            mask = pd.Series([True] * len(trades_df))

        trades_df = trades_df[mask]

    if trades_df.empty:
        st.info(f"No trades for {timeframe.lower()}")
        return

    # Add date column for grouping
    trades_df['Trade Date'] = trades_df['Entry Time'].dt.date

    # Get unique dates sorted descending (most recent first)
    unique_dates = sorted(trades_df['Trade Date'].unique(), reverse=True)

    # Overall summary for the period
    total_trades = len(trades_df)
    total_pnl = trades_df['P&L'].sum() if 'P&L' in trades_df.columns else 0
    total_fees = trades_df['Fees'].sum() if 'Fees' in trades_df.columns else 0
    total_gross = trades_df['Gross P&L'].sum() if 'Gross P&L' in trades_df.columns else total_pnl
    wins = len(trades_df[trades_df['P&L'] > 0]) if 'P&L' in trades_df.columns else 0
    losses = len(trades_df[trades_df['P&L'] < 0]) if 'P&L' in trades_df.columns else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(f"{timeframe} Trades", total_trades)
    col2.metric("Net P&L", f"${total_pnl:+,.2f}")
    col3.metric("Total Fees", f"${total_fees:.2f}")
    col4.metric("Win Rate", f"{win_rate:.1f}%")
    col5.metric("Wins", wins)
    col6.metric("Losses", losses)

    # Show cost breakdown
    if total_fees > 0:
        st.caption(f"💰 Gross P&L: ${total_gross:+,.2f} → After Fees: ${total_pnl:+,.2f} (Fees: -${total_fees:.2f})")

    st.markdown("---")

    # Display trades grouped by day
    for trade_date in unique_dates:
        day_trades = trades_df[trades_df['Trade Date'] == trade_date].copy()

        # Calculate daily stats
        day_pnl = day_trades['P&L'].sum() if 'P&L' in day_trades.columns else 0
        day_wins = len(day_trades[day_trades['P&L'] > 0]) if 'P&L' in day_trades.columns else 0
        day_losses = len(day_trades[day_trades['P&L'] < 0]) if 'P&L' in day_trades.columns else 0
        day_total = len(day_trades)
        day_win_rate = (day_wins / day_total * 100) if day_total > 0 else 0

        # Format date header
        date_str = trade_date.strftime("%A, %B %d, %Y")
        pnl_icon = "🟢" if day_pnl >= 0 else "🔴"

        # Create expandable section for each day
        with st.expander(
            f"{pnl_icon} **{date_str}** — {day_total} trades | P&L: ${day_pnl:+,.2f} | Win Rate: {day_win_rate:.0f}%",
            expanded=(trade_date == unique_dates[0])  # Expand most recent day
        ):
            # Calculate transaction cost totals for the day
            day_fees = day_trades['Fees'].sum() if 'Fees' in day_trades.columns else 0
            day_slippage = day_trades['Slippage'].sum() if 'Slippage' in day_trades.columns else 0
            day_gross = day_trades['Gross P&L'].sum() if 'Gross P&L' in day_trades.columns else day_pnl

            # Daily stats row - include transaction costs
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Trades", day_total)
            c2.metric("Net P&L", f"${day_pnl:+,.2f}")
            c3.metric("Win Rate", f"{day_win_rate:.1f}%")
            c4.metric("Fees", f"${day_fees:.2f}")
            c5.metric("Wins", day_wins)
            c6.metric("Losses", day_losses)

            # Show cost breakdown if we have the data
            if day_fees > 0 or day_slippage > 0:
                st.caption(f"📊 Gross P&L: ${day_gross:+,.2f} | Fees: -${day_fees:.2f} | Slippage: -${day_slippage:.4f}")

            # Format the trades table for this day - include cost columns
            display_cols = [
                'Strategy', 'Entry Time', 'Exit Time', 'Entry', 'Exit',
                'Gross P&L', 'Fees', 'Net P&L', 'P&L %',
                'Exit Reason', 'Quantity', 'SPY Price', 'VIX'
            ]
            available_cols = [c for c in display_cols if c in day_trades.columns]

            if available_cols:
                display_df = day_trades[available_cols].copy()

                # Format Entry Time to show only time (date is in header)
                if 'Entry Time' in display_df.columns:
                    display_df['Entry Time'] = pd.to_datetime(display_df['Entry Time']).dt.strftime('%H:%M:%S')
                if 'Exit Time' in display_df.columns:
                    display_df['Exit Time'] = pd.to_datetime(display_df['Exit Time'], errors='coerce').dt.strftime('%H:%M:%S')

                # Format P&L columns
                if 'P&L %' in display_df.columns:
                    display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")

                # Format currency columns
                for col in ['Gross P&L', 'Net P&L', 'P&L']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "")

                for col in ['Entry', 'Exit', 'SPY Price']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")

                # Format fees (negative cost)
                if 'Fees' in display_df.columns:
                    display_df['Fees'] = display_df['Fees'].apply(lambda x: f"-${x:.2f}" if pd.notna(x) and x > 0 else "$0.00")

                # Color P&L
                def color_pnl(val):
                    try:
                        if isinstance(val, str):
                            num = float(val.replace('$', '').replace('%', '').replace('+', '').replace(',', ''))
                        else:
                            num = float(val)
                        if num > 0:
                            return 'color: #00c853; font-weight: 600'
                        elif num < 0:
                            return 'color: #ff1744; font-weight: 600'
                    except (ValueError, TypeError, AttributeError):
                        pass
                    return ''

                styled_df = display_df.style.applymap(color_pnl, subset=[c for c in ['P&L', 'P&L %'] if c in display_df.columns])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(day_trades, use_container_width=True, hide_index=True)

    # Export all filtered trades
    st.markdown("---")
    csv = trades_df.drop(columns=['Trade Date'], errors='ignore').to_csv(index=False)
    st.download_button("📥 Download All Trades (CSV)", csv, f"trades_{timeframe.lower().replace(' ', '_')}.csv", "text/csv")
    
    # Calculate comprehensive stats for today
    total_trades = len(trades_df)
    if 'P&L' in trades_df.columns:
        total_pnl = trades_df['P&L'].sum()
        wins = len(trades_df[trades_df['P&L'] > 0])
        losses = len(trades_df[trades_df['P&L'] < 0])
        breakeven = len(trades_df[trades_df['P&L'] == 0])
        avg_win = trades_df[trades_df['P&L'] > 0]['P&L'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['P&L'] < 0]['P&L'].mean() if losses > 0 else 0
        max_win = trades_df['P&L'].max() if not trades_df.empty else 0
        max_loss = trades_df['P&L'].min() if not trades_df.empty else 0
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor: gross profits / gross losses
        gross_profit = trades_df[trades_df['P&L'] > 0]['P&L'].sum() if wins > 0 else 0
        gross_loss = abs(trades_df[trades_df['P&L'] < 0]['P&L'].sum()) if losses > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        loss_rate = losses / total_trades if total_trades > 0 else 0
        expectancy = (win_rate/100 * avg_win) + (loss_rate * avg_loss) if total_trades > 0 else 0
    else:
        total_pnl = wins = losses = breakeven = avg_win = avg_loss = 0
        max_win = max_loss = win_rate = profit_factor = expectancy = 0
        gross_profit = gross_loss = 0
    
    # Row 1: Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", total_trades)
    col2.metric("Total P&L", f"${total_pnl:+,.2f}", delta=f"{'▲' if total_pnl > 0 else '▼' if total_pnl < 0 else ''}")
    col3.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins}W / {losses}L")
    col4.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
    col5.metric("Expectancy", f"${expectancy:+.2f}/trade")
    
    # Row 2: Win/Loss details
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Wins", wins, delta=f"${gross_profit:+,.2f}")
    col2.metric("Losses", losses, delta=f"${-gross_loss:,.2f}" if gross_loss > 0 else "$0.00")
    col3.metric("Avg Win", f"${avg_win:+.2f}")
    col4.metric("Avg Loss", f"${avg_loss:.2f}")
    col5.metric("Breakeven", breakeven)
    
    # Row 3: Extremes
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Trade", f"${max_win:+.2f}")
    col2.metric("Worst Trade", f"${max_loss:.2f}")
    
    # Strategy breakdown if available
    if 'Strategy' in trades_df.columns and 'P&L' in trades_df.columns:
        strategy_stats = trades_df.groupby('Strategy').agg({
            'P&L': ['count', 'sum', 'mean', lambda x: (x > 0).sum()]
        }).round(2)
        strategy_stats.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Wins']
        strategy_stats['Win Rate'] = (strategy_stats['Wins'] / strategy_stats['Trades'] * 100).round(1)
        
        # Show best and worst strategy
        if len(strategy_stats) > 0:
            best_strat = strategy_stats['Total P&L'].idxmax()
            best_strat_pnl = strategy_stats.loc[best_strat, 'Total P&L']
            col3.metric("Best Strategy", best_strat, delta=f"${best_strat_pnl:+.2f}")
            
            if len(strategy_stats) > 1:
                worst_strat = strategy_stats['Total P&L'].idxmin()
                worst_strat_pnl = strategy_stats.loc[worst_strat, 'Total P&L']
                col4.metric("Worst Strategy", worst_strat, delta=f"${worst_strat_pnl:+.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Strategy breakdown table
    if 'Strategy' in trades_df.columns and 'P&L' in trades_df.columns:
        st.markdown("#### Strategy Breakdown")
        strategy_display = strategy_stats.reset_index()
        strategy_display.columns = ['Strategy', 'Trades', 'Total P&L', 'Avg P&L', 'Wins', 'Win Rate %']
        st.dataframe(strategy_display, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Trade history table
    st.markdown("#### All Trades")
    
    # Define columns to display in order of importance
    display_cols = [
        'Strategy', 'Entry Time', 'Exit Time', 'Entry', 'Exit', 'P&L', 'P&L %',
        'Exit Reason', 'Quantity', 'SPY Price', 'VIX', 'Delta', 'Market', 'Fees'
    ]
    available_cols = [c for c in display_cols if c in trades_df.columns]
    
    # Fallback columns if above not available
    if not available_cols:
        display_cols = ['Strategy', 'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'P&L', 'P&L %', 'Exit Reason', 'Type']
        available_cols = [c for c in display_cols if c in trades_df.columns]
    
    if available_cols:
        display_df = trades_df[available_cols].copy()
        
        # Style function for coloring wins/losses
        def color_pnl(val):
            try:
                if isinstance(val, str):
                    # Extract number from formatted string like "$10.50" or "+5.2%"
                    num = float(val.replace('$', '').replace('%', '').replace('+', '').replace(',', ''))
                else:
                    num = float(val)
                if num > 0:
                    return 'color: #00c853; font-weight: 600'  # Green
                elif num < 0:
                    return 'color: #ff1744; font-weight: 600'  # Red
            except:
                pass
            return ''
        
        # Format P&L % column if present
        if 'P&L %' in display_df.columns:
            display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) and x != 0 else "0.0%")
        
        # Format currency columns
        for col in ['P&L', 'Entry', 'Exit', 'SPY Price', 'Fees']:
            if col in display_df.columns:
                if col == 'P&L':
                    display_df[col] = display_df[col].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "")
                else:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        # Apply styling for P&L columns (green/red)
        styled_df = display_df.style.applymap(color_pnl, subset=[c for c in ['P&L', 'P&L %'] if c in display_df.columns])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True, height=400)
    
    # Export
    csv = trades_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "trades_today.csv", "text/csv")


def render_ai_page():
    """AI decisions page - Shows today's decisions only"""
    provider = get_data_provider()
    decisions_df = provider.get_ai_decisions(limit=100)
    
    st.markdown("### AI Decisions")
    st.caption("Today's AI decisions • Use Reports for historical data")
    
    # Filter to today only
    today = datetime.now().strftime("%Y-%m-%d")
    if not decisions_df.empty and 'Time' in decisions_df.columns:
        # Time column is in HH:MM:SS format, so we need to check differently
        # Get today's decisions from the provider (already filtered by recent)
        pass  # The get_ai_decisions already returns recent ones, but let's be safe
    
    if decisions_df.empty:
        st.info("No AI decisions today • Use Reports to view historical data")
        return
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(decisions_df)
    executes = len(decisions_df[decisions_df['Decision'] == 'EXECUTE']) if 'Decision' in decisions_df.columns else 0
    holds = len(decisions_df[decisions_df['Decision'] == 'HOLD']) if 'Decision' in decisions_df.columns else 0
    rejects = len(decisions_df[decisions_df['Decision'] == 'REJECT']) if 'Decision' in decisions_df.columns else 0
    
    col1.metric("Today's Decisions", total)
    col2.metric("Execute", executes)
    col3.metric("Hold", holds)
    col4.metric("Reject", rejects)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(decisions_df, use_container_width=True, hide_index=True, height=400)


def render_reports_page():
    """Reports page - Pull historical data by date for analysis"""
    provider = get_data_provider()
    
    st.markdown("### Reports")
    st.caption("View historical trades and AI decisions for any date")
    
    # Date selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Get available dates
        available_dates = provider.get_available_dates(days_back=60)
        
        if available_dates:
            # Default to most recent date with data
            min_date = min(available_dates) if available_dates else datetime.now().date() - timedelta(days=30)
            max_date = datetime.now().date()
            
            selected_date = st.date_input(
                "Select Date",
                value=available_dates[0] if available_dates else datetime.now().date(),
                min_value=min_date,
                max_value=max_date,
                key="report_date"
            )
            
            # Report type selection
            report_type = st.radio(
                "Report Type",
                ["Summary", "All Trades", "AI Decisions", "Full Export"],
                key="report_type"
            )
        else:
            st.info("No historical data available yet")
            return
    
    with col2:
        if selected_date:
            # Get summary for the selected date
            summary = provider.get_daily_summary(selected_date)
            
            # Summary metrics row
            st.markdown(f"#### {selected_date.strftime('%A, %B %d, %Y')}")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            pnl_color = "normal" if summary["total_pnl"] >= 0 else "inverse"
            m1.metric("Total P&L", f"${summary['total_pnl']:+,.2f}")
            m2.metric("Trades", summary["closed_trades"])
            m3.metric("Win Rate", f"{summary['win_rate']:.1f}%")
            m4.metric("AI Decisions", summary["total_ai_decisions"])
            m5.metric("Executes", summary["execute_decisions"])
            
            st.divider()
            
            if report_type == "Summary":
                # Detailed summary
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("##### Trade Statistics")
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Total Trades | {summary['total_trades']} |
                    | Closed Trades | {summary['closed_trades']} |
                    | Winning Trades | {summary['winning_trades']} |
                    | Losing Trades | {summary['losing_trades']} |
                    | Win Rate | {summary['win_rate']:.1f}% |
                    | Average Win | ${summary['avg_win']:+.2f} |
                    | Average Loss | ${summary['avg_loss']:.2f} |
                    | Best Trade | ${summary['best_trade']:+.2f} |
                    | Worst Trade | ${summary['worst_trade']:.2f} |
                    """)
                
                with col_b:
                    st.markdown("##### AI Decision Statistics")
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Total Decisions | {summary['total_ai_decisions']} |
                    | Execute | {summary['execute_decisions']} |
                    | Reject | {summary['reject_decisions']} |
                    | Hold | {summary['hold_decisions']} |
                    """)
                    
                    if summary["strategies_used"]:
                        st.markdown("##### Strategies Used")
                        for strat in summary["strategies_used"]:
                            st.markdown(f"- {format_strategy_name(strat)}")
            
            elif report_type == "All Trades":
                # Full trade details
                st.markdown("##### Trade Details")
                trades_df = provider.get_trades_for_date(selected_date)
                
                if trades_df.empty:
                    st.info("No trades found for this date")
                else:
                    # Show key columns
                    display_cols = ["Entry Time", "Exit Time", "Strategy", "Status", 
                                    "Entry Price", "Exit Price", "P&L", "P&L %", "Exit Reason"]
                    available_cols = [c for c in display_cols if c in trades_df.columns]
                    
                    st.dataframe(trades_df[available_cols], use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        "Download Trades CSV",
                        csv,
                        f"trades_{selected_date.strftime('%Y-%m-%d')}.csv",
                        "text/csv"
                    )
            
            elif report_type == "AI Decisions":
                # Full AI decision details
                st.markdown("##### AI Decision Details")
                decisions_df = provider.get_ai_decisions_for_date(selected_date)
                
                if decisions_df.empty:
                    st.info("No AI decisions found for this date")
                else:
                    # Show key columns
                    display_cols = ["Time", "Decision", "Confidence", "Strategy", 
                                    "Market Regime", "Risk Approval", "Consensus Reasoning"]
                    available_cols = [c for c in display_cols if c in decisions_df.columns]
                    
                    st.dataframe(decisions_df[available_cols], use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = decisions_df.to_csv(index=False)
                    st.download_button(
                        "Download AI Decisions CSV",
                        csv,
                        f"ai_decisions_{selected_date.strftime('%Y-%m-%d')}.csv",
                        "text/csv"
                    )
            
            elif report_type == "Full Export":
                # Export all data for the date
                st.markdown("##### Full Data Export")
                
                trades_df = provider.get_trades_for_date(selected_date)
                decisions_df = provider.get_ai_decisions_for_date(selected_date)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.markdown(f"**Trades**: {len(trades_df)} records")
                    if not trades_df.empty:
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            "Download All Trades",
                            csv,
                            f"full_trades_{selected_date.strftime('%Y-%m-%d')}.csv",
                            "text/csv",
                            key="dl_trades"
                        )
                
                with col_dl2:
                    st.markdown(f"**AI Decisions**: {len(decisions_df)} records")
                    if not decisions_df.empty:
                        csv = decisions_df.to_csv(index=False)
                        st.download_button(
                            "Download All AI Decisions",
                            csv,
                            f"full_ai_decisions_{selected_date.strftime('%Y-%m-%d')}.csv",
                            "text/csv",
                            key="dl_decisions"
                        )
                
                st.divider()
                
                # Preview both datasets
                if not trades_df.empty:
                    st.markdown("##### Trades Preview")
                    st.dataframe(trades_df.head(10), use_container_width=True, hide_index=True)
                
                if not decisions_df.empty:
                    st.markdown("##### AI Decisions Preview")
                    st.dataframe(decisions_df.head(10), use_container_width=True, hide_index=True)


def render_settings_page():
    """Settings page - configuration only"""
    config = load_dashboard_config()
    
    st.markdown("### Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Trading Mode")
        mode = st.radio(
            "Select trading mode",
            ["Paper", "Live"],
            index=0 if config.get("trading_mode") == "Paper" else 1,
            help="Paper: Simulated trades | Live: Real money"
        )
        
        if mode == "Live":
            st.warning("⚠️ Live trading uses real money!")
    
    with col2:
        st.markdown("#### Dashboard Refresh")
        auto_refresh = st.toggle("Enable auto-refresh", value=config.get("auto_refresh", True))
        
        if auto_refresh:
            refresh_rate = st.slider("Refresh interval (seconds)", 1, 10, config.get("refresh_rate", 2))
            st.caption("⚡ Lower = faster updates (1-2s recommended)")
        else:
            refresh_rate = config.get("refresh_rate", 2)
            st.info("Auto-refresh is disabled")
    
    with col3:
        st.markdown("#### Bot Control")
        bot_running = is_bot_running()
        
        if bot_running:
            st.success("🟢 Bot is currently running")
            if st.button("Stop Bot", type="secondary", use_container_width=True):
                stop_bot()
                st.rerun()
        else:
            st.error("🔴 Bot is currently stopped")
            if st.button("Start Bot", type="primary", use_container_width=True):
                start_bot()
                st.rerun()
    
    st.divider()
    
    # Save button
    if st.button("💾 Save All Settings", type="primary"):
        config["trading_mode"] = mode
        config["auto_refresh"] = auto_refresh
        config["refresh_rate"] = refresh_rate
        save_dashboard_config(config)
        st.success("✅ Settings saved successfully!")
        st.rerun()


def main():
    """Main dashboard entry point"""
    config = load_dashboard_config()
    
    # Auto-refresh using streamlit-autorefresh (reliable, fast)
    # Don't auto-refresh on Reports or Settings pages
    if config.get("auto_refresh", True):
        refresh_rate = config.get("refresh_rate", 2)  # Default to 2 seconds for speed
        # Only refresh on active pages, not settings/reports
        if "current_page" not in st.session_state or st.session_state.current_page not in ["Reports", "Settings"]:
            st_autorefresh(interval=refresh_rate * 1000, limit=None, key="dashboard_autorefresh")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = config.get("current_page", "Dashboard")
    
    # Top Navigation
    selected_page = render_top_nav()
    
    # Update page if changed
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        config["current_page"] = selected_page
        save_dashboard_config(config)
        st.rerun()
    
    st.divider()
    
    # Render selected page
    if st.session_state.current_page == "Dashboard":
        render_dashboard_page()
    elif st.session_state.current_page == "Open Positions":
        render_positions_page()
    elif st.session_state.current_page == "Trade History":
        render_trades_page()
    elif st.session_state.current_page == "AI Decisions":
        render_ai_page()
    elif st.session_state.current_page == "Reports":
        render_reports_page()
    elif st.session_state.current_page == "Settings":
        render_settings_page()
    
    # Footer with refresh indicator
    if config.get("auto_refresh", True) and st.session_state.current_page not in ["Reports", "Settings"]:
        refresh_rate = config.get("refresh_rate", 2)
        st.markdown(
            f'<div style="position: fixed; bottom: 10px; right: 20px; font-size: 0.7rem; color: #22c55e; opacity: 0.7;">'
            f'🔄 Auto-refresh: {refresh_rate}s</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
