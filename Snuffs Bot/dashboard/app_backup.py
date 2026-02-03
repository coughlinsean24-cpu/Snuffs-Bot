"""
Spywave AI Trading Dashboard

Single-page customizable dashboard for autonomous 0DTE trading.
Uses Streamlit fragments for smooth, partial updates without full page reloads.
Supports multi-user authentication and personalized settings.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import time
import subprocess
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from snuffs_bot.config.settings import get_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

from data_provider import get_data_provider

# Dashboard config file path
DASHBOARD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "dashboard_config.json")

def load_dashboard_config():
    """Load dashboard configuration from file"""
    default_config = {
        "trading_mode": "Paper",
        "show_chart": True,
        "show_positions": True,
        "show_trades": True,
        "show_ai": True,
        "show_risk": True,
        "auto_refresh": True,  # Enable auto-refresh by default
        "refresh_rate": 2  # 2 second refresh for near real-time updates
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

def apply_trading_mode(mode: str):
    """Apply trading mode to the .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    # Mode mapping:
    # Paper = Live data + simulated trades
    # Live = Live data + real trades
    if mode == "Paper":
        paper_trading = "true"
        live_enabled = "false"
    else:  # Live
        paper_trading = "false"
        live_enabled = "true"

    try:
        # Read current .env
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()

        # Update or add settings
        paper_found = False
        live_found = False
        new_lines = []

        for line in lines:
            if line.strip().startswith("PAPER_TRADING="):
                new_lines.append(f"PAPER_TRADING={paper_trading}\n")
                paper_found = True
            elif line.strip().startswith("LIVE_TRADING_ENABLED="):
                new_lines.append(f"LIVE_TRADING_ENABLED={live_enabled}\n")
                live_found = True
            else:
                new_lines.append(line)

        # Add if not found
        if not paper_found:
            new_lines.append(f"PAPER_TRADING={paper_trading}\n")
        if not live_found:
            new_lines.append(f"LIVE_TRADING_ENABLED={live_enabled}\n")

        # Write back
        with open(env_path, 'w') as f:
            f.writelines(new_lines)

        return True
    except Exception as e:
        st.error(f"Failed to update mode: {e}")
        return False

# Page configuration - no menu
st.set_page_config(
    page_title="Spywave AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# Professional green/dark blue theme CSS
st.markdown("""
<style>
    /* Hide default menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f3c 0%, #132744 100%);
        border-right: 1px solid #1e3a5f;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #4ade80 !important;
    }

    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }

    .sub-header {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Bot control buttons */
    .bot-control {
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        margin: 8px 0;
    }

    .bot-running {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.2), rgba(34, 197, 94, 0.1));
        border: 2px solid #4ade80;
    }

    .bot-stopped {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.2), rgba(239, 68, 68, 0.1));
        border: 2px solid #f87171;
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .status-running {
        background: rgba(74, 222, 128, 0.15);
        border: 1px solid #4ade80;
        color: #4ade80;
    }

    .status-stopped {
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid #f87171;
        color: #f87171;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    .status-dot-green {
        background: #4ade80;
        box-shadow: 0 0 8px #4ade80;
    }

    .status-dot-red {
        background: #f87171;
        box-shadow: 0 0 8px #f87171;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Mode banner */
    .mode-banner {
        padding: 10px 20px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }

    .mode-paper {
        background: linear-gradient(90deg, rgba(74, 222, 128, 0.1), rgba(34, 197, 94, 0.05));
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    .mode-live {
        background: linear-gradient(90deg, rgba(248, 113, 113, 0.1), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    .mode-title {
        font-size: 1rem;
        font-weight: 700;
    }

    .mode-paper .mode-title { color: #4ade80; }
    .mode-live .mode-title { color: #f87171; }

    .mode-desc {
        color: #64748b;
        font-size: 0.8rem;
        margin-left: auto;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Colors */
    .profit { color: #4ade80 !important; }
    .loss { color: #f87171 !important; }

    /* Section panels */
    .dashboard-section {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.3), rgba(13, 40, 71, 0.2));
        border: 1px solid rgba(74, 222, 128, 0.15);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    .section-header {
        color: #4ade80;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 58, 95, 0.3) !important;
        border-radius: 8px;
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #0d2847);
        border: 1px solid #4ade80;
        color: #4ade80;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4ade80, #22c55e);
        color: #0a1628;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        border-color: #f87171;
        color: white;
    }

    /* Start button special styling */
    .start-btn button {
        background: linear-gradient(135deg, #059669, #047857) !important;
        border-color: #4ade80 !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 12px 24px !important;
    }

    .stop-btn button {
        background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
        border-color: #f87171 !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 12px 24px !important;
    }

    /* Dividers */
    hr { border-color: rgba(74, 222, 128, 0.15); }

    /* Time */
    .time-display {
        text-align: right;
        color: #e2e8f0;
    }

    .time-display .time {
        font-size: 1.3rem;
        font-weight: 600;
        font-family: monospace;
    }

    .time-display .date {
        color: #64748b;
        font-size: 0.8rem;
    }

    /* Market badge */
    .market-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .market-open {
        background: rgba(74, 222, 128, 0.1);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    .market-closed {
        background: rgba(100, 116, 139, 0.1);
        color: #64748b;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def load_settings():
    """Load application settings (fresh from .env each time)"""
    if not SETTINGS_AVAILABLE:
        return None
    try:
        # Always reload to get latest values from .env
        return get_settings(reload=True)
    except:
        return None


def get_provider():
    """Get data provider (fresh instance for accurate data)"""
    # Don't use @st.cache_resource - we need fresh data on each refresh
    # The data provider has its own short-lived cache (2 seconds)
    return get_data_provider()


# Initialize trading mode from saved config
if 'trading_mode' not in st.session_state:
    _saved = load_dashboard_config()
    st.session_state.trading_mode = _saved.get("trading_mode", "Paper")


def is_bot_running():
    """Check if the bot process is actually running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_bot.py"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def start_bot():
    """Start the trading bot"""
    if is_bot_running():
        return True  # Already running

    try:
        venv_python = "/home/coughlinsean24/Snuffs Bot/.venv/bin/python"
        bot_script = "/home/coughlinsean24/Snuffs Bot/run_bot.py"

        if os.path.exists(bot_script):
            # Start detached so it survives page refresh
            subprocess.Popen(
                [venv_python, bot_script],
                stdout=open("/tmp/snuffs_bot.log", "a"),
                stderr=subprocess.STDOUT,
                cwd="/home/coughlinsean24/Snuffs Bot",
                start_new_session=True
            )
            time.sleep(2)  # Give it time to start
            return is_bot_running()
        else:
            st.error("Bot script not found")
            return False
    except Exception as e:
        st.error(f"Failed to start bot: {e}")
        return False


def stop_bot():
    """Stop the trading bot"""
    try:
        subprocess.run(["pkill", "-f", "run_bot.py"], capture_output=True)
        time.sleep(1)
        return not is_bot_running()
    except Exception as e:
        st.error(f"Failed to stop bot: {e}")
        return False


def render_bot_controls():
    """Render bot start/stop controls"""
    is_running = is_bot_running()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if is_running:
            st.markdown("""
            <div class="bot-control bot-running">
                <div style="font-size: 2rem; margin-bottom: 8px;">🟢</div>
                <div style="color: #4ade80; font-size: 1.2rem; font-weight: 700;">BOT IS RUNNING</div>
                <div style="color: #64748b; font-size: 0.85rem;">Actively monitoring and trading</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("STOP BOT", use_container_width=True, key="stop_bot"):
                if stop_bot():
                    st.success("Bot stopped")
                    time.sleep(0.5)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="bot-control bot-stopped">
                <div style="font-size: 2rem; margin-bottom: 8px;">🔴</div>
                <div style="color: #f87171; font-size: 1.2rem; font-weight: 700;">BOT IS STOPPED</div>
                <div style="color: #64748b; font-size: 0.85rem;">Click below to start trading</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="start-btn">', unsafe_allow_html=True)
            if st.button("START BOT", use_container_width=True, key="start_bot"):
                if start_bot():
                    st.success("Bot started!")
                    time.sleep(0.5)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_header():
    """Render compact header"""
    # Get mode from session state
    trading_mode = st.session_state.get('trading_mode', 'Paper Trading')

    # Check market hours
    current_time = datetime.now().time()
    from datetime import time as dt_time
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    is_market_open = market_open <= current_time <= market_close and datetime.now().weekday() < 5

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown('<p class="main-header">Spywave AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Autonomous 0DTE SPY Options Trading</p>', unsafe_allow_html=True)

    with col2:
        # Mode and market status based on selected trading mode
        if trading_mode == "Paper":
            st.markdown('<span class="status-badge status-running"><span class="status-dot status-dot-green"></span>PAPER MODE</span>', unsafe_allow_html=True)
        else:  # Live
            st.markdown('<span class="status-badge status-stopped"><span class="status-dot status-dot-red"></span>LIVE MODE</span>', unsafe_allow_html=True)

        market_class = "market-open" if is_market_open else "market-closed"
        market_text = "MARKET OPEN" if is_market_open else "MARKET CLOSED"
        st.markdown(f'<span class="market-badge {market_class}">{market_text}</span>', unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="time-display">
            <div class="time">{datetime.now().strftime('%H:%M:%S')}</div>
            <div class="date">{datetime.now().strftime('%a, %b %d')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_metrics(portfolio):
    """Render key metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pnl = float(portfolio["daily_pnl"])
        delta_color = "normal" if pnl >= 0 else "inverse"
        st.metric("Today's P&L", f"${pnl:+,.2f}", f"{pnl/portfolio['starting_capital']*100:+.2f}%", delta_color=delta_color)

    with col2:
        account_value = float(portfolio['account_value'])
        starting_capital = float(portfolio['starting_capital'])
        change_pct = (account_value - starting_capital) / starting_capital * 100
        delta_color = "normal" if change_pct >= 0 else "inverse"
        st.metric("Account", f"${account_value:,.0f}", f"{change_pct:+.2f}%", delta_color=delta_color)

    with col3:
        st.metric("Positions", f"{portfolio['open_positions']}", f"${portfolio['total_exposure']:,.0f} risk", delta_color="off")

    with col4:
        win_rate = portfolio["winning_trades"] / portfolio["total_trades"] * 100 if portfolio["total_trades"] else 0
        st.metric("Win Rate", f"{win_rate:.0f}%", f"{portfolio['winning_trades']}W/{portfolio['total_trades']-portfolio['winning_trades']}L", delta_color="off")

    with col5:
        st.metric("Buying Power", f"${portfolio['buying_power']:,.0f}", delta_color="off")


def render_pnl_chart():
    """Render P&L chart"""
    provider = get_provider()
    intraday_df = provider.get_intraday_pnl()

    if intraday_df.empty or intraday_df["pnl"].sum() == 0:
        times = pd.date_range(start='09:30', end='16:00', freq='15min')
        pnl_data = [0] * len(times)
    else:
        times = intraday_df["time"]
        pnl_data = intraday_df["pnl"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=pnl_data, mode='lines', fill='tozeroy',
        line=dict(color='#4ade80', width=2),
        fillcolor='rgba(74, 222, 128, 0.1)'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.3)", line_width=1)

    fig.update_layout(
        title=dict(text="Intraday P&L", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(gridcolor="rgba(30, 58, 95, 0.5)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(30, 58, 95, 0.5)", tickfont=dict(color="#94a3b8"), tickformat="$,.0f"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color="#94a3b8")
    )
    st.plotly_chart(fig, use_container_width=True)


def render_positions_section(positions_df, provider):
    """Render positions section with export and force sell buttons"""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="section-header">📊 Open Positions</div>', unsafe_allow_html=True)
    with col2:
        if not positions_df.empty and len(positions_df) > 0:
            csv = positions_df.to_csv(index=False)
            st.download_button("📥 Export", csv, "positions.csv", "text/csv", key="export_positions")
    
    if positions_df.empty or (len(positions_df) == 1 and positions_df.iloc[0].get("Position ID") == "No open positions"):
        st.info("No open positions")
        return

    # Format numeric columns to 2 decimal places
    numeric_cols = ['Entry Price', 'Current Price', 'Unrealized P&L', 'P&L %', 'Max Loss', 'Max Profit', 'Strike', 'SPY']
    for col in numeric_cols:
        if col in positions_df.columns:
            positions_df[col] = positions_df[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

    # Header row with clear labels
    st.markdown("""
    <div style="display: flex; padding: 8px 0; border-bottom: 2px solid #334155; margin-bottom: 10px; font-weight: 600; color: #94a3b8; font-size: 11px;">
        <div style="flex: 1.5; min-width: 70px;">STRATEGY</div>
        <div style="flex: 0.8; min-width: 50px; text-align: center;">QTY</div>
        <div style="flex: 1; min-width: 55px; text-align: right;">STRIKE</div>
        <div style="flex: 1; min-width: 55px; text-align: right;">SPY</div>
        <div style="flex: 1; min-width: 55px; text-align: right;">ENTRY</div>
        <div style="flex: 1; min-width: 55px; text-align: right;">CURR</div>
        <div style="flex: 1.2; min-width: 65px; text-align: right;">P&L</div>
        <div style="flex: 0.8; min-width: 45px; text-align: right;">%</div>
        <div style="flex: 0.8; min-width: 50px; text-align: center;">TIME</div>
        <div style="flex: 1.2; min-width: 70px; text-align: center;">ACTION</div>
    </div>
    """, unsafe_allow_html=True)

    # Display each position
    for idx, row in positions_df.iterrows():
        position_id = str(row.get('Position ID', ''))
        strategy = row.get('Strategy', 'N/A')
        qty = row.get('Qty', 1)
        strike = float(row.get('Strike', 0))
        spy_price = float(row.get('SPY', 0))
        entry_price = float(row.get('Entry Price', 0))
        current_price = float(row.get('Current Price', 0))
        pnl = float(row.get('Unrealized P&L', 0))
        pnl_pct = float(row.get('P&L %', 0)) if 'P&L %' in row else 0
        entry_time = row.get('Entry Time', '')
        
        # Determine P&L color
        pnl_color = "#4ade80" if pnl > 0 else "#f87171" if pnl < 0 else "#94a3b8"
        pnl_icon = "▲" if pnl > 0 else "▼" if pnl < 0 else "●"
        
        # Create a container for each position row
        with st.container():
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns([1.5, 0.8, 1, 1, 1, 1, 1.2, 0.8, 0.8, 1.2])
            
            with c1:
                st.markdown(f"**{strategy}**")
            with c2:
                st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: 600;'>{qty}</div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div style='text-align: right;'>${strike:.0f}</div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div style='text-align: right;'>${spy_price:.2f}</div>", unsafe_allow_html=True)
            with c5:
                st.markdown(f"<div style='text-align: right;'>${entry_price:.2f}</div>", unsafe_allow_html=True)
            with c6:
                st.markdown(f"<div style='text-align: right;'>${current_price:.2f}</div>", unsafe_allow_html=True)
            with c7:
                st.markdown(f"<div style='text-align: right; color: {pnl_color}; font-weight: 600;'>{pnl_icon} ${pnl:.2f}</div>", unsafe_allow_html=True)
            with c8:
                st.markdown(f"<div style='text-align: right; color: {pnl_color};'>{pnl_pct:+.1f}%</div>", unsafe_allow_html=True)
            with c9:
                st.markdown(f"<div style='text-align: center; font-size: 10px;'>{entry_time}</div>", unsafe_allow_html=True)
            with c10:
                if st.button("🚨 SELL", key=f"force_sell_{position_id}_{idx}", type="primary", use_container_width=True):
                    with st.spinner("Closing..."):
                        result = provider.force_close_position(position_id)
                        if result.get("success"):
                            st.success(f"✅ Sold! P&L: ${result.get('pnl', 0):.2f}")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('error', 'Failed')}")


def render_trades_section(trades_df, provider):
    """Render trades section with export"""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="section-header">📈 Recent Trades</div>', unsafe_allow_html=True)
    with col2:
        if not trades_df.empty:
            # Get full trade data for export
            full_trades = provider.get_recent_trades(limit=100, full_details=True)
            csv = full_trades.to_csv(index=False)
            st.download_button("📥 Export All", csv, "trades_export.csv", "text/csv", key="export_trades")
    
    if trades_df.empty:
        st.info("No trades yet")
        return

    # Format numeric columns to 2 decimal places
    numeric_cols = ['Entry', 'Exit', 'P&L']
    for col in numeric_cols:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

    if "P&L" in trades_df.columns:
        def color_pnl(val):
            try:
                val = float(val)
                color = '#4ade80' if val > 0 else '#f87171' if val < 0 else '#64748b'
            except:
                color = '#64748b'
            return f'color: {color}; font-weight: 600'

        # Format numbers to 2 decimal places in display
        format_dict = {col: '{:.2f}' for col in numeric_cols if col in trades_df.columns}
        styled_df = trades_df.style.applymap(color_pnl, subset=['P&L']).format(format_dict)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=250)
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True, height=250)


def render_ai_section(decisions_df):
    """Render AI decisions section with export"""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="section-header">🤖 AI Decisions</div>', unsafe_allow_html=True)
    with col2:
        if not decisions_df.empty:
            csv = decisions_df.to_csv(index=False)
            st.download_button("📥 Export", csv, "ai_decisions.csv", "text/csv", key="export_ai")
    
    if decisions_df.empty:
        st.info("No AI decisions yet")
        return

    def color_decision(val):
        colors = {'EXECUTE': '#4ade80', 'HOLD': '#fbbf24', 'REJECT': '#f87171', 'DEFER': '#64748b'}
        return f'color: {colors.get(val, "#64748b")}; font-weight: 600'
    
    def color_risk(val):
        colors = {'APPROVE': '#4ade80', 'REJECT': '#f87171'}
        return f'color: {colors.get(val, "#64748b")}; font-weight: 600'

    # Show columns including Strike and Reason
    display_cols = ["Time", "Decision", "Confidence", "Strategy", "Strike", "Market", "Risk", "Reason"]
    display_df = decisions_df[[c for c in display_cols if c in decisions_df.columns]].copy()
    
    # Format confidence to 1 decimal place
    if 'Confidence' in display_df.columns:
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{float(x):.1f}%")
    
    styled_df = display_df.style.applymap(color_decision, subset=['Decision'])
    if 'Risk' in display_df.columns:
        styled_df = styled_df.applymap(color_risk, subset=['Risk'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=250)


def render_sidebar():
    """Render sidebar"""
    # Load saved config
    saved_config = load_dashboard_config()

    # Initialize session state from saved config
    if 'trading_mode' not in st.session_state:
        st.session_state.trading_mode = saved_config.get("trading_mode", "Paper")

    mode_options = ["Paper", "Live"]

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 15px 0;">
            <span style="font-size: 1.3rem; font-weight: 700; background: linear-gradient(90deg, #4ade80, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">SPYWAVE AI</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Trading Mode")

        # Get current index
        current_mode = st.session_state.trading_mode
        current_index = mode_options.index(current_mode) if current_mode in mode_options else 0

        mode = st.selectbox(
            "Mode",
            mode_options,
            index=current_index,
            help="Paper: Live data + simulated trades | Live: Live data + real trades",
            key="mode_selector"
        )

        # Check if mode changed
        if mode != st.session_state.trading_mode:
            st.session_state.trading_mode = mode
            # Apply the mode change
            if apply_trading_mode(mode):
                # Save to config
                saved_config["trading_mode"] = mode
                save_dashboard_config(saved_config)
                st.success(f"Mode changed to {mode}")
                time.sleep(0.5)
                st.rerun()

        st.divider()

        st.markdown("### Market Data")
        provider = get_provider()
        market_data = provider.get_market_data()

        col1, col2 = st.columns(2)
        with col1:
            spy_price = market_data.get("spy_price", 0)
            spy_change = market_data.get("spy_change_pct", 0)
            delta_color = "normal" if spy_change >= 0 else "inverse"
            st.metric("SPY", f"${spy_price:.2f}", f"{spy_change:+.2f}%", delta_color=delta_color)
        with col2:
            vix = market_data.get("vix", 0)
            vix_change = market_data.get("vix_change", 0)
            delta_color = "inverse" if vix_change >= 0 else "normal"
            st.metric("VIX", f"{vix:.2f}", f"{vix_change:+.2f}", delta_color=delta_color)

        data_source = market_data.get("data_source", "DEFAULT")
        if data_source == "LIVE":
            st.success("Live data", icon="📡")
        else:
            st.warning("Cached data", icon="⏸️")

        st.divider()

        st.markdown("### Dashboard Layout")
        show_chart = st.toggle("P&L Chart", value=saved_config.get("show_chart", True), key="toggle_chart")
        show_positions = st.toggle("Positions", value=saved_config.get("show_positions", True), key="toggle_positions")
        show_trades = st.toggle("Trade History", value=saved_config.get("show_trades", True), key="toggle_trades")
        show_ai = st.toggle("AI Decisions", value=saved_config.get("show_ai", True), key="toggle_ai")
        show_risk = st.toggle("Risk Controls", value=saved_config.get("show_risk", True), key="toggle_risk")

        st.divider()

        st.markdown("### Settings")
        auto_refresh = st.toggle("Auto-Refresh", value=saved_config.get("auto_refresh", False), key="toggle_refresh")
        if auto_refresh:
            refresh_rate = st.select_slider("Rate", [2, 3, 5, 10, 15, 30], value=saved_config.get("refresh_rate", 3), format_func=lambda x: f"{x}s")
        else:
            refresh_rate = saved_config.get("refresh_rate", 3)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True):
                provider.clear_cache()
                st.rerun()
        with col2:
            if st.button("Save Layout", use_container_width=True):
                new_config = {
                    "trading_mode": mode,
                    "show_chart": show_chart,
                    "show_positions": show_positions,
                    "show_trades": show_trades,
                    "show_ai": show_ai,
                    "show_risk": show_risk,
                    "auto_refresh": auto_refresh,
                    "refresh_rate": refresh_rate
                }
                if save_dashboard_config(new_config):
                    st.success("Layout saved!")

        return {
            "trading_mode": mode,
            "auto_refresh": auto_refresh,
            "refresh_rate": refresh_rate,
            "show_chart": show_chart,
            "show_positions": show_positions,
            "show_trades": show_trades,
            "show_ai": show_ai,
            "show_risk": show_risk,
        }


def update_env_setting(key: str, value: str) -> bool:
    """Update a single setting in the .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    
    try:
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        
        # Update or add the setting
        found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            new_lines.append(f"{key}={value}\n")
        
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        return True
    except Exception as e:
        st.error(f"Failed to update {key}: {e}")
        return False


def get_current_guardrails() -> dict:
    """Get current guardrail values from settings or defaults"""
    settings = load_settings()
    if settings:
        return {
            "starting_capital": getattr(settings, 'starting_capital', 100000.0),
            "max_daily_loss": settings.max_daily_loss,
            "max_position_size": settings.max_position_size,
            "max_concurrent_positions": settings.max_concurrent_positions,
            "risk_per_trade_percent": settings.risk_per_trade_percent,
            "min_profit_target": getattr(settings, 'min_profit_target', 0.15),
            "max_stop_loss": getattr(settings, 'max_stop_loss', 0.25),
        }
    return {
        "starting_capital": 100000.0,
        "max_daily_loss": 500.0,
        "max_position_size": 5000.0,
        "max_concurrent_positions": 3,
        "risk_per_trade_percent": 0.04,
        "min_profit_target": 0.15,
        "max_stop_loss": 0.25,
    }


def render_risk_section():
    """Render editable risk controls section with guardrails"""
    
    # Get current values
    guardrails = get_current_guardrails()
    
    # Initialize session state for editing
    if 'editing_guardrails' not in st.session_state:
        st.session_state.editing_guardrails = False
    
    # Header with edit toggle
    col_header, col_edit = st.columns([4, 1])
    with col_edit:
        if st.button("✏️ Edit" if not st.session_state.editing_guardrails else "✓ Done", key="toggle_edit_guardrails"):
            st.session_state.editing_guardrails = not st.session_state.editing_guardrails
            st.rerun()
    
    if st.session_state.editing_guardrails:
        # Editable mode
        st.markdown("**Edit Account & Risk Settings** - Changes sync with AI trading limits")
        
        # Account Capital Section
        st.markdown("##### 💰 Account Capital")
        col_cap1, col_cap2 = st.columns(2)
        
        with col_cap1:
            new_starting_capital = st.number_input(
                "Starting Capital ($)",
                min_value=100.0,
                max_value=10000000.0,
                value=min(float(guardrails["starting_capital"]), 10000000.0),
                step=100.0,
                help="Total account value for paper trading. Bot will size trades based on this."
            )
        
        with col_cap2:
            st.info(f"Position sizing will use {guardrails['risk_per_trade_percent']*100:.0f}% of this capital per trade")
        
        st.markdown("##### ⚠️ Risk Guardrails")
        col1, col2 = st.columns(2)
        
        with col1:
            # Cap max values based on starting capital
            max_daily_limit = min(float(guardrails["max_daily_loss"]), new_starting_capital)
            new_daily_loss = st.number_input(
                "Max Daily Loss ($)",
                min_value=10.0,
                max_value=new_starting_capital,
                value=min(max_daily_limit, new_starting_capital * 0.5),
                step=10.0,
                help="Maximum loss allowed in a single day before trading stops"
            )
            
            max_pos_limit = min(float(guardrails["max_position_size"]), new_starting_capital)
            new_position_size = st.number_input(
                "Max Position Size ($)",
                min_value=10.0,
                max_value=new_starting_capital,
                value=min(max_pos_limit, new_starting_capital * 0.5),
                step=10.0,
                help="Maximum risk per single position"
            )
            
            new_profit_target = st.slider(
                "Profit Target (%)",
                min_value=10,
                max_value=100,
                value=int(guardrails["min_profit_target"] * 100),
                step=5,
                help="Take profit target as percentage of max profit"
            )
        
        with col2:
            new_max_positions = st.number_input(
                "Max Concurrent Positions",
                min_value=1,
                max_value=10,
                value=int(guardrails["max_concurrent_positions"]),
                step=1,
                help="Maximum number of open positions at once"
            )
            
            new_risk_per_trade = st.slider(
                "Risk Per Trade (%)",
                min_value=1,
                max_value=50,
                value=min(int(guardrails["risk_per_trade_percent"] * 100), 50),
                step=1,
                help="Maximum percentage of capital risked per trade"
            )
            
            new_stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=10,
                max_value=75,
                value=int(guardrails["max_stop_loss"] * 100),
                step=5,
                help="Stop loss as percentage of max loss"
            )
        
        # Save button
        if st.button("💾 Save Account & Risk Settings", type="primary", use_container_width=True):
            updates = [
                ("STARTING_CAPITAL", str(new_starting_capital)),
                ("MAX_DAILY_LOSS", str(new_daily_loss)),
                ("MAX_POSITION_SIZE", str(new_position_size)),
                ("MAX_CONCURRENT_POSITIONS", str(new_max_positions)),
                ("RISK_PER_TRADE_PERCENT", str(new_risk_per_trade / 100)),
                ("MIN_PROFIT_TARGET", str(new_profit_target / 100)),
                ("MAX_STOP_LOSS", str(new_stop_loss / 100)),
            ]
            
            success = True
            for key, value in updates:
                if not update_env_setting(key, value):
                    success = False
            
            if success:
                # Force reload settings from updated .env file
                try:
                    from snuffs_bot.config.settings import get_settings
                    get_settings(reload=True)
                except:
                    pass
                    
                st.success("✓ Guardrails saved! Bot will use new settings on next decision.")
                st.session_state.editing_guardrails = False
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save some settings")
    
    else:
        # Display mode - show current guardrails
        st.markdown(f"**💰 Account Capital: ${guardrails['starting_capital']:,.0f}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Daily Loss Limit",
                f"${guardrails['max_daily_loss']:,.0f}",
                help="Trading stops when daily loss exceeds this"
            )
            st.metric(
                "Max Position Size",
                f"${guardrails['max_position_size']:,.0f}",
                help="Maximum risk per single position"
            )
        
        with col2:
            st.metric(
                "Max Positions",
                f"{guardrails['max_concurrent_positions']}",
                help="Maximum concurrent open positions"
            )
            st.metric(
                "Risk Per Trade",
                f"{guardrails['risk_per_trade_percent']*100:.0f}%",
                help="Percentage of capital risked per trade"
            )
        
        with col3:
            st.metric(
                "Profit Target",
                f"{guardrails['min_profit_target']*100:.0f}%",
                help="Take profit target percentage"
            )
            st.metric(
                "Stop Loss",
                f"{guardrails['max_stop_loss']*100:.0f}%",
                help="Stop loss percentage"
            )


@st.fragment(run_every=timedelta(seconds=3))
def live_metrics_fragment():
    """Auto-updating metrics section - refreshes every 3 seconds for real-time feel"""
    provider = get_provider()
    provider.clear_cache()  # Always get fresh data
    portfolio = provider.get_portfolio_summary()
    render_metrics(portfolio)


@st.fragment(run_every=timedelta(seconds=3))
def live_time_fragment():
    """Auto-updating time display"""
    current_time = datetime.now().time()
    from datetime import time as dt_time
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    is_market_open = market_open <= current_time <= market_close and datetime.now().weekday() < 5
    
    market_class = "market-open" if is_market_open else "market-closed"
    market_text = "MARKET OPEN" if is_market_open else "MARKET CLOSED"
    
    st.markdown(f"""
    <div style="text-align: right;">
        <span class="market-badge {market_class}">{market_text}</span>
        <div class="time-display">
            <div class="time">{datetime.now().strftime('%H:%M:%S')}</div>
            <div class="date">{datetime.now().strftime('%a, %b %d')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=1))
def live_positions_fragment(show_positions: bool):
    """Auto-updating positions section - 1 second refresh for real-time"""
    if not show_positions:
        return
    provider = get_provider()
    provider.clear_cache()
    positions_df = provider.get_open_positions()
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    render_positions_section(positions_df, provider)
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=3))
def live_trades_fragment(show_trades: bool):
    """Auto-updating trades section - 3 second refresh"""
    if not show_trades:
        return
    provider = get_provider()
    provider.clear_cache()
    trades_df = provider.get_recent_trades(limit=50)
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    render_trades_section(trades_df, provider)
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=5))
def live_ai_fragment(show_ai: bool):
    """Auto-updating AI decisions section - 5 second refresh"""
    if not show_ai:
        return
    provider = get_provider()
    provider.clear_cache()
    decisions_df = provider.get_ai_decisions(limit=50)
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    render_ai_section(decisions_df)
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=30))
def live_chart_fragment(show_chart: bool):
    """Auto-updating P&L chart - less frequent since chart is heavy"""
    if not show_chart:
        return
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    render_pnl_chart()
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main dashboard with streaming-like updates using fragments"""
    
    # Sidebar (static, only updates on user interaction)
    config = render_sidebar()

    # Header - static parts
    trading_mode = st.session_state.get('trading_mode', 'Paper')
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<p class="main-header">Spywave AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Autonomous 0DTE SPY Options Trading</p>', unsafe_allow_html=True)
    with col2:
        if trading_mode == "Paper":
            st.markdown('<span class="status-badge status-running"><span class="status-dot status-dot-green"></span>PAPER MODE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-stopped"><span class="status-dot status-dot-red"></span>LIVE MODE</span>', unsafe_allow_html=True)
    with col3:
        # Live updating time
        if config["auto_refresh"]:
            live_time_fragment()
        else:
            st.markdown(f"""
            <div class="time-display">
                <div class="time">{datetime.now().strftime('%H:%M:%S')}</div>
                <div class="date">{datetime.now().strftime('%a, %b %d')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Bot Controls - static, updates on button click
    render_bot_controls()

    st.divider()

    # Key Metrics - live updating fragment
    if config["auto_refresh"]:
        live_metrics_fragment()
    else:
        provider = get_provider()
        portfolio = provider.get_portfolio_summary()
        render_metrics(portfolio)

    st.divider()

    # P&L Chart
    if config["auto_refresh"]:
        live_chart_fragment(config["show_chart"])
    elif config["show_chart"]:
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        render_pnl_chart()
        st.markdown('</div>', unsafe_allow_html=True)

    # Stacked card layout (vertical)
    
    # Positions Section
    if config["auto_refresh"]:
        live_positions_fragment(config["show_positions"])
    elif config["show_positions"]:
        provider = get_provider()
        positions_df = provider.get_open_positions()
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        render_positions_section(positions_df, provider)
        st.markdown('</div>', unsafe_allow_html=True)

    # Trades Section
    if config["auto_refresh"]:
        live_trades_fragment(config["show_trades"])
    elif config["show_trades"]:
        provider = get_provider()
        trades_df = provider.get_recent_trades(limit=50)
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        render_trades_section(trades_df, provider)
        st.markdown('</div>', unsafe_allow_html=True)

    # AI Decisions Section
    if config["auto_refresh"]:
        live_ai_fragment(config["show_ai"])
    elif config["show_ai"]:
        provider = get_provider()
        decisions_df = provider.get_ai_decisions(limit=50)
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        render_ai_section(decisions_df)
        st.markdown('</div>', unsafe_allow_html=True)

    # Risk Controls Section (with editable guardrails)
    if config["show_risk"]:
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚠️ Risk Controls & Guardrails</div>', unsafe_allow_html=True)
        render_risk_section()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
