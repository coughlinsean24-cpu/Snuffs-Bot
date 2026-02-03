"""
Table Components

Reusable table components with styling.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List


def _color_pnl(val):
    """Color P&L values"""
    try:
        if isinstance(val, str):
            val = float(val.replace('$', '').replace(',', ''))
        if val > 0:
            return 'color: #00ff88'
        elif val < 0:
            return 'color: #ff4444'
        return 'color: #888888'
    except:
        return ''


def _color_decision(val):
    """Color decision values"""
    colors = {
        'EXECUTE': '#00ff88',
        'REJECT': '#ff4444',
        'PAPER_ONLY': '#ffaa00',
        'DEFER': '#888888',
        'APPROVE': '#00ff88',
        'MODIFY': '#ffaa00'
    }
    return f'color: {colors.get(val, "#888888")}'


def _color_confidence(val):
    """Color confidence values"""
    try:
        if val >= 70:
            return 'color: #00ff88'
        elif val >= 50:
            return 'color: #ffaa00'
        return 'color: #ff4444'
    except:
        return ''


def render_positions_table(
    positions: pd.DataFrame,
    title: str = "Open Positions",
    show_title: bool = True
):
    """
    Render a styled positions table

    Args:
        positions: DataFrame with position data
        title: Table title
        show_title: Whether to show the title
    """
    if show_title:
        st.subheader(title)

    if positions.empty:
        st.info("No open positions")
        return

    # Style the dataframe
    pnl_columns = [c for c in positions.columns if 'P&L' in c or 'PnL' in c]

    styled = positions.style

    for col in pnl_columns:
        styled = styled.applymap(_color_pnl, subset=[col])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(400, 50 + len(positions) * 35)
    )


def render_trades_table(
    trades: pd.DataFrame,
    title: str = "Recent Trades",
    show_title: bool = True,
    max_rows: int = 20
):
    """
    Render a styled trades table

    Args:
        trades: DataFrame with trade data
        title: Table title
        show_title: Whether to show the title
        max_rows: Maximum rows to display
    """
    if show_title:
        st.subheader(title)

    if trades.empty:
        st.info("No trades to display")
        return

    # Limit rows
    display_df = trades.head(max_rows)

    # Style the dataframe
    styled = display_df.style

    if 'P&L' in display_df.columns:
        styled = styled.applymap(_color_pnl, subset=['P&L'])

    if 'Exit Reason' in display_df.columns:
        styled = styled.applymap(_color_decision, subset=['Exit Reason'])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(500, 50 + len(display_df) * 35)
    )

    if len(trades) > max_rows:
        st.caption(f"Showing {max_rows} of {len(trades)} trades")


def render_decisions_table(
    decisions: pd.DataFrame,
    title: str = "AI Decisions",
    show_title: bool = True,
    max_rows: int = 20
):
    """
    Render a styled AI decisions table

    Args:
        decisions: DataFrame with decision data
        title: Table title
        show_title: Whether to show the title
        max_rows: Maximum rows to display
    """
    if show_title:
        st.subheader(title)

    if decisions.empty:
        st.info("No decisions to display")
        return

    display_df = decisions.head(max_rows)

    # Style the dataframe
    styled = display_df.style

    if 'Decision' in display_df.columns:
        styled = styled.applymap(_color_decision, subset=['Decision'])

    if 'Confidence' in display_df.columns:
        styled = styled.applymap(_color_confidence, subset=['Confidence'])

    if 'Risk' in display_df.columns:
        styled = styled.applymap(_color_decision, subset=['Risk'])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(500, 50 + len(display_df) * 35)
    )

    if len(decisions) > max_rows:
        st.caption(f"Showing {max_rows} of {len(decisions)} decisions")


def render_risk_limits_table(risk_limits: List[dict]):
    """
    Render risk limits as a table

    Args:
        risk_limits: List of risk limit dictionaries
    """
    if not risk_limits:
        st.info("No risk limits configured")
        return

    df = pd.DataFrame(risk_limits)

    # Add status column
    def get_status(row):
        current = row.get('current_exposure', 0)
        limit = row.get('limit_value', 1)
        pct = current / limit * 100 if limit else 0
        if pct >= 100:
            return '🔴 Exceeded'
        elif pct >= 80:
            return '🟡 Warning'
        return '🟢 OK'

    if 'current_exposure' in df.columns and 'limit_value' in df.columns:
        df['Status'] = df.apply(get_status, axis=1)

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_position_detail(position: dict):
    """
    Render detailed view of a single position

    Args:
        position: Position data dictionary
    """
    st.subheader(f"Position: {position.get('position_id', 'Unknown')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Strategy", position.get('strategy_type', 'N/A'))
        st.metric("Entry Price", f"${position.get('entry_price', 0):.2f}")
        st.metric("Entry Time", position.get('entry_time', 'N/A'))

    with col2:
        st.metric("Current Price", f"${position.get('current_price', 0):.2f}")
        pnl = position.get('unrealized_pnl', 0)
        st.metric(
            "Unrealized P&L",
            f"${pnl:.2f}",
            delta_color="normal" if pnl >= 0 else "inverse"
        )
        st.metric("Contracts", position.get('contracts', 1))

    with col3:
        st.metric("Max Profit", f"${position.get('max_profit', 0):.2f}")
        st.metric("Max Loss", f"${position.get('max_loss', 0):.2f}")
        st.metric("Profit Target", f"{position.get('profit_target_percent', 50)}%")

    # Legs detail
    if 'legs' in position and position['legs']:
        st.subheader("Option Legs")
        legs_df = pd.DataFrame(position['legs'])
        st.dataframe(legs_df, use_container_width=True, hide_index=True)
