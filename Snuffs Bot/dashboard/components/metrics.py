"""
Metric Components

Reusable metric display components.
"""

import streamlit as st
from typing import Optional


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    icon: Optional[str] = None,
    color: Optional[str] = None
):
    """
    Render a styled metric card

    Args:
        label: Metric label
        value: Main value to display
        delta: Optional delta/change value
        delta_color: 'normal', 'inverse', or 'off'
        icon: Optional emoji icon to prepend to label
        color: Optional color hint ('green', 'red') - affects delta_color
    """
    # Prepend icon if provided
    display_label = f"{icon} {label}" if icon else label

    # Handle color hint
    if color == "green":
        delta_color = "normal"
    elif color == "red":
        delta_color = "inverse"

    st.metric(
        label=display_label,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def render_pnl_metric(pnl: float, label: str = "P&L"):
    """
    Render a P&L metric with appropriate coloring

    Args:
        pnl: Profit/loss value
        label: Metric label
    """
    color = "normal" if pnl >= 0 else "inverse"
    sign = "+" if pnl > 0 else ""

    st.metric(
        label=label,
        value=f"${abs(pnl):,.2f}",
        delta=f"{sign}{pnl:,.2f}",
        delta_color=color
    )


def render_account_metrics(portfolio: dict):
    """
    Render all account metrics in a row

    Args:
        portfolio: Portfolio data dictionary
    """
    cols = st.columns(5)

    with cols[0]:
        pnl = portfolio.get("daily_pnl", 0)
        render_pnl_metric(pnl, "Today's P&L")

    with cols[1]:
        value = portfolio.get("account_value", 0)
        starting = portfolio.get("starting_capital", 100000)
        change = (value - starting) / starting * 100
        st.metric(
            "Account Value",
            f"${value:,.2f}",
            f"{change:+.2f}%"
        )

    with cols[2]:
        positions = portfolio.get("open_positions", 0)
        exposure = portfolio.get("total_exposure", 0)
        st.metric(
            "Open Positions",
            positions,
            f"${exposure:,.0f} exposure"
        )

    with cols[3]:
        wins = portfolio.get("winning_trades", 0)
        total = portfolio.get("total_trades", 1)
        rate = wins / total * 100 if total else 0
        st.metric(
            "Win Rate",
            f"{rate:.1f}%",
            f"{wins}/{total} trades"
        )

    with cols[4]:
        bp = portfolio.get("buying_power", 0)
        st.metric("Buying Power", f"${bp:,.2f}")


def render_risk_metrics(risk_data: dict):
    """
    Render risk metrics in sidebar style

    Args:
        risk_data: Risk limit data
    """
    st.subheader("Risk Limits")

    st.metric(
        "Daily Loss Limit",
        f"${risk_data.get('max_daily_loss', 500):,.0f}",
        f"${risk_data.get('daily_loss_remaining', 500):,.0f} remaining"
    )

    st.metric(
        "Position Limit",
        f"${risk_data.get('max_position_size', 5000):,.0f}"
    )

    st.metric(
        "Max Positions",
        f"{risk_data.get('current_positions', 0)}/{risk_data.get('max_positions', 3)}"
    )

    # Progress bar for daily loss
    used = risk_data.get('max_daily_loss', 500) - risk_data.get('daily_loss_remaining', 500)
    limit = risk_data.get('max_daily_loss', 500)
    progress = min(1.0, used / limit) if limit else 0

    st.progress(progress, text=f"Daily Loss Used: {progress*100:.0f}%")
