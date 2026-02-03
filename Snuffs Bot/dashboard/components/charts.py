"""
Chart Components

Reusable chart components using Plotly.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta


def render_pnl_chart(
    times: List[datetime],
    pnl_values: List[float],
    title: str = "Intraday P&L",
    height: int = 300
):
    """
    Render an intraday P&L line chart

    Args:
        times: List of timestamps
        pnl_values: List of cumulative P&L values
        title: Chart title
        height: Chart height in pixels
    """
    fig = go.Figure()

    # Determine color based on final value
    final_pnl = pnl_values[-1] if pnl_values else 0
    line_color = '#00d4aa' if final_pnl >= 0 else '#ff4444'
    fill_color = 'rgba(0, 212, 170, 0.2)' if final_pnl >= 0 else 'rgba(255, 68, 68, 0.2)'

    fig.add_trace(go.Scatter(
        x=times,
        y=pnl_values,
        mode='lines',
        fill='tozeroy',
        line=dict(color=line_color, width=2),
        fillcolor=fill_color,
        name='P&L'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_strategy_chart(
    strategy_data: pd.DataFrame,
    title: str = "P&L by Strategy",
    height: int = 250
):
    """
    Render a bar chart of P&L by strategy

    Args:
        strategy_data: DataFrame with 'Strategy' and 'P&L' columns
        title: Chart title
        height: Chart height
    """
    if strategy_data.empty:
        st.info("No strategy data available")
        return

    fig = px.bar(
        strategy_data,
        x="Strategy",
        y="P&L",
        title=title,
        color="P&L",
        color_continuous_scale=["#ff4444", "#888888", "#00ff88"],
        color_continuous_midpoint=0
    )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_win_loss_chart(
    wins: int,
    losses: int,
    title: str = "Win/Loss Distribution",
    height: int = 250
):
    """
    Render a stacked bar chart of wins vs losses

    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        title: Chart title
        height: Chart height
    """
    fig = go.Figure(data=[
        go.Bar(
            name='Wins',
            x=['Trades'],
            y=[wins],
            marker_color='#00ff88',
            text=[f'{wins}'],
            textposition='inside'
        ),
        go.Bar(
            name='Losses',
            x=['Trades'],
            y=[losses],
            marker_color='#ff4444',
            text=[f'{losses}'],
            textposition='inside'
        )
    ])

    fig.update_layout(
        barmode='stack',
        template="plotly_dark",
        height=height,
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_decision_pie(
    decisions: pd.DataFrame,
    title: str = "Decision Distribution",
    height: int = 300
):
    """
    Render a pie chart of AI decisions

    Args:
        decisions: DataFrame with 'Decision' column
        title: Chart title
        height: Chart height
    """
    if decisions.empty:
        st.info("No decision data available")
        return

    decision_counts = decisions["Decision"].value_counts()

    color_map = {
        'EXECUTE': '#00ff88',
        'REJECT': '#ff4444',
        'PAPER_ONLY': '#ffaa00',
        'DEFER': '#888888'
    }

    colors = [color_map.get(d, '#888888') for d in decision_counts.index]

    fig = px.pie(
        values=decision_counts.values,
        names=decision_counts.index,
        title=title,
        color_discrete_sequence=colors
    )

    fig.update_layout(
        template="plotly_dark",
        height=height
    )

    st.plotly_chart(fig, use_container_width=True)


def render_confidence_histogram(
    decisions: pd.DataFrame,
    title: str = "Confidence Distribution",
    height: int = 250
):
    """
    Render a histogram of AI confidence scores

    Args:
        decisions: DataFrame with 'Confidence' column
        title: Chart title
        height: Chart height
    """
    if decisions.empty or 'Confidence' not in decisions.columns:
        st.info("No confidence data available")
        return

    fig = px.histogram(
        decisions,
        x="Confidence",
        nbins=10,
        title=title,
        color_discrete_sequence=['#00d4aa']
    )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        xaxis_title="Confidence (%)",
        yaxis_title="Count"
    )

    # Add vertical lines for thresholds
    fig.add_vline(x=50, line_dash="dash", line_color="#ffaa00", annotation_text="Low")
    fig.add_vline(x=70, line_dash="dash", line_color="#00ff88", annotation_text="High")

    st.plotly_chart(fig, use_container_width=True)


def render_cumulative_pnl(
    trades: pd.DataFrame,
    title: str = "Cumulative P&L",
    height: int = 300
):
    """
    Render cumulative P&L over time

    Args:
        trades: DataFrame with 'Time' and 'P&L' columns
        title: Chart title
        height: Chart height
    """
    if trades.empty:
        st.info("No trade data available")
        return

    # Calculate cumulative P&L
    trades = trades.sort_values('Time')
    trades['Cumulative'] = trades['P&L'].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trades['Time'],
        y=trades['Cumulative'],
        mode='lines+markers',
        line=dict(color='#00d4aa', width=2),
        marker=dict(size=8),
        name='Cumulative P&L'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#666666")

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        xaxis_title="Time",
        yaxis_title="Cumulative P&L ($)"
    )

    st.plotly_chart(fig, use_container_width=True)
