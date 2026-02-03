#!/bin/bash
# Stop All - Stops both the trading bot and dashboard
# Usage: ./stop_all.sh

echo "⏹️  Stopping Spywave AI Trading System..."
echo ""

# Stop the bot
if pgrep -f "run_bot.py" > /dev/null; then
    echo "🤖 Stopping trading bot..."
    pkill -f "run_bot.py"
    echo "   ✓ Bot stopped"
else
    echo "🤖 Bot was not running"
fi

# Stop the dashboard
if pgrep -f "streamlit" > /dev/null; then
    echo "📊 Stopping dashboard..."
    pkill -f "streamlit"
    echo "   ✓ Dashboard stopped"
else
    echo "📊 Dashboard was not running"
fi

echo ""
echo "================================================"
echo "  ✓ All processes stopped"
echo "================================================"
echo ""
echo "  To start again: ./start_all.sh"
echo ""
