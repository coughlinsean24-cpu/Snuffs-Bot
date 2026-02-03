#!/bin/bash
# Start All - Starts both the trading bot and dashboard
# Usage: ./start_all.sh

cd "$(dirname "$0")"

echo "🚀 Starting Spywave AI Trading System..."
echo ""

# Check if PostgreSQL is running
if ! docker ps | grep -q trading_bot_db; then
    echo "📦 Starting database..."
    docker start trading_bot_db 2>/dev/null || echo "   Database already running or not found"
    sleep 2
fi

# Stop any existing processes
pkill -f "run_bot.py" 2>/dev/null
pkill -f "streamlit" 2>/dev/null
sleep 1

# Clear old cache
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

# Activate virtual environment
source .venv/bin/activate

# Start the trading bot
echo "🤖 Starting trading bot..."
nohup python run_bot.py > logs/bot.log 2>&1 &
BOT_PID=$!
sleep 2

# Check if bot started
if ps -p $BOT_PID > /dev/null 2>&1; then
    echo "   ✓ Bot started (PID: $BOT_PID)"
else
    echo "   ✗ Bot failed to start. Check logs/bot.log"
fi

# Start the dashboard
echo "📊 Starting dashboard..."
cd dashboard
nohup streamlit run app.py --server.port 8501 --server.headless true > ../logs/dashboard.log 2>&1 &
DASH_PID=$!
cd ..
sleep 3

# Check if dashboard started
if pgrep -f "streamlit" > /dev/null; then
    echo "   ✓ Dashboard started"
else
    echo "   ✗ Dashboard failed to start. Check logs/dashboard.log"
fi

echo ""
echo "================================================"
echo "  🎉 Spywave AI is now running!"
echo "================================================"
echo ""
echo "  📊 Dashboard: http://localhost:8501"
echo "  📝 Bot Log:   tail -f logs/bot.log"
echo ""
echo "  To stop: ./stop_all.sh"
echo ""
