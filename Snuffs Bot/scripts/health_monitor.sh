#!/bin/bash
# Health monitor - runs every 5 minutes via cron to ensure bot and learner stay running

LOG_FILE="/home/coughlinsean24/Snuffs Bot/logs/health.log"
BOT_DIR="/home/coughlinsean24/Snuffs Bot"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$LOG_FILE"
}

# Check background learner
if ! pgrep -f "background_learner.py" > /dev/null; then
    log "⚠️ Background learner not running - restarting..."
    cd "$BOT_DIR" && source venv/bin/activate && nohup python scripts/background_learner.py >> logs/learner.log 2>&1 &
    sleep 2
    if pgrep -f "background_learner.py" > /dev/null; then
        log "✅ Background learner restarted successfully"
    else
        log "❌ Failed to restart background learner"
    fi
else
    log "✅ Background learner running"
fi

# Check trading bot
if ! pgrep -f "run_bot.py" > /dev/null; then
    log "⚠️ Trading bot not running - restarting..."
    cd "$BOT_DIR" && source venv/bin/activate && nohup python run_bot.py >> logs/bot.log 2>&1 &
    sleep 3
    if pgrep -f "run_bot.py" > /dev/null; then
        log "✅ Trading bot restarted successfully"
    else
        log "❌ Failed to restart trading bot"
    fi
else
    log "✅ Trading bot running"
fi

# Log quick stats
SNAPSHOTS=$(sqlite3 "$BOT_DIR/data/local_ai/market_data.db" "SELECT COUNT(*) FROM market_snapshots" 2>/dev/null || echo "0")
TRADES=$(sqlite3 "$BOT_DIR/data/local_ai/market_data.db" "SELECT COUNT(*) FROM trade_records WHERE exit_time IS NOT NULL" 2>/dev/null || echo "0")
log "📊 Stats: $SNAPSHOTS snapshots, $TRADES closed trades"
