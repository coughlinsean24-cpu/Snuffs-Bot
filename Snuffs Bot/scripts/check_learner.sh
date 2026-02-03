#!/bin/bash
# Check Snuffs Bot Background Learner Status
# Run this anytime to see if the AI is learning

echo "=============================================="
echo "🧠 SNUFFS BOT - AI LEARNER STATUS"
echo "=============================================="
echo ""

# Current time
echo "📅 Current time: $(TZ=America/New_York date '+%I:%M:%S %p %Z - %A')"
echo ""

# Service status
echo "📊 Service Status:"
if systemctl --user is-active snuffs-learner >/dev/null 2>&1; then
    echo "   ✅ Background Learner: RUNNING"
    uptime=$(systemctl --user show snuffs-learner --property=ActiveEnterTimestamp | cut -d= -f2)
    echo "   ⏱️  Started: $uptime"
else
    echo "   ❌ Background Learner: NOT RUNNING"
    echo ""
    echo "   To start: systemctl --user start snuffs-learner"
fi
echo ""

# Status file
status_file="/home/coughlinsean24/Snuffs Bot/data/local_ai/learner_status.json"
if [ -f "$status_file" ]; then
    echo "📈 Learning Stats:"
    cat "$status_file" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"   Total Snapshots: {data.get('total_snapshots', 0):,}\")
print(f\"   Today's Snapshots: {data.get('snapshots_collected', 0)}\")
print(f\"   Simulations Run: {data.get('simulations_run', 0)}\")
win_rate = data.get('simulation_win_rate', 0) * 100
print(f\"   Simulation Win Rate: {win_rate:.1f}%\")
print(f\"   Pending Simulations: {data.get('pending_simulations', 0)}\")
print(f\"   Market Hours: {'Yes' if data.get('is_market_hours') else 'No'}\")
print(f\"   Last Update: {data.get('last_update', 'Unknown')}\")
"
else
    echo "   ⚠️  No status file yet (will be created when market opens)"
fi
echo ""

# Database stats
db_file="/home/coughlinsean24/Snuffs Bot/data/local_ai/market_data.db"
if [ -f "$db_file" ]; then
    echo "💾 Database Stats:"
    sqlite3 "$db_file" "SELECT COUNT(*) FROM market_snapshots;" 2>/dev/null | while read count; do
        echo "   Market Snapshots: $count"
    done
    sqlite3 "$db_file" "SELECT COUNT(*) FROM trade_outcomes;" 2>/dev/null | while read count; do
        echo "   Trade Records: $count"
    done
    sqlite3 "$db_file" "SELECT COUNT(*) FROM simulated_trades;" 2>/dev/null | while read count; do
        echo "   Simulated Trades: $count"
    done
fi
echo ""

# Recent log
echo "📋 Recent Activity (last 5 entries):"
tail -5 "/home/coughlinsean24/Snuffs Bot/logs/background_learner.log" 2>/dev/null | grep -v "^$" | while read line; do
    echo "   $line"
done
echo ""
echo "=============================================="
