#!/bin/bash
# Start Background Learner
# Runs independently to collect data and train AI

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting Background Learner..."
echo "This runs independently of the trading bot."
echo ""

python scripts/background_learner.py
