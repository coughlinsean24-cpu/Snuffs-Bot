#!/bin/bash
# Wrapper script for systemd to run background learner
cd /home/coughlinsean24/snuffs_bot_link

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

exec python scripts/background_learner.py
