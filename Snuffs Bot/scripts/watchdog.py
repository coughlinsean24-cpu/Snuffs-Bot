#!/usr/bin/env python3
"""
Watchdog - Monitors bot and learner, restarts if crashed
Runs as a background daemon
"""
import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path

BOT_DIR = Path("/home/coughlinsean24/Snuffs Bot")
LOG_FILE = BOT_DIR / "logs" / "watchdog.log"
CHECK_INTERVAL = 300  # 5 minutes

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"{timestamp} | {msg}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def is_running(process_name):
    result = subprocess.run(
        ["pgrep", "-f", process_name],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def start_process(script, log_name):
    cmd = f'cd "{BOT_DIR}" && source venv/bin/activate && nohup python {script} >> logs/{log_name}.log 2>&1 &'
    subprocess.Popen(cmd, shell=True, executable="/bin/bash")

def main():
    log("🐕 Watchdog started - monitoring bot and learner")
    
    while True:
        try:
            # Check background learner
            if not is_running("background_learner.py"):
                log("⚠️ Background learner crashed - restarting...")
                start_process("scripts/background_learner.py", "learner")
                time.sleep(3)
                if is_running("background_learner.py"):
                    log("✅ Background learner restarted")
                else:
                    log("❌ Failed to restart learner")
            
            # Check trading bot
            if not is_running("run_bot.py"):
                log("⚠️ Trading bot crashed - restarting...")
                start_process("run_bot.py", "bot")
                time.sleep(5)
                if is_running("run_bot.py"):
                    log("✅ Trading bot restarted")
                else:
                    log("❌ Failed to restart bot")
            
            # Log status every hour
            if datetime.now().minute == 0:
                log("✅ All systems running normally")
            
        except Exception as e:
            log(f"❌ Watchdog error: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
