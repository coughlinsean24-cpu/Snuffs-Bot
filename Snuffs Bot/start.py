#!/usr/bin/env python3
"""
Snuffs Bot - All-in-One Launcher

Starts both the trading engine and dashboard simultaneously.
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Change to project directory
PROJECT_DIR = Path(__file__).parent
os.chdir(PROJECT_DIR)

processes = []


def cleanup(signum=None, frame=None):
    """Clean up all processes on exit"""
    print("\n🛑 Shutting down Snuffs Bot...")
    for name, proc in processes:
        if proc.poll() is None:
            print(f"   Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("✅ Shutdown complete")
    sys.exit(0)


def main():
    print("=" * 60)
    print("🤖 SNUFFS BOT - All-in-One Launcher")
    print("=" * 60)
    print()

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Activate virtual environment if exists
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable

    # Start the trading engine (paper mode)
    print("📈 Starting Trading Engine (Paper Mode)...")
    engine_proc = subprocess.Popen(
        [python_cmd, "-m", "snuffs_bot", "trade", "--paper", "-v"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(("Trading Engine", engine_proc))
    time.sleep(2)

    # Start the dashboard
    print("📊 Starting Dashboard...")
    dashboard_proc = subprocess.Popen(
        [python_cmd, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", "8501",
         "--server.headless", "true"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(("Dashboard", dashboard_proc))
    time.sleep(3)

    print()
    print("=" * 60)
    print("✅ Snuffs Bot is running!")
    print("=" * 60)
    print()
    print("📊 Dashboard:  http://localhost:8501")
    print("📈 Engine:     Running in paper trading mode")
    print()
    print("Press Ctrl+C to stop everything")
    print("=" * 60)
    print()

    # Monitor processes and print output
    while True:
        for name, proc in processes:
            if proc.poll() is not None:
                print(f"⚠️  {name} stopped unexpectedly (exit code: {proc.returncode})")
                # Read any remaining output
                output, _ = proc.communicate()
                if output:
                    print(f"   Last output: {output.decode()[-500:]}")
        time.sleep(1)


if __name__ == "__main__":
    main()
