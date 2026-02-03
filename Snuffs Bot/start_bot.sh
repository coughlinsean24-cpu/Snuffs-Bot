#!/bin/bash
# Snuffs Bot Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     SNUFFS BOT - Autonomous 0DTE SPY Options Trading   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating from example...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${RED}Please edit .env with your credentials before running.${NC}"
        exit 1
    fi
fi

# Check Docker services
echo "Checking Docker services..."
if command -v docker &> /dev/null; then
    if ! docker ps | grep -q "snuffs-postgres"; then
        echo -e "${YELLOW}Starting Docker services...${NC}"
        docker-compose up -d
        sleep 5
    else
        echo -e "${GREEN}Docker services already running${NC}"
    fi
else
    echo -e "${YELLOW}Docker not found. Database may not be available.${NC}"
fi

# Parse arguments
COMMAND=${1:-"trade"}

case $COMMAND in
    trade)
        echo -e "${GREEN}Starting trading bot in PAPER mode...${NC}"
        python -m snuffs_bot.main trade --paper -v
        ;;
    dashboard)
        echo -e "${GREEN}Starting dashboard...${NC}"
        streamlit run dashboard/app.py --server.port 8501
        ;;
    both)
        echo -e "${GREEN}Starting bot and dashboard...${NC}"
        # Start dashboard in background
        streamlit run dashboard/app.py --server.port 8501 &
        DASH_PID=$!
        # Start bot
        python -m snuffs_bot.main trade --paper -v
        # Cleanup
        kill $DASH_PID 2>/dev/null
        ;;
    test)
        echo -e "${GREEN}Running tests...${NC}"
        python -m snuffs_bot.main test --all
        ;;
    status)
        echo -e "${GREEN}Checking system status...${NC}"
        python -m snuffs_bot.main status
        ;;
    *)
        echo "Usage: $0 {trade|dashboard|both|test|status}"
        echo ""
        echo "Commands:"
        echo "  trade     - Start the trading bot (paper mode)"
        echo "  dashboard - Start the web dashboard"
        echo "  both      - Start both bot and dashboard"
        echo "  test      - Run system tests"
        echo "  status    - Check system status"
        exit 1
        ;;
esac
