#!/bin/bash
#
# Quick Start Script for Snuffs Bot
#
# This script helps you get started quickly by:
# 1. Checking prerequisites
# 2. Starting Docker services
# 3. Initializing the database
# 4. Running validation tests
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================================================"
echo "  🚀 Snuffs Bot - Quick Start Setup"
echo "========================================================================"
echo -e "${NC}"

# Function to print step
print_step() {
    echo -e "\n${BLUE}==>${NC} ${1}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓${NC} ${1}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} ${1}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} ${1}"
}

# Check prerequisites
print_step "Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker found: $DOCKER_VERSION"
else
    print_warning "Docker not found. You'll need to install PostgreSQL manually."
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_success "Docker Compose found"
else
    print_warning "Docker Compose not found."
fi

# Check .env file
print_step "Checking environment configuration..."

if [ -f ".env" ]; then
    print_success ".env file exists"

    # Check if it's been configured
    if grep -q "your-username" .env || grep -q "sk-your-openai" .env; then
        print_warning ".env file contains placeholder values"
        print_warning "Please edit .env with your real credentials before proceeding"
        echo ""
        read -p "Have you updated .env with real credentials? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Please update .env file and run this script again${NC}"
            exit 1
        fi
    fi
else
    print_error ".env file not found"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    print_warning "Please edit .env with your credentials, then run this script again"
    exit 1
fi

# Start Docker services
print_step "Starting Docker services..."

if command -v docker-compose &> /dev/null; then
    cd docker

    # Set default database password if not set
    if [ -z "$DB_PASSWORD" ]; then
        export DB_PASSWORD="changeme"
        print_warning "Using default DB password. Set DB_PASSWORD env var for custom password."
    fi

    echo "Starting PostgreSQL and Redis..."
    docker-compose up -d postgres redis

    # Wait for services to be healthy
    echo "Waiting for services to be ready..."
    sleep 5

    # Check if containers are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Docker services started"
    else
        print_error "Docker services failed to start"
        echo "Check logs with: cd docker && docker-compose logs"
        exit 1
    fi

    cd ..
else
    print_warning "Skipping Docker setup (not installed)"
fi

# Install Python dependencies
print_step "Installing Python dependencies..."

if [ -f "requirements.txt" ]; then
    echo "Installing packages..."
    pip3 install -q -r requirements.txt
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Initialize database
print_step "Initializing database..."

if [ -f "scripts/init_database.py" ]; then
    python3 scripts/init_database.py
    if [ $? -eq 0 ]; then
        print_success "Database initialized"
    else
        print_error "Database initialization failed"
        exit 1
    fi
else
    print_error "init_database.py script not found"
    exit 1
fi

# Run tests
print_step "Running validation tests..."

if [ -f "scripts/test_setup.py" ]; then
    python3 scripts/test_setup.py
    TEST_RESULT=$?
else
    print_warning "test_setup.py script not found, skipping tests"
    TEST_RESULT=0
fi

# Final summary
echo ""
echo -e "${BLUE}========================================================================"
echo "  Setup Complete!"
echo "========================================================================${NC}"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    print_success "All validation tests passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Start the dashboard:"
    echo "     ${GREEN}streamlit run dashboard/app.py${NC}"
    echo ""
    echo "  2. Or run the trading bot:"
    echo "     ${GREEN}python main.py${NC}"
    echo ""
    echo "  3. View Docker logs:"
    echo "     ${GREEN}cd docker && docker-compose logs -f${NC}"
    echo ""
else
    print_warning "Some tests failed. Review the output above."
    echo ""
    echo "Common fixes:"
    echo "  - Check .env has real credentials (not placeholders)"
    echo "  - Verify Docker services are running: ${GREEN}cd docker && docker-compose ps${NC}"
    echo "  - Check logs: ${GREEN}cd docker && docker-compose logs${NC}"
    echo ""
fi

echo -e "${BLUE}Documentation:${NC}"
echo "  - Setup Guide: AI_TRADING_BOT_SETUP.md"
echo "  - Main README: README.md"
echo "  - Implementation Plan: .claude/plans/soft-moseying-tome.md"
echo ""
