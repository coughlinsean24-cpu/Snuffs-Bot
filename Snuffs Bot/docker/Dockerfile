FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/logs /app/data /app/data/local_ai

# Create non-root user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Expose dashboard port
EXPOSE 8501

# Default command - trading bot in paper mode
CMD ["python", "-m", "snuffs_bot.main", "trade", "--paper"]
