-- Initial database setup script
-- This runs when the PostgreSQL container first starts

-- Create database if not exists (handled by POSTGRES_DB env var)
-- Set proper encoding and locale
ALTER DATABASE trading_db SET timezone TO 'America/New_York';

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trader;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Trading bot database initialized successfully';
END $$;
