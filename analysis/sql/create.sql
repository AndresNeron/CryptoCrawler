/* Setting up the cryptocurrency database */

CREATE DATABASE cryptocurrency;

/* Create another user and grant permissions*/
CREATE USER crypto_user WITH PASSWORD 'HrEvPIaoSN53gG04Sb';
ALTER ROLE crypto_user WITH REPLICATION;

\c cryptocurrency

CREATE TABLE crypto_data (
    id SERIAL PRIMARY KEY,                 -- Auto-incremented ID
    cryptocurrency VARCHAR(50) NOT NULL,  -- Name of the cryptocurrency
    timestamp TIMESTAMP NOT NULL,          -- Timestamp of the record
    price NUMERIC(12, 2) NOT NULL,         -- Price in dollars
    market_cap VARCHAR(20),			       -- Market capitalization with units
    volume VARCHAR(20),			           -- Volume with units
    fdv VARCHAR(20)				           -- Fully Diluted Valuation (FDV) with units
);

CREATE INDEX idx_crypto_timestamp ON crypto_data (cryptocurrency, timestamp);

ALTER TABLE crypto_data
ADD CONSTRAINT unique_entry UNIQUE (cryptocurrency, timestamp, price);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE crypto_data TO crypto_user;


/* Another table creation */
CREATE TABLE personal_assets (
    id SERIAL PRIMARY KEY,             -- Auto-incremented unique ID
    cryptocurrency VARCHAR(50) ,  -- Name of the cryptocurrency
    timestamp TIMESTAMP ,      -- Time of the asset snapshot
    amount NUMERIC(20, 8) ,    -- Amount of cryptocurrency owned (up to 8 decimal places for precision)
    value_usd NUMERIC(20, 8),          -- Total value in USD at the timestamp (optional)
    notes TEXT                         -- Optional notes or comments
);

CREATE INDEX idx_assets_crypto_timestamp ON personal_assets (cryptocurrency, timestamp);


-- Grant privileges for crypto_data table
GRANT INSERT, UPDATE, DELETE ON TABLE crypto_data TO crypto_user;
GRANT INSERT, UPDATE, DELETE ON TABLE personal_assets TO crypto_user;
GRANT SELECT ON TABLE crypto_data, personal_assets TO crypto_user;
-- Grant usage and update privileges on the sequences
GRANT USAGE, SELECT, UPDATE ON SEQUENCE crypto_data_id_seq TO crypto_user;
GRANT USAGE, SELECT, UPDATE ON SEQUENCE personal_assets_id_seq TO crypto_user;

-- Create a new table to store timestamp and total assets in USD
CREATE TABLE total_assets (
    id SERIAL PRIMARY KEY,         -- Auto-incremented unique ID
    timestamp TIMESTAMP NOT NULL,  -- Timestamp of the record
    total_assets_usd NUMERIC(20, 2) NOT NULL  -- Total assets value in USD
);

-- Grant privileges for total_assets table
GRANT INSERT, UPDATE, DELETE ON TABLE total_assets TO crypto_user;
GRANT SELECT ON TABLE total_assets TO crypto_user;

-- Grant usage and update privileges on the sequence to crypto_user
GRANT USAGE, SELECT ON SEQUENCE total_assets_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE total_assets_id_seq TO crypto_user;

ALTER TABLE total_assets
ALTER COLUMN total_assets_usd TYPE NUMERIC(20, 8);

ALTER TABLE total_assets
ADD COLUMN total_assets_mxm NUMERIC(20, 8);

CREATE TABLE exchange_rate (
    id SERIAL PRIMARY KEY,               -- Auto-incrementing unique identifier
    timestamp TIMESTAMPTZ DEFAULT NOW(), -- Timestamp with timezone, defaulting to the current time
    currency1 CHAR(3) NOT NULL,          -- The base currency, e.g., 'USD'
    currency2 CHAR(3) NOT NULL,          -- The target currency, e.g., 'MXN'
    exchange_value NUMERIC(20, 8) NOT NULL -- The exchange rate value
);


-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON TABLE exchange_rate TO crypto_user;

-- Grant permissions on the sequence
GRANT USAGE, SELECT ON SEQUENCE exchange_rate_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE exchange_rate_id_seq TO crypto_user;


CREATE TABLE assets_per_crypto (
    id SERIAL PRIMARY KEY,             -- Auto-incremented unique identifier
    cryptocurrency VARCHAR(50) NOT NULL, -- Name of the cryptocurrency
    timestamp TIMESTAMP NOT NULL,      -- Timestamp of the calculation
    total_value_usd NUMERIC(20, 8) NOT NULL, -- Total value in USD for this cryptocurrency
    total_value_mxn NUMERIC(20, 8) NOT NULL -- Total value in MXN for this cryptocurrency
);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON TABLE assets_per_crypto TO crypto_user;

-- Grant usage and update privileges on the sequence to crypto_user
GRANT USAGE, SELECT ON SEQUENCE assets_per_crypto_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE assets_per_crypto_id_seq TO crypto_user;

GRANT USAGE, SELECT ON SEQUENCE assets_per_crypto_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE assets_per_crypto_id_seq TO crypto_user;

CREATE TABLE crypto_volatility (
    cryptocurrency TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    volatility DOUBLE PRECISION,
    id BIGSERIAL PRIMARY KEY -- Add id column as part of table definition
);

GRANT SELECT, INSERT, UPDATE ON TABLE crypto_volatility TO crypto_user;

-- Grant permissions on the automatically created sequence
GRANT USAGE, SELECT, UPDATE ON SEQUENCE crypto_volatility_id_seq TO crypto_user;
--GRANT USAGE, SELECT, UPDATE ON SEQUENCE crypto_volatility_id_seq1 TO crypto_user;

CREATE TABLE crawled_cryptos (
    id SERIAL PRIMARY KEY,                -- Auto-incremented unique identifier
    cryptocurrency VARCHAR(50) NOT NULL, -- Name of the cryptocurrency
    timestamp TIMESTAMP DEFAULT NOW()    -- Timestamp of the crawl
);

-- Grant permissions for crawled_cryptos
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE crawled_cryptos TO crypto_user;

-- Grant usage and update privileges on the sequence
GRANT USAGE, SELECT, UPDATE ON SEQUENCE crawled_cryptos_id_seq TO crypto_user;

ALTER TABLE crawled_cryptos
ADD CONSTRAINT unique_cryptocurrency UNIQUE (cryptocurrency);


/* Trading Simulation */
CREATE TABLE strategy_results (
    id SERIAL PRIMARY KEY,                  -- Unique ID for each strategy simulation
    strategy_name VARCHAR(100) NOT NULL,    -- Name or identifier of the strategy
    cryptocurrency VARCHAR(50),            -- Cryptocurrency used in the simulation
    start_date TIMESTAMP NOT NULL,          -- Simulation start date
    end_date TIMESTAMP NOT NULL,            -- Simulation end date
    initial_balance NUMERIC(20, 8),         -- Starting balance in USD
    final_balance NUMERIC(20, 8),           -- Ending balance in USD
    total_trades INT,                       -- Number of trades executed
    win_rate NUMERIC(5, 2),                 -- Percentage of profitable trades
    max_drawdown NUMERIC(5, 2),             -- Maximum drawdown percentage
    annualized_return NUMERIC(5, 2),        -- Annualized return percentage
    volatility NUMERIC(5, 2),               -- Volatility measure
    notes TEXT                              -- Additional notes or logs
);

GRANT SELECT, INSERT, UPDATE ON TABLE strategy_results TO crypto_user;
GRANT USAGE, SELECT, UPDATE ON SEQUENCE strategy_results_id_seq TO crypto_user;


CREATE TABLE binance_symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL
);

GRANT SELECT, INSERT, UPDATE, DELETE ON binance_symbols TO crypto_user;
GRANT USAGE, SELECT ON SEQUENCE binance_symbols_id_seq TO crypto_user;

-- Create a new table for storing slug-symbol pairs
CREATE TABLE crypto_slugs (
    id SERIAL PRIMARY KEY,        -- Auto-incremented unique identifier
    slug VARCHAR(100) NOT NULL,   -- Cryptocurrency slug (e.g., "0x", "1inch")
    symbol VARCHAR(20) NOT NULL   -- Cryptocurrency symbol (e.g., "ZRX", "1INCH")
);

-- Add a unique constraint to ensure no duplicate slug-symbol pairs
ALTER TABLE crypto_slugs
ADD CONSTRAINT unique_slug_symbol UNIQUE (slug, symbol);

-- Grant permissions for the new table
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE crypto_slugs TO crypto_user;

-- Grant usage and update privileges on the sequence
GRANT USAGE, SELECT ON SEQUENCE crypto_slugs_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE crypto_slugs_id_seq TO crypto_user;

-- Delete a unused column from personal_assets table, this data is contained in assets_per_crypto
ALTER TABLE personal_assets
DROP COLUMN value_usd;

CREATE TABLE ranked_cryptos (
    id SERIAL PRIMARY KEY,               -- Auto-incrementing unique ID
    symbol VARCHAR(20) NOT NULL,         -- Cryptocurrency symbol
    cumulative_return FLOAT NOT NULL,    -- Cumulative return from backtest
    rank INTEGER,                        -- Rank based on cumulative returns
    execution_timestamp TIMESTAMP NOT NULL -- Timestamp for each execution
);
-- Grant full permissions on the ranked_cryptos table
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE ranked_cryptos TO crypto_user;


-- Grant usage and update privileges on the sequence associated with ranked_cryptos
GRANT USAGE, SELECT ON SEQUENCE ranked_cryptos_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE ranked_cryptos_id_seq TO crypto_user;
GRANT TRUNCATE ON TABLE ranked_cryptos TO crypto_user;
GRANT USAGE ON SCHEMA public TO crypto_user;

-- Create the table for storing cryptocurrency rankings
CREATE TABLE crypto_rankings (
    id SERIAL PRIMARY KEY,                 -- Auto-incremented unique identifier
    cryptocurrency VARCHAR(50) NOT NULL,  -- Name of the cryptocurrency
    rank INT NOT NULL,                     -- Rank of the cryptocurrency
    score NUMERIC(10, 2) NOT NULL,         -- Calculated score
    timestamp TIMESTAMP DEFAULT NOW()      -- Timestamp of the ranking calculation
);

-- Add a unique constraint to prevent duplicate rankings for the same cryptocurrency and timestamp
ALTER TABLE crypto_rankings
ADD CONSTRAINT unique_crypto_rank UNIQUE (cryptocurrency, timestamp);

-- Grant permissions to crypto_user
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE crypto_rankings TO crypto_user;

-- Grant usage and update privileges on the sequence
GRANT USAGE, SELECT ON SEQUENCE crypto_rankings_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE crypto_rankings_id_seq TO crypto_user;

CREATE TABLE portfolio_data (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(50) NOT NULL,
    high_stop_portfolio NUMERIC(10, 2) NOT NULL,
    current_portfolio NUMERIC(10, 2) NOT NULL,
    low_stop_portfolio NUMERIC(10, 2) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions to crypto_user
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE portfolio_data TO crypto_user;

-- Grant usage and update privileges on the sequence
GRANT USAGE, SELECT ON SEQUENCE portfolio_data_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE portfolio_data_id_seq TO crypto_user;

CREATE TABLE training_results (
    id SERIAL PRIMARY KEY,
    crypto_symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_start TIMESTAMP,
    training_end TIMESTAMP,
    predicted_price NUMERIC,
    actual_price NUMERIC,
    mse NUMERIC,
    r2_score NUMERIC,
    model_type VARCHAR(50),
    additional_info JSONB
);

-- Grant permissions to crypto_user
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE training_results TO crypto_user;

-- Grant usage and update privileges on the sequence
GRANT USAGE, SELECT ON SEQUENCE training_results_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE training_results_id_seq TO crypto_user;


-- Store results related to lstm model hyper parameters for training
CREATE TABLE model_results (
    id SERIAL PRIMARY KEY,
    cryptocurrency VARCHAR(50),
    learning_rate FLOAT,
    batch_size INT,
    epochs INT,
    loss FLOAT,
    mae FLOAT,
    mse FLOAT,
    rmse FLOAT,
    mape FLOAT,
    r2 FLOAT,
    model_file_path TEXT,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions to crypto_user
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE model_results TO crypto_user;

-- Grant usage and update privileges on the sequence
GRANT USAGE, SELECT ON SEQUENCE model_results_id_seq TO crypto_user;
GRANT UPDATE ON SEQUENCE model_results_id_seq TO crypto_user;


-- Create a temporary table
CREATE TABLE temp_crypto_data (
  cryptocurrency TEXT,
  "timestamp" TIMESTAMP,
  price NUMERIC,
  market_cap TEXT,
  volume TEXT,
  fdv TEXT
);

-- Grant permissions to crypto_user
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE temp_crypto_data TO crypto_user;
