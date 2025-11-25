-- bronze_vpc_metrics.sql
-- Raw VPC and networking metrics ingestion table (bronze layer)
-- Includes VPC Flow Logs, NAT Gateway, VPN, and Network-related metrics

CREATE TABLE IF NOT EXISTS __schema__.bronze_vpc_metrics (
    vpc_id TEXT,
    vpc_name TEXT,
    resource_id TEXT,  -- Could be NAT Gateway ID, VPN ID, etc.
    resource_type TEXT,  -- 'vpc', 'nat_gateway', 'vpn_connection', 'vpc_endpoint'
    region TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    dimensions_json JSONB,

    -- deterministic hash key
    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now()
);

-- Prevent duplicate raw rows by hash_key
CREATE UNIQUE INDEX IF NOT EXISTS ux_bronze_vpc_hash ON __schema__.bronze_vpc_metrics (hash_key);

-- convenience view for quick inspection
CREATE OR REPLACE VIEW __schema__.v_bronze_vpc_recent AS
SELECT * FROM __schema__.bronze_vpc_metrics ORDER BY ingested_at DESC LIMIT 1000;
