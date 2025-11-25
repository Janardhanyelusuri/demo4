-- silver_vpc_metrics.sql
-- Silver layer for VPC metrics: cleaned, deduplicated data

CREATE TABLE IF NOT EXISTS __schema__.silver_vpc_metrics (
    vpc_id TEXT,
    vpc_name TEXT,
    resource_id TEXT,
    resource_type TEXT,
    region TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    dimensions_json JSONB,

    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now(),

    UNIQUE(hash_key)
);

CREATE INDEX IF NOT EXISTS ix_silver_vpc_resource_time ON __schema__.silver_vpc_metrics (LOWER(resource_id), timestamp);
CREATE INDEX IF NOT EXISTS ix_silver_vpc_metric_name ON __schema__.silver_vpc_metrics (metric_name);
CREATE INDEX IF NOT EXISTS ix_silver_vpc_type ON __schema__.silver_vpc_metrics (resource_type);

-- Insert from bronze to silver (deduplication)
INSERT INTO __schema__.silver_vpc_metrics (
    vpc_id, vpc_name, resource_id, resource_type, region, account_id,
    timestamp, metric_name, value, unit,
    dimensions_json, hash_key, ingested_at
)
SELECT DISTINCT
    COALESCE(vpc_id, '') AS vpc_id,
    COALESCE(vpc_name, '') AS vpc_name,
    COALESCE(resource_id, '') AS resource_id,
    COALESCE(resource_type, 'vpc') AS resource_type,
    COALESCE(region, '') AS region,
    COALESCE(account_id, '') AS account_id,
    COALESCE(timestamp, now())::timestamp AS timestamp,
    COALESCE(metric_name, '') AS metric_name,
    COALESCE(value, 0.0) AS value,
    COALESCE(unit, '') AS unit,
    dimensions_json,
    hash_key,
    ingested_at
FROM __schema__.bronze_vpc_metrics
ON CONFLICT (hash_key) DO NOTHING;
