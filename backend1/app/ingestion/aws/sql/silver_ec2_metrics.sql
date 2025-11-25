-- silver_ec2_metrics.sql
-- Silver layer for EC2 metrics: cleaned, deduplicated data

CREATE TABLE IF NOT EXISTS __schema__.silver_ec2_metrics (
    instance_id TEXT,
    instance_name TEXT,
    instance_type TEXT,
    region TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    availability_zone TEXT,
    dimensions_json JSONB,

    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now(),

    UNIQUE(hash_key)
);

CREATE INDEX IF NOT EXISTS ix_silver_ec2_instance_time ON __schema__.silver_ec2_metrics (LOWER(instance_id), timestamp);
CREATE INDEX IF NOT EXISTS ix_silver_ec2_metric_name ON __schema__.silver_ec2_metrics (metric_name);

-- Insert from bronze to silver (deduplication)
INSERT INTO __schema__.silver_ec2_metrics (
    instance_id, instance_name, instance_type, region, account_id,
    timestamp, metric_name, value, unit, availability_zone,
    dimensions_json, hash_key, ingested_at
)
SELECT DISTINCT
    COALESCE(instance_id, '') AS instance_id,
    COALESCE(instance_name, '') AS instance_name,
    COALESCE(instance_type, '') AS instance_type,
    COALESCE(region, '') AS region,
    COALESCE(account_id, '') AS account_id,
    COALESCE(timestamp, now())::timestamp AS timestamp,
    COALESCE(metric_name, '') AS metric_name,
    COALESCE(value, 0.0) AS value,
    COALESCE(unit, '') AS unit,
    COALESCE(availability_zone, '') AS availability_zone,
    dimensions_json,
    hash_key,
    ingested_at
FROM __schema__.bronze_ec2_instance_metrics
ON CONFLICT (hash_key) DO NOTHING;
