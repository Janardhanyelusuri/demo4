-- bronze_ec2_instance_metrics.sql
-- Raw EC2 CloudWatch metrics ingestion table (bronze layer)

CREATE TABLE IF NOT EXISTS __schema__.bronze_ec2_instance_metrics (
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

    -- deterministic hash key (instance-centered)
    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now()
);

-- Prevent duplicate raw rows by hash_key
CREATE UNIQUE INDEX IF NOT EXISTS ux_bronze_ec2_hash ON __schema__.bronze_ec2_instance_metrics (hash_key);

-- convenience view for quick inspection
CREATE OR REPLACE VIEW __schema__.v_bronze_ec2_recent AS
SELECT * FROM __schema__.bronze_ec2_instance_metrics ORDER BY ingested_at DESC LIMIT 1000;
