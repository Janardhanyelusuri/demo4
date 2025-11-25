-- bronze_s3_bucket_metrics.sql
-- Raw S3 ingestion table (bronze). Adjust columns if your producer emits extra fields.

CREATE TABLE IF NOT EXISTS __schema__.bronze_s3_bucket_metrics (
    bucket_name TEXT,
    region TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    storage_class TEXT,
    storage_type TEXT,
    dimensions_json JSONB,
    arn TEXT,
    storage_classes_sample_json JSONB,

    -- deterministic hash key (bucket-centered)
    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now()
);

-- Prevent duplicate raw rows by hash_key
CREATE UNIQUE INDEX IF NOT EXISTS ux_bronze_s3_hash ON __schema__.bronze_s3_bucket_metrics (hash_key);

-- convenience view for quick inspection
CREATE OR REPLACE VIEW __schema__.v_bronze_s3_recent AS
SELECT * FROM __schema__.bronze_s3_bucket_metrics ORDER BY ingested_at DESC LIMIT 1000;