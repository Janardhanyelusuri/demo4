-- silver_s3_metrics.sql
-- Normalized silver table for S3 metrics (recomputes canonical hash_key and avoids duplicates)

CREATE TABLE IF NOT EXISTS __schema__.silver_s3_metrics (
    bucket_name TEXT,
    region TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    event_date DATE,
    event_hour TIMESTAMP,
    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    storage_class TEXT,
    storage_type TEXT,
    dimensions_json JSONB,
    arn TEXT,
    storage_classes_sample_json JSONB,
    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_silver_s3_bucket_time ON __schema__.silver_s3_metrics (LOWER(bucket_name), timestamp);

-- Insert unique rows from bronze into silver
INSERT INTO __schema__.silver_s3_metrics (
    bucket_name, region, account_id, timestamp, event_date, event_hour,
    metric_name, value, unit, storage_class, storage_type, dimensions_json, arn, storage_classes_sample_json, hash_key
)
SELECT
    COALESCE(b.bucket_name,'') AS bucket_name,
    COALESCE(b.region,'') AS region,
    COALESCE(b.account_id,'') AS account_id,
    COALESCE(b.timestamp, now())::timestamp AS timestamp,
    (COALESCE(b.timestamp, now())::timestamp)::date AS event_date,
    date_trunc('hour', COALESCE(b.timestamp, now())::timestamp) AS event_hour,
    COALESCE(b.metric_name,'') AS metric_name,
    b.value::double precision AS value,
    COALESCE(b.unit,'') AS unit,
    COALESCE(b.storage_class,'') AS storage_class,
    COALESCE(b.storage_type,'') AS storage_type,
    b.dimensions_json,
    b.arn,
    b.storage_classes_sample_json,
    -- canonical hash - second precision, lower(bucket_name)
    md5(
        LOWER(COALESCE(b.bucket_name,'')) || '|' ||
        COALESCE(to_char(date_trunc('second', COALESCE(b.timestamp, now())::timestamp),'YYYY-MM-DD HH24:MI:SS'),'') || '|' ||
        COALESCE(b.metric_name,'') || '|' ||
        COALESCE(round(COALESCE(b.value,0.0)::numeric,6)::text,'')
    ) AS hash_key
FROM __schema__.bronze_s3_bucket_metrics b
WHERE NOT EXISTS (
    SELECT 1 FROM __schema__.bronze_s3_bucket_metrics s
    WHERE s.hash_key = md5(
        LOWER(COALESCE(b.bucket_name,'')) || '|' ||
        COALESCE(to_char(date_trunc('second', COALESCE(b.timestamp, now())::timestamp),'YYYY-MM-DD HH24:MI:SS'),'') || '|' ||
        COALESCE(b.metric_name,'') || '|' ||
        COALESCE(round(COALESCE(b.value,0.0)::numeric,6)::text,'')
    )
);
