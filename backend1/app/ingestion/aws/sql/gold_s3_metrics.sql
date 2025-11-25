-- gold_s3_metrics.sql
-- Gold layer: dimensions + metrics fact (grain = bucket_name + timestamp + metric_name)

-- DIM: s3 bucket
CREATE TABLE IF NOT EXISTS __schema__.dim_s3_bucket (
    bucket_id SERIAL PRIMARY KEY,
    bucket_name TEXT UNIQUE,
    region TEXT,
    account_id TEXT,
    first_seen TIMESTAMP DEFAULT now()
);

-- DIM: metric
CREATE TABLE IF NOT EXISTS __schema__.dim_s3_metric (
    metric_key SERIAL PRIMARY KEY,
    metric_name TEXT UNIQUE,
    unit TEXT,
    description TEXT
);

-- DIM: time (hour)
CREATE TABLE IF NOT EXISTS __schema__.dim_time_hour (
    time_key SERIAL PRIMARY KEY,
    event_hour TIMESTAMP UNIQUE,
    event_date DATE,
    year INT,
    month INT,
    day INT,
    hour INT
);

-- FACT: metrics (grain = bucket_name + timestamp(second) + metric_name)
CREATE TABLE IF NOT EXISTS __schema__.fact_s3_metrics (
    fact_id BIGSERIAL PRIMARY KEY,
    bucket_id INT REFERENCES __schema__.dim_s3_bucket(bucket_id),
    time_key INT REFERENCES __schema__.dim_time_hour(time_key),

    bucket_name TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    event_hour TIMESTAMP,
    event_date DATE,

    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    storage_class TEXT,
    storage_type TEXT,
    dimensions_json JSONB,

    samples INT DEFAULT 1,
    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now(),

    UNIQUE(hash_key)
);

CREATE INDEX IF NOT EXISTS ix_fact_s3_bucket_time ON __schema__.fact_s3_metrics (LOWER(bucket_name), timestamp);

-- UPSERT dims from silver
INSERT INTO __schema__.dim_s3_bucket (bucket_name, region, account_id, first_seen)
SELECT DISTINCT
    COALESCE(bucket_name,'') AS bucket_name,
    COALESCE(region,'') AS region,
    COALESCE(account_id,'') AS account_id,
    MIN(ingested_at) OVER (PARTITION BY COALESCE(bucket_name,'')) AS first_seen
FROM __schema__.silver_s3_metrics s
ON CONFLICT (bucket_name) DO UPDATE
  SET region = EXCLUDED.region,
      account_id = EXCLUDED.account_id
;

INSERT INTO __schema__.dim_s3_metric (metric_name, unit, description)
SELECT DISTINCT metric_name, unit, '' FROM __schema__.silver_s3_metrics
ON CONFLICT (metric_name) DO NOTHING
;

INSERT INTO __schema__.dim_time_hour (event_hour, event_date, year, month, day, hour)
SELECT DISTINCT
    date_trunc('hour', COALESCE(timestamp, now())::timestamp) AS event_hour,
    (date_trunc('hour', COALESCE(timestamp, now())::timestamp))::date AS event_date,
    EXTRACT(YEAR FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS year,
    EXTRACT(MONTH FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS month,
    EXTRACT(DAY FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS day,
    EXTRACT(HOUR FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS hour
FROM __schema__.silver_s3_metrics s
ON CONFLICT (event_hour) DO NOTHING
;

-- Insert aggregated metric facts from silver (deduplicated by hash_key)
-- Use sample aggregations to compress duplicates: AVG value per (bucket,timestamp,metric_name)
WITH metric_agg AS (
    SELECT
        LOWER(COALESCE(bucket_name,'')) AS bucket_name,
        COALESCE(account_id,'') AS account_id,
        COALESCE(timestamp, now())::timestamp AS timestamp,
        date_trunc('hour', COALESCE(timestamp, now())::timestamp) AS event_hour,
        (date_trunc('hour', COALESCE(timestamp, now())::timestamp))::date AS event_date,
        metric_name,
        AVG(value::float) AS value,
        MAX(unit) AS unit,
        MAX(storage_class) AS storage_class,
        MAX(storage_type) AS storage_type,
        jsonb_agg(DISTINCT dimensions_json) FILTER (WHERE dimensions_json IS NOT NULL) AS dimensions_agg,

        COUNT(*) AS samples,

        md5(
            LOWER(COALESCE(bucket_name,'')) || '|' ||
            COALESCE(to_char(date_trunc('second', COALESCE(timestamp, now())::timestamp),'YYYY-MM-DD HH24:MI:SS'),'') || '|' ||
            COALESCE(metric_name,'')
        ) AS fact_hash_key
    FROM __schema__.silver_s3_metrics s
    GROUP BY 1,2,3,4,5,6
)

INSERT INTO __schema__.fact_s3_metrics (
    bucket_id, time_key, bucket_name, account_id, timestamp, event_hour, event_date,
    metric_name, value, unit, storage_class, storage_type, dimensions_json, samples, hash_key
)
SELECT
    bd.bucket_id,
    th.time_key,
    a.bucket_name,
    a.account_id,
    a.timestamp,
    a.event_hour,
    a.event_date,
    a.metric_name,
    a.value,
    a.unit,
    a.storage_class,
    a.storage_type,
    CASE WHEN a.dimensions_agg IS NULL THEN NULL
         WHEN jsonb_array_length(a.dimensions_agg) = 1 THEN a.dimensions_agg->0
         ELSE a.dimensions_agg END AS dimensions_json,
    a.samples,
    a.fact_hash_key
FROM metric_agg a
LEFT JOIN __schema__.dim_s3_bucket bd ON bd.bucket_name = a.bucket_name
LEFT JOIN __schema__.dim_time_hour th ON th.event_hour = a.event_hour
WHERE NOT EXISTS (
    SELECT 1 FROM __schema__.fact_s3_metrics f WHERE f.hash_key = a.fact_hash_key
);

-- convenient view: flattened metrics for query / LLMs
CREATE OR REPLACE VIEW __schema__.gold_s3_fact_metrics AS
SELECT
    bucket_name,
    timestamp,
    metric_name,
    value,
    unit,
    storage_class,
    storage_type,
    dimensions_json
FROM __schema__.fact_s3_metrics;
