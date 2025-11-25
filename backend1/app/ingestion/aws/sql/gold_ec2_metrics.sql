-- gold_ec2_metrics.sql
-- Gold layer: dimensions + metrics fact for EC2 instances

-- DIM: EC2 instance
CREATE TABLE IF NOT EXISTS __schema__.dim_ec2_instance (
    instance_key SERIAL PRIMARY KEY,
    instance_id TEXT UNIQUE,
    instance_name TEXT,
    instance_type TEXT,
    region TEXT,
    account_id TEXT,
    availability_zone TEXT,
    first_seen TIMESTAMP DEFAULT now()
);

-- DIM: metric
CREATE TABLE IF NOT EXISTS __schema__.dim_ec2_metric (
    metric_key SERIAL PRIMARY KEY,
    metric_name TEXT UNIQUE,
    unit TEXT,
    description TEXT
);

-- DIM: time (hour)
CREATE TABLE IF NOT EXISTS __schema__.dim_time_hour_ec2 (
    time_key SERIAL PRIMARY KEY,
    event_hour TIMESTAMP UNIQUE,
    event_date DATE,
    year INT,
    month INT,
    day INT,
    hour INT
);

-- FACT: EC2 metrics (grain = instance_id + timestamp + metric_name)
CREATE TABLE IF NOT EXISTS __schema__.fact_ec2_metrics (
    fact_id BIGSERIAL PRIMARY KEY,
    instance_key INT REFERENCES __schema__.dim_ec2_instance(instance_key),
    time_key INT REFERENCES __schema__.dim_time_hour_ec2(time_key),

    instance_id TEXT,
    instance_name TEXT,
    account_id TEXT,
    timestamp TIMESTAMP,
    event_hour TIMESTAMP,
    event_date DATE,
    region TEXT,

    metric_name TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    dimensions_json JSONB,

    samples INT DEFAULT 1,
    hash_key VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMP DEFAULT now(),

    UNIQUE(hash_key)
);

CREATE INDEX IF NOT EXISTS ix_fact_ec2_instance_time ON __schema__.fact_ec2_metrics (LOWER(instance_id), timestamp);
CREATE INDEX IF NOT EXISTS ix_fact_ec2_metric_name ON __schema__.fact_ec2_metrics (metric_name);

-- UPSERT dimensions from silver
INSERT INTO __schema__.dim_ec2_instance (instance_id, instance_name, instance_type, region, account_id, availability_zone, first_seen)
SELECT DISTINCT
    COALESCE(instance_id,'') AS instance_id,
    COALESCE(instance_name,'') AS instance_name,
    COALESCE(instance_type,'') AS instance_type,
    COALESCE(region,'') AS region,
    COALESCE(account_id,'') AS account_id,
    COALESCE(availability_zone,'') AS availability_zone,
    MIN(ingested_at) OVER (PARTITION BY COALESCE(instance_id,'')) AS first_seen
FROM __schema__.silver_ec2_metrics s
ON CONFLICT (instance_id) DO UPDATE
  SET instance_name = EXCLUDED.instance_name,
      instance_type = EXCLUDED.instance_type,
      region = EXCLUDED.region,
      account_id = EXCLUDED.account_id,
      availability_zone = EXCLUDED.availability_zone
;

INSERT INTO __schema__.dim_ec2_metric (metric_name, unit, description)
SELECT DISTINCT metric_name, unit, '' FROM __schema__.silver_ec2_metrics
ON CONFLICT (metric_name) DO NOTHING
;

INSERT INTO __schema__.dim_time_hour_ec2 (event_hour, event_date, year, month, day, hour)
SELECT DISTINCT
    date_trunc('hour', COALESCE(timestamp, now())::timestamp) AS event_hour,
    (date_trunc('hour', COALESCE(timestamp, now())::timestamp))::date AS event_date,
    EXTRACT(YEAR FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS year,
    EXTRACT(MONTH FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS month,
    EXTRACT(DAY FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS day,
    EXTRACT(HOUR FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS hour
FROM __schema__.silver_ec2_metrics s
ON CONFLICT (event_hour) DO NOTHING
;

-- Insert aggregated metric facts from silver (deduplicated by hash_key)
WITH metric_agg AS (
    SELECT
        LOWER(COALESCE(instance_id,'')) AS instance_id,
        COALESCE(instance_name,'') AS instance_name,
        COALESCE(account_id,'') AS account_id,
        COALESCE(timestamp, now())::timestamp AS timestamp,
        date_trunc('hour', COALESCE(timestamp, now())::timestamp) AS event_hour,
        (date_trunc('hour', COALESCE(timestamp, now())::timestamp))::date AS event_date,
        COALESCE(region,'') AS region,
        metric_name,
        AVG(value::float) AS value,
        MAX(unit) AS unit,
        jsonb_agg(DISTINCT dimensions_json) FILTER (WHERE dimensions_json IS NOT NULL) AS dimensions_agg,

        COUNT(*) AS samples,

        md5(
            LOWER(COALESCE(instance_id,'')) || '|' ||
            COALESCE(to_char(date_trunc('second', COALESCE(timestamp, now())::timestamp),'YYYY-MM-DD HH24:MI:SS'),'') || '|' ||
            COALESCE(metric_name,'')
        ) AS fact_hash_key
    FROM __schema__.silver_ec2_metrics s
    GROUP BY 1,2,3,4,5,6,7,8
)

INSERT INTO __schema__.fact_ec2_metrics (
    instance_key, time_key, instance_id, instance_name, account_id, timestamp, event_hour, event_date,
    region, metric_name, value, unit, dimensions_json, samples, hash_key
)
SELECT
    inst.instance_key,
    th.time_key,
    a.instance_id,
    a.instance_name,
    a.account_id,
    a.timestamp,
    a.event_hour,
    a.event_date,
    a.region,
    a.metric_name,
    a.value,
    a.unit,
    CASE WHEN a.dimensions_agg IS NULL THEN NULL
         WHEN jsonb_array_length(a.dimensions_agg) = 1 THEN a.dimensions_agg->0
         ELSE a.dimensions_agg END AS dimensions_json,
    a.samples,
    a.fact_hash_key
FROM metric_agg a
LEFT JOIN __schema__.dim_ec2_instance inst ON inst.instance_id = a.instance_id
LEFT JOIN __schema__.dim_time_hour_ec2 th ON th.event_hour = a.event_hour
WHERE NOT EXISTS (
    SELECT 1 FROM __schema__.fact_ec2_metrics f WHERE f.hash_key = a.fact_hash_key
);

-- Convenient view: flattened metrics for queries / LLMs
CREATE OR REPLACE VIEW __schema__.gold_ec2_fact_metrics AS
SELECT
    instance_id,
    instance_name,
    timestamp,
    event_date,
    region,
    metric_name,
    value,
    unit,
    dimensions_json
FROM __schema__.fact_ec2_metrics;
