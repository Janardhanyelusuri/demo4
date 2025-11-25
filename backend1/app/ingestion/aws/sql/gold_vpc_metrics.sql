-- gold_vpc_metrics.sql
-- Gold layer: dimensions + metrics fact for VPC resources

-- DIM: VPC resource (VPCs, NAT Gateways, VPN connections, etc.)
CREATE TABLE IF NOT EXISTS __schema__.dim_vpc_resource (
    resource_key SERIAL PRIMARY KEY,
    resource_id TEXT UNIQUE,
    vpc_id TEXT,
    vpc_name TEXT,
    resource_type TEXT,
    region TEXT,
    account_id TEXT,
    first_seen TIMESTAMP DEFAULT now()
);

-- DIM: metric
CREATE TABLE IF NOT EXISTS __schema__.dim_vpc_metric (
    metric_key SERIAL PRIMARY KEY,
    metric_name TEXT UNIQUE,
    unit TEXT,
    description TEXT
);

-- DIM: time (hour)
CREATE TABLE IF NOT EXISTS __schema__.dim_time_hour_vpc (
    time_key SERIAL PRIMARY KEY,
    event_hour TIMESTAMP UNIQUE,
    event_date DATE,
    year INT,
    month INT,
    day INT,
    hour INT
);

-- FACT: VPC metrics (grain = resource_id + timestamp + metric_name)
CREATE TABLE IF NOT EXISTS __schema__.fact_vpc_metrics (
    fact_id BIGSERIAL PRIMARY KEY,
    resource_key INT REFERENCES __schema__.dim_vpc_resource(resource_key),
    time_key INT REFERENCES __schema__.dim_time_hour_vpc(time_key),

    resource_id TEXT,
    vpc_id TEXT,
    vpc_name TEXT,
    resource_type TEXT,
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

CREATE INDEX IF NOT EXISTS ix_fact_vpc_resource_time ON __schema__.fact_vpc_metrics (LOWER(resource_id), timestamp);
CREATE INDEX IF NOT EXISTS ix_fact_vpc_metric_name ON __schema__.fact_vpc_metrics (metric_name);
CREATE INDEX IF NOT EXISTS ix_fact_vpc_type ON __schema__.fact_vpc_metrics (resource_type);

-- UPSERT dimensions from silver
INSERT INTO __schema__.dim_vpc_resource (resource_id, vpc_id, vpc_name, resource_type, region, account_id, first_seen)
SELECT DISTINCT
    COALESCE(resource_id,'') AS resource_id,
    COALESCE(vpc_id,'') AS vpc_id,
    COALESCE(vpc_name,'') AS vpc_name,
    COALESCE(resource_type,'vpc') AS resource_type,
    COALESCE(region,'') AS region,
    COALESCE(account_id,'') AS account_id,
    MIN(ingested_at) OVER (PARTITION BY COALESCE(resource_id,'')) AS first_seen
FROM __schema__.silver_vpc_metrics s
ON CONFLICT (resource_id) DO UPDATE
  SET vpc_id = EXCLUDED.vpc_id,
      vpc_name = EXCLUDED.vpc_name,
      resource_type = EXCLUDED.resource_type,
      region = EXCLUDED.region,
      account_id = EXCLUDED.account_id
;

INSERT INTO __schema__.dim_vpc_metric (metric_name, unit, description)
SELECT DISTINCT metric_name, unit, '' FROM __schema__.silver_vpc_metrics
ON CONFLICT (metric_name) DO NOTHING
;

INSERT INTO __schema__.dim_time_hour_vpc (event_hour, event_date, year, month, day, hour)
SELECT DISTINCT
    date_trunc('hour', COALESCE(timestamp, now())::timestamp) AS event_hour,
    (date_trunc('hour', COALESCE(timestamp, now())::timestamp))::date AS event_date,
    EXTRACT(YEAR FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS year,
    EXTRACT(MONTH FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS month,
    EXTRACT(DAY FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS day,
    EXTRACT(HOUR FROM date_trunc('hour', COALESCE(timestamp, now())::timestamp))::int AS hour
FROM __schema__.silver_vpc_metrics s
ON CONFLICT (event_hour) DO NOTHING
;

-- Insert aggregated metric facts from silver (deduplicated by hash_key)
WITH metric_agg AS (
    SELECT
        LOWER(COALESCE(resource_id,'')) AS resource_id,
        COALESCE(vpc_id,'') AS vpc_id,
        COALESCE(vpc_name,'') AS vpc_name,
        COALESCE(resource_type,'vpc') AS resource_type,
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
            LOWER(COALESCE(resource_id,'')) || '|' ||
            COALESCE(to_char(date_trunc('second', COALESCE(timestamp, now())::timestamp),'YYYY-MM-DD HH24:MI:SS'),'') || '|' ||
            COALESCE(metric_name,'')
        ) AS fact_hash_key
    FROM __schema__.silver_vpc_metrics s
    GROUP BY 1,2,3,4,5,6,7,8,9,10
)

INSERT INTO __schema__.fact_vpc_metrics (
    resource_key, time_key, resource_id, vpc_id, vpc_name, resource_type, account_id,
    timestamp, event_hour, event_date, region, metric_name, value, unit,
    dimensions_json, samples, hash_key
)
SELECT
    res.resource_key,
    th.time_key,
    a.resource_id,
    a.vpc_id,
    a.vpc_name,
    a.resource_type,
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
LEFT JOIN __schema__.dim_vpc_resource res ON res.resource_id = a.resource_id
LEFT JOIN __schema__.dim_time_hour_vpc th ON th.event_hour = a.event_hour
WHERE NOT EXISTS (
    SELECT 1 FROM __schema__.fact_vpc_metrics f WHERE f.hash_key = a.fact_hash_key
);

-- Convenient view: flattened metrics for queries / LLMs
CREATE OR REPLACE VIEW __schema__.gold_vpc_fact_metrics AS
SELECT
    resource_id,
    vpc_id,
    vpc_name,
    resource_type,
    timestamp,
    event_date,
    region,
    metric_name,
    value,
    unit,
    dimensions_json
FROM __schema__.fact_vpc_metrics;
