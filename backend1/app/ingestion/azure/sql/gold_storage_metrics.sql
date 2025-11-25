-- gold_storage_metrics.sql

-- =========================================================================
-- STEP 1: DEFINE STAR SCHEMA TABLES (No changes here, definitions are correct)
-- =========================================================================

-- dim_date.sql: Stores date attributes (REQUIRED for fact table foreign key)
CREATE TABLE IF NOT EXISTS __schema__.dim_date (
    date_key                  INTEGER PRIMARY KEY, -- Surrogate Key, typically YYYYMMDD
    full_date                 DATE NOT NULL UNIQUE,
    year                      INTEGER NOT NULL,
    quarter                   INTEGER NOT NULL,
    month_name                TEXT,
    day_of_month              INTEGER NOT NULL,
    day_name                  TEXT
    -- Other date attributes (e.g., is_holiday, fiscal_week) can be added here
);


-- dim_storage_account.sql: Stores resource metadata.
CREATE TABLE IF NOT EXISTS __schema__.dim_storage_account (
    storage_account_key       SERIAL PRIMARY KEY, -- Surrogate Key
    storage_account_name      TEXT NOT NULL,
    resource_group            TEXT,
    subscription_id           TEXT,
    resource_id               TEXT NOT NULL UNIQUE, -- Natural Key
    region                    TEXT,
    sku                       TEXT,
    access_tier               TEXT,
    replication_type          TEXT, -- The column name used in the DIM table
    kind                      TEXT,
    status                    TEXT,
    last_updated_at           TIMESTAMP DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_storage_resource_id ON __schema__.dim_storage_account (resource_id);


-- dim_metric.sql: Stores metric type and classification.
CREATE TABLE IF NOT EXISTS __schema__.dim_metric (
    metric_key                SERIAL PRIMARY KEY, -- Surrogate Key
    metric_name               TEXT NOT NULL UNIQUE, -- Natural Key
    unit                      TEXT,
    namespace                 TEXT,
    metric_type_category      TEXT, -- e.g., 'Capacity', 'Transaction'
    recommendation_focus      TEXT -- e.g., 'Optimization', 'Cost Tracking'
);


-- fact_storage_daily_usage.sql: Aggregated usage data.
CREATE TABLE IF NOT EXISTS __schema__.fact_storage_daily_usage (
    -- Foreign Keys
    date_key                  INTEGER NOT NULL REFERENCES __schema__.dim_date (date_key),
    storage_account_key       INTEGER NOT NULL REFERENCES __schema__.dim_storage_account (storage_account_key),
    metric_key                INTEGER NOT NULL REFERENCES __schema__.dim_metric (metric_key),

    -- Degenerate Dimension
    resource_group_name       TEXT,

    -- Fact Metrics
    daily_value_sum           DOUBLE PRECISION,
    daily_value_avg           DOUBLE PRECISION,
    daily_value_max           DOUBLE PRECISION,
    observation_count         BIGINT,
    daily_cost_sum            DOUBLE PRECISION,

    PRIMARY KEY (date_key, storage_account_key, metric_key)
);
CREATE INDEX IF NOT EXISTS ix_fact_daily_usage ON __schema__.fact_storage_daily_usage (storage_account_key, date_key);


-- =========================================================================
-- STEP 2: ETL - POPULATE DIMENSION TABLES (FIXED)
-- =========================================================================
-- dim_date_load.sql: Script to populate the dim_date table.

DO $$
DECLARE
    -- Start date for generation (adjust as needed, e.g., 2025-01-01)
    start_date DATE := '2025-08-01'::DATE; 
    -- End date for generation (adjust as needed)
    end_date DATE := '2025-12-31'::DATE; 
BEGIN
    -- This CTE generates a series of dates
    EXECUTE '
        INSERT INTO __schema__.dim_date (
            date_key,
            full_date,
            year,
            quarter,
            month_name,
            day_of_month,
            day_name
        )
        SELECT
            CAST(TO_CHAR(d.day, ''YYYYMMDD'') AS INTEGER) AS date_key,
            d.day AS full_date,
            EXTRACT(YEAR FROM d.day)::INTEGER AS year,
            EXTRACT(QUARTER FROM d.day)::INTEGER AS quarter,
            TO_CHAR(d.day, ''Month'') AS month_name,
            EXTRACT(DAY FROM d.day)::INTEGER AS day_of_month,
            TO_CHAR(d.day, ''Day'') AS day_name
        FROM (
            SELECT generate_series($1::DATE, $2::DATE, ''1 day''::interval) AS day
        ) d
        ON CONFLICT (full_date) DO NOTHING
    ' USING start_date, end_date;
END $$;


-- 2.1 Load/Update dim_storage_account (SCD Type 1 logic)
INSERT INTO __schema__.dim_storage_account (
    storage_account_name,
    resource_group,
    subscription_id,
    resource_id,
    region,
    sku,
    access_tier,
    replication_type,  -- Column in DIM table
    kind,
    status
)
SELECT DISTINCT ON (t1.resource_id)
    t1.storage_account_name,
    t1.resource_group,
    t1.subscription_id,
    t1.resource_id,
    t1.resourceregion,
    t1.sku,
    t1.access_tier,
    t1.replication AS replication_type,  -- Alias source column to match DIM column
    t1.kind,
    t1.storage_account_status AS status  -- Alias source column to match DIM column

FROM
    __schema__.silver_azure_storage_metrics_clean t1
ORDER BY t1.resource_id, t1.observation_timestamp DESC NULLS LAST
ON CONFLICT (resource_id) DO UPDATE SET
    resource_group = EXCLUDED.resource_group,
    subscription_id = EXCLUDED.subscription_id,
    region = EXCLUDED.region,
    sku = EXCLUDED.sku,
    access_tier = EXCLUDED.access_tier,
    -- FIX: Reference the correct aliases from the SELECT statement:
    replication_type = EXCLUDED.replication_type, 
    kind = EXCLUDED.kind,
    status = EXCLUDED.status, 
    last_updated_at = now();


-- 2.2 Load dim_metric (Standard lookup/insert for new metrics)
INSERT INTO __schema__.dim_metric (
    metric_name,
    unit,
    namespace
)
SELECT DISTINCT
    metric_name,
    unit,
    namespace
FROM
    __schema__.silver_azure_storage_metrics_clean
ON CONFLICT (metric_name) DO NOTHING;


-- =========================================================================
-- STEP 3: ETL - AGGREGATE AND LOAD FACT TABLE (No changes needed here)
-- =========================================================================

INSERT INTO __schema__.fact_storage_daily_usage (
    date_key,
    storage_account_key,
    metric_key,
    resource_group_name,
    daily_value_sum,
    daily_value_avg,
    daily_value_max,
    observation_count,
    daily_cost_sum
)
SELECT
    -- 1. Date Key Lookup (Casting the date to INTEGER YYYYMMDD format)
    CAST(TO_CHAR(t1.observation_date, 'YYYYMMDD') AS INTEGER) AS date_key,

    -- 2. Storage Account Key Lookup (JOIN on Natural Key: resource_id)
    storage_dim.storage_account_key,

    -- 3. Metric Key Lookup (JOIN on Natural Key: metric_name)
    metric_dim.metric_key,

    -- 4. Degenerate Dimension
    t1.resource_group AS resource_group_name,

    -- 5. Aggregated Fact Metrics
    SUM(t1.value) AS daily_value_sum,
    AVG(t1.value) AS daily_value_avg,
    MAX(t1.value) AS daily_value_max,
    COUNT(*) AS observation_count,
    SUM(t1.cost) AS daily_cost_sum
FROM
    __schema__.silver_azure_storage_metrics_clean t1
JOIN
    __schema__.dim_storage_account storage_dim ON t1.resource_id = storage_dim.resource_id
JOIN
    __schema__.dim_metric metric_dim ON t1.metric_name = metric_dim.metric_name
GROUP BY
    1, 2, 3, 4
ON CONFLICT (date_key, storage_account_key, metric_key)
DO UPDATE SET
    daily_value_sum = EXCLUDED.daily_value_sum,
    daily_value_avg = EXCLUDED.daily_value_avg,
    daily_value_max = EXCLUDED.daily_value_max,
    observation_count = EXCLUDED.observation_count,
    daily_cost_sum = EXCLUDED.daily_cost_sum;