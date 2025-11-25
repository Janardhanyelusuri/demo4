-- gold_public_ip_metrics.sql

-- =========================================================================
-- STEP 1: DEFINE STAR SCHEMA TABLES
-- =========================================================================

-- Note: dim_date and dim_metric are shared. 
-- We assume dim_date is already created by the storage script.
-- If not, refer to gold_storage_metrics.sql for dim_date creation.

-- dim_public_ip.sql: specific dimension for Public IP resources
CREATE TABLE IF NOT EXISTS __schema__.dim_public_ip (
    public_ip_key             SERIAL PRIMARY KEY,
    public_ip_name            TEXT NOT NULL,
    resource_group            TEXT,
    subscription_id           TEXT,
    resource_id               TEXT NOT NULL UNIQUE, -- Natural Key
    region                    TEXT,
    ip_address                TEXT,
    ip_version                TEXT,
    sku                       TEXT,
    tier                      TEXT,
    allocation_method         TEXT,
    last_updated_at           TIMESTAMP DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_public_ip_rid ON __schema__.dim_public_ip (resource_id);

-- fact_public_ip_daily_usage.sql
CREATE TABLE IF NOT EXISTS __schema__.fact_public_ip_daily_usage (
    -- Foreign Keys
    date_key                  INTEGER NOT NULL REFERENCES __schema__.dim_date (date_key),
    public_ip_key             INTEGER NOT NULL REFERENCES __schema__.dim_public_ip (public_ip_key),
    metric_key                INTEGER NOT NULL REFERENCES __schema__.dim_metric (metric_key),

    -- Degenerate Dimension
    resource_group_name       TEXT,

    -- Fact Metrics
    daily_value_sum           DOUBLE PRECISION,
    daily_value_avg           DOUBLE PRECISION,
    daily_value_max           DOUBLE PRECISION,
    observation_count         BIGINT,

    PRIMARY KEY (date_key, public_ip_key, metric_key)
);
CREATE INDEX IF NOT EXISTS ix_fact_pip_daily ON __schema__.fact_public_ip_daily_usage (public_ip_key, date_key);

-- =========================================================================
-- STEP 2: ETL - POPULATE DIMENSIONS
-- =========================================================================

-- 2.1 Load/Update dim_public_ip (SCD Type 1)
INSERT INTO __schema__.dim_public_ip (
    public_ip_name,
    resource_group,
    subscription_id,
    resource_id,
    region,
    ip_address,
    ip_version,
    sku,
    tier,
    allocation_method
)
SELECT DISTINCT
    t1.public_ip_name,
    t1.resource_group,
    t1.subscription_id,
    t1.resource_id,
    t1.resourceregion,
    t1.ip_address,
    t1.ip_version,
    t1.sku,
    t1.tier,
    t1.ip_allocation_method
FROM
    __schema__.silver_azure_public_ip_metrics t1
ON CONFLICT (resource_id) DO UPDATE SET
    resource_group = EXCLUDED.resource_group,
    ip_address = EXCLUDED.ip_address,
    sku = EXCLUDED.sku,
    tier = EXCLUDED.tier,
    last_updated_at = now();

-- 2.2 Update dim_metric (Add new metrics like PacketCount)
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
    __schema__.silver_azure_public_ip_metrics
ON CONFLICT (metric_name) DO NOTHING;

-- =========================================================================
-- STEP 3: ETL - AGGREGATE AND LOAD FACT TABLE
-- =========================================================================

INSERT INTO __schema__.fact_public_ip_daily_usage (
    date_key,
    public_ip_key,
    metric_key,
    resource_group_name,
    daily_value_sum,
    daily_value_avg,
    daily_value_max,
    observation_count
)
SELECT
    CAST(TO_CHAR(t1.observation_date, 'YYYYMMDD') AS INTEGER) AS date_key,
    pip_dim.public_ip_key,
    metric_dim.metric_key,
    t1.resource_group AS resource_group_name,
    
    SUM(t1.value) AS daily_value_sum,
    AVG(t1.value) AS daily_value_avg,
    MAX(t1.value) AS daily_value_max,
    COUNT(*) AS observation_count
FROM
    __schema__.silver_azure_public_ip_metrics t1
JOIN
    __schema__.dim_public_ip pip_dim ON t1.resource_id = pip_dim.resource_id
JOIN
    __schema__.dim_metric metric_dim ON t1.metric_name = metric_dim.metric_name
GROUP BY
    1, 2, 3, 4
ON CONFLICT (date_key, public_ip_key, metric_key)
DO UPDATE SET
    daily_value_sum = EXCLUDED.daily_value_sum,
    daily_value_avg = EXCLUDED.daily_value_avg,
    daily_value_max = EXCLUDED.daily_value_max,
    observation_count = EXCLUDED.observation_count;