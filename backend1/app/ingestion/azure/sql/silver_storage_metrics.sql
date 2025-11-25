-- silver_azure_storage_metrics_clean.sql
-- Cleansed and enriched storage account metrics (silver)

CREATE TABLE IF NOT EXISTS __schema__.silver_azure_storage_metrics_clean (
    -- Primary Keys/Identifiers
    metric_observation_id    VARCHAR(64) NOT NULL PRIMARY KEY, -- Using hash_key from bronze
    storage_account_name     TEXT,
    resource_group           TEXT,
    subscription_id          TEXT,
    resource_id              TEXT,

    -- Time dimensions
    observation_timestamp    TIMESTAMP,
    observation_date         DATE,
    observation_hour         INTEGER,

    -- Metric details
    metric_name              TEXT,
    value                    DOUBLE PRECISION,
    unit                     TEXT,
    namespace                TEXT,

    -- Resource properties (Dimension attributes)
    resourceregion           TEXT,
    sku                      TEXT,
    access_tier              TEXT,
    replication              TEXT,
    kind                     TEXT,
    storage_account_status   TEXT,

    -- Cost/Billing
    cost                     DOUBLE PRECISION,

    -- Audit/Lineage
    bronze_hash_key          VARCHAR(64), -- Link back to bronze for lineage
    processed_at             TIMESTAMP DEFAULT now()
);

-- Index for quick joins and time-based filtering
CREATE INDEX IF NOT EXISTS ix_silver_metrics_time ON __schema__.silver_azure_storage_metrics_clean (observation_date, storage_account_name);
-- silver_azure_storage_metrics_load.sql
-- ETL script to insert/merge data from Bronze to Silver

INSERT INTO __schema__.silver_azure_storage_metrics_clean (
    metric_observation_id,
    storage_account_name,
    resource_group,
    subscription_id,
    resource_id,
    observation_timestamp,
    observation_date,
    observation_hour,
    metric_name,
    value,
    unit,
    namespace,
    resourceregion,
    sku,
    access_tier,
    replication,
    kind,
    storage_account_status,
    cost,
    bronze_hash_key
)
-- Use SELECT ... ON CONFLICT DO NOTHING for idempotent upserts based on hash_key
SELECT
    -- 1. Identifiers/Keys
    t1.hash_key AS metric_observation_id,
    t1.storage_account_name,
    t1.resource_group,
    t1.subscription_id,
    t1.resource_id,

    -- 2. Time Dimensions (Enrichment and Casting)
    t1.timestamp AS observation_timestamp,
    DATE(t1.timestamp) AS observation_date,  -- Extracting Date part
    EXTRACT(HOUR FROM t1.timestamp) AS observation_hour, -- Extracting Hour part

    -- 3. Metric Details (Basic Cleansing/Validation)
    t1.metric_name,
    t1.value,
    t1.unit,
    t1.namespace,

    -- 4. Resource Properties
    t1.resourceregion,
    t1.sku,
    t1.access_tier,
    t1.replication,
    t1.kind,
    t1.storage_account_status,

    -- 5. Cost/Billing
    t1.cost,

    -- 6. Lineage
    t1.hash_key AS bronze_hash_key
FROM
    __schema__.bronze_azure_storage_account_metrics t1
-- 7. Cleansing/Filtering Logic: Only include records with essential data
WHERE
    t1.value IS NOT NULL
    AND t1.timestamp IS NOT NULL
    AND t1.resource_id IS NOT NULL
    AND t1.hash_key IS NOT NULL
    -- OPTIONAL: Add filtering here if you only want to process new records
    -- e.g., WHERE t1.ingested_at > (SELECT MAX(processed_at) FROM __schema__.silver_azure_storage_metrics_clean)

-- Handling duplicates: Ensures the insert is idempotent based on the primary key
ON CONFLICT (metric_observation_id) DO NOTHING;