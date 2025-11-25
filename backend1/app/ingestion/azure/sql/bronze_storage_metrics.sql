-- bronze_storage_metrics.sql
-- Raw ingestion table for storage account metrics (bronze)
-- Uses __schema__ placeholder replaced by run_sql_file()

CREATE TABLE IF NOT EXISTS __schema__.bronze_azure_storage_account_metrics (
    storage_account_name     TEXT,
    resource_group           TEXT,
    subscription_id          TEXT,
    timestamp                TIMESTAMP,
    value                    DOUBLE PRECISION,
    metric_name              TEXT,
    unit                     TEXT,
    displaydescription       TEXT,
    namespace                TEXT,
    resourceregion           TEXT,
    resource_id              TEXT,
    sku                      TEXT,
    access_tier              TEXT,
    replication              TEXT,
    location                 TEXT,
    kind                     TEXT,
    storage_account_status   TEXT,
    cost                     DOUBLE PRECISION,
    -- deterministic hash key for dedupe (MD5 hex)
    hash_key                 VARCHAR(64) NOT NULL,
    ingested_at              TIMESTAMP DEFAULT now()
);

-- Unique index on hash_key prevents duplicates at DB level
CREATE UNIQUE INDEX IF NOT EXISTS ux_bronze_storage_hash ON __schema__.bronze_azure_storage_account_metrics (hash_key);

-- Helpful view: recent ingestions
CREATE OR REPLACE VIEW __schema__.v_bronze_storage_recent AS
SELECT * FROM __schema__.bronze_azure_storage_account_metrics
ORDER BY ingested_at DESC
LIMIT 1000;
