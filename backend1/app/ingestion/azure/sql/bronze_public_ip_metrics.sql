-- bronze_public_ip_metrics.sql

CREATE TABLE IF NOT EXISTS __schema__.bronze_azure_public_ip_metrics (
    public_ip_name           TEXT,
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
    
    -- Public IP Specific Props
    sku                      TEXT,
    tier                     TEXT,
    ip_address               TEXT,
    ip_version               TEXT,
    ip_allocation_method     TEXT,
    location                 TEXT,
    provisioning_state       TEXT,

    -- Deterministic hash key
    hash_key                 VARCHAR(64) NOT NULL,
    ingested_at              TIMESTAMP DEFAULT now()
);

-- Unique index on hash_key prevents duplicates at DB level
CREATE UNIQUE INDEX IF NOT EXISTS ux_bronze_public_ip_hash ON __schema__.bronze_azure_public_ip_metrics (hash_key);