-- silver_azure_public_ip_metrics.sql

CREATE TABLE IF NOT EXISTS __schema__.silver_azure_public_ip_metrics (
    -- Primary Keys/Identifiers
    metric_observation_id    VARCHAR(64) NOT NULL PRIMARY KEY,
    public_ip_name           TEXT,
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

    -- Resource properties
    resourceregion           TEXT,
    sku                      TEXT,
    tier                     TEXT,
    ip_address               TEXT,
    ip_version               TEXT,
    ip_allocation_method     TEXT,
    provisioning_state       TEXT,

    -- Audit/Lineage
    bronze_hash_key          VARCHAR(64),
    processed_at             TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_silver_pip_metrics_time ON __schema__.silver_azure_public_ip_metrics (observation_date, public_ip_name);

-- -----------------------------------------------------
-- LOAD SCRIPT
-- -----------------------------------------------------

INSERT INTO __schema__.silver_azure_public_ip_metrics (
    metric_observation_id,
    public_ip_name,
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
    tier,
    ip_address,
    ip_version,
    ip_allocation_method,
    provisioning_state,
    bronze_hash_key
)
SELECT
    t1.hash_key AS metric_observation_id,
    t1.public_ip_name,
    t1.resource_group,
    t1.subscription_id,
    t1.resource_id,
    
    t1.timestamp AS observation_timestamp,
    DATE(t1.timestamp) AS observation_date,
    EXTRACT(HOUR FROM t1.timestamp) AS observation_hour,

    t1.metric_name,
    t1.value,
    t1.unit,
    t1.namespace,

    t1.resourceregion,
    t1.sku,
    t1.tier,
    t1.ip_address,
    t1.ip_version,
    t1.ip_allocation_method,
    t1.provisioning_state,
    
    t1.hash_key AS bronze_hash_key
FROM
    __schema__.bronze_azure_public_ip_metrics t1
WHERE
    t1.value IS NOT NULL
    AND t1.timestamp IS NOT NULL
ON CONFLICT (metric_observation_id) DO NOTHING;