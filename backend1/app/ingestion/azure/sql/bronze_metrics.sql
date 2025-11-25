-- bronze_metrics.sql

CREATE TABLE if not exists __schema__.bronze_azure_vm_metrics (
    vm_name TEXT,
    resource_group TEXT,
    subscription_id TEXT,
    timestamp TIMESTAMP,
    value FLOAT,                        -- Metric value (normalized if CPU)
    metric_name TEXT,
    unit TEXT,
    displaydescription TEXT,             -- Human-readable description of the metric
    namespace TEXT,
    resourceregion TEXT,                 -- Region of the resource (e.g., eastus)
    resource_id TEXT,                    -- Full Azure resource ID
    instance_type TEXT,                  -- VM SKU like Standard_D4s_v3
    cost FLOAT,                         -- Cost placeholder (nullable)
    hash_key TEXT UNIQUE
);