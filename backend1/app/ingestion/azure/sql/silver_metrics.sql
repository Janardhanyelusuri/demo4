-- silver_metrics.sql: Cleaned, deduplicated, incremental table.

DO $$
BEGIN
    -- STEP 1: Ensure the Silver Table Exists
    -- Uses IF NOT EXISTS to prevent dropping/recreating the table on subsequent runs.
    EXECUTE '
        CREATE TABLE IF NOT EXISTS __schema__.silver_azure_vm_metrics (
            vm_name TEXT,
            resource_group TEXT,
            subscription_id TEXT,
            timestamp TIMESTAMP,
            value DOUBLE PRECISION,
            metric_name TEXT,
            unit TEXT,
            displaydescription TEXT,
            namespace TEXT,
            resourceregion TEXT,
            resource_id TEXT,
            instance_type TEXT,
            -- hash_key is the PRIMARY KEY for deduplication
            hash_key TEXT NOT NULL UNIQUE 
        )
    ';

    -- STEP 2: Ensure the Unique Index Exists
    -- This is essential for the efficiency of the ON CONFLICT clause.
    EXECUTE '
        CREATE UNIQUE INDEX IF NOT EXISTS ux_silver_vm_metrics_hash ON __schema__.silver_azure_vm_metrics (hash_key)
    ';

    -- STEP 3: Incremental Insert from Bronze to Silver
    EXECUTE '
        INSERT INTO __schema__.silver_azure_vm_metrics (
            vm_name,
            resource_group,
            subscription_id,
            timestamp,
            value,
            metric_name,
            unit,
            displaydescription,
            namespace,
            resourceregion,
            resource_id,
            instance_type,
            hash_key
        )
        SELECT
            vm_name,
            resource_group,
            subscription_id,
            timestamp,
            value,
            metric_name,
            unit,
            displaydescription,
            namespace,
            resourceregion,
            resource_id,
            instance_type,
            hash_key
        FROM __schema__.bronze_azure_vm_metrics
        -- 💡 INCREMENTAL LOGIC: If a record with this hash_key exists in Silver, skip insertion.
        ON CONFLICT (hash_key) DO NOTHING
    ';

    -- STEP 4: Clean up Bronze
    -- Truncate the Bronze table after the successful load into Silver.

END $$;