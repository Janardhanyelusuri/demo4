CREATE TABLE IF NOT EXISTS __schema__.genai_response (
    vm_name TEXT,
    instance_type TEXT,
    total_cost DOUBLE PRECISION,
    average_percentage_cpu DOUBLE PRECISION,
    average_network_in DOUBLE PRECISION,
    average_network_out DOUBLE PRECISION,
    average_disk_read_bytes DOUBLE PRECISION,
    average_disk_write_bytes DOUBLE PRECISION,
    average_burst_iops DOUBLE PRECISION,
    suggested_instance_type TEXT,
    suggested_reason TEXT,
    suggested_cost DOUBLE PRECISION,
    cost_saving DOUBLE PRECISION,
    hash_key TEXT PRIMARY KEY 
);

truncate table  __schema__.genai_response;


-- DO $$
-- BEGIN
--   IF EXISTS (
--     SELECT FROM information_schema.tables
--     WHERE table_schema = __schema__
--       AND table_name = 'genai_response'
--   ) THEN
--     EXECUTE format('TRUNCATE TABLE __schema__.genai_response');
--   END IF;
-- END $$;