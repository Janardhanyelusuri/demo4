-- Ensure the role exists and grant the required privileges directly
DO
$$
BEGIN
    -- Create role if it does not exist
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cloudmeterpsqladmin') THEN
        EXECUTE 'CREATE ROLE cloudmeterpsqladmin WITH LOGIN PASSWORD ''CloudMeter12''';
    END IF;
    
    -- Grant privileges to the role
    IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'cloudmeter-db') THEN
        EXECUTE 'GRANT ALL PRIVILEGES ON DATABASE "cloudmeter-db" TO cloudmeterpsqladmin';
    END IF;
    
    -- Grant privileges to the role on schema and existing objects
    EXECUTE 'GRANT USAGE, CREATE ON SCHEMA public TO cloudmeterpsqladmin';
    EXECUTE 'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cloudmeterpsqladmin';
    EXECUTE 'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cloudmeterpsqladmin';
    EXECUTE 'GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO cloudmeterpsqladmin';
END
$$;

-- DO block to truncate or create the __schema__.silver_aws_cur_standard table and insert data
DO
$$
BEGIN
    -- Check if the table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '__schema__' AND table_name = 'silver_aws_cur_standard') THEN
        -- If the table exists, truncate it
        EXECUTE 'TRUNCATE TABLE __schema__.silver_aws_cur_standard';

        -- Insert new data
        EXECUTE '
            INSERT INTO __schema__.silver_aws_cur_standard (
                bill_payer_account_id,
                bill_payer_account_name,
                line_item_availability_zone,
                line_item_blended_cost,
                line_item_blended_rate,
                line_item_operation,
                line_item_product_code,
                line_item_resource_id,
                line_item_unblended_cost,
                line_item_unblended_rate,
                line_item_usage_account_id,
                line_item_usage_account_name,
                line_item_usage_amount,
                line_item_usage_date,
                line_item_usage_type,
                pricing_public_on_demand_cost,
                pricing_public_on_demand_rate,
                product_name,
                region,
                service_name,
                product_from_location,
                product_from_region_code,
                product_location,
                product_location_type,
                product_operation,
                product_product_family,
                product_region_code,
                product_servicecode,
                product_to_location_type,
                product_to_region_code,
                product_usagetype,
                reservation_subscription_id,
                resource_tags
            )
            SELECT
                bill_payer_account_id,
                bill_payer_account_name,
                line_item_availability_zone,
                line_item_blended_cost::DOUBLE PRECISION,
                line_item_blended_rate::DOUBLE PRECISION,
                line_item_operation,
                line_item_product_code,
                line_item_resource_id,
                line_item_unblended_cost::DOUBLE PRECISION,
                line_item_unblended_rate::DOUBLE PRECISION,
                line_item_usage_account_id,
                line_item_usage_account_name,
                line_item_usage_amount::DOUBLE PRECISION,
                line_item_usage_start_date::DATE AS line_item_usage_date,
                line_item_usage_type,
                pricing_public_on_demand_cost::DOUBLE PRECISION,
                pricing_public_on_demand_rate::DOUBLE PRECISION,
                extract_col_from_product(product, ''product_name'') AS product_name,
                extract_col_from_product(product, ''region'') AS region,
                extract_col_from_product(product, ''servicename'') AS service_name,
                product_from_location,
                product_from_region_code,
                product_location,
                product_location_type,
                product_operation,
                product_product_family,
                product_region_code,
                product_servicecode,
                product_to_location_type,
                product_to_region_code,
                product_usagetype,
                reservation_subscription_id,
                convert_and_cleanup_json(resource_tags) AS resource_tags
            FROM __schema__.bronze_aws_cur_standard';
    ELSE
        -- If the table does not exist, create it and insert data
        EXECUTE '
            CREATE TABLE __schema__.silver_aws_cur_standard AS
            SELECT
                bill_payer_account_id,
                bill_payer_account_name,
                line_item_availability_zone,
                line_item_blended_cost::DOUBLE PRECISION,
                line_item_blended_rate::DOUBLE PRECISION,
                line_item_operation,
                line_item_product_code,
                line_item_resource_id,
                line_item_unblended_cost::DOUBLE PRECISION,
                line_item_unblended_rate::DOUBLE PRECISION,
                line_item_usage_account_id,
                line_item_usage_account_name,
                line_item_usage_amount::DOUBLE PRECISION,
                line_item_usage_start_date::DATE AS line_item_usage_date,
                line_item_usage_type,
                pricing_public_on_demand_cost::DOUBLE PRECISION,
                pricing_public_on_demand_rate::DOUBLE PRECISION,
                extract_col_from_product(product, ''product_name'') AS product_name,
                extract_col_from_product(product, ''region'') AS region,
                extract_col_from_product(product, ''servicename'') AS service_name,
                product_from_location,
                product_from_region_code,
                product_location,
                product_location_type,
                product_operation,
                product_product_family,
                product_region_code,
                product_servicecode,
                product_to_location_type,
                product_to_region_code,
                product_usagetype,
                reservation_subscription_id,
                convert_and_cleanup_json(resource_tags) AS resource_tags
            FROM __schema__.bronze_aws_cur_standard';
    END IF;
END
$$;

-- Add tags column if does not exists

ALTER TABLE __schema__.silver_aws_cur_standard
ADD COLUMN IF NOT EXISTS tags JSONB;

-- Create a temporary table to hold JSONB values
CREATE TEMP TABLE temp_tags (tags JSONB);

-- Insert unique JSONB values into the temporary table
INSERT INTO temp_tags (tags) VALUES
('{"team": "devops", "product": "cm"}'::jsonb),
('{"team": "marketing", "product": "crm"}'::jsonb),
('{"team": "finance", "product": "erp"}'::jsonb),
('{"team": "hr", "product": "workday"}'::jsonb),
('{"team": "engineering", "product": "jira"}'::jsonb),
('{"team": "support", "product": "zendesk"}'::jsonb),
('{"team": "design", "product": "figma"}'::jsonb),
('{"team": "legal", "product": "clio"}'::jsonb),
('{"team": "it", "product": "service_now"}'::jsonb),
('{"team": "security", "product": "splunk"}'::jsonb),
('{"team": "product", "product": "productboard"}'::jsonb),
('{"team": "research", "product": "qualtrics"}'::jsonb),
('{"team": "growth", "product": "hubspot"}'::jsonb),
('{"team": "media", "product": "sprinklr"}'::jsonb),
('{"team": "data", "product": "snowflake"}'::jsonb);

WITH row_numbers AS (
    SELECT ctid, ROW_NUMBER() OVER () AS rn
    FROM __schema__.silver_aws_cur_standard
),
tag_rows AS (
    SELECT tags, ROW_NUMBER() OVER () AS tag_rn
    FROM temp_tags
),
all_tags AS (
    SELECT ctid, rn, tags
    FROM row_numbers
    JOIN tag_rows
    ON (rn - 1) % (SELECT COUNT(*) FROM temp_tags) + 1 = tag_rn
)
UPDATE __schema__.silver_aws_cur_standard
SET tags = all_tags.tags
FROM all_tags
WHERE __schema__.silver_aws_cur_standard.ctid = all_tags.ctid;
