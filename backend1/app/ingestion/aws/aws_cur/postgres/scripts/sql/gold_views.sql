-- Create or replace the aws_dim_account view
CREATE OR REPLACE VIEW __schema__.gold_aws_dim_account AS
SELECT DISTINCT ON (line_item_usage_account_id)
    line_item_usage_account_id AS usage_account_id,
    line_item_usage_account_name AS usage_account_name,
    bill_payer_account_id,
    bill_payer_account_name
FROM
    __schema__.silver_aws_cur_standard;

-- Create or replace the aws_fact_cost view
CREATE OR REPLACE VIEW __schema__.gold_aws_fact_cost AS
SELECT
    line_item_usage_account_id AS usage_account_id,
    md5(resource_tags::text) AS tags_key,
    line_item_usage_date AS usage_date,
    line_item_unblended_cost AS unblended_cost,
    line_item_blended_cost AS blended_cost,
    line_item_operation AS operation,
    line_item_usage_type AS usage_type,
    case
		when product_servicecode = 'AmazonEC2' and product_product_family = 'Compute Instance'
		or product_servicecode = 'AmazonRDS' and product_product_family = 'Database Instance'
		then split_part(line_item_usage_type, ':', -1)
		else 'not applicable'
	end as instance_usage_type,
    line_item_resource_id AS resource_id,
    split_part(split_part(line_item_resource_id, ':'::text, '-1'::integer), '/'::text, '-1'::integer) AS resource_name,
    line_item_product_code AS product_code,
    product_name,
    product_product_family AS product_family,
    COALESCE(region, product_region_code) AS product_region,CASE
        WHEN resource_tags IS NULL THEN 'untagged'::text
        ELSE 'tagged'::text
    END AS is_tagged,
    '__budget__'::integer AS monthly_budget
FROM
    __schema__.silver_aws_cur_standard;

CREATE OR REPLACE VIEW __schema__.gold_aws_monthly_budget AS
SELECT
    '__budget__' as monthly_budget;

-- Create the function to generate the aws_tags view
-- CREATE OR REPLACE FUNCTION aws_tags_view_generation()
-- RETURNS text AS $$
-- DECLARE
--     record_tagkey record;
--     q_statement text = format(E'CREATE OR REPLACE VIEW __schema__.gold_aws_tags AS\nSELECT DISTINCT\n    substr(cast(digest(cast(resource_tags AS text), \'sha256\') AS text), 3, 64) AS tags_key,');
-- BEGIN
--     -- Loop through each distinct tag key
--     FOR record_tagkey IN
--         SELECT DISTINCT jsonb_object_keys(resource_tags) AS tagkey FROM __schema__.silver_aws_cur_standard
--     LOOP
--         q_statement := q_statement || format(E'\n    resource_tags->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
--     END LOOP;

--     -- Remove the trailing comma and complete the query
--     q_statement := rtrim(q_statement, ',');
--     q_statement := q_statement || E'\nFROM __schema__.silver_aws_cur_standard\nWHERE\n    resource_tags IS NOT NULL';
--     RAISE NOTICE E'\n%', q_statement;
--     RETURN q_statement;
-- END;
-- $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION aws_tags_view_generation()
RETURNS text AS $$
DECLARE
    record_tagkey record;
    q_statement text = format(E'CREATE OR REPLACE VIEW __schema__.gold_aws_tags AS\nSELECT DISTINCT\n    md5(cast(resource_tags AS text)) AS tags_key,');
BEGIN
    -- Loop through each distinct tag key from resource_tags
    FOR record_tagkey IN
        SELECT DISTINCT jsonb_object_keys(resource_tags) AS tagkey FROM __schema__.silver_aws_cur_standard
    LOOP
        q_statement := q_statement || format(E'\n    resource_tags->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
    END LOOP;

    -- Loop through each distinct tag key from tags
    FOR record_tagkey IN
        SELECT DISTINCT jsonb_object_keys(tags) AS tagkey FROM __schema__.silver_aws_cur_standard
    LOOP
        q_statement := q_statement || format(E'\n    tags->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
    END LOOP;

    -- Remove the trailing comma and complete the query
    q_statement := rtrim(q_statement, ',');
    q_statement := q_statement || E'\nFROM __schema__.silver_aws_cur_standard\nWHERE\n    resource_tags IS NOT NULL OR tags IS NOT NULL';
    RAISE NOTICE E'\n%', q_statement;
    RETURN q_statement;
END;
$$ LANGUAGE plpgsql;


-- Execute the function to generate and create the aws_tags view
DO $$
DECLARE
    del_statement text := 'DROP VIEW IF EXISTS __schema__.gold_aws_tags';
    q_statement text;
BEGIN
    -- Generate the view creation query
    q_statement := aws_tags_view_generation();
    
    -- Drop the existing view if it exists
    EXECUTE del_statement;
    
    -- Execute the dynamically generated query
    EXECUTE q_statement;
END;
$$ LANGUAGE plpgsql;
