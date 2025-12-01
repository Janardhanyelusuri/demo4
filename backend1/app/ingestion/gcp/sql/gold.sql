-- CREATE OR REPLACE VIEW __schema__.gold_gcp_location_dim AS
-- SELECT
--     DISTINCT region_id,
--     region_name,
--     x_location
-- FROM
--     __schema__.silver_focus_gcp_data;

-- CREATE OR REPLACE VIEW __schema__.gold_gcp_service_dim AS
-- SELECT
--     DISTINCT x_service_id,
--     sku_id,
--     service_name,
--     NULL AS service_category -- Placeholder for service_category to be added or derived
-- FROM
--     __schema__.silver_focus_gcp_data;

-- --CREATE OR REPLACE VIEW __schema__.gold_gcp_cost_dim AS
-- --SELECT
-- --    DISTINCT contracted_cost,
-- --    effective_cost,
-- --    list_cost
-- --FROM
-- --    __schema__.silver_focus_gcp_data;

-- CREATE OR REPLACE VIEW __schema__.gold_gcp_fact_dim AS
-- SELECT
--     billed_cost,
--     resource_name,
--     resource_type,
--     billing_period_start,
--     x_project_id,
--     region_id,
--     x_service_id,
--     charge_period_start,
--     contracted_cost,
--     charge_description,
--     charge_category,
--     substr(cast(digest(tags::text, 'sha256') as text), 3, 64) as tags_key,
--     __budget__::integer AS monthly_budget,
--     consumed_quantity,
--     pricing_quantity,
--     provider_name,
--     list_cost,
-- 	effective_cost
-- FROM
--    __schema__.silver_focus_gcp_data;

-- CREATE OR REPLACE FUNCTION gcp_tags_view_generation()
-- RETURNS text AS $$
-- DECLARE
--     record_tagkey record;
--     q_statement text = format(E'CREATE OR REPLACE VIEW __schema__.gold_gcp_tags_dim AS\nSELECT DISTINCT\n    substr(cast(digest(cast(tags AS text), ''sha256'') AS text), 3, 64) AS tags_key,');
-- BEGIN
--     -- Loop through each distinct tag key
--     FOR record_tagkey IN
--         SELECT DISTINCT jsonb_object_keys(tags) AS tagkey FROM __schema__.silver_focus_gcp_data
--     LOOP
--         q_statement := q_statement || format(E'\n    tags->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
--     END LOOP;

--     -- Remove the trailing comma and complete the query
--     q_statement := rtrim(q_statement, ',');
--     q_statement := q_statement || E'\nFROM __schema__.silver_focus_gcp_data\nWHERE\n    tags IS NOT NULL';
--     RAISE NOTICE E'\n%', q_statement;
--     RETURN q_statement;
-- END;
-- $$ LANGUAGE plpgsql;

-- DO $$
-- DECLARE
--     del_statement text := 'DROP VIEW IF EXISTS __schema__.gold_gcp_tags_dim';
--     q_statement text;
-- BEGIN
--     -- Generate the view creation query
--     q_statement := gcp_tags_view_generation();

--     -- Drop the existing view if it exists
--     EXECUTE del_statement;

--     -- Execute the dynamically generated query
--     EXECUTE q_statement;
-- END;
-- $$ LANGUAGE plpgsql;


CREATE OR REPLACE VIEW __schema__.gold_gcp_billing_dim AS
SELECT
   DISTINCT billing_account_id,
   sub_account_id
FROM
   __schema__.silver_focus_gcp_data;



--CREATE OR REPLACE VIEW __schema__.gold_gcp_cost_dim AS
--SELECT
--    DISTINCT contracted_cost,
--    effective_cost,
--    list_cost
--FROM
--    __schema__.silver_focus_gcp_data;



CREATE OR REPLACE VIEW __schema__.gold_gcp_fact_dim AS
SELECT
    billed_cost,
	billing_account_id,
    resource_name,
    resource_type,
    billing_period_start,
	billing_period_end,
    x_project_id,
    region_id,
    x_service_id,
    charge_period_start,
	charge_period_end,
    contracted_cost,
    charge_description,
    charge_category,
    md5(tags::text) as tags_key,
    __budget__::integer AS monthly_budget,
    consumed_quantity,
    pricing_quantity,
    provider_name,
    list_cost,
	effective_cost,
	region_name,
    x_location,
	sku_id,
	service_name,
	service_category,
    hash_key,
    resource_id
FROM
   __schema__.silver_focus_gcp_data;

CREATE OR REPLACE FUNCTION gcp_tags_view_generation()
RETURNS text AS $$
DECLARE
    record_tagkey record;
    q_statement text = format(E'CREATE OR REPLACE VIEW __schema__.gold_gcp_tags_dim AS\nSELECT DISTINCT\n    md5(cast(tags AS text)) AS tags_key,');
BEGIN
    -- Loop through each distinct tag key
    FOR record_tagkey IN
        SELECT DISTINCT jsonb_object_keys(tags) AS tagkey FROM __schema__.silver_focus_gcp_data
    LOOP
        q_statement := q_statement || format(E'\n    tags->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
    END LOOP;

    -- Remove the trailing comma and complete the query
    q_statement := rtrim(q_statement, ',');
    q_statement := q_statement || E'\nFROM __schema__.silver_focus_gcp_data\nWHERE\n    tags IS NOT NULL';
    RAISE NOTICE E'\n%', q_statement;
    RETURN q_statement;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    del_statement text := 'DROP VIEW IF EXISTS __schema__.gold_gcp_tags_dim CASCADE';
    q_statement text;
BEGIN
    -- Generate the view creation query
    q_statement := gcp_tags_view_generation();

    -- Drop the existing view if it exists
    EXECUTE del_statement;

    -- Execute the dynamically generated query
    EXECUTE q_statement;
END;
$$ LANGUAGE plpgsql;


