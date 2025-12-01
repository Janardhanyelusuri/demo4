-- Create or replace view for gold_aws_billing_dim
CREATE OR REPLACE VIEW __schema__.gold_aws_billing_dim AS
SELECT DISTINCT
    "BillingAccountId" AS billing_account_id,
    "BillingAccountName" AS billing_account_name,
    "SubAccountId" AS sub_account_id,
    "SubAccountName" AS sub_account_name
FROM __schema__.silver_focus_aws;

CREATE OR REPLACE VIEW __schema__.gold_aws_fact_focus AS
SELECT
    "BilledCost" AS billed_cost,
    "ConsumedUnit" AS consumed_unit,
    "ConsumedQuantity" AS consumed_quantity,
    "ChargePeriodStart"::timestamp AS charge_period_start,  -- Convert to timestamp
    "ChargePeriodEnd"::timestamp AS charge_period_end,      -- Convert to timestamp
    "ContractedCost" AS contracted_cost,
    "EffectiveCost" AS effective_cost,
    "ListCost" AS list_cost,
    "ListUnitPrice" AS list_unit_price,
    "RegionId" AS region_id,
    "RegionName" AS region_name,
    "PricingCategory" AS pricing_category,
    "PricingQuantity" AS pricing_quantity,
    "PricingUnit" AS pricing_unit,
    "ContractedUnitPrice" AS contracted_unit_price,
    "ProviderName" AS provider_name,
    "ResourceId" AS resource_id,
    "BillingPeriodStart"::timestamp AS billing_period_start,  -- Convert to timestamp
    "BillingPeriodEnd"::timestamp AS billing_period_end,      -- Convert to timestamp
    "BillingAccountName" AS billing_account_name,
    "ChargeCategory" AS charge_category,
    "ChargeClass" AS charge_class,
    "ChargeDescription" AS charge_description,
    "ChargeFrequency" AS charge_frequency,
    "ServiceName" AS service_name,
    "ServiceCategory" AS service_category,
    "x_Operation" AS x_operation,
    "x_UsageType" AS x_usage_type,
    "SkuPriceId" AS sku_price_id,
    "SkuId" AS sku_id,
    "BillingAccountId" AS billing_account_id,
    __budget__::integer AS monthly_budget,
	"x_ServiceCode" AS x_service_code,
    md5("Tags"::text) as tags_key,
    "hash_key" as hash_key,
    "ResourceName" as resource_name
FROM __schema__.silver_focus_aws;

-- drop view __schema__.gold_aws_fact_focus;


CREATE OR REPLACE FUNCTION aws_tags_view_generation()
RETURNS text AS $$
DECLARE
    record_tagkey record;
    q_statement text = format(E'CREATE OR REPLACE VIEW __schema__.gold_aws_tags AS\nSELECT DISTINCT\n    md5(cast("Tags" AS text)) AS tags_key,');
BEGIN
    -- Loop through each distinct tag key from Tags
    FOR record_tagkey IN
        WITH tag_keys AS (
            SELECT DISTINCT jsonb_object_keys("Tags"::jsonb) AS tagkey
            FROM __schema__.silver_focus_aws
        )
        SELECT tagkey
        FROM tag_keys
        WHERE tagkey <> ''
    LOOP
        q_statement := q_statement || format(E'\n    ("Tags"::jsonb)->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
    END LOOP;

    -- Remove the trailing comma and complete the query
    q_statement := rtrim(q_statement, ',');
    q_statement := q_statement || E'\nFROM __schema__.silver_focus_aws\nWHERE\n    "Tags" IS NOT NULL';
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
