
--resource dim
CREATE OR REPLACE VIEW __schema__.gold_azure_resource_dim AS
SELECT DISTINCT 
    "ResourceId" AS resource_id, 
    "ResourceName" AS resource_name,  
    "RegionId" AS region_id, 
    "RegionName" AS region_name, 
    "ServiceCategory" AS service_category, 
    "ServiceName" AS service_name
FROM __schema__.silver_azure_focus;


--charge_summary_dim
CREATE OR REPLACE VIEW __schema__.gold_azure_charge_summary_dim AS
SELECT DISTINCT 
    "SkuId" AS sku_id,
    "ChargeCategory" AS charge_category,
    "ChargeClass" AS charge_class,
    "ChargeDescription" AS charge_description,
    "ChargeFrequency" AS charge_frequency,
    "x_SkuDescription" AS x_sku_description
FROM __schema__.silver_azure_focus;

--azure_account_dim
CREATE OR REPLACE VIEW __schema__.gold_azure_account_dim AS
SELECT DISTINCT 
    "SubAccountId" AS sub_account_id,
    "SubAccountName" AS sub_account_name,
    "SubAccountType" AS sub_account_type,
    "x_AccountId" AS x_account_id,
    "x_AccountName" AS x_account_name,
    "x_AccountOwnerId" AS x_account_owner_id,
    "x_BillingProfileId" AS x_billing_profile_id,
    "x_BillingProfileName" AS x_billing_profile_name,
    "BillingAccountId" AS billing_account_id,
    "BillingAccountName" AS billing_account_name,
    "BillingAccountType" AS billing_account_type
FROM __schema__.silver_azure_focus;


-- METRICS DIM TABLE (NEW)
CREATE OR REPLACE VIEW __schema__.gold_azure_metric_dim AS
SELECT DISTINCT
    metric_name,
    unit,
    displaydescription,
    namespace
FROM __schema__.silver_azure_vm_metrics
WHERE metric_name IS NOT NULL;

-- METRICS FACT TABLE (NEW)
CREATE OR REPLACE VIEW __schema__.gold_azure_fact_vm_metrics AS
SELECT
    resource_id,
    resource_group,
    subscription_id,
    timestamp,
    metric_name,
    value,
    unit,
    vm_name,
    instance_type,
    namespace,
    resourceregion
FROM __schema__.silver_azure_vm_metrics
WHERE resource_id IS NOT NULL;
-- fact
CREATE OR REPLACE VIEW __schema__.gold_azure_fact_cost AS
SELECT 
    md5(cast("Tags" AS text)) AS tags_key,
    "SubAccountId" AS sub_account_id,
    "ResourceId" AS resource_id,
    "SkuId" AS sku_id,
    "x_ResourceGroupName" AS resource_group_name,
    "ChargePeriodStart" AS charge_period_start,
    "PricingCategory" AS pricing_category,
    "PricingUnit" AS pricing_unit,
    "ListUnitPrice" AS list_unit_price,
    "ContractedUnitPrice" AS contracted_unit_price,
    "PricingQuantity" AS pricing_quantity,
    "BilledCost" AS billed_cost,
    "ConsumedQuantity" AS consumed_quantity,
    "ConsumedUnit" AS consumed_unit,
    "EffectiveCost" AS effective_cost,
    "ContractedCost" AS contracted_cost,
    "ListCost" AS list_cost,
    "x_EffectiveUnitPrice" AS effective_unit_price,
    "x_BilledCostInUsd" AS billed_cost_in_usd,
    "x_EffectiveCostInUsd" AS effective_cost_in_usd,
    "x_ListCostInUsd" AS list_cost_in_usd,
    "SkuPriceId" AS sku_price_id,
    "x_SkuMeterName" AS sku_meter_name,
    "x_SkuMeterSubcategory" AS sku_meter_subcategory,
    "x_SkuServiceFamily" AS sku_service_family,
    __budget__::integer as monthly_budget,
    "hash_key" as hash_key
FROM __schema__.silver_azure_focus;  

CREATE OR REPLACE FUNCTION azure_tags_view_generation()
RETURNS text AS $$
DECLARE
    record_tagkey record;
    q_statement text = format(E'CREATE OR REPLACE VIEW __schema__.gold_azure_tags_dim AS\nSELECT DISTINCT\n    md5(cast("Tags" AS text)) AS tags_key,');
BEGIN
    -- Loop through each distinct tag key, filtering for JSON objects
    FOR record_tagkey IN
        SELECT DISTINCT jsonb_object_keys("Tags") AS tagkey 
        FROM __schema__.silver_azure_focus
        WHERE jsonb_typeof("Tags") = 'object'
    LOOP
        q_statement := q_statement || format(E'\n    "Tags"->>%L AS %I,', record_tagkey.tagkey, record_tagkey.tagkey);
    END LOOP;

    -- Remove the trailing comma and complete the query
    q_statement := rtrim(q_statement, ',');
    q_statement := q_statement || E'\nFROM __schema__.silver_azure_focus\nWHERE\n    jsonb_typeof("Tags") = ''object''';
    RAISE NOTICE E'\n%', q_statement;
    RETURN q_statement;
END;
$$ LANGUAGE plpgsql;
