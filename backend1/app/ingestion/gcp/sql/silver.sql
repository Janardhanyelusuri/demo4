DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '__schema__' AND table_name = 'silver_focus_gcp_data') THEN
        -- Create the table if it does not exist
        CREATE TABLE __schema__.silver_focus_gcp_data (
            availability_zone VARCHAR(255),
            billed_cost FLOAT,
            billing_account_id VARCHAR(255),
            billing_currency VARCHAR(10),
            billing_period_start TIMESTAMP,
            billing_period_end TIMESTAMP,
            charge_category VARCHAR(255),
            charge_class VARCHAR(255),
            charge_description TEXT,
            charge_period_start TIMESTAMP,
            charge_period_end TIMESTAMP,
            commitment_discount_category VARCHAR(255),
            commitment_discount_id VARCHAR(255),
            commitment_discount_name VARCHAR(255),
            consumed_quantity FLOAT,
            consumed_unit VARCHAR(50),
            contracted_cost FLOAT,
            contracted_unit_price FLOAT,
            effective_cost FLOAT,
            list_cost FLOAT,
            list_unit_price FLOAT,
            pricing_category VARCHAR(255),
            pricing_quantity FLOAT,
            pricing_unit VARCHAR(50),
            provider_name VARCHAR(255),
            publisher_name VARCHAR(255),
            region_id VARCHAR(255),
            region_name VARCHAR(255),
            resource_id VARCHAR(255),
            resource_name VARCHAR(255),
            resource_type VARCHAR(255),
            service_category VARCHAR(255),
            service_name VARCHAR(255),
            sku_id VARCHAR(255),
            sku_price_id VARCHAR(255),
            sub_account_id VARCHAR(255),
            tags JSONB,
            x_cost_type VARCHAR(50),
            x_currency_conversion_rate FLOAT,
            x_export_time TIMESTAMP,
            x_location VARCHAR(255),
            x_project_id VARCHAR(255),
            x_project_number VARCHAR(255),
            x_project_name VARCHAR(255),
            x_project_ancestry_numbers TEXT,
            x_project_ancestors TEXT,
            x_project VARCHAR(255),
            x_service_id VARCHAR(255),
            "hash_key" TEXT PRIMARY KEY
        );

        -- Insert data from bronze_focus_gcp_data into silver_focus_gcp_data
        INSERT INTO __schema__.silver_focus_gcp_data
        SELECT
            "AvailabilityZone"::VARCHAR(255) AS availability_zone,
            NULLIF("BilledCost", 'None')::FLOAT AS billed_cost,
            "BillingAccountId"::VARCHAR(255) AS billing_account_id,
            "BillingCurrency"::VARCHAR(10) AS billing_currency,
            "BillingPeriodStart"::TIMESTAMP AS billing_period_start,
            "BillingPeriodEnd"::TIMESTAMP AS billing_period_end,
            "ChargeCategory"::VARCHAR(255) AS charge_category,
            "ChargeClass"::VARCHAR(255) AS charge_class,
            "ChargeDescription"::TEXT AS charge_description,
            "ChargePeriodStart"::TIMESTAMP AS charge_period_start,
            "ChargePeriodEnd"::TIMESTAMP AS charge_period_end,
            "CommitmentDiscountCategory"::VARCHAR(255) AS commitment_discount_category,
            "CommitmentDiscountId"::VARCHAR(255) AS commitment_discount_id,
            "CommitmentDiscountName"::VARCHAR(255) AS commitment_discount_name,
            NULLIF("ConsumedQuantity", 'None')::FLOAT AS consumed_quantity,
            "ConsumedUnit"::VARCHAR(50) AS consumed_unit,
            NULLIF("ContractedCost", 'None')::FLOAT AS contracted_cost,
            NULLIF("ContractedUnitPrice", 'None')::FLOAT AS contracted_unit_price,
            NULLIF("EffectiveCost", 'None')::FLOAT AS effective_cost,
            NULLIF("ListCost", 'None')::FLOAT AS list_cost,
            NULLIF("ListUnitPrice", 'None')::FLOAT AS list_unit_price,
            "PricingCategory"::VARCHAR(255) AS pricing_category,
            NULLIF("PricingQuantity", 'None')::FLOAT AS pricing_quantity,
            "PricingUnit"::VARCHAR(50) AS pricing_unit,
            "ProviderName"::VARCHAR(255) AS provider_name,
            "PublisherName"::VARCHAR(255) AS publisher_name,
            "RegionId"::VARCHAR(255) AS region_id,
            "RegionName"::VARCHAR(255) AS region_name,
            "ResourceId"::VARCHAR(255) AS resource_id,
            "ResourceName"::VARCHAR(255) AS resource_name,
            "ResourceType"::VARCHAR(255) AS resource_type,
            "ServiceCategory"::VARCHAR(255) AS service_category,
            "ServiceName"::VARCHAR(255) AS service_name,
            "SkuId"::VARCHAR(255) AS sku_id,
            "SkuPriceId"::VARCHAR(255) AS sku_price_id,
            "SubAccountId"::VARCHAR(255) AS sub_account_id,
            CASE
                WHEN "Tags" IS NULL OR "Tags" = 'None' THEN NULL::jsonb
                ELSE (
                    SELECT jsonb_object_agg(
                        trim(both '"' from key),
                        CASE
                            WHEN value ~ '^[-]?[0-9]+$' THEN value::jsonb
                            WHEN value ~ '^[-]?[0-9]+[.][0-9]+$' THEN value::jsonb
                            ELSE to_jsonb(trim(both '"' from value))
                        END
                    )
                    FROM (
                        SELECT
                            trim(both '{' from trim(both '}' from trim(split_part(kv, ':', 1)))) AS key,
                            trim(both '{' from trim(both '}' from trim(split_part(kv, ':', 2)))) AS value
                        FROM regexp_split_to_table(
                            regexp_replace("Tags", '^\[{|}\]$', '', 'g'),
                            ',(?=(?:[^'']*''[^'']*'')*[^'']*$)'
                        ) AS kv
                    ) AS kvs
                )
            END AS tags,
            "x_CostType"::VARCHAR(50) AS x_cost_type,
            NULLIF("x_CurrencyConversionRate", 'None')::FLOAT AS x_currency_conversion_rate,
            "x_ExportTime"::TIMESTAMP AS x_export_time,
            "x_Location"::VARCHAR(255) AS x_location,
            "x_ProjectId"::VARCHAR(255) AS x_project_id,
            "x_ProjectNumber"::VARCHAR(255) AS x_project_number,
            "x_ProjectName"::VARCHAR(255) AS x_project_name,
            "x_ProjectAncestryNumbers"::TEXT AS x_project_ancestry_numbers,
            "x_ProjectAncestors"::TEXT AS x_project_ancestors,
            "x_Project"::VARCHAR(255) AS x_project,
            "x_ServiceId"::VARCHAR(255) AS x_service_id,
            "hash_key" as hash_key
        FROM __schema__.bronze_focus_gcp_data;
    ELSE
        -- Truncate the table if it already exists
        TRUNCATE TABLE __schema__.silver_focus_gcp_data;

        -- Insert data from bronze_focus_gcp_data into silver_focus_gcp_data
        INSERT INTO __schema__.silver_focus_gcp_data
        SELECT
            "AvailabilityZone"::VARCHAR(255) AS availability_zone,
            NULLIF("BilledCost", 'None')::FLOAT AS billed_cost,
            "BillingAccountId"::VARCHAR(255) AS billing_account_id,
            "BillingCurrency"::VARCHAR(10) AS billing_currency,
            "BillingPeriodStart"::TIMESTAMP AS billing_period_start,
            "BillingPeriodEnd"::TIMESTAMP AS billing_period_end,
            "ChargeCategory"::VARCHAR(255) AS charge_category,
            "ChargeClass"::VARCHAR(255) AS charge_class,
            "ChargeDescription"::TEXT AS charge_description,
            "ChargePeriodStart"::TIMESTAMP AS charge_period_start,
            "ChargePeriodEnd"::TIMESTAMP AS charge_period_end,
            "CommitmentDiscountCategory"::VARCHAR(255) AS commitment_discount_category,
            "CommitmentDiscountId"::VARCHAR(255) AS commitment_discount_id,
            "CommitmentDiscountName"::VARCHAR(255) AS commitment_discount_name,
            NULLIF("ConsumedQuantity", 'None')::FLOAT AS consumed_quantity,
            "ConsumedUnit"::VARCHAR(50) AS consumed_unit,
            NULLIF("ContractedCost", 'None')::FLOAT AS contracted_cost,
            NULLIF("ContractedUnitPrice", 'None')::FLOAT AS contracted_unit_price,
            NULLIF("EffectiveCost", 'None')::FLOAT AS effective_cost,
            NULLIF("ListCost", 'None')::FLOAT AS list_cost,
            NULLIF("ListUnitPrice", 'None')::FLOAT AS list_unit_price,
            "PricingCategory"::VARCHAR(255) AS pricing_category,
            NULLIF("PricingQuantity", 'None')::FLOAT AS pricing_quantity,
            "PricingUnit"::VARCHAR(50) AS pricing_unit,
            "ProviderName"::VARCHAR(255) AS provider_name,
            "PublisherName"::VARCHAR(255) AS publisher_name,
            "RegionId"::VARCHAR(255) AS region_id,
            "RegionName"::VARCHAR(255) AS region_name,
            "ResourceId"::VARCHAR(255) AS resource_id,
            "ResourceName"::VARCHAR(255) AS resource_name,
            "ResourceType"::VARCHAR(255) AS resource_type,
            "ServiceCategory"::VARCHAR(255) AS service_category,
            "ServiceName"::VARCHAR(255) AS service_name,
            "SkuId"::VARCHAR(255) AS sku_id,
            "SkuPriceId"::VARCHAR(255) AS sku_price_id,
            "SubAccountId"::VARCHAR(255) AS sub_account_id,
            CASE
                WHEN "Tags" IS NULL OR "Tags" = 'None' THEN NULL::jsonb
                ELSE (
                    SELECT jsonb_object_agg(
                        trim(both '"' from key),
                        CASE
                            WHEN value ~ '^[-]?[0-9]+$' THEN value::jsonb
                            WHEN value ~ '^[-]?[0-9]+[.][0-9]+$' THEN value::jsonb
                            ELSE to_jsonb(trim(both '"' from value))
                        END
                    )
                    FROM (
                        SELECT
                            trim(both '{' from trim(both '}' from trim(split_part(kv, ':', 1)))) AS key,
                            trim(both '{' from trim(both '}' from trim(split_part(kv, ':', 2)))) AS value
                        FROM regexp_split_to_table(
                            regexp_replace("Tags", '^\[{|}\]$', '', 'g'),
                            ',(?=(?:[^'']*''[^'']*'')*[^'']*$)'
                        ) AS kv
                    ) AS kvs
                )
            END AS tags,
            "x_CostType"::VARCHAR(50) AS x_cost_type,
            NULLIF("x_CurrencyConversionRate", 'None')::FLOAT AS x_currency_conversion_rate,
            "x_ExportTime"::TIMESTAMP AS x_export_time,
            "x_Location"::VARCHAR(255) AS x_location,
            "x_ProjectId"::VARCHAR(255) AS x_project_id,
            "x_ProjectNumber"::VARCHAR(255) AS x_project_number,
            "x_ProjectName"::VARCHAR(255) AS x_project_name,
            "x_ProjectAncestryNumbers"::TEXT AS x_project_ancestry_numbers,
            "x_ProjectAncestors"::TEXT AS x_project_ancestors,
            "x_Project"::VARCHAR(255) AS x_project,
            "x_ServiceId"::VARCHAR(255) AS x_service_id,
            "hash_key" as hash_key
        FROM __schema__.bronze_focus_gcp_data;
    END IF;
END $$;
