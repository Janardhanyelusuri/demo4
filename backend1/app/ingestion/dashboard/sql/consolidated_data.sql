DO $$
DECLARE
    schemas_and_tables JSON := '__schema__'; 

    target_schema TEXT := '__dashboardname__';
    target_table TEXT := 'target_table';

    insert_query TEXT;
    schema_rec RECORD;

BEGIN
    -- Create the target schema if it doesn't exist
    EXECUTE 'CREATE SCHEMA IF NOT EXISTS ' || quote_ident(target_schema);

    -- Drop and recreate the consolidated table
    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(target_schema) || '.' || quote_ident(target_table) || ' CASCADE';
    -- Create the consolidated table if it doesn't exist
    EXECUTE '
    CREATE TABLE IF NOT EXISTS ' || quote_ident(target_schema) || '.' || quote_ident(target_table) || ' (
        cloud_source TEXT,
        hash_key TEXT,
        BilledCost NUMERIC,
        BillingAccountId TEXT,
        BillingCurrency TEXT,
        BillingPeriodStart DATE,
        BillingPeriodEnd DATE,
        ChargeCategory TEXT,
        ChargeClass TEXT,
        ChargeDescription TEXT,
        ChargePeriodStart DATE,
        ChargePeriodEnd DATE,
        CommitmentDiscountCategory TEXT,
        CommitmentDiscountId TEXT,
        CommitmentDiscountName TEXT,
        ConsumedQuantity NUMERIC,
        ConsumedUnit TEXT,
        ContractedCost NUMERIC,
        ContractedUnitPrice NUMERIC,
        EffectiveCost NUMERIC,
        ListCost NUMERIC,
        ListUnitPrice NUMERIC,
        PricingCategory TEXT,
        PricingQuantity NUMERIC,
        PricingUnit TEXT,
        ProviderName TEXT,
        PublisherName TEXT,
        RegionId TEXT,
        RegionName TEXT,
        ResourceId TEXT,
        ResourceName TEXT,
        ResourceType TEXT,
        ServiceCategory TEXT,
        ServiceName TEXT,
        SkuId TEXT,
        SkuPriceId TEXT,
        SubAccountId TEXT,
        Tags TEXT
    )';

    -- Loop through each schema and table
    FOR schema_rec IN 
        SELECT 
            (json_array_elements(schemas_and_tables)->>'schema') AS source_schema,
            (json_array_elements(schemas_and_tables)->>'table') AS source_table,
            (json_array_elements(schemas_and_tables)->>'cloud') AS cloud_source
    LOOP
        -- Dynamically build the INSERT query
        insert_query := '
            INSERT INTO ' || quote_ident(target_schema) || '.' || quote_ident(target_table) || ' (
                cloud_source,
                hash_key,
                BilledCost,
                BillingAccountId,
                BillingCurrency,
                BillingPeriodStart,
                BillingPeriodEnd,
                ChargeCategory,
                ChargeClass,
                ChargeDescription,
                ChargePeriodStart,
                ChargePeriodEnd,
                CommitmentDiscountCategory,
                CommitmentDiscountId,
                CommitmentDiscountName,
                ConsumedQuantity,
                ConsumedUnit,
                ContractedCost,
                ContractedUnitPrice,
                EffectiveCost,
                ListCost,
                ListUnitPrice,
                PricingCategory,
                PricingQuantity,
                PricingUnit,
                ProviderName,
                PublisherName,
                RegionId,
                RegionName,
                ResourceId,
                ResourceName,
                ResourceType,
                ServiceCategory,
                ServiceName,
                SkuId,
                SkuPriceId,
                SubAccountId,
                Tags
            )
            SELECT 
                ''' || schema_rec.cloud_source || ''' AS cloud_source,
                hash_key,
                CAST("BilledCost" AS NUMERIC),
                "BillingAccountId",
                "BillingCurrency",
                "BillingPeriodStart"::DATE AS "BillingPeriodStart",  -- Convert TIMESTAMP to DATE
                "BillingPeriodEnd"::DATE AS "BillingPeriodEnd",      -- Convert TIMESTAMP to DATE
                "ChargeCategory",
                "ChargeClass",
                "ChargeDescription",
                "ChargePeriodStart"::DATE AS "ChargePeriodStart",    -- Convert TIMESTAMP to DATE
                "ChargePeriodEnd"::DATE AS "ChargePeriodEnd",        -- Convert TIMESTAMP to DATE
                "CommitmentDiscountCategory",
                "CommitmentDiscountId",
                "CommitmentDiscountName",
                CAST("ConsumedQuantity" AS NUMERIC) AS "ConsumedQuantity",  -- Cast TEXT to NUMERIC
                "ConsumedUnit",
                CAST("ContractedCost" AS NUMERIC) AS "ContractedCost",
                CAST("ContractedUnitPrice" AS NUMERIC) AS "ContractedUnitPrice",
                CAST("EffectiveCost" AS NUMERIC) AS "EffectiveCost",
                CAST("ListCost" AS NUMERIC) AS "ListCost",
                CAST("ListUnitPrice" AS NUMERIC) AS "ListUnitPrice",
                "PricingCategory",
                CAST("PricingQuantity" AS NUMERIC) AS "PricingQuantity",
                "PricingUnit",
                "ProviderName",
                "PublisherName",
                "RegionId",
                "RegionName",
                "ResourceId",
                "ResourceName",
                "ResourceType",
                "ServiceCategory",
                "ServiceName",
                "SkuId",
                "SkuPriceId",
                "SubAccountId",
                "Tags"
             FROM ' || quote_ident(schema_rec.source_schema) || '.' || quote_ident(schema_rec.source_table) || '
             ';
        EXECUTE insert_query;
    END LOOP;
END $$;