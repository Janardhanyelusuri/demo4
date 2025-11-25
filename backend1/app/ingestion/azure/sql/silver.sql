-- DO $$
-- BEGIN
--     -- Check if the silver table exists
--     IF NOT EXISTS (
--         SELECT FROM information_schema.tables 
--         WHERE table_schema = '__schema__' 
--         AND table_name = 'silver_azure_focus'
--     ) THEN
--         -- Create the silver table if it does not exist
--         CREATE TABLE __schema__.silver_azure_focus (
--             "BilledCost" DOUBLE PRECISION,
--             "BillingAccountId" TEXT,
--             "BillingAccountName" TEXT,
--             "BillingAccountType" TEXT,
--             "ChargePeriodStart" DATE,
--             "ChargeCategory" TEXT,
--             "ChargeClass" TEXT,
--             "ChargeDescription" TEXT,
--             "ChargeFrequency" TEXT,
--             "ConsumedQuantity" DOUBLE PRECISION,
--             "ConsumedUnit" TEXT,
--             "ContractedCost" DOUBLE PRECISION,
--             "ContractedUnitPrice" DOUBLE PRECISION,
--             "EffectiveCost" DOUBLE PRECISION,
--             "ListCost" DOUBLE PRECISION,
--             "ListUnitPrice" DOUBLE PRECISION,
--             "PricingCategory" TEXT,
--             "PricingQuantity" DOUBLE PRECISION,
--             "PricingUnit" TEXT,
--             "RegionId" TEXT,
--             "RegionName" TEXT,
--             "ResourceId" TEXT,
--             "ResourceName" TEXT,
--             "ServiceCategory" TEXT,
--             "ServiceName" TEXT,
--             "SkuId" TEXT,
--             "SkuPriceId" TEXT,
--             "SubAccountId" TEXT,
--             "SubAccountName" TEXT,
--             "SubAccountType" TEXT,
--             "Tags" JSONB,
--             "x_AccountId" TEXT,
--             "x_AccountName" TEXT,
--             "x_AccountOwnerId" TEXT,
--             "x_BilledCostInUsd" DOUBLE PRECISION,
--             "x_BillingProfileId" TEXT,
--             "x_BillingProfileName" TEXT,
--             "x_EffectiveCostInUsd" DOUBLE PRECISION,
--             "x_EffectiveUnitPrice" DOUBLE PRECISION,
--             "x_ListCostInUsd" DOUBLE PRECISION,
--             "x_ResourceGroupName" TEXT,
--             "x_SkuDescription" TEXT,
--             "x_SkuMeterName" TEXT,
--             "x_SkuMeterSubcategory" TEXT,
--             "x_SkuServiceFamily" TEXT,
--             "hash_key" TEXT PRIMARY KEY 
--         );
--     ELSE
--         -- Truncate the silver table if it already exists
--         TRUNCATE TABLE __schema__.silver_azure_focus;
--     END IF;

--     -- Insert data from the bronze table into the silver table
--     INSERT INTO __schema__.silver_azure_focus (
--         "BilledCost", "BillingAccountId", "BillingAccountName", "BillingAccountType", "ChargePeriodStart", 
--         "ChargeCategory", "ChargeClass", "ChargeDescription", "ChargeFrequency", "ConsumedQuantity", "ConsumedUnit", 
--         "ContractedCost", "ContractedUnitPrice", "EffectiveCost", "ListCost", "ListUnitPrice", 
--         "PricingCategory", "PricingQuantity", "PricingUnit", "RegionId", "RegionName", "ResourceId", 
--         "ResourceName", "ServiceCategory", "ServiceName", "SkuId", "SkuPriceId", "SubAccountId", 
--         "SubAccountName", "SubAccountType", "Tags", "x_AccountId", "x_AccountName", "x_AccountOwnerId", 
--         "x_BilledCostInUsd", "x_BillingProfileId", "x_BillingProfileName", "x_EffectiveCostInUsd", 
--         "x_EffectiveUnitPrice", "x_ListCostInUsd", "x_ResourceGroupName", "x_SkuDescription", 
--         "x_SkuMeterName", "x_SkuMeterSubcategory", "x_SkuServiceFamily",  "hash_key" 
--     )
--     SELECT 
--         "BilledCost", "BillingAccountId", "BillingAccountName", "BillingAccountType",
--         -- Extract the date part from the timestamp string
--         SUBSTRING("ChargePeriodStart", 1, 10)::DATE AS ChargePeriodStart,
--         "ChargeCategory", "ChargeClass", "ChargeDescription", "ChargeFrequency", "ConsumedQuantity", "ConsumedUnit", 
--         "ContractedCost", "ContractedUnitPrice", "EffectiveCost", "ListCost", "ListUnitPrice", 
--         "PricingCategory", "PricingQuantity", "PricingUnit", "RegionId", "RegionName", "ResourceId", 
--         "ResourceName", "ServiceCategory", "ServiceName", "SkuId", "SkuPriceId", "SubAccountId", 
--         "SubAccountName", "SubAccountType", "Tags"::jsonb, "x_AccountId", "x_AccountName", "x_AccountOwnerId", 
--         "x_BilledCostInUsd", "x_BillingProfileId", "x_BillingProfileName", "x_EffectiveCostInUsd", 
--         "x_EffectiveUnitPrice", "x_ListCostInUsd", "x_ResourceGroupName", "x_SkuDescription", 
--         "x_SkuMeterName", "x_SkuMeterSubcategory", "x_SkuServiceFamily",  "hash_key" 
--     FROM __schema__.bronze_azure_focus;

-- END $$;



DO $$
BEGIN
    -- Check if the silver table exists
    IF NOT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = '__schema__' 
        AND table_name = 'silver_azure_focus'
    ) THEN
        -- Create the silver table if it does not exist
        CREATE TABLE __schema__.silver_azure_focus (
            "BilledCost" DOUBLE PRECISION,
            "BillingAccountId" TEXT,
            "BillingAccountName" TEXT,
            "BillingAccountType" TEXT,
            "ChargePeriodStart" DATE,
            "ChargeCategory" TEXT,
            "ChargeClass" TEXT,
            "ChargeDescription" TEXT,
            "ChargeFrequency" TEXT,
            "ConsumedQuantity" DOUBLE PRECISION,
            "ConsumedUnit" TEXT,
            "ContractedCost" DOUBLE PRECISION,
            "ContractedUnitPrice" DOUBLE PRECISION,
            "EffectiveCost" DOUBLE PRECISION,
            "ListCost" DOUBLE PRECISION,
            "ListUnitPrice" DOUBLE PRECISION,
            "PricingCategory" TEXT,
            "PricingQuantity" DOUBLE PRECISION,
            "PricingUnit" TEXT,
            "RegionId" TEXT,
            "RegionName" TEXT,
            "ResourceId" TEXT,
            "ResourceName" TEXT,
            "ServiceCategory" TEXT,
            "ServiceName" TEXT,
            "SkuId" TEXT,
            "SkuPriceId" TEXT,
            "SubAccountId" TEXT,
            "SubAccountName" TEXT,
            "SubAccountType" TEXT,
            "Tags" JSONB,
            "x_AccountId" TEXT,
            "x_AccountName" TEXT,
            "x_AccountOwnerId" TEXT,
            "x_BilledCostInUsd" DOUBLE PRECISION,
            "x_BillingProfileId" TEXT,
            "x_BillingProfileName" TEXT,
            "x_EffectiveCostInUsd" DOUBLE PRECISION,
            "x_EffectiveUnitPrice" DOUBLE PRECISION,
            "x_ListCostInUsd" DOUBLE PRECISION,
            "x_ResourceGroupName" TEXT,
            "x_SkuDescription" TEXT,
            "x_SkuMeterName" TEXT,
            "x_SkuMeterSubcategory" TEXT,
            "x_SkuServiceFamily" TEXT,
            "hash_key" TEXT PRIMARY KEY 
        );
    ELSE
        -- Truncate the silver table if it already exists
        TRUNCATE TABLE __schema__.silver_azure_focus;
    END IF;

    -- Insert data with robust JSON sanitization and validation
    INSERT INTO __schema__.silver_azure_focus (
        "BilledCost", "BillingAccountId", "BillingAccountName", "BillingAccountType", 
        "ChargePeriodStart", "ChargeCategory", "ChargeClass", "ChargeDescription", 
        "ChargeFrequency", "ConsumedQuantity", "ConsumedUnit", "ContractedCost", 
        "ContractedUnitPrice", "EffectiveCost", "ListCost", "ListUnitPrice", 
        "PricingCategory", "PricingQuantity", "PricingUnit", "RegionId", 
        "RegionName", "ResourceId", "ResourceName", "ServiceCategory", 
        "ServiceName", "SkuId", "SkuPriceId", "SubAccountId", "SubAccountName", 
        "SubAccountType", "Tags", "x_AccountId", "x_AccountName", "x_AccountOwnerId", 
        "x_BilledCostInUsd", "x_BillingProfileId", "x_BillingProfileName", 
        "x_EffectiveCostInUsd", "x_EffectiveUnitPrice", "x_ListCostInUsd", 
        "x_ResourceGroupName", "x_SkuDescription", "x_SkuMeterName", 
        "x_SkuMeterSubcategory", "x_SkuServiceFamily", "hash_key"
    )
    SELECT 
        "BilledCost", "BillingAccountId", "BillingAccountName", "BillingAccountType",
        SUBSTRING("ChargePeriodStart", 1, 10)::DATE AS ChargePeriodStart,
        "ChargeCategory", "ChargeClass", "ChargeDescription", "ChargeFrequency", 
        "ConsumedQuantity", "ConsumedUnit", "ContractedCost", "ContractedUnitPrice", 
        "EffectiveCost", "ListCost", "ListUnitPrice", "PricingCategory", 
        "PricingQuantity", "PricingUnit", "RegionId", "RegionName", "ResourceId", 
        "ResourceName", "ServiceCategory", "ServiceName", "SkuId", "SkuPriceId", 
        "SubAccountId", "SubAccountName", "SubAccountType", 
        CASE 
            WHEN "Tags" IS NULL THEN '{}'::JSONB -- Default to empty JSON for NULL
            WHEN jsonb_typeof(to_jsonb("Tags")) IS NOT NULL THEN to_jsonb("Tags") -- If valid, cast to JSONB
            ELSE '{}'::JSONB -- Replace invalid JSON with empty JSON
        END AS "Tags",
        "x_AccountId", "x_AccountName", "x_AccountOwnerId", "x_BilledCostInUsd", 
        "x_BillingProfileId", "x_BillingProfileName", "x_EffectiveCostInUsd", 
        "x_EffectiveUnitPrice", "x_ListCostInUsd", "x_ResourceGroupName", 
        "x_SkuDescription", "x_SkuMeterName", "x_SkuMeterSubcategory", 
        "x_SkuServiceFamily", "hash_key"
    FROM __schema__.bronze_azure_focus;

END $$;
