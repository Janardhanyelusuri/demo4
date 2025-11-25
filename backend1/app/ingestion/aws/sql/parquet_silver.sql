-- -- select "Tags" from __schema__.silver_focus_aws limit 100;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '__schema__' AND table_name = 'silver_focus_aws_2') THEN
        -- Create the table if it does not exist
        CREATE TABLE __schema__.silver_focus_aws_2 AS
        SELECT
            "AvailabilityZone",
            "BilledCost",
            "BillingAccountId",
            "BillingAccountName",
            "BillingCurrency",
            "BillingPeriodEnd",
            "BillingPeriodStart",
            "ChargeCategory",
            "ChargeClass",
            "ChargeDescription",
            "ChargeFrequency",
            "ChargePeriodEnd",
            "ChargePeriodStart",
            "CommitmentDiscountCategory",
            "CommitmentDiscountId",
            "CommitmentDiscountName",
            "CommitmentDiscountStatus",
            "CommitmentDiscountType",
            "ConsumedQuantity",
            "ConsumedUnit",
            "ContractedCost",
            "ContractedUnitPrice",
            "EffectiveCost",
            "InvoiceIssuerName",
            "ListCost",
            "ListUnitPrice",
            "PricingCategory",
            "PricingQuantity",
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
            "SubAccountName",
            -- Transform Tags column into JSON and store it in the new table
            (
                SELECT
                    json_object_agg(
                        -- Extract key and remove unwanted characters
                        TRIM(BOTH '()' FROM REGEXP_REPLACE(split_part(tag, ',', 1), '^"|"$', '')),
                        -- Extract value and remove unwanted characters
                        TRIM(BOTH '()' FROM REGEXP_REPLACE(split_part(tag, ',', 2), '^"|"$', ''))
                    )
                FROM
                    unnest(
                        string_to_array(
                            TRIM(BOTH '{}' FROM "Tags"),
                            '","'
                        )
                    ) AS tag
            ) AS "Tags",
            "x_CostCategories",
            "x_Discounts",
            "x_Operation",
            "x_ServiceCode",
            "x_UsageType",
            "hash_key"
        FROM
            __schema__.silver_focus_aws;
    ELSE
        -- If the table exists, you can truncate or perform other operations as needed
        -- TRUNCATE TABLE __schema__.silver_focus_aws_2;
        -- Optionally, you could re-insert data from the original table
        INSERT INTO __schema__.silver_focus_aws_2
        SELECT
            "AvailabilityZone",
            "BilledCost",
            "BillingAccountId"::bigint AS "BillingAccountId",
            "BillingAccountName",
            "BillingCurrency",
            "BillingPeriodEnd",
            "BillingPeriodStart",
            "ChargeCategory",
            "ChargeClass"::double precision AS "ChargeClass",
            "ChargeDescription",
            "ChargeFrequency",
            "ChargePeriodEnd",
            "ChargePeriodStart",
            -- "CommitmentDiscountCategory"::double precision AS "CommitmentDiscountCategory",
            -- "CommitmentDiscountId"::double precision AS "CommitmentDiscountId",
            -- "CommitmentDiscountName"::double precision AS "CommitmentDiscountName",
            -- "CommitmentDiscountStatus"::double precision AS "CommitmentDiscountStatus",
            -- "CommitmentDiscountType"::double precision AS "CommitmentDiscountType",
            "CommitmentDiscountCategory" AS "CommitmentDiscountCategory",
            "CommitmentDiscountId" AS "CommitmentDiscountId",
            "CommitmentDiscountName" AS "CommitmentDiscountName",
            "CommitmentDiscountStatus" AS "CommitmentDiscountStatus",
            "CommitmentDiscountType" AS "CommitmentDiscountType",
            "ConsumedQuantity",
            "ConsumedUnit",
            "ContractedCost",
            "ContractedUnitPrice",
            "EffectiveCost",
            "InvoiceIssuerName",
            "ListCost",
            "ListUnitPrice",
            "PricingCategory",
            "PricingQuantity",
            "PricingUnit",
            "ProviderName",
            "PublisherName",
            "RegionId",
            "RegionName",
            "ResourceId",
            "ResourceName"::double precision AS "ResourceName",
            "ResourceType",
            "ServiceCategory",
            "ServiceName",
            "SkuId",
            "SkuPriceId",
            "SubAccountId"::bigint AS "SubAccountId",
            "SubAccountName",
            (
                SELECT
                    json_object_agg(
                        TRIM(BOTH '()' FROM REGEXP_REPLACE(split_part(tag, ',', 1), '^"|"$', '')),
                        TRIM(BOTH '()' FROM REGEXP_REPLACE(split_part(tag, ',', 2), '^"|"$', ''))
                    )
                FROM
                    unnest(
                        string_to_array(
                            TRIM(BOTH '{}' FROM "Tags"),
                            '","'
                        )
                    ) AS tag
            ) AS "Tags",
            "x_CostCategories",
            "x_Discounts",
            "x_Operation",
            "x_ServiceCode",
            "x_UsageType",
            "hash_key"
        FROM
            __schema__.silver_focus_aws;
    END IF;
END $$;
