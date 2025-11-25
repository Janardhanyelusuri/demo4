-- DO
-- $$
-- BEGIN
--    -- Check if the table exists
--    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '__schema__' AND table_name = 'silver_aws_ce') THEN
--       -- If the table exists, truncate it
--       EXECUTE 'TRUNCATE TABLE __schema__.silver_aws_ce';
      
--       EXECUTE '
--          INSERT INTO __schema__.silver_aws_ce ( 
--             usage_date,
--             operation,
--             service,
--             amortized_cost,
--             blended_cost,
--             net_amortized_cost,
--             net_unblended_cost,
--             normalized_usage_amount,
--             unblended_cost,
--             usage_quantity
--          )
--          SELECT 
--             LEFT("TimePeriod", 10)::DATE AS usage_date,
--             "OPERATION" AS operation,
--             "SERVICE" AS service,
--             "AmortizedCost"::DOUBLE PRECISION AS amortized_cost,
--             "BlendedCost"::DOUBLE PRECISION AS blended_cost,
--             "NetAmortizedCost"::DOUBLE PRECISION AS net_amortized_cost,
--             "NetUnblendedCost"::DOUBLE PRECISION AS net_unblended_cost,
--             "NormalizedUsageAmount"::DOUBLE PRECISION AS normalized_usage_amount,
--             "UnblendedCost"::DOUBLE PRECISION AS unblended_cost,
--             "UsageQuantity"::DOUBLE PRECISION AS usage_quantity
--          FROM __schema__.bronze_aws_ce';
--    ELSE
--       -- If the table does not exist, create it and insert data
--       EXECUTE '
--          CREATE TABLE __schema__.silver_aws_ce (
--             usage_date DATE,
--             operation TEXT,
--             service TEXT,
--             amortized_cost DOUBLE PRECISION,
--             blended_cost DOUBLE PRECISION,
--             net_amortized_cost DOUBLE PRECISION,
--             net_unblended_cost DOUBLE PRECISION,
--             normalized_usage_amount DOUBLE PRECISION,
--             unblended_cost DOUBLE PRECISION,
--             usage_quantity DOUBLE PRECISION
--          );

--          INSERT INTO __schema__.silver_aws_ce (
--             usage_date::DATE,
--             operation,
--             service,
--             amortized_cost,
--             blended_cost,
--             net_amortized_cost,
--             net_unblended_cost,
--             normalized_usage_amount,
--             unblended_cost,
--             usage_quantity
--          )
--          SELECT 
--             LEFT("TimePeriod", 10)::DATE,
--             "OPERATION",
--             "SERVICE",
--             "AmortizedCost"::DOUBLE PRECISION,
--             "BlendedCost"::DOUBLE PRECISION,
--             "NetAmortizedCost"::DOUBLE PRECISION,
--             "NetUnblendedCost"::DOUBLE PRECISION,
--             "NormalizedUsageAmount"::DOUBLE PRECISION,
--             "UnblendedCost"::DOUBLE PRECISION,
--             "UsageQuantity"::DOUBLE PRECISION
--          FROM __schema__.bronze_aws_ce';
--    END IF;
-- END
-- $$;

DO
$$
BEGIN
   -- Check if the table exists
   IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '__schema__' AND table_name = 'silver_aws_ce') THEN
      -- If the table exists, truncate it
      EXECUTE 'TRUNCATE TABLE __schema__.silver_aws_ce';
      
      EXECUTE '
         INSERT INTO __schema__.silver_aws_ce ( 
            usage_date,
            operation,
            service,
            amortized_cost,
            blended_cost,
            net_amortized_cost,
            net_unblended_cost,
            normalized_usage_amount,
            unblended_cost,
            usage_quantity
         )
         SELECT 
            LEFT("TimePeriod", 10)::DATE AS usage_date,
            "OPERATION" AS operation,
            "SERVICE" AS service,
            "AmortizedCost"::DOUBLE PRECISION AS amortized_cost,
            "BlendedCost"::DOUBLE PRECISION AS blended_cost,
            "NetAmortizedCost"::DOUBLE PRECISION AS net_amortized_cost,
            "NetUnblendedCost"::DOUBLE PRECISION AS net_unblended_cost,
            "NormalizedUsageAmount"::DOUBLE PRECISION AS normalized_usage_amount,
            "UnblendedCost"::DOUBLE PRECISION AS unblended_cost,
            "UsageQuantity"::DOUBLE PRECISION AS usage_quantity
         FROM __schema__.bronze_aws_ce';
   ELSE
      -- If the table does not exist, create it and insert data
      EXECUTE '
         CREATE TABLE __schema__.silver_aws_ce (
            usage_date DATE,
            operation TEXT,
            service TEXT,
            amortized_cost DOUBLE PRECISION,
            blended_cost DOUBLE PRECISION,
            net_amortized_cost DOUBLE PRECISION,
            net_unblended_cost DOUBLE PRECISION,
            normalized_usage_amount DOUBLE PRECISION,
            unblended_cost DOUBLE PRECISION,
            usage_quantity DOUBLE PRECISION
         );

         INSERT INTO __schema__.silver_aws_ce (
            usage_date,
            operation,
            service,
            amortized_cost,
            blended_cost,
            net_amortized_cost,
            net_unblended_cost,
            normalized_usage_amount,
            unblended_cost,
            usage_quantity
         )
         SELECT 
            LEFT("TimePeriod", 10)::DATE AS usage_date,
            "OPERATION" AS operation,
            "SERVICE" AS service,
            "AmortizedCost"::DOUBLE PRECISION AS amortized_cost,
            "BlendedCost"::DOUBLE PRECISION AS blended_cost,
            "NetAmortizedCost"::DOUBLE PRECISION AS net_amortized_cost,
            "NetUnblendedCost"::DOUBLE PRECISION AS net_unblended_cost,
            "NormalizedUsageAmount"::DOUBLE PRECISION AS normalized_usage_amount,
            "UnblendedCost"::DOUBLE PRECISION AS unblended_cost,
            "UsageQuantity"::DOUBLE PRECISION AS usage_quantity
         FROM __schema__.bronze_aws_ce';
   END IF;
END
$$;