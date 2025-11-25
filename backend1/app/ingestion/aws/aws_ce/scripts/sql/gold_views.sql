CREATE OR REPLACE VIEW __schema__.gold_aws_ce AS
SELECT
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
FROM
    __schema__.silver_aws_ce;