const { securityContext } = COMPILE_CONTEXT

cube(`aws_fact_focus`, {
  sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_aws_fact_focus`,
  
  data_source: `default`,
  
  joins: {
    aws_billing_dim: {
      relationship: `many_to_one`,
      sql: `${CUBE}.billing_account_id = ${aws_billing_dim.billing_account_id}`
    },
    aws_tags: {
      relationship: `many_to_one`,
      sql: `${CUBE}.tags_key = ${aws_tags.tags_key}`
    }
  },
  
  dimensions: {
    billing_account_id: {
      sql: `billing_account_id`,
      type: `string`,
      primaryKey: true,
      public: true,
    },
    tags_key: {
      sql: `tags_key`,
      type: `string`,
      primaryKey: true,
      public: true,
    },
    x_operation: {
      sql: `x_operation`,
      type: `string`
    },
    x_usage_type: {
      sql: `x_usage_type`,
      type: `string`
    },
    resource_id: {
      sql: `resource_id`,
      type: `string`
    },
    consumed_unit: {
      sql: `consumed_unit`,
      type: `string`
    },
    consumed_quantity:{
      sql: 'consumed_quantity',
      type: 'number'
    },
    charge_period_start: {
      sql: `charge_period_start`,
      type: `time`
    },
    charge_period_end: {
      sql: `charge_period_end`,
      type: `time`
    },
    contracted_cost: {
      sql: `contracted_cost`,
      type: `number`
    },
    effective_cost:{
      sql: 'effective_cost',
      type: 'number'
    },
    list_cost: {
      sql: `list_cost`,
      type: `number`
    },
    list_unit_price: {
      sql: `list_unit_price`,
      type: `number`
    },
    region_id: {
      sql: `region_id`,
      type: `string`
    },
    region_name: {
      sql: `region_name`,
      type: `string`
    },
    service_name: {
      sql: `service_name`,
      type: `string`
    },
    pricing_category:{
      sql: 'pricing_category',
      type: 'string'
    },
    pricing_quantity: {
      sql: `pricing_quantity`,
      type: `number`
    },
    pricing_unit: {
      sql: `pricing_unit`,
      type: `string`
    },
    contracted_unit_price: {
      sql: `contracted_unit_price`,
      type: `number`
    },
    provider_name: {
      sql: `provider_name`,
      type: `string`
    },
    billing_period_start:{
      sql: 'billing_period_start',
      type: 'time'
    },
    billing_period_end: {
      sql: `billing_period_end`,
      type: `time`
    },
    billing_account_name: {
      sql: `billing_account_name`,
      type: `string`
    },
    charge_category: {
      sql: `charge_category`,
      type: `string`
    },
    charge_class: {
      sql: `charge_class`,
      type: `string`
    },
    charge_description:{
      sql: 'charge_description',
      type: 'string'
    },
    charge_frequency: {
      sql: `charge_frequency`,
      type: `number`
    },
    monthly_budget: {
      sql: `monthly_budget`,
      type: `number`
    },
    x_service_code: {
      sql: `x_service_code`,
      type: `string`
    },
    service_category: {
      sql: `service_category`,
      type: `string`
    },
    sku_price_id: {
      sql: `sku_price_id`,
      type: `string`
    },
    sku_id:{
      sql: 'sku_id',
      type: 'string'
    }
  },
  
  measures: {
    count: {
      type: `count`
    },
    // list_cost: {
    //   sql: `list_cost`,
    //   type: `sum`
    // },
    // blended_cost: {
    //   sql: `blended_cost`,
    //   type: `sum`
    // },
    total_list_cost: {
      sql: `list_cost`,
      type: `sum`,
      title: `Total List Cost`
    },
    month_to_date_list_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` }
      ],
      title: `Month-to-Date List Cost`
    },
    quarter_to_date_list_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` }
      ],
      title: `Quarter-to-Date List Cost`
    },
    year_to_date_list_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` }
      ],
      title: `Year-to-Date List Cost`
    },
    quarterly_budget: {
      sql: `MAX(${CUBE}.monthly_budget) * 3`,
      type: `number`,
      title: `Quarterly Budget`
    },
    yearly_budget: {
      sql: `MAX(${CUBE}.monthly_budget) * 12`,
      type: `number`,
      title: `Yearly Budget`
    },
    monthly_budget_drift: {
      sql: `SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END) 
      - (MAX(${CUBE}.monthly_budget) * EXTRACT(DAY FROM CURRENT_TIMESTAMP) / EXTRACT(DAY FROM DATE_TRUNC('month', CURRENT_TIMESTAMP + INTERVAL '1 month' - INTERVAL '1 day')))
      `,
      type: `number`,
      title: `Monthly Budget Drift Percentage`
    },
    monthly_budget_drift_value: {
      sql: `(
        SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
        - (
            MAX(${CUBE}.monthly_budget) * 
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
            EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP))
          )
      )`,
      type: `number`,
      title: `Monthly Budget Drift Value`
    },          
    monthly_budget_drift_percentage: {
      sql: `(
        100.0 * (
          SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
          - (
              MAX(${CUBE}.monthly_budget) * 
              EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
              EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP))
            )
        )
        / (
          MAX(${CUBE}.monthly_budget) * 
          EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
          EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP))
        )
      )`,
      type: `number`,
      title: `Monthly Budget Drift Percentage`
    },                       
    quarterly_budget_drift: {
      sql: `100.0 * SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END) / (MAX(${CUBE}.monthly_budget)*3)`,
      type: `number`,
      title: `Quarterly Budget Drift Percentage`
    },
    quarterly_budget_drift_value: {
      sql: `(
        SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
        - (
            MAX(${CUBE}.monthly_budget) * 3 * 
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
            EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP))
          )
      )`,
      type: `number`,
      title: `Quarterly Budget Drift Value`
    },                           
    quarterly_budget_drift_percentage: {
      sql: `(
        100.0 * (
          SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
          - (
              MAX(${CUBE}.monthly_budget) * 3 * 
              EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
              EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP))
            )
        )
        / (
          MAX(${CUBE}.monthly_budget) * 3 * 
          EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
          EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP))
        )
      )`,
      type: `number`,
      title: `Quarterly Budget Drift Percentage`
    },                           
    yearly_budget_drift: {
      sql: `100.0 * SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END) / (MAX(${CUBE}.monthly_budget)*12)`,
      type: `number`,
      title: `Yearly Budget Drift Percentage`
    },
    yearly_budget_drift_value: {
      sql: `(
        SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
        - (
            MAX(${CUBE}.monthly_budget) * 12 * 
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
            EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP))
          )
      )`,
      type: `number`,
      title: `Yearly Budget Drift Value`
    },                     
    yearly_budget_drift_percentage: {
      sql: `(
        100.0 * (
          SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
          - (
              MAX(${CUBE}.monthly_budget) * 12 * 
              EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
              EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP))
            )
        )
        / (
          MAX(${CUBE}.monthly_budget) * 12 * 
          EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
          EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP))
        )
      )`,
      type: `number`,
      title: `Yearly Budget Drift Percentage`
    },                  
    ecc_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonEC2'` } // Add this filter
      ],
      title: `EC2 Month-to-Date Cost`
    },
    ecc_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonEC2'` } // Add this filter
      ],
      title: `EC2 QUARTER-to-Date Cost`
    },
    ecc_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonEC2'` } // Add this filter
      ],
      title: `EC2 Year-to-Date Cost`
    },
    year_to_date_cost: {
      sql: `list_cost::numeric`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` }
      ],
      title: `Year-to-Date Cost`
    },
    quarter_to_date_cost: {
      sql: `list_cost::numeric`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` }
      ],
      title: `Quarter-to-Date Cost`
    },
    month_to_date_cost: {
      sql: `list_cost::numeric`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` }
      ],
      title: `Month-to-Date Cost`
    },
    storage_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonS3'` } // Add this filter
      ],
      title: `S3 Month-to-Date Cost`
    },
    ecs_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonEKS'` } // Add this filter
      ],
      title: `ECS Month-to-Date Cost`
    },
    load_balancing_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSELB'` } // Add this filter
      ],
      title: `AWSELB Month-to-Date Cost`
    },
    vpc_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonVPC'` } // Add this filter
      ],
      title: `AmazonVPC Month-to-Date Cost`
    },
    cloud_watch_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonCloudWatch'` } // Add this filter
      ],
      title: `AmazonCloudWatch Month-to-Date Cost`
    },
    kms_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'awskms'` } // Add this filter
      ],
      title: `awskms Month-to-Date Cost`
    },
    cost_explorer_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSCostExplorer'` } // Add this filter
      ],
      title: `AWSCostExplorer Month-to-Date Cost`
    },
    ecr_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonECR'` } // Add this filter
      ],
      title: `AmazonECR Month-to-Date Cost`
    },
    secret_manager_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSSecretsManager'` } // Add this filter
      ],
      title: `AWSSecretsManager Month-to-Date Cost`
    },
    ecs_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonEKS'` } // Add this filter
      ],
      title: `ECS year-to-Date Cost`
    },
    load_balancing_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSELB'` } // Add this filter
      ],
      title: `AWSELB year-to-Date Cost`
    },
    vpc_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonVPC'` } // Add this filter
      ],
      title: `AmazonVPC year-to-Date Cost`
    },
    cloud_watch_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonCloudWatch'` } // Add this filter
      ],
      title: `AmazonCloudWatch year-to-Date Cost`
    },
    kms_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'awskms'` } // Add this filter
      ],
      title: `awskms year-to-Date Cost`
    },
    cost_explorer_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSCostExplorer'` } // Add this filter
      ],
      title: `AWSCostExplorer year-to-Date Cost`
    },
    ecr_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonECR'` } // Add this filter
      ],
      title: `AmazonECR year-to-Date Cost`
    },
    secret_manager_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSSecretsManager'` } // Add this filter
      ],
      title: `AWSSecretsManager year-to-Date Cost`
    },
    ecs_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonEKS'` } // Add this filter
      ],
      title: `ECS quarter-to-Date Cost`
    },
    load_balancing_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSELB'` } // Add this filter
      ],
      title: `AWSELB quarter-to-Date Cost`
    },
    vpc_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonVPC'` } // Add this filter
      ],
      title: `AmazonVPC quarter-to-Date Cost`
    },
    cloud_watch_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonCloudWatch'` } // Add this filter
      ],
      title: `AmazonCloudWatch quarter-to-Date Cost`
    },
    kms_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'awskms'` } // Add this filter
      ],
      title: `awskms quarter-to-Date Cost`
    },
    cost_explorer_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSCostExplorer'` } // Add this filter
      ],
      title: `AWSCostExplorer quarter-to-Date Cost`
    },
    ecr_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonECR'` } // Add this filter
      ],
      title: `AmazonECR quarter-to-Date Cost`
    },
    secret_manager_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AWSSecretsManager'` } // Add this filter
      ],
      title: `AWSSecretsManager quarter-to-Date Cost`
    },
    storage_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonS3'` } // Add this filter
      ],
      title: `S3 Quarter-to-Date Cost`
    },
    storage_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonS3'` } // Add this filter
      ],
      title: `S3 Year-to-Date Cost`
    },
    rds_month_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonRDS'` } // Add this filter
      ],
      title: `RDS Month-to-Date Cost`
    },
    rds_quarter_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonRDS'` } // Add this filter
      ],
      title: `RDS Quarter-to-Date Cost`
    },
    rds_year_to_date_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
        { sql: `${CUBE}.x_service_code = 'AmazonRDS'` } // Add this filter
      ],
      title: `RDS Year-to-Date Cost`
    },
    ecc_total_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.x_service_code = 'AmazonEC2'` } // Add this filter
      ],
      title: `EC2 Total Cost`
    },
    storage_total_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.x_service_code = 'AmazonS3'` } // Add this filter
      ],
      title: `S3 Total Cost`
    },
    earliest_charge_period_date: {
      sql: `MIN(${CUBE}.charge_period_start)`,
      type: `time`,
      title: `Earliest Charge Period Start`
    },      
    latest_charge_period_date: {
      sql: `MAX(${CUBE}.charge_period_start)`,
      type: `time`,
      title: `Latest Charge Period Start`
    },
    rds_total_cost: {
      sql: `list_cost`,
      type: `sum`,
      filters: [
        { sql: `${CUBE}.x_service_code = 'AmazonRDS'` } // Add this filter
      ],
      title: `RDS Total Cost`
    },
    percentage_tagged_services: {
      sql: `
        CASE
          WHEN COUNT(${CUBE}.x_service_code) = 0 THEN 0
          ELSE 100.0 * SUM(CASE WHEN ${CUBE}.is_tagged = 'tagged' THEN 1 ELSE 0 END) / COUNT(${CUBE}.x_service_code)
        END
      `,
      type: `number`,
      title: `Percentage of Tagged Services`
    },
    max_monthly_budget: {
      sql: `MAX(${CUBE}.monthly_budget)`,
      type: `number`,
      title: `Max Monthly Budget`
    },
    monthly_budget_utilization: {
      sql: `100.0 * SUM(
        CASE 
          WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / MAX(${CUBE}.monthly_budget)`,
      type: `number`,
      title: `Monthly Budget Utilization`
    },   
    monthly_budget_utilization_actual_value: {
      sql: `SUM(
        CASE 
          WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      )`,
      type: `number`,
      title: `Monthly Budget Utilized (Actual Value)`
    },
    monthly_budget_utilization_percentage_value: {
      sql: `100.0 * SUM(
        CASE 
          WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / MAX(${CUBE}.monthly_budget)`,
      type: `number`,
      title: `Monthly Budget Utilized (Percentage)`
    },               
    quarterly_budget_utilization: {
      sql: `100.0 * SUM(
        CASE
          WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / (MAX(${CUBE}.monthly_budget) * 3)`,
      type: `number`,
      title: `Quarterly Budget Utilization`
    },
    quarterly_budget_utilization_actual_value: {
      sql: `SUM(
        CASE 
          WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      )`,
      type: `number`,
      title: `Quarterly Budget Utilized (Actual Value)`
    },
    quarterly_budget_utilization_percentage_value: {
      sql: `100.0 * SUM(
        CASE 
          WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / (MAX(${CUBE}.monthly_budget) * 3)`,
      type: `number`,
      title: `Quaterly Budget Utilized (Percentage)`
    },
    yearly_budget_utilization: {
      sql: `100.0 * SUM(
        CASE
          WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / (MAX(${CUBE}.monthly_budget) * 12)`,
      type: `number`,
      title: `Yearly Budget Utilization`
    },    
    yearly_budget_utilization_actual_value: {
      sql: `SUM(
        CASE 
          WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      )`,
      type: `number`,
      title: `Yearly Budget Utilized (Actual Value)`
    },
    yearly_budget_utilization_percentage_value: {
      sql: `100.0 * SUM(
        CASE 
          WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / (MAX(${CUBE}.monthly_budget) * 12)`,
      type: `number`,
      title: `Yearly Budget Utilized (Percentage)`
    },                   
    forecast_next_month_cost: {
      sql: `
        SUM(
          CASE 
            WHEN DATE_TRUNC('month', CURRENT_DATE) = DATE_TRUNC('month', ${CUBE}.charge_period_start)
            THEN ${CUBE}.list_cost
            ELSE 0
          END
        ) / DATE_PART('day', CURRENT_DATE) * 30
      `,
      type: `number`,
      title: `Forecasted Cost for the Next Month`
    },  

  forecast_next_quarter_cost: {
    sql: `
      SUM(
        CASE 
          WHEN DATE_TRUNC('quarter', CURRENT_DATE) = DATE_TRUNC('quarter', ${CUBE}.charge_period_start)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / DATE_PART('day', CURRENT_DATE) * 90
    `,
    type: `number`,
    title: `Forecasted Cost for the Next Quarter`
  },
  forecast_next_year_cost: {
    sql: `
      SUM(
        CASE 
          WHEN DATE_TRUNC('year', CURRENT_DATE) = DATE_TRUNC('year', ${CUBE}.charge_period_start)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / DATE_PART('day', CURRENT_DATE) * 365
    `,
    type: `number`,
    title: `Forecasted Cost for the Next Year`
  }, 
  tags_yearly_budget: {
    sql: `${COMPILE_CONTEXT.securityContext.tagsBudget}`,
    type: `number`,
    title: `Tags Yearly Budget`
  },
  tags_quarterly_budget: {
    sql: `${COMPILE_CONTEXT.securityContext.tagsBudget} / 4`,
    type: `number`,
    title: `Quarterly Budget (based on YEARLY TAGS Budget)`
  },
  tags_monthly_budget: {
    sql: `${COMPILE_CONTEXT.securityContext.tagsBudget} / 12`,
    type: `number`,
    title: `Monthly Budget (based on YEARLY TAGS Budget)`
  },
  tags_monthly_budget_drift_value: {
    sql: `(
      SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
      - (
          ${COMPILE_CONTEXT.securityContext.tagsBudget} / 12 *
          EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
          EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP))
        )
    )`,
    type: `number`,
    title: `Monthly Budget Drift Value`
  },
  tags_quarterly_budget_drift_value: {
    sql: `(
      SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
      - (
          ${COMPILE_CONTEXT.securityContext.tagsBudget} * 3 *
          EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
          EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP))
        )
    )`,
    type: `number`,
    title: `Quarterly Budget Drift Value`
  },
  tags_yearly_budget_drift_value: {
    sql: `(
      SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
      - (
          ${COMPILE_CONTEXT.securityContext.tagsBudget} *
          EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
          EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP))
        )
    )`,
    type: `number`,
    title: `Yearly Budget Drift Value`
  },
  tags_monthly_budget_drift_percentage: {
    sql: `(
      100.0 * (
        SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
        - (
            ${COMPILE_CONTEXT.securityContext.tagsBudget} / 12 *
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
            NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP)), 0)
          )
      )
      /
      NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget} / 12 *
      EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
      NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP)), 0), 0)
    )`,
    type: `number`,
    title: `Monthly Budget Drift Percentage`
  },
  tags_quarterly_budget_drift_percentage: {
    sql: `(
      100.0 * (
        SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
        - (
            ${COMPILE_CONTEXT.securityContext.tagsBudget} * 3 *
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
            NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP)), 0)
          )
      )
      /
      NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget} * 3 *
      EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
      NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP)), 0), 0)
    )`,
    type: `number`,
    title: `Quarterly Budget Drift Percentage`
  },
  tags_yearly_budget_drift_percentage: {
    sql: `(
      100.0 * (
        SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.list_cost ELSE 0 END)
        - (
            ${COMPILE_CONTEXT.securityContext.tagsBudget} *
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
            NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0)
          )
      )
      /
      NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget} *
      EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
      NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0), 0)
    )`,
    type: `number`,
    title: `Yearly Budget Drift Percentage`
  },
  tags_monthly_budget_utilization_actual_value: {
    sql: `SUM(
      CASE
        WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
        THEN ${CUBE}.list_cost
        ELSE 0
      END
    )`,
    type: `number`,
    title: `Monthly Budget Utilized (Actual Value)`
  },
  tags_quarterly_budget_utilization_actual_value: {
    sql: `SUM(
      CASE
        WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
        THEN ${CUBE}.list_cost
        ELSE 0
      END
    )`,
    type: `number`,
    title: `Quarterly Budget Utilized (Actual Value)`
  },
  tags_yearly_budget_utilization_actual_value: {
    sql: `SUM(
      CASE
        WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
        THEN ${CUBE}.list_cost
        ELSE 0
      END
    )`,
    type: `number`,
    title: `Yearly Budget Utilized (Actual Value)`
  },
  //utilization queries
  tags_yearly_budget_utilization_percentage: {
    sql: `(
      SUM(
        CASE
          WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / ${COMPILE_CONTEXT.securityContext.tagsBudget} * 100
    )`,
    type: `number`,
    title: `Yearly Budget Utilization (%)`
  },
  tags_monthly_budget_utilization_percentage: {
    sql: `(
      SUM(
        CASE
          WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget}, 0) / 12 * 100
    )`,
    type: `number`,
    title: `Monthly Budget Utilization (%)`
  },

  tags_quarterly_budget_utilization_percentage: {
    sql: `(
      SUM(
        CASE
          WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
          THEN ${CUBE}.list_cost
          ELSE 0
        END
      ) / NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget}, 0) / 4 * 100
    )`,
    type: `number`,
    title: `Quarterly Budget Utilization (%)`
  },
    
  },
  
  pre_aggregations: {

    // main: {
    //   measures: [
    //     aws_fact_focus.total_cost,
    //     aws_fact_focus.month_to_date_cost,
    //     aws_fact_focus.quarter_to_date_cost,
    //     aws_fact_focus.year_to_date_cost,
    //     aws_fact_focus.quarterly_budget,
    //     aws_fact_focus.yearly_budget,
    //     aws_fact_focus.monthly_budget_drift,
    //     aws_fact_focus.quarterly_budget_drift,
    //     aws_fact_focus.yearly_budget_drift,
    //     aws_fact_focus.ecc_month_to_date_cost,
    //     aws_fact_focus.ecc_quarter_to_date_cost,
    //     aws_fact_focus.ecc_year_to_date_cost,
    //     aws_fact_focus.rds_month_to_date_cost,
    //     aws_fact_focus.rds_quarter_to_date_cost,
    //     aws_fact_focus.rds_year_to_date_cost,
    //     aws_fact_focus.storage_month_to_date_cost,
    //     aws_fact_focus.storage_quarter_to_date_cost,
    //     aws_fact_focus.storage_year_to_date_cost,
    //     aws_fact_focus.ecc_total_cost,
    //     aws_fact_focus.rds_total_cost,
    //     aws_fact_focus.storage_total_cost


    //   ],
    //   dimensions: [
    //     aws_fact_focus.tags_key,
    //     aws_fact_focus.usage_account_id,
    //     aws_fact_focus.operation,
    //     aws_fact_focus.usage_type,
    //     aws_fact_focus.resource_id,
    //     aws_fact_focus.x_service_code,
    //     aws_fact_focus.service_name,
    //     aws_fact_focus.product_family,
    //     aws_fact_focus.product_region,
    //     aws_fact_focus.resource_name,
    //     aws_fact_focus.is_tagged,
    //     aws_fact_focus.monthly_budget,
    //     aws_fact_focus.instance_usage_type,
    //     aws_dim_account.usage_account_name
    //   ],
    //   timeDimension: aws_fact_focus.charge_period_start,
    //   granularity: `month`
    // }
    // Pre-aggregation definitions go here.
    // Learn more in the documentation: https://cube.dev/docs/caching/pre-aggregations/getting-started
  }
});
