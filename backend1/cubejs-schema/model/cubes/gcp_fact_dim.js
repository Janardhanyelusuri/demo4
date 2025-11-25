const { securityContext } = COMPILE_CONTEXT

cube(`gcp_fact_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_gcp_fact_dim`,
    
    data_source: `default`,
    
    joins: {
        gcp_tags_dim: {
            relationship: `many_to_one`,
            sql: `${CUBE}.tags_key = ${gcp_tags_dim.tags_key}`
          }  ,
          gcp_billing_dim: {
            relationship: `many_to_one`,
            sql: `${CUBE}.billing_account_id = ${gcp_billing_dim.billing_account_id}`
          }  ,
    },
    
    dimensions: {
      region_id: {
        sql: `region_id`,
        type: `string`,
      },

      region_name: {
        sql: `region_name`,
        type: `string`
      },
      
      x_location: {
        sql: `x_location`,
        type: `string`
      },

      sku_id: {
        sql: `sku_id`,
        type: `string`
      },
      
      service_name: {
        sql: `service_name`,
        type: `string`
      },

      service_category: {
        sql: `service_category`,
        type: `string`
      },

      x_service_id: {
        sql: `x_service_id`,
        type: `string`,
      },

      contracted_cost: {
        sql: `contracted_cost`,
        type: `number`,
      },
    
      tags_key: {
        sql: `tags_key`,
        type: `string`,
        primaryKey: true,
        public: true,
      },

      billing_account_id: {
        sql: `billing_account_id`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      resource_name: {
        sql: `resource_name`,
        type: `string`
      },
      
      resource_type: {
        sql: `resource_type`,
        type: `string`
      },

      consumed_quantity: {
        sql: `consumed_quantity`,
        type: `number`
      },

      pricing_quantity: {
        sql: `pricing_quantity`,
        type: `number`
      },

      provider_name: {
        sql: `provider_name`,
        type: `string`
      },

      billed_cost: {
        sql: `billed_cost`,
        type: `number`
      },

      list_cost: {
        sql: `list_cost`,
        type: `number`
      },

      effective_cost: {
        sql: `effective_cost`,
        type: `number`
      },
      
      billing_period_start: {
        sql: `billing_period_start`,
        type: `time`
      },

      billing_period_end: {
        sql: `billing_period_end`,
        type: `time`
      },

      x_project_id: {
        sql: `x_project_id`,
        type: `string`
      },

      charge_period_start: {
        sql: `charge_period_start`,
        type: `time`
      },
    
      charge_period_end: {
        sql: `charge_period_end`,
        type: `time`
      },
      
      charge_description: {
        sql: `charge_description`,
        type: `string`
      },

      charge_category: {
        sql: `charge_category`,
        type: `string`
      },

      monthly_budget: {
        sql: `monthly_budget`, // or the appropriate SQL expression
        type: `number`, // or `sum` or `avg`, depending on your use case
      },

    },
    
    measures: {
      count: {
        type: `count`
      },
      csql_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud SQL'` } // Add this filter
        ],
        title: `Cloud SQL Year-to-Date Cost`
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
      csql_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud SQL'` } // Add this filter
        ],
        title: `Cloud SQL Year-to-Date Cost`
      },

      csql_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud SQL'` } // Add this filter
        ],
        title: `Cloud SQL Year-to-Date Cost`
      },

      cstorage_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Storage'` } // Add this filter
        ],
        title: `Cloud Storage Year-to-Date Cost`
      },
      cstorage_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Storage'` } // Add this filter
        ],
        title: `Cloud Storage Month-to-Date Cost`
      },

      cstorage_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Storage'` } // Add this filter
        ],
        title: `Cloud Storage Quarter-to-Date Cost`
      },

      bq_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'BigQuery'` } // Add this filter
        ],
        title: `BigQuery Year-to-Date Cost`
      },
      bq_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'BigQuery'` } // Add this filter
        ],
        title: `BigQuery Month-to-Date Cost`
      },

      bq_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'BigQuery'` } // Add this filter
        ],
        title: `BigQuery Quarter-to-Date Cost`
      },

      areg_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Artifact Registry'` } // Add this filter
        ],
        title: `Artifact Registry Year-to-Date Cost`
      },
      areg_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Artifact Registry'` } // Add this filter
        ],
        title: `Artifact Registry Month-to-Date Cost`
      },

      areg_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Artifact Registry'` } // Add this filter
        ],
        title: `Artifact Registry Quarter-to-Date Cost`
      },

      logging_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Logging'` } // Add this filter
        ],
        title: `Cloud Logging Year-to-Date Cost`
      },
      logging_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Logging'` } // Add this filter
        ],
        title: `Cloud Logging Month-to-Date Cost`
      },

      logging_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Logging'` } // Add this filter
        ],
        title: `Cloud Logging Year-to-Date Cost`
      },

      networking_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Networking'` } // Add this filter
        ],
        title: `Networking Year-to-Date Cost`
      },
      networking_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Networking'` } // Add this filter
        ],
        title: `Networking Month-to-Date Cost`
      },

      networking_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Networking'` } // Add this filter
        ],
        title: `Networking Year-to-Date Cost`
      },

      kubengine_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Kubernetes Engine'` } // Add this filter
        ],
        title: `Kubernetes Engine Year-to-Date Cost`
      },
      kubengine_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Kubernetes Engine'` } // Add this filter
        ],
        title: `Kubernetes Engine Month-to-Date Cost`
      },

      kubengine_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Kubernetes Engine'` } // Add this filter
        ],
        title: `Kubernetes Engine Year-to-Date Cost`
      },

      kms_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Key Management Service (KMS)'` } // Add this filter
        ],
        title: `Cloud Key Management Service (KMS) Year-to-Date Cost`
      },
      kms_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Key Management Service (KMS)'` } // Add this filter
        ],
        title: `Cloud Key Management Service (KMS) Month-to-Date Cost`
      },

      kms_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Cloud Key Management Service (KMS)'` } // Add this filter
        ],
        title: `Cloud Key Management Service (KMS) Year-to-Date Cost`
      },

      ce_year_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Compute Engine'` } // Add this filter
        ],
        title: `Compute Engine Year-to-Date Cost`
      },
      ce_month_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Compute Engine'` } // Add this filter
        ],
        title: `Compute Engine Year-to-Date Cost`
      },
      ce_quarter_to_date_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `gcp_fact_dim.service_name = 'Compute Engine'` } // Add this filter
        ],
        title: `Compute Engine Year-to-Date Cost`
      },
      total_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        title: `Total Billed Cost`
      },
  
      total_list_cost: {
        sql: `list_cost::numeric`,
        type: `sum`,
        title: `Total List Cost`
      },

      total_effective_cost: {
        sql: `effective_cost::numeric`,
        type: `sum`,
        title: `Total Effective Cost`
      },
      average_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `avg`,
        title: `Average Billed Cost`
      },
  
      max_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `max`,
        title: `Max Billed Cost`
      },
  
      min_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `min`,
        title: `Min Billed Cost`
      },
  
      monthly_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` }
        ],
        title: `Monthly Billed Cost`
      },
  
      quarterly_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` }
        ],
        title: `Quarterly Billed Cost`
      },
  
      yearly_billed_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` }
        ],
        title: `Yearly Billed Cost`
      },
  
      project_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.x_project_id IS NOT NULL` }
        ],
        title: `Total Project Cost`
      },
  
      service_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.x_service_id IS NOT NULL` }
        ],
        title: `Total Service Cost`
      },
      
      count_of_services: {
        sql: `DISTINCT ${CUBE}.x_service_id`,
        type: `countDistinct`,
        title: `Count of Services`
      },
      
      count_of_projects: {
        sql: `DISTINCT ${CUBE}.x_project_id`,
        type: `countDistinct`,
        title: `Count of Projects`
      },
      
      count_of_resource_types: {
        sql: `DISTINCT ${CUBE}.resource_type`,
        type: `countDistinct`,
        title: `Count of Resource Types`
      },

      month_to_date_list_cost: {
        sql: `list_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` }
        ],
        title: `Month-to-Date Cost`
      },

      // Quarter-to-Date Cost
      quarter_to_date_list_cost: {
        sql: `list_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` }
        ],
        title: `Quarter-to-Date Cost`
      },
  
      // Year-to-Date Cost
      year_to_date_list_cost: {
        sql: `list_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.billing_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` }
        ],
        title: `Year-to-Date Cost`
      },
  
      // Quarterly Budget
      quarterly_budget: {
        sql: `MAX(${CUBE}.monthly_budget) * 3`,
        type: `number`,
        title: `Quarterly Budget`
      },
  
      // Yearly Budget
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
    
      // Percentage of Tagged Resources
      percentage_tagged_resources: {
        sql: `
          CASE
            WHEN COUNT(${CUBE}.resource_name) = 0 THEN 0
            ELSE 100.0 * SUM(CASE WHEN ${CUBE}.x_project_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(${CUBE}.resource_name)
          END
        `,
        type: `number`,
        title: `Percentage of Tagged Resources`
      },

      max_monthly_budget: {
        sql: `MAX(${CUBE}.monthly_budget)`,
        type: `number`,
        title: `Max Monthly Budget`
      },
  
      total_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        title: `Total Cost`
      },

      total_cost_by_provider: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        title: `Total Cost by Provider`
      },
      
  
      total_consumed_quantity: {
        sql: `consumed_quantity::numeric`,
        type: `sum`,
        title: `Total Consumed Quantity`
      },
  
      total_pricing_quantity: {
        sql: `pricing_quantity::numeric`,
        type: `sum`,
        title: `Total Pricing Quantity`
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

  }});
  