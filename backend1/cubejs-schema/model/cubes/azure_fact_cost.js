const { securityContext } = COMPILE_CONTEXT

cube(`azure_fact_cost`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_azure_fact_cost`,
  //  sql: `SELECT * FROM ${COMPILE_CONTEXT.securityContext.schemaName}.gold_azure_fact_cost`, // Ensure this matches your table name and format
  
    dataSource: `default`,

    joins: {
      azure_account_dim: {
        relationship: `many_to_one`,
        sql: `${CUBE}.sub_account_id = ${azure_account_dim}.sub_account_id`
      },
      azure_resource_dim: {
        relationship: `many_to_one`,
        sql: `${CUBE}.resource_id = ${azure_resource_dim}.resource_id`
      },
      azure_charge_summary_dim: {
        relationship: `many_to_one`,
        sql: `${CUBE}.sku_id = ${azure_charge_summary_dim}.sku_id`
      },
    },
  
    dimensions: {
      sub_account_id: {
        sql: `sub_account_id`,
        type: `string`,
        primaryKey: true,
        public: true
      },
      resource_id: {
        sql: `resource_id`,
        type: `string`,
        primaryKey: true,
        public: true
      },
      sku_id: {
        sql: `sku_id`,
        type: `string`,
        primaryKey: true,
        public: true
      },
      resource_group_name: {
        sql: `resource_group_name`,
        type: `string`
      },
      charge_period_start: {
        sql: `charge_period_start`,
        type: `time`
      },
      pricing_category: {
        sql: `pricing_category`,
        type: `string`
      },
      pricing_unit: {
        sql: `pricing_unit`,
        type: `string`
      },
      list_unit_price: {
        sql: `list_unit_price`,
        type: `number`
      },
      contracted_unit_price: {
        sql: `contracted_unit_price`,
        type: `number`
      },
      pricing_quantity: {
        sql: `pricing_quantity`,
        type: `number`
      },
      billed_cost: {
        sql: `billed_cost`,
        type: `number`
      },
      consumed_quantity: {
        sql: `consumed_quantity`,
        type: `number`
      },
      consumed_unit: {
        sql: `consumed_unit`,
        type: `string`
      },
      effective_cost: {
        sql: `effective_cost`,
        type: `number`
      },
      contracted_cost: {
        sql: `contracted_cost`,
        type: `number`
      },
      list_cost: {
        sql: `list_cost`,
        type: `number`
      },
      effective_unit_price: {
        sql: `effective_unit_price`,
        type: `number`
      },
      billed_cost_in_usd: {
        sql: `billed_cost_in_usd`,
        type: `number`
      },
      effective_cost_in_usd: {
        sql: `effective_cost_in_usd`,
        type: `number`
      },
      list_cost_in_usd: {
        sql: `list_cost_in_usd`,
        type: `number`
      },
      sku_price_id: {
        sql: `sku_price_id`,
        type: `string`
      },
      sku_meter_name: {
        sql: `sku_meter_name`,
        type: `string`
      },
      sku_meter_subcategory: {
        sql: `sku_meter_subcategory`,
        type: `string`
      },
      sku_service_family: {
        sql: `sku_service_family`,
        type: `string`
      },
      monthly_budget: {
        sql: `monthly_budget`,
        type: `number`
      },
      budget_drift: {
        sql: `${CUBE}.effective_cost - ${CUBE}.monthly_budget`, // Calculate drift as difference
        type: `number`,
        title: `Budget Drift`
      },
    },
  
    measures: {
      count: {
        type: `count`
      },
      total_billed_cost: {
        sql: `billed_cost`,
        type: `sum`,
        title: `Total Billed Cost`
      },
      average_billed_cost: {
        sql: `billed_cost`,
        type: `avg`,
        title: `Average Billed Cost`
      },
      max_billed_cost: {
        sql: `billed_cost`,
        type: `max`,
        title: `Max Billed Cost`
      },
      min_billed_cost: {
        sql: `billed_cost`,
        type: `min`,
        title: `Min Billed Cost`
      },
      max_monthly_budget: {
        sql: `MAX(${CUBE}.monthly_budget)`,
        type: `number`,
        title: `Max Monthly Budget`
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
           SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
           SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
           SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
             SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
             SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
             SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
             THEN ${CUBE}.billed_cost
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
             THEN ${CUBE}.billed_cost
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
             THEN ${CUBE}.billed_cost
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
               THEN ${CUBE}.billed_cost
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
               THEN ${CUBE}.billed_cost
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
               THEN ${CUBE}.billed_cost
               ELSE 0
             END
           ) / NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget}, 0) / 4 * 100
         )`,
         type: `number`,
         title: `Quarterly Budget Utilization (%)`
       },
//      tags_yearly_budget_drift_percentage: {
//        sql: `(
//          100.0 * (
//            SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
//            - (
//                200 *
//                EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
//                NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0)
//              )
//          )
//          /
//          NULLIF(200 *
//          EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
//          NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0), 0)
//        )`,
//        type: `number`,
//        title: `Yearly Budget Drift Percentage`
//      },
//      tags_monthly_budget_utilization_actual_value: {
//        sql: `SUM(
//          CASE
//            WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
//            THEN ${CUBE}.billed_cost
//            ELSE 0
//          END
//        )`,
//        type: `number`,
//        title: `Monthly Budget Utilized (Actual Value)`
//      },
//      tags_quarterly_budget_utilization_actual_value: {
//        sql: `SUM(
//          CASE
//            WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
//            THEN ${CUBE}.billed_cost
//            ELSE 0
//          END
//        )`,
//        type: `number`,
//        title: `Quarterly Budget Utilized (Actual Value)`
//      },
//      tags_yearly_budget_utilization_actual_value: {
//        sql: `SUM(
//          CASE
//            WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
//            THEN ${CUBE}.billed_cost
//            ELSE 0
//          END
//        )`,
//        type: `number`,
//        title: `Yearly Budget Utilized (Actual Value)`
//      },
//      tags_yearly_budget_utilization_percentage: {
//        sql: `(
//          SUM(
//            CASE
//              WHEN DATE_TRUNC('year', ${CUBE}.charge_period_start) = DATE_TRUNC('year', CURRENT_DATE)
//              THEN ${CUBE}.billed_cost
//              ELSE 0
//            END
//          ) / 200 * 100
//        )`,
//        type: `number`,
//        title: `Yearly Budget Utilization (%)`
//      },
//      tags_monthly_budget_utilization_percentage: {
//        sql: `(
//          SUM(
//            CASE
//              WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
//              THEN ${CUBE}.billed_cost
//              ELSE 0
//            END
//          ) / NULLIF(200, 0) / 12 * 100
//        )`,
//        type: `number`,
//        title: `Monthly Budget Utilization (%)`
//      },
//      tags_quarterly_budget_utilization_percentage: {
//        sql: `(
//          SUM(
//            CASE
//              WHEN DATE_TRUNC('quarter', ${CUBE}.charge_period_start) = DATE_TRUNC('quarter', CURRENT_DATE)
//              THEN ${CUBE}.billed_cost
//              ELSE 0
//            END
//          ) / NULLIF(200, 0) / 4 * 100
//        )`,
//        type: `number`,
//        title: `Quarterly Budget Utilization (%)`
//      },
      month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` }
        ],
        title: `Month-to-Date Cost`
      },
      quarterly_budget: {
        sql: `MAX(${CUBE}.monthly_budget) * 3`,
        type: `number`,
        title: `Quarterly Budget`
      },
      total_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        title: `Total Cost`
      },
      yearly_budget: {
        sql: `MAX(${CUBE}.monthly_budget) * 12`,
        type: `number`,
        title: `Yearly Budget`
      },
      quarter_to_date_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` }
        ],
        title: `Quarter-to-Date Cost`
      },
      year_to_date_cost: {
        sql: `billed_cost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` }
        ],
        title: `Year-to-Date Cost`
      },
      // monthly_budget_drift: {
      //   sql: `100.0 * SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END) / MAX(${CUBE}.monthly_budget)`,
      //   type: `number`,
      //   title: `Monthly Budget Drift Percentage`
      // },
      monthly_budget_drift: {
        sql: `SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END) 
        - (MAX(${CUBE}.monthly_budget) * EXTRACT(DAY FROM CURRENT_TIMESTAMP) / EXTRACT(DAY FROM DATE_TRUNC('month', CURRENT_TIMESTAMP + INTERVAL '1 month' - INTERVAL '1 day')))
        `,
        type: `number`,
        title: `Monthly Budget Drift Percentage`
      },
      earliest_charge_period_start: {
        sql: `MIN(${CUBE}.charge_period_start)`,
        type: `time`,
        title: `Earliest Charge Period Date`
      },      
      latest_charge_period_start: {
        sql: `MAX(${CUBE}.charge_period_start)`,
        type: `time`,
        title: `Latest Charge Period Date`
      },              
      monthly_budget_drift_value: {
        sql: `(
          SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
            SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
        sql: `100.0 * SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END) / (MAX(${CUBE}.monthly_budget)*3)`,
        type: `number`,
        title: `Quarterly Budget Drift Percentage`
      },
      quarterly_budget_drift_value: {
        sql: `(
          SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
            SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
        sql: `100.0 * SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END) / (MAX(${CUBE}.monthly_budget)*12)`,
        type: `number`,
        title: `Yearly Budget Drift Percentage`
      },
      yearly_budget_drift_value: {
        sql: `(
          SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
            SUM(CASE WHEN ${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billed_cost ELSE 0 END)
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
      dq_demo_workspace_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.resource_name = 'dq-demo-workspace'` } // Add this filter
        ],
        title: `DQ Demo Workspace Year-to-Date Cost`
      },
      nat_gateway_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.resource_name = 'nat-gateway'` } // Add this filter
        ],
        title: `Nat Gateway Year-to-Date Cost`
      },
      sigmoid_devops_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `${azure_resource_dim.resource_name} = 'Sigmoid-DevOps'` } // Update to match dimension name
        ],
        title: `Sigmoid Devops Year-to-Date Cost`
      },
      dq_demo_workspace_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.resource_name = 'dq-demo-workspace'` } // Add this filter
        ],
        title: `DQ Demo Workspace Quarter-to-Date Cost`
      },
      nat_gateway_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.resource_name = 'nat-gateway'` } // Add this filter
        ],
        title: `Nat Gateway Quarter-to-Date Cost`
      },
      sigmoid_devops_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `${azure_resource_dim.resource_name} = 'Sigmoid-DevOps'` } // Update to match dimension name
        ],
        title: `Sigmoid Devops Quarter-to-Date Cost`
      },
      dq_demo_workspace_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.resource_name = 'dq-demo-workspace'` } // Add this filter
        ],
        title: `DQ Demo Workspace Month-to-Date Cost`
      },
      nat_gateway_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.resource_name = 'nat-gateway'` } // Add this filter
        ],
        title: `Nat Gateway Month-to-Date Cost`
      },
      sigmoid_devops_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `${azure_resource_dim.resource_name} = 'Sigmoid-DevOps'` } // Update to match dimension name
        ],
        title: `Sigmoid Devops Month-to-Date Cost`
      },

      nat_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure NAT Gateway'` } // Add this filter
        ],
        title: `Azure NAT Gateway Year-to-Date Cost`
      },

      vs_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Visual Studio'` } // Add this filter
        ],
        title: `Visual Studio Year-to-Date Cost`
      },

      db_for_postgres_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure DB for PostgreSQL'` } // Add this filter
        ],
        title: `Azure DB for PostgreSQL Year-to-Date Cost`
      },

      container_registry_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Container Registry'` } // Add this filter
        ],
        title: `Azure Container Registry Year-to-Date Cost`
      },

      vnet_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Network'` } // Add this filter
        ],
        title: `Virtual Network Year-to-Date Cost`
      },

      ml_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Machine Learning'` } // Add this filter
        ],
        title: `Azure Machine Learning Year-to-Date Cost`
      },

      vmss_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Machine Scale Sets'` } // Add this filter
        ],
        title: `Virtual Machine Scale Sets Year-to-Date Cost`
      },

      monitor_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Monitor'` } // Add this filter
        ],
        title: `Azure Monitor Year-to-Date Cost`
      },

      load_balancer_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Load Balancer'` } // Add this filter
        ],
        title: `Load Balancer Year-to-Date Cost`
      },

      vs_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Visual Studio'` } // Add this filter
        ],
        title: `Visual Studio Quarter-to-Date Cost`
      },

      db_for_postgres_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure DB for PostgreSQL'` } // Add this filter
        ],
        title: `Azure DB for PostgreSQL quarter-to-Date Cost`
      },

      container_registry_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Container Registry'` } // Add this filter
        ],
        title: `Azure Container Registry quarter-to-Date Cost`
      },

      vnet_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Network'` } // Add this filter
        ],
        title: `Virtual Network quarter-to-Date Cost`
      },

      ml_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Machine Learning'` } // Add this filter
        ],
        title: `Azure Machine Learning quarter-to-Date Cost`
      },

      vmss_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Machine Scale Sets'` } // Add this filter
        ],
        title: `Virtual Machine Scale Sets quarter-to-Date Cost`
      },

      monitor_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Monitor'` } // Add this filter
        ],
        title: `Azure Monitor quarter-to-Date Cost`
      },

      load_balancer_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Load Balancer'` } // Add this filter
        ],
        title: `Load Balancer quarter-to-Date Cost`
      },

      vs_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Visual Studio'` } // Add this filter
        ],
        title: `Visual Studio month-to-Date Cost`
      },

      db_for_postgres_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure DB for PostgreSQL'` } // Add this filter
        ],
        title: `Azure DB for PostgreSQL month-to-Date Cost`
      },

      container_registry_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Container Registry'` } // Add this filter
        ],
        title: `Azure Container Registry month-to-Date Cost`
      },

      vnet_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Network'` } // Add this filter
        ],
        title: `Virtual Network month-to-Date Cost`
      },

      ml_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Machine Learning'` } // Add this filter
        ],
        title: `Azure Machine Learning month-to-Date Cost`
      },

      vmss_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Machine Scale Sets'` } // Add this filter
        ],
        title: `Virtual Machine Scale Sets month-to-Date Cost`
      },

      monitor_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure Monitor'` } // Add this filter
        ],
        title: `Azure Monitor month-to-Date Cost`
      },

      load_balancer_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Load Balancer'` } // Add this filter
        ],
        title: `Load Balancer month-to-Date Cost`
      },

      vm_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Machines'` } // Add this filter
        ],
        title: `Virtual Machines' Year-to-Date Cost`
      },
      databricks_year_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` },
          { sql: `${azure_resource_dim.service_name} = 'Azure Databricks'` } // Update to match dimension name
        ],
        title: `Azure Databricks Year-to-Date Cost`
      },
      nat_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure NAT Gateway'` } // Add this filter
        ],
        title: `Azure NAT Gateway Quarter-to-Date Cost`
      },
      vm_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Machines'` } // Add this filter
        ],
        title: `Virtual Machines' Quarter-to-Date Cost`
      },
      databricks_quarter_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` },
          { sql: `${azure_resource_dim.service_name} = 'Azure Databricks'` } // Update to match dimension name
        ],
        title: `Azure Databricks Quarter-to-Date Cost`
      },
      nat_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Azure NAT Gateway'` } // Add this filter
        ],
        title: `Azure NAT Gateway Month-to-Date Cost`
      },
      vm_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `azure_resource_dim.service_name = 'Virtual Machines'` } // Add this filter
        ],
        title: `Virtual Machines' Month-to-Date Cost`
      },
      databricks_month_to_date_cost: {
        sql: `billed_cost`,
        type: `sum`,
        filters: [
          { sql: `${CUBE}.charge_period_start >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` },
          { sql: `${azure_resource_dim.service_name} = 'Azure Databricks'` } // Update to match dimension name
        ],
        title: `Azure Databricks Month-to-Date Cost`
      },
      monthly_budget_utilization: {
        sql: `100.0 * SUM(
          CASE 
            WHEN DATE_TRUNC('month', ${CUBE}.charge_period_start) = DATE_TRUNC('month', CURRENT_DATE)
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
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
            THEN ${CUBE}.billed_cost
            ELSE 0
          END
        ) / (MAX(${CUBE}.monthly_budget) * 12)`,
        type: `number`,
        title: `Yearly Budget Utilized (Percentage)`
      },              
      // forecast_next_month_cost: {
      //   sql: `
      //     SUM(
      //       CASE 
      //         WHEN DATE_TRUNC('month', CURRENT_DATE) = DATE_TRUNC('month', ${CUBE}.charge_period_start)
      //         THEN ${CUBE}.billed_cost
      //         ELSE 0
      //       END
      //     ) / DATE_PART('day', CURRENT_DATE) * 30
      //   `,
      //   type: `number`,
      //   title: `Forecasted Cost for the Next Month`
      // },  
      forecast_next_month_cost: {
        sql: `
          (
            SUM(
              CASE 
                WHEN DATE_TRUNC('month', CURRENT_DATE) = DATE_TRUNC('month', ${CUBE}.charge_period_start)
                THEN ${CUBE}.billed_cost
                ELSE 0
              END
            ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('month', CURRENT_DATE)) + 1, 0)  -- Days passed in the current month
          ) * 30  -- Projected over the next month
        `,
        type: `number`,
        title: `Forecasted Cost for the Next Month`
      },
      


    // forecast_next_quarter_cost: {
    //   sql: `
    //     SUM(
    //       CASE 
    //         WHEN DATE_TRUNC('quarter', CURRENT_DATE) = DATE_TRUNC('quarter', ${CUBE}.charge_period_start)
    //         THEN ${CUBE}.billed_cost
    //         ELSE 0
    //       END
    //     ) / DATE_PART('day', CURRENT_DATE) * 90
    //   `,
    //   type: `number`,
    //   title: `Forecasted Cost for the Next Quarter`
    // },

    // forecast_next_quarter_cost: {
    //   sql: `
    //     (
    //       SUM(
    //         CASE 
    //           WHEN DATE_TRUNC('quarter', CURRENT_DATE) = DATE_TRUNC('quarter', ${CUBE}.charge_period_start)
    //           THEN ${CUBE}.billed_cost
    //           ELSE 0
    //         END
    //       ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('quarter', CURRENT_DATE)) + 1, 0)
    //     ) * 90
    //   `,
    //   type: `number`,
    //   title: `Forecasted Cost for the Next Quarter`
    // },
  
    forecast_next_quarter_cost: {
      sql: `
        (
          SUM(
            CASE 
              WHEN DATE_TRUNC('quarter', CURRENT_DATE) = DATE_TRUNC('quarter', ${CUBE}.charge_period_start)
              THEN ${CUBE}.billed_cost
              ELSE 0
            END
          ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('quarter', CURRENT_DATE)) + 1, 0)
        ) * 90  -- Adjusted to 92 to account for the typical length of a quarter
      `,
      type: `number`,
      title: `Forecasted Cost for the Next Quarter`
    },
    
    // forecast_next_year_cost: {
    //   sql: `
    //     SUM(
    //       CASE 
    //         WHEN DATE_TRUNC('year', CURRENT_DATE) = DATE_TRUNC('year', ${CUBE}.charge_period_start)
    //         THEN ${CUBE}.billed_cost
    //         ELSE 0
    //       END
    //     ) / DATE_PART('day', CURRENT_DATE) * 365
    //   `,
    //   type: `number`,
    //   title: `Forecasted Cost for the Next Year`
    // },   
     
    forecast_next_year_cost: {
      sql: `
        (
          SUM(
            CASE 
              WHEN DATE_TRUNC('year', CURRENT_DATE) = DATE_TRUNC('year', ${CUBE}.charge_period_start)
              THEN ${CUBE}.billed_cost
              ELSE 0
            END
          ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('year', CURRENT_DATE)) + 1, 0)  -- Days passed in the current year
        ) * 365  -- Projected over the next year
      `,
      type: `number`,
      title: `Forecasted Cost for the Next Year`
    },
    
      cost_variance: {
        sql: `SUM(${CUBE}.billed_cost) - MAX(${CUBE}.monthly_budget)`,
        type: `number`,
        title: `Cost Variance`
      },
      idle_resource_cost: {
        sql: `SUM(CASE WHEN ${CUBE}.consumed_quantity = 0 THEN ${CUBE}.billed_cost ELSE 0 END)`,
        type: `number`,
        title: `Idle Resource Cost`
      },
      effective_cost_per_unit: {
        sql: `SUM(${CUBE}.effective_cost) / SUM(${CUBE}.consumed_quantity)`,
        type: `number`,
        title: `Effective Cost Per Unit`
      },
      configuration_drift: {
        sql: `SUM(${CUBE}.effective_cost) - SUM(${CUBE}.contracted_cost)`,
        type: `number`,
        title: `Configuration Drift`
      },
      critical_resource_alert: {
        sql: `CASE WHEN MAX(${CUBE}.billed_cost) > 1000 THEN 'High' ELSE 'Normal' END`,
        type: `string`,
        title: `Critical Resource Alert`
      },
      total_consumed_quantity: {
        sql: `SUM(${CUBE}.consumed_quantity)`,
        type: `number`,
        title: `Total Consumed Quantity`
      }     
        
      },
      
  
    preAggregations: {
      // Pre-aggregation definitions go here.
      // Learn more in the documentation: https://cube.dev/docs/caching/pre-aggregations/getting-started
    }
  });
  