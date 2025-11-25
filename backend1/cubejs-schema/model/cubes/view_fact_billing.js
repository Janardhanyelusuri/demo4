const { securityContext } = COMPILE_CONTEXT

cube(`view_fact_billing`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_fact_billing`,
    
    data_source: `default`,
    
    joins: {
        view_dim_pricing: {
          relationship: `many_to_one`,
          sql: `${CUBE}.pricingcategory = ${view_dim_pricing.pricingcategory}`
        },
        view_dim_provider: {
          relationship: `many_to_one`,
          sql: `${CUBE}.providername = ${view_dim_provider.providername}`
        },
        view_dim_region: {
            relationship: `many_to_one`,
            sql: `${CUBE}.regionid = ${view_dim_region.regionid}`
          },
        view_dim_resource: {
            relationship: `many_to_one`,
            sql: `${CUBE}.resourceid = ${view_dim_resource.resourceid}`
        },
        view_dim_service: {
            relationship: `many_to_one`,
            sql: `${CUBE}.servicename = ${view_dim_service.servicename}`
          },
        view_dim_time: {
            relationship: `many_to_one`,
            sql: `${CUBE}.billingperiodstart = ${view_dim_time.billingperiodstart}`
        }
    },
    
    dimensions: {
      hash_key: {
            sql: `hash_key`,
            type: `string`,
            primaryKey: true,
            public: true,
          },

      pricingcategory: {
        sql: `pricingcategory`,
        type: `string`,
        public: true,
      },

      providername: {
        sql: `providername`,
        type: `string`,
        public: true,
      },

      regionid: {
        sql: `regionid`,
        type: `string`,
        public: true,
      },

      resourceid: {
        sql: `resourceid`,
        type: `string`,
        public: true,
      },

      servicename: {
        sql: `servicename`,
        type: `string`,
        public: true,
      },

      billingperiodstart: {
        sql: `billingperiodstart`,
        type: `time`,
        public: true,
      },

      //new columns(not in views)
      cloud_source: {
        sql: `cloud_source`,
        type: `string`
      },
      
      effectivecost: {
        sql: `effectivecost`,
        type: `number`
      },

      billedcost: {
        sql: `billedcost`,
        type: `number`
      },

      contractedcost: {
        sql: `contractedcost`,
        type: `number`
      },

      listcost: {
        sql: `listcost`,
        type: `number`
      },

      consumedquantity: {
        sql: `consumedquantity`,
        type: `number`
      },

      billingcurrency: {
        sql: `billingcurrency`,
        type: `string`
      },

      chargedescription: {
        sql: `chargedescription`,
        type: `string`
      },

      chargeclass: {
        sql: `chargeclass`,
        type: `string`
      },

      commitmentdiscountcategory: {
        sql: `commitmentdiscountcategory`,
        type: `string`
      },
      
      commitmentdiscountid: {
        sql: `commitmentdiscountid`,
        type: `string`
      },

      commitmentdiscountname: {
        sql: `commitmentdiscountname`,
        type: `string`
      },

      subaccountid: {
        sql: `subaccountid`,
        type: `string`
      },

      monthly_budget: {
        sql: `monthly_budget`,
        type: `number`
      },
    },
    
    measures: {
      count: {
        type: `count`
      },

      total_billed_cost: {
        sql: `billedcost`,
        type: `sum`,
        title: `Total Billed Cost`
      },

      max_monthly_budget: {
        sql: `MAX(${CUBE}.monthly_budget)`,
        type: `number`,
        title: `Max Monthly Budget`
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

      monthly_budget_drift_value: {
        sql: `(
          SUM(CASE WHEN ${view_dim_time.chargeperiodstart} >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
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
        sql: `COALESCE(
            100.0 * (
                SUM(CASE WHEN ${view_dim_time.chargeperiodstart} >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
                - (
                    MAX(${CUBE}.monthly_budget) * 
                    EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
                    NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP)), 0)
                )
            ) / NULLIF(
                MAX(${CUBE}.monthly_budget) * 
                EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
                NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP)), 0),
                0
            ),
            0
        )`,
        type: `number`,
        title: `Monthly Budget Drift Percentage`
    },                         

      quarterly_budget_drift_value: {
        sql: `(
          SUM(CASE WHEN ${view_dim_time.chargeperiodstart} >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
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
        sql: `COALESCE(
            100.0 * (
                SUM(CASE WHEN ${view_dim_time.chargeperiodstart} >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
                - (
                    MAX(${CUBE}.monthly_budget) * 3 * 
                    EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
                    NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP)), 0)
                )
            ) / NULLIF(
                MAX(${CUBE}.monthly_budget) * 3 * 
                EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
                NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP)), 0),
                0
            ),
            0
        )`,
        type: `number`,
        title: `Quarterly Budget Drift Percentage`
    },                           

      yearly_budget_drift_value: {
        sql: `(
          SUM(CASE WHEN ${view_dim_time.chargeperiodstart} >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
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
        sql: `COALESCE(
            100.0 * (
                SUM(CASE WHEN ${view_dim_time.chargeperiodstart} >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
                - (
                    MAX(${CUBE}.monthly_budget) * 12 * 
                    EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
                    NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0)
                )
            ) / NULLIF(
                MAX(${CUBE}.monthly_budget) * 12 * 
                EXTRACT(DAY FROM CURRENT_TIMESTAMP) / 
                NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0),
                0
            ),
            0
        )`,
        type: `number`,
        title: `Yearly Budget Drift Percentage`
    },  

      monthly_budget_utilization_actual_value: {
        sql: `SUM(
          CASE 
            WHEN DATE_TRUNC('month', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('month', CURRENT_DATE)
            THEN ${CUBE}.billedcost
            ELSE 0
          END
        )`,
        type: `number`,
        title: `Monthly Budget Utilized (Actual Value)`
      },

      monthly_budget_utilization_percentage: {
        sql: `COALESCE(
            100.0 * SUM(
                CASE 
                    WHEN DATE_TRUNC('month', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('month', CURRENT_DATE)
                    THEN ${CUBE}.billedcost
                    ELSE 0
                END
            ) / NULLIF(MAX(${CUBE}.monthly_budget), 0),
            0
        )`,
        type: `number`,
        title: `Monthly Budget Utilized (Percentage)`
    },               
          

      quarterly_budget_utilization_actual_value: {
        sql: `SUM(
          CASE 
            WHEN DATE_TRUNC('quarter', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('quarter', CURRENT_DATE)
            THEN ${CUBE}.billedcost
            ELSE 0
          END
        )`,
        type: `number`,
        title: `Quarterly Budget Utilized (Actual Value)`
      },

      quarterly_budget_utilization_percentage: {
        sql: `COALESCE(
            100.0 * SUM(
                CASE 
                    WHEN DATE_TRUNC('quarter', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('quarter', CURRENT_DATE)
                    THEN ${CUBE}.billedcost
                    ELSE 0
                END
            ) / NULLIF(MAX(${CUBE}.monthly_budget) * 3, 0),
            0
        )`,
        type: `number`,
        title: `Quarterly Budget Utilized (Percentage)`
    },
  
      yearly_budget_utilization_actual_value: {
        sql: `SUM(
          CASE 
            WHEN DATE_TRUNC('year', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('year', CURRENT_DATE)
            THEN ${CUBE}.billedcost
            ELSE 0
          END
        )`,
        type: `number`,
        title: `Yearly Budget Utilized (Actual Value)`
      },

      yearly_budget_utilization_percentage: {
        sql: `COALESCE(
            100.0 * SUM(
                CASE 
                    WHEN DATE_TRUNC('year', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('year', CURRENT_DATE)
                    THEN ${CUBE}.billedcost
                    ELSE 0
                END
            ) / NULLIF(MAX(${CUBE}.monthly_budget) * 12, 0),
            0
        )`,
        type: `number`,
        title: `Yearly Budget Utilized (Percentage)`
    },         

      forecast_next_month_cost: {
        sql: `
          (
            SUM(
              CASE 
                WHEN DATE_TRUNC('month', CURRENT_DATE) = DATE_TRUNC('month', ${view_dim_time.chargeperiodstart})
                THEN ${CUBE}.billedcost
                ELSE 0
              END
            ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('month', CURRENT_DATE)) + 1, 0)  -- Days passed in the current month
          ) * 30  -- Projected over the next month
        `,
        type: `number`,
        title: `Forecasted Cost for the Next Month`
      },

      forecast_next_quarter_cost: {
        sql: `
          (
            SUM(
              CASE 
                WHEN DATE_TRUNC('quarter', CURRENT_DATE) = DATE_TRUNC('quarter', ${view_dim_time.chargeperiodstart})
                THEN ${CUBE}.billedcost
                ELSE 0
              END
            ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('quarter', CURRENT_DATE)) + 1, 0)
          ) * 90  -- Adjusted to 92 to account for the typical length of a quarter
        `,
        type: `number`,
        title: `Forecasted Cost for the Next Quarter`
      },

      forecast_next_year_cost: {
        sql: `
          (
            SUM(
              CASE 
                WHEN DATE_TRUNC('year', CURRENT_DATE) = DATE_TRUNC('year', ${view_dim_time.chargeperiodstart})
                THEN ${CUBE}.billedcost
                ELSE 0
              END
            ) / NULLIF(DATE_PART('day', CURRENT_DATE - DATE_TRUNC('year', CURRENT_DATE)) + 1, 0)  -- Days passed in the current year
          ) * 365  -- Projected over the next year
        `,
        type: `number`,
        title: `Forecasted Cost for the Next Year`
      },

      month_to_date_cost: {
        sql: `billedcost`,
        type: `sum`,
        filters: [
          { sql: `${view_dim_time.chargeperiodstart} >= DATE_TRUNC('month', CURRENT_TIMESTAMP)` }
        ],
        title: `Month-to-Date Cost`
      },

      quarter_to_date_cost: {
        sql: `billedcost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${view_dim_time.chargeperiodstart} >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP)` }
        ],
        title: `Quarter-to-Date Cost`
      },

      year_to_date_cost: {
        sql: `billedcost::numeric`,
        type: `sum`,
        filters: [
          { sql: `${view_dim_time.chargeperiodstart} >= DATE_TRUNC('year', CURRENT_TIMESTAMP)` }
        ],
        title: `Year-to-Date Cost`
      },


      //tags measures
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
          SUM(CASE WHEN ${CUBE}.charge_period_date >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
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
          SUM(CASE WHEN ${CUBE}.charge_period_date >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
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
          SUM(CASE WHEN ${CUBE}.charge_period_date >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
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
        sql: `COALESCE(
            100.0 * (
                SUM(CASE WHEN ${CUBE}.charge_period_date >= DATE_TRUNC('month', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
                - (
                    ${COMPILE_CONTEXT.securityContext.tagsBudget} / 12 *
                    EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
                    NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP)), 0)
                )
            ) / NULLIF(
                ${COMPILE_CONTEXT.securityContext.tagsBudget} / 12 *
                EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
                NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_TIMESTAMP) + INTERVAL '1 month' - INTERVAL '1 day') - DATE_TRUNC('month', CURRENT_TIMESTAMP)), 0),
                0
            ),
            0
        )`,
        type: `number`,
        title: `Monthly Budget Drift Percentage`
    },
    tags_quarterly_budget_drift_percentage: {
      sql: `COALESCE(
          100.0 * (
              SUM(CASE WHEN ${CUBE}.charge_period_date >= DATE_TRUNC('quarter', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
              - (
                  ${COMPILE_CONTEXT.securityContext.tagsBudget} * 3 *
                  EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
                  NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP)), 0)
              )
          ) / NULLIF(
              ${COMPILE_CONTEXT.securityContext.tagsBudget} * 3 *
              EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
              NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('quarter', CURRENT_TIMESTAMP) + INTERVAL '3 months' - INTERVAL '1 day') - DATE_TRUNC('quarter', CURRENT_TIMESTAMP)), 0),
              0
          ),
          0
      )`,
      type: `number`,
      title: `Quarterly Budget Drift Percentage`
    },
    tags_yearly_budget_drift_percentage: {
      sql: `COALESCE(
        100.0 * (
            SUM(CASE WHEN ${CUBE}.charge_period_date >= DATE_TRUNC('year', CURRENT_TIMESTAMP) THEN ${CUBE}.billedcost ELSE 0 END)
            - (
                ${COMPILE_CONTEXT.securityContext.tagsBudget} *
                EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
                NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0)
            )
         ) / NULLIF(
            ${COMPILE_CONTEXT.securityContext.tagsBudget} *
            EXTRACT(DAY FROM CURRENT_TIMESTAMP) /
            NULLIF(EXTRACT(DAY FROM (DATE_TRUNC('year', CURRENT_TIMESTAMP) + INTERVAL '1 year' - INTERVAL '1 day') - DATE_TRUNC('year', CURRENT_TIMESTAMP)), 0),
            0
          ),
         0
      )`,
      type: `number`,
      title: `Yearly Budget Drift Percentage`
      },
      tags_monthly_budget_utilization_actual_value: {
        sql: `SUM(
          CASE
            WHEN DATE_TRUNC('month', ${view_dim_time.chargeperiodstart} = DATE_TRUNC('month', CURRENT_DATE)
            THEN ${CUBE}.billedcost
            ELSE 0
          END
        )`,
        type: `number`,
        title: `Monthly Budget Utilized (Actual Value)`
      },
      tags_quarterly_budget_utilization_actual_value: {
        sql: `SUM(
          CASE
            WHEN DATE_TRUNC('quarter', ${view_dim_time.chargeperiodstart} = DATE_TRUNC('quarter', CURRENT_DATE)
            THEN ${CUBE}.billedcost
            ELSE 0
          END
        )`,
        type: `number`,
        title: `Quarterly Budget Utilized (Actual Value)`
      },
      tags_yearly_budget_utilization_actual_value: {
        sql: `SUM(
          CASE
            WHEN DATE_TRUNC('year', ${view_dim_time.chargeperiodstart} = DATE_TRUNC('year', CURRENT_DATE)
            THEN ${CUBE}.billedcost
            ELSE 0
          END
        )`,
        type: `number`,
        title: `Yearly Budget Utilized (Actual Value)`
      },
      //utilization queries
      tags_yearly_budget_utilization_percentage: {
        sql: `COALESCE(
            SUM(
                CASE
                    WHEN DATE_TRUNC('year', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('year', CURRENT_DATE)
                    THEN ${CUBE}.billedcost
                    ELSE 0
                END
            ) / NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget}, 0) * 100,
            0
        )`,
        type: `number`,
        title: `Yearly Budget Utilization (%)`
    },
    tags_monthly_budget_utilization_percentage: {
      sql: `COALESCE(
          SUM(
              CASE
                  WHEN DATE_TRUNC('month', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('month', CURRENT_DATE)
                  THEN ${CUBE}.billedcost
                  ELSE 0
              END
          ) / NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget} / 12, 0) * 100,
          0
      )`,
      type: `number`,
      title: `Monthly Budget Utilization (%)`
      },

      tags_quarterly_budget_utilization_percentage: {
        sql: `COALESCE(
            SUM(
                CASE
                    WHEN DATE_TRUNC('quarter', ${view_dim_time.chargeperiodstart}) = DATE_TRUNC('quarter', CURRENT_DATE)
                    THEN ${CUBE}.billedcost
                    ELSE 0
                END
            ) / NULLIF(${COMPILE_CONTEXT.securityContext.tagsBudget} / 4, 0) * 100,
            0
        )`,
        type: `number`,
        title: `Quarterly Budget Utilization (%)`
        },
      },
    
    pre_aggregations: {

    }
  });
  