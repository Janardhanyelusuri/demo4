const { securityContext } = COMPILE_CONTEXT

cube(`azure_charge_summary_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_azure_charge_summary_dim`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      sku_id: {
        sql: `sku_id`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      charge_category: {
        sql: `charge_category`,
        type: `string`
      },
      
      charge_class: {
        sql: `charge_class`,
        type: `string`
      },
      
      charge_description: {
        sql: `charge_description`,
        type: `string`
      },

      charge_frequency: {
        sql: `charge_frequency`,
        type: `string`
      },

      x_sku_description: {
        sql: `x_sku_description`,
        type: `string`
      },

    },
    
    measures: {
      count: {
        type: `count`
      }
    },
    
    pre_aggregations: {
    
      // Pre-aggregation definitions go here.
      // Learn more in the documentation: https://cube.dev/docs/caching/pre-aggregations/getting-started
    }
  });
  