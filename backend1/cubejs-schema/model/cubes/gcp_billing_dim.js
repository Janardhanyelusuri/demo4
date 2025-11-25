const { securityContext } = COMPILE_CONTEXT

cube(`gcp_billing_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_gcp_billing_dim`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      billing_account_id: {
        sql: `billing_account_id`,
        type: `number`,
        primaryKey: true,
        public: true,
      },
      
      sub_account_id: {
        sql: `sub_account_id`,
        type: `number`
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
  