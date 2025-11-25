const { securityContext } = COMPILE_CONTEXT

cube(`aws_billing_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_aws_billing_dim`,
    
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
      
      billing_account_name: {
        sql: `billing_account_name`,
        type: `string`
      },
      
      sub_account_id: {
        sql: `sub_account_id`,
        type: `number`
      },

      sub_account_name: {
        sql: `sub_account_name`,
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
  