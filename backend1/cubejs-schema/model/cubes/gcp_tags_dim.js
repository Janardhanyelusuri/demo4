const { securityContext } = COMPILE_CONTEXT

cube(`gcp_tags_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_gcp_tags_dim`,
    
    data_source: `default`,
    
    joins: {
   
    },
    
    dimensions: {
      tags_key: {
        sql: `tags_key`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      x_inherited: {
        sql: `x_inherited`,
        type: `string`
      },
      
      x_type: {
        sql: `x_type`,
        type: `string`
      },

      x_namespace: {
        sql: `x_namespace`,
        type: `string`
      },
      
      value: {
        sql: `value`,
        type: `string`
      },

      key: {
        sql: `key`,
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
  