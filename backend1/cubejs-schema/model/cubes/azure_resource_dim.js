const { securityContext } = COMPILE_CONTEXT

cube(`azure_resource_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_azure_resource_dim`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      resource_id: {
        sql: `resource_id`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      resource_name: {
        sql: `resource_name`,
        type: `string`
      },
      
      region_id: {
        sql: `region_id`,
        type: `string`
      },
      
      region_name: {
        sql: `region_name`,
        type: `string`
      },

      service_category: {
        sql: `service_category`,
        type: `string`
      },

      service_name: {
        sql: `service_name`,
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
  