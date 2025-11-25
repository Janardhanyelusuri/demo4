const { securityContext } = COMPILE_CONTEXT

cube(`view_dim_resource`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_dim_resource`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      resourceid: {
        sql: `resourceid`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      resourcename: {
        sql: `resourcename`,
        type: `string`
      },

      resourcetype: {
        sql: `resourcetype`,
        type: `string`
      },
    },
    
    measures: {
      count: {
        type: `count`
      },
    },
    
    pre_aggregations: {

    }
  });
  