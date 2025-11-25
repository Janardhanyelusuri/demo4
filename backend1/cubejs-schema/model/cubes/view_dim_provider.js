const { securityContext } = COMPILE_CONTEXT

cube(`view_dim_provider`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_dim_provider`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      providername: {
        sql: `providername`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      publishername: {
        sql: `publishername`,
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
  