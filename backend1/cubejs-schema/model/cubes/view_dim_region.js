const { securityContext } = COMPILE_CONTEXT

cube(`view_dim_region`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_dim_region`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      regionid: {
        sql: `regionid`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      regionname: {
        sql: `regionname`,
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
  