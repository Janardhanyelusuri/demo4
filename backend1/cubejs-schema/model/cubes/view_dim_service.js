const { securityContext } = COMPILE_CONTEXT

cube(`view_dim_service`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_dim_service`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      servicename: {
        sql: `servicename`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      servicecategory: {
        sql: `servicecategory`,
        type: `string`
      },

      skuid: {
        sql: `skuid`,
        type: `string`
      },

      skupriceid: {
        sql: `skupriceid`,
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
  