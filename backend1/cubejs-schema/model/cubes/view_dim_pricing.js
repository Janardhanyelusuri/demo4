const { securityContext } = COMPILE_CONTEXT

cube(`view_dim_pricing`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_dim_pricing`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      pricingcategory: {
        sql: `pricingcategory`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      pricingunit: {
        sql: `pricingunit`,
        type: `string`
      },
      
      contractedunitprice: {
        sql: `contractedunitprice`,
        type: `number`
      },

      listunitprice: {
        sql: `listunitprice`,
        type: `number`
      },

      pricingquantity: {
        sql: `pricingquantity`,
        type: `number`
      },

    //   ml_workspace_link_update_time: {
    //     sql: `MLWorkspaceLinkUpdateTime`,
    //     type: `string`
    //   },
    },
    
    measures: {
      count: {
        type: `count`
      },
    },
    
    pre_aggregations: {

    }
  });
  