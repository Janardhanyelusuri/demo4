const { securityContext } = COMPILE_CONTEXT

cube(`view_dim_time`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.view_dim_time`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      billingperiodstart: {
        sql: `billingperiodstart`,
        type: `time`,
        primaryKey: true,
        public: true,
      },
      
      billingperiodend: {
        sql: `billingperiodend`,
        type: `time`
      },

      chargeperiodstart: {
        sql: `chargeperiodstart`,
        type: `time`
      },

      chargeperiodend: {
        sql: `chargeperiodend`,
        type: `time`
      },
    },
    
    measures: {
      count: {
        type: `count`
      },

      earliest_charge_period_date: {
        sql: `MIN(${CUBE}.chargeperiodstart)`,
        type: `time`,
        title: `Earliest Charge Period Date`
      },  
          
      latest_charge_period_date: {
        sql: `MAX(${CUBE}.chargeperiodstart)`,
        type: `time`,
        title: `Latest Charge Period Date`
      }, 
    },
    
    pre_aggregations: {

    }
  });
  