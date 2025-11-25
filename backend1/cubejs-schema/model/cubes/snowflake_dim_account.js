cube(`snowflake_dim_account`, {
  sql_table: `gold.snowflake_dim_account`,
  
  data_source: `default`,
  
  joins: {
    
  },
  
  dimensions: {
    account_name: {
      sql: `${CUBE}."account_name"`,
      type: `string`,
      primaryKey: true,
      public: true,
    },
    
    contract_number: {
      sql: `${CUBE}."contract_number"`,
      type: `string`
    },
    
    account_locator: {
      sql: `${CUBE}."account_locator"`,
      type: `string`
    },
    
    service_level: {
      sql: `${CUBE}."service_level"`,
      type: `string`
    },
    
    organization_name: {
      sql: `${CUBE}."organization_name"`,
      type: `string`
    },
    
    region: {
      sql: `${CUBE}."region"`,
      type: `string`
    }
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
