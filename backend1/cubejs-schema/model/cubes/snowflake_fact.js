cube(`snowflake_fact`, {
  sql_table: `gold.snowflake_fact`,
  
  data_source: `default`,
  
  joins: {
    snowflake_dim_date: {
      relationship: `many_to_one`,
      sql: `${CUBE}.date = ${snowflake_dim_date.date}`
    },
    snowflake_dim_account: {
      relationship: `many_to_one`,
      sql: `${CUBE}.account_name = ${snowflake_dim_account.account_name}`
    }
    
  },
  
  dimensions: {
    account_name: {
      sql: `${CUBE}."account_name"`,
      type: `string`,
      primaryKey: true,
      public: true,
    },
    
    usage_type: {
      sql: `${CUBE}."usage_type"`,
      type: `string`
    },
    
    balance_source: {
      sql: `${CUBE}."balance_source"`,
      type: `string`
    },

    usage_in_currency: {
      sql: `${CUBE}."usage_in_currency"`,
      type: `string`
    },
    
    rating_type: {
      sql: `${CUBE}."rating_type"`,
      type: `string`
    },

    service_type: {
      sql: `${CUBE}."service_type"`,
      type: `string`
    },

    
    
    date: {
      sql: `date`,
      type: `time`,
      primaryKey: true,
      public: true,
    }
  },
  
  measures: {
    totalcost: {
      sql: `usage_in_currency`,
      type: `sum`
    },
    count: {
      type: `count`
    }
  },
  
  pre_aggregations: {
    // Pre-aggregation definitions go here.
    // Learn more in the documentation: https://cube.dev/docs/caching/pre-aggregations/getting-started
  }
});
