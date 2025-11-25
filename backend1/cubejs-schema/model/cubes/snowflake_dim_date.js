cube(`snowflake_dim_date`, {
  sql_table: `gold.snowflake_dim_date`,
  
  data_source: `default`,
  
  joins: {
    
  },
  
  dimensions: {
    date: {
      sql: `date`,
      type: `time`,
      primaryKey: true,
      public: true,
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
