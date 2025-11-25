const { securityContext } = COMPILE_CONTEXT

cube(`azure_account_dim`, {
  sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_azure_account_dim`,

  data_source: `default`,

  joins: {},

  dimensions: {
    sub_account_id: {
      sql: `sub_account_id`,
      type: `string`,
      primaryKey: true,
      public: true,
    },

    sub_account_name: {
      sql: `sub_account_name`,
      type: `string`,
    },

    sub_account_type: {
      sql: `sub_account_type`,
      type: `string`,
    },

    x_account_id: {
      sql: `x_account_id`,
      type: `string`,
    },

    x_account_name: {
      sql: `x_account_name`,
      type: `string`,
    },

    x_account_owner_id: {
      sql: `x_account_owner_id`,
      type: `string`,
    },

    x_billing_profile_id: {
      sql: `x_billing_profile_id`,
      type: `string`,
    },

    x_billing_profile_name: {
      sql: `x_billing_profile_name`,
      type: `string`,
    },

    billing_account_id: {
      sql: `billing_account_id`,
      type: `string`,
    },

    billing_account_name: {
      sql: `billing_account_name`,
      type: `string`,
    },

    billing_account_type: {
      sql: `billing_account_type`,
      type: `string`,
    },
  },

  measures: {
    count: {
      type: `count`,
    },
  },

  pre_aggregations: {
    // Pre-aggregation definitions go here.
    // Learn more in the documentation: https://cube.dev/docs/caching/pre-aggregations/getting-started
  },
});
