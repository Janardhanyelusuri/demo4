const { securityContext } = COMPILE_CONTEXT

cube(`azure_tags_dim`, {
    sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_azure_tags_dim`,
    
    data_source: `default`,
    
    joins: {
      
    },
    
    dimensions: {
      tags_key: {
        sql: `tags_key`,
        type: `string`,
        primaryKey: true,
        public: true,
      },
      
      datadrift: {
        sql: `DataDrift`,
        type: `string`
      },
      
      ml_workspace_link_update_time: {
        sql: `MLWorkspaceLinkUpdateTime`,
        type: `string`
      },
      
      aks_managed_pool_name: {
        sql: `aks-managed-poolName`,
        type: `string`
      },

      aks_managed_orchestrator: {
        sql: `aks-managed-orchestrator`,
        type: `string`
      },

      aks_managed_kubelet_identity_client_id: {
        sql: `aks-managed-kubeletIdentityClientID`,
        type: `string`
      },

    },
    
    measures: {
      count: {
        type: `count`
      },

      distinctAksManagedPoolNameCount: {
        sql: `aks-managed-poolName`,
        type: `countDistinct`,
        title: `Distinct AKS Managed Pool Names`
      },
    },
    
    pre_aggregations: {
    
      // Pre-aggregation definitions go here.
      // Learn more in the documentation: https://cube.dev/docs/caching/pre-aggregations/getting-started
    }
  });
  