const { securityContext } = COMPILE_CONTEXT

cube(`aws_tags`, {
  sql_table: `${COMPILE_CONTEXT.securityContext.schemaName}.gold_aws_tags`,
  
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
    
    aws_cloudformation_logical_id: {
      sql: `aws_cloudformation_logical_id`,
      type: `string`
    },
    
    aws_elasticfilesystem_default_backup: {
      sql: `aws_elasticfilesystem_default_backup`,
      type: `string`
    },
    
    user_r: {
      sql: `user_r`,
      type: `string`
    },
    
    user_eks_cluster_name: {
      sql: `user_eks_cluster_name`,
      type: `string`
    },
    
    user_kubernetes_io_cluster_d_q_cluster_demo: {
      sql: `user_kubernetes_io_cluster_d_q_cluster_demo`,
      type: `string`
    },
    
    user_k8s_io_cluster_autoscaler_d_q_cluster_demo: {
      sql: `user_k8s_io_cluster_autoscaler_d_q_cluster_demo`,
      type: `string`
    },
    
    user_node_k8s_amazonaws_com_instance_id: {
      sql: `user_node_k8s_amazonaws_com_instance_id`,
      type: `string`
    },
    
    user_environment: {
      sql: `user_environment`,
      type: `string`
    },
    
    aws_eks_cluster_name: {
      sql: `aws_eks_cluster_name`,
      type: `string`
    },
    
    aws_ec2launchtemplate_version: {
      sql: `aws_ec2launchtemplate_version`,
      type: `string`
    },
    
    user_kubernetes_io_service_name: {
      sql: `user_kubernetes_io_service_name`,
      type: `string`
    },
    
    user_awscodestar_project_arn: {
      sql: `user_awscodestar_project_arn`,
      type: `string`
    },
    
    user_sagemaker_project_name: {
      sql: `user_sagemaker_project_name`,
      type: `string`
    },
    
    // user_redshift: {
    //   sql: `user_redshift`,
    //   type: `string`
    // },
    
    user_managed_by_amazon_sage_maker_resource: {
      sql: `user_managed_by_amazon_sage_maker_resource`,
      type: `string`
    },
    
    user_owner: {
      sql: `user_owner`,
      type: `string`
    },
    
    aws_autoscaling_group_name: {
      sql: `aws_autoscaling_group_name`,
      type: `string`
    },
    
    aws_cloudformation_stack_id: {
      sql: `aws_cloudformation_stack_id`,
      type: `string`
    },
    
    user_created_by: {
      sql: `user_created_by`,
      type: `string`
    },
    
    aws_created_by: {
      sql: `aws_created_by`,
      type: `string`
    },
    
    aws_ec2launchtemplate_id: {
      sql: `aws_ec2launchtemplate_id`,
      type: `string`
    },
    
    user_k8s_io_cluster_autoscaler_enabled: {
      sql: `user_k8s_io_cluster_autoscaler_enabled`,
      type: `string`
    },
    
    aws_cloudformation_stack_name: {
      sql: `aws_cloudformation_stack_name`,
      type: `string`
    },
    
    user_name: {
      sql: `user_name`,
      type: `string`
    },
    
    aws_ec2_fleet_id: {
      sql: `aws_ec2_fleet_id`,
      type: `string`
    },
    
    user_terraform: {
      sql: `user_terraform`,
      type: `string`
    },
    
    user_cluster_k8s_amazonaws_com_name: {
      sql: `user_cluster_k8s_amazonaws_com_name`,
      type: `string`
    },
    
    user_eks_nodegroup_name: {
      sql: `user_eks_nodegroup_name`,
      type: `string`
    },
    
    user_sagemaker_project_id: {
      sql: `user_sagemaker_project_id`,
      type: `string`
    },

    //  team: {
    //    sql: `team`,
    //    type: `string`
    //  },
    
    //  product: {
    //    sql: `product`,
    //    type: `string`
    //  }
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
