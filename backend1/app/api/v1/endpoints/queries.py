import json
from fastapi import APIRouter, HTTPException
import httpx
import os
from dotenv import load_dotenv
from app.models.project import Project
from app.models.tags import Tag
from app.models.dashboard import Dashboard
from app.models.resources_tags import ResourceTag
from app.schemas.connection import QueriesRequest, GenerateRecommendationRequest
import jwt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Load environment variables from .env file
load_dotenv()

queriesrouter = APIRouter()

CUBEJS_API_URL = os.getenv("CUBEJS_API_URL", "http://localhost:4000/cubejs-api/v1")
CUBEJS_API_SECRET = os.getenv("CUBEJS_API_SECRET")


@queriesrouter.post("/queries")
async def post_tagging_data(payload: QueriesRequest):
    if payload.cloud_provider:
        if payload.cloud_provider not in ["aws", "gcp", "azure"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid cloud provider. Only aws, azure, and gcp are supported.",
            )
    elif payload.project_id:  # Cloud provider is required only for project dashboards
        raise HTTPException(
            status_code=400,
            detail="cloud_provider is required when project_id is specified.",
        )

    schema_name = ""
    tags_budget = ""
    start_date = ""
    today = datetime.today()

    if payload.duration:
        try:
            duration = payload.duration

            if duration == "today":
                start_date = today
                end_date = today

            elif duration == "yesterday":
                start_date = today - timedelta(days=1)
                end_date = start_date

            elif duration == "last_7_days":
                start_date = today - timedelta(days=7)
                end_date = today

            elif duration == "last_30_days":
                start_date = today - timedelta(days=30)
                end_date = today

            elif duration == "last_90_days":
                start_date = today - timedelta(days=90)
                end_date = today

            elif duration == "this_month":
                start_date = today.replace(day=1)
                end_date = today

            elif duration == "last_month":
                first_day_this_month = today.replace(day=1)
                start_date = first_day_this_month - relativedelta(months=1)
                end_date = first_day_this_month - timedelta(days=1)

            elif duration == "this_week":
                start_date = today - timedelta(days=today.weekday())  # Monday
                end_date = today

            elif duration == "last_week":
                start_of_this_week = today - timedelta(days=today.weekday())
                start_date = start_of_this_week - timedelta(weeks=1)
                end_date = start_of_this_week - timedelta(days=1)

            elif duration == "this_year":
                start_date = today.replace(month=1, day=1)
                end_date = today

            elif duration == "last_year":
                start_date = today.replace(year=today.year - 1, month=1, day=1)
                end_date = today.replace(year=today.year - 1, month=12, day=31)

            # Format to ISO strings for API
            start_date_str = start_date.strftime("%Y-%m-%dT00:00:00.000")
            end_date_str = end_date.strftime("%Y-%m-%dT23:59:59.999")

        except Exception as e:
            print(f"Error while setting date range: {e}")

    if payload.project_id:
        try:
            obj = await Project.filter(id=payload.project_id).first()
            if not obj:
                raise HTTPException(status_code=404, detail="Project not found.")
            schema_name = obj.name
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Error fetching project: {ex}")

    if payload.tag_id:
        try:
            obj = await Tag.filter(tag_id=payload.tag_id).first()
            if obj:
                tags_budget = obj.budget
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Error fetching tag: {ex}")

    # Handle dashboard_id-specific logic
    if payload.dashboard_id:
        try:
            obj = await Dashboard.filter(id=payload.dashboard_id).first()
            if not obj:
                raise HTTPException(status_code=404, detail="Dashboard not found.")
            schema_name = obj.name  # Overwrite schema_name if both IDs are present
        except Exception as ex:
            raise HTTPException(
                status_code=500, detail=f"Error fetching dashboard: {ex}"
            )

    # Generate the JWT token
    token_payload = {"schemaName": schema_name, "tagsBudget": tags_budget}
    print(token_payload)
    token = jwt.encode(token_payload, CUBEJS_API_SECRET, algorithm="HS256")

    headers = {
        "Authorization": f"Bearer {token}",
    }

    # Placeholder for handling Cube.js queries (implementation not shown in original code)
    query_type = payload.query_type
    granularity = payload.granularity

    # Validate query_type
    if query_type not in [
        "storage_month_to_date_cost",
        "storage_quarter_to_date_cost",
        "storage_year_to_date_cost",
        "aws_cost_by_bucket",
        "aws_cost_by_storageclass",
        "aws_cost_by_time_storage",
        "rds_month_to_date_cost",
        "rds_quarter_to_date_cost",
        "rds_year_to_date_cost",
        "aws_cost_by_region_rds",
        "aws_cost_by_instance_rds",
        "aws_cost_by_instance_ecr",
        "aws_cost_by_time_rds",
        "ecc_month_to_date_cost",
        "ecc_quarter_to_date_cost",
        "ecc_year_to_date_cost",
        "aws_cost_by_region_ecc",
        "aws_cost_by_instance_ecc",
        "aws_cost_by_time_ecc",
        "aws_cost_by_service_category",
        "aws_dynamic_services_cost",
        "aws_month_to_date_list_cost",
        "aws_monthly_budget",
        "aws_quarter_to_date_list_cost",
        "aws_quarterly_budget",
        "aws_year_to_date_list_cost",
        "aws_yearly_budget",
        "aws_monthly_budget_drift",
        "aws_quarterly_budget_drift",
        "aws_yearly_budget_drift",
        "aws_cost_by_product",
        "aws_cost_by_time",
        "aws_total_list_cost_card",
        "aws_budgets_data",
        "ecs_month_to_date_cost",
        "load_balancing_month_to_date_cost",
        "vpc_month_to_date_cost",
        "cloud_watch_month_to_date_cost",
        "kms_month_to_date_cost",
        "cost_explorer_month_to_date_cost",
        "ecr_month_to_date_cost",
        "secret_manager_month_to_date_cost",
        "ecs_quarter_to_date_cost",
        "load_balancing_quarter_to_date_cost",
        "vpc_quarter_to_date_cost",
        "cloud_watch_quarter_to_date_cost",
        "kms_quarter_to_date_cost",
        "cost_explorer_quarter_to_date_cost",
        "ecr_quarter_to_date_cost",
        "secret_manager_quarter_to_date_cost",
        "ecs_year_to_date_cost",
        "load_balancing_year_to_date_cost",
        "vpc_year_to_date_cost",
        "cloud_watch_year_to_date_cost",
        "kms_year_to_date_cost",
        "cost_explorer_year_to_date_cost",
        "ecr_year_to_date_cost",
        "secret_manager_year_to_date_cost",
        "aws_cost_by_time_ecs",
        "aws_cost_by_time_kms",
        "aws_cost_by_time_load_balancing",
        "aws_cost_by_time_cost_explorer",
        "aws_cost_by_time_vpc",
        "aws_cost_by_time_ecr",
        "aws_cost_by_time_cloud_watch",
        "aws_cost_by_time_secret_manager",
        "aws_cost_by_region_ecs",
        "aws_cost_by_region_kms",
        "aws_cost_by_region_load_balancing",
        "aws_cost_by_region_cost_explorer",
        "aws_cost_by_region_vpc",
        "aws_cost_by_region_ecr",
        "aws_cost_by_region_cloud_watch",
        "aws_cost_by_region_secret_manager",
        "azure_all_services_cost",
        "azure_dynamic_services_cost",
        "aws_all_services_cost",
        "aws_service_name_cost",
        "aws_service_category_name_cost",
        "aws_cost_by_region",
        "aws_year_to_date_list_cost",
        "aws_quarter_to_date_list_cost",
        "aws_month_to_date_list_cost",
        "aws_cost_by_instance_cloud_watch",
        "aws_cost_by_instance_load_balancing",
        "aws_cost_by_instance_vpc",
        "aws_resource_id_cost",
        "aws_cost_by_region_s3",
        "aws_cost_by_instance_ecs",
        "aws_cost_by_instance_kms",
        "aws_cost_by_instance_cost_explorer",
        "azure_services_billed_cost",
        "aws_services_list_cost",
        "azure_resources_quarter_to_date_cost",
        "aws_services_cost_by_instance",
        "gcp_services_billed_cost",
        # AZURE
        "azure_service_name_cost_tags",
        "aws_services_year_to_date_cost",
        "aws_services_quarter_to_date_cost",
        "aws_services_month_to_date_cost",
        "aws_services_cost_by_time",
        "aws_services_cost_by_region",
        "azure_resource_name_cost",
        "azure_total_billed_cost",
        "azure_services_cost_by_instance",
        "azure_service_name_cost",
        "azure_sku_meter_name_cost",
        "azure_max_monthly_budget",
        "azure_month_to_date_cost",
        "azure_monthly_budget",
        "azure_quarter_to_date_cost",
        "azure_quarterly_budget",
        "azure_year_to_date_cost",
        "azure_yearly_budget",
        "azure_total_cost_card",
        "azure_monthly_budget_drift",
        "azure_yearly_budget_drift",
        "azure_quarterly_budget_drift",
        "azure_sigmoid_devops_month_to_date_cost",
        "azure_nat_gateway_month_to_date_cost",
        "azure_dq_demo_workspace_month_to_date_cost",
        "azure_dq_demo_workspace_quarter_to_date_cost",
        "azure_sigmoid_devops_quarter_to_date_cost",
        "azure_nat_gateway_quarter_to_date_cost",
        "azure_dq_demo_workspace_year_to_date_cost",
        "azure_sigmoid_devops_year_to_date_cost",
        "azure_nat_gateway_year_to_date_cost",
        "azure_effective_list_cost_resource",
        "azure_cost_by_region",
        "azure_cost_by_service_category",
        "azure_effective_list_cost_service_category",
        "azure_cost_trends_over_time",
        "azure_vm_year_to_date_cost",
        "azure_vm_quarter_to_date_cost",
        "azure_vm_month_to_date_cost",
        "azure_databricks_month_to_date_cost",
        "azure_databricks_year_to_date_cost",
        "azure_databricks_quarter_to_date_cost",
        "azure_nat_year_to_date_cost",
        "azure_nat_month_to_date_cost",
        "azure_nat_quarter_to_date_cost",
        "azure_services_year_to_date_cost",
        "azure_services_quarter_to_date_cost",
        "azure_services_month_to_date_cost",
        "azure_services_cost_by_time",
        "azure_services_cost_by_sku",
        "azure_services_cost_by_region",
        "azure_vs_year_to_date_cost",
        "azure_db_for_postgres_year_to_date_cost",
        "azure_container_registry_year_to_date_cost",
        "azure_vnet_year_to_date_cost",
        "azure_ml_year_to_date_cost",
        "azure_vmss_year_to_date_cost",
        "azure_monitor_year_to_date_cost",
        "azure_load_balancer_year_to_date_cost",
        "azure_vs_quarter_to_date_cost",
        "azure_db_for_postgres_quarter_to_date_cost",
        "azure_container_registry_quarter_to_date_cost",
        "azure_vnet_quarter_to_date_cost",
        "azure_ml_quarter_to_date_cost",
        "azure_vmss_quarter_to_date_cost",
        "azure_monitor_quarter_to_date_cost",
        "azure_load_balancer_quarter_to_date_cost",
        "azure_vs_month_to_date_cost",
        "azure_db_for_postgres_month_to_date_cost",
        "azure_container_registry_month_to_date_cost",
        "azure_vnet_month_to_date_cost",
        "azure_ml_month_to_date_cost",
        "azure_vmss_month_to_date_cost",
        "azure_monitor_month_to_date_cost",
        "azure_load_balancer_month_to_date_cost",
        "azure_cost_by_time_nat",
        "azure_cost_by_time_vm",
        "azure_cost_by_time_databricks",
        "azure_cost_by_region_vm",
        "azure_cost_by_region_nat",
        "azure_cost_by_region_databricks",
        "azure_cost_by_instance_databricks",
        "azure_cost_by_instance_nat",
        "azure_cost_by_instance_vm",
        "gcp_charge_period_dates",
        "aws_charge_period_dates",
        "azure_cost_by_time_vs",
        "azure_cost_by_time_db_for_postgres",
        "azure_cost_by_time_container_registry",
        "azure_cost_by_time_vnet",
        "azure_cost_by_time_ml",
        "azure_cost_by_time_vmss",
        "azure_cost_by_time_monitor",
        "azure_cost_by_time_load_balancer",
        "azure_cost_by_region_vs",
        "azure_cost_by_region_db_for_postgres",
        "azure_cost_by_region_container_registry",
        "azure_cost_by_region_vnet",
        "azure_cost_by_region_ml",
        "azure_cost_by_region_vmss",
        "azure_cost_by_region_monitor",
        "azure_cost_by_region_load_balancer",
        "azure_cost_by_instance_vs",
        "azure_cost_by_instance_db_for_postgres",
        "azure_cost_by_instance_container_registry",
        "azure_cost_by_instance_vnet",
        "azure_cost_by_instance_ml",
        "azure_cost_by_instance_vmss",
        "azure_cost_by_instance_monitor",
        "azure_cost_by_instance_load_balancer",
        "azure_monthly_budget_utilization",
        "azure_total_consumed_quantity",
        "azure_configuration_drift",
        "azure_critical_resource_alert",
        "azure_effective_cost_per_unit",
        "azure_idle_resource_cost",
        "azure_cost_variance",
        "azure_forecast_next_year_cost",
        "azure_forecast_next_quarter_cost",
        "azure_forecast_next_month_cost",
        "azure_yearly_budget_utilization",
        "azure_quarterly_budget_utilization",
        "azure_cost_by_time_services",
        "azure_nat_combined_cost",
        "azure_databricks_combined_cost",
        "azure_vm_combined_cost",
        "azure_vnet_combined_cost",
        "azure_db_for_postgres_combined_cost",
        "azure_container_registry_combined_cost",
        "azure_ml_combined_cost",
        "azure_vmss_combined_cost",
        "azure_budgets",
        "azure_budgets_drifts",
        "azure_budgets_utilizations",
        "azure_budgets_data",
        "azure_costs_data",
        # cost by sku azure
        "azure_cost_by_sku_vs",
        "azure_cost_by_sku_db_for_postgres",
        "azure_cost_by_sku_container_registry",
        "azure_cost_by_sku_vnet",
        "azure_cost_by_sku_ml",
        "azure_cost_by_sku_vmss",
        "azure_cost_by_sku_monitor",
        "azure_cost_by_sku_load_balancer",
        "azure_cost_by_sku_vm",
        "azure_cost_by_sku_nat",
        "azure_cost_by_sku_databricks",
        "azure_budgets_data_tags",
        "azure_resource_name_cost_tags",
        "azure_tags_cost_by_region",
        "azure_tags_cost_by_service_category",
        "azure_tags_cost_by_resource_group_name",
        "azure_tags_sku_meter_name_cost",
        "azure_cost_trends_over_time_tags",
        # GCP
        "gcp_total_billed_cost",
        "gcp_service_name_cost",
        "gcp_provider_name_cost",
        "gcp_month_to_date_cost",
        "gcp_resource_name_cost",
        "gcp_monthly_budget",
        "gcp_quarter_to_date_cost",
        "gcp_quarterly_budget",
        "gcp_yearly_budget",
        "gcp_cost_by_region",
        "gcp_total_cost_card",
        "gcp_year_to_date_cost",
        "gcp_monthly_budget_drift",
        "gcp_yearly_budget_drift",
        "gcp_quarterly_budget_drift",
        "gcp_list_cost_trends_over_quarter",
        "gcp_list_cost_trends_over_year",
        "gcp_list_cost_trends_over_month",
        "gcp_budgets_data",
        "gcp_cost_trends_over_time",
        "gcp_total_list_cost",
        "gcp_cost_by_time",
        "gcp_all_services_cost",
        "gcp_service_name_list_cost",
        "gcp_provider_name_list_cost",
        "gcp_resource_name_list_cost",
        "gcp_quarter_to_date_list_cost",
        "gcp_year_to_date_list_cost",
        "gcp_month_to_date_list_cost",
        "gcp_csql_year_to_date_cost",
        "gcp_csql_quarter_to_date_cost",
        "gcp_csql_month_to_date_cost",
        "gcp_cstorage_year_to_date_cost",
        "gcp_cstorage_quarter_to_date_cost",
        "gcp_cstorage_month_to_date_cost",
        "gcp_bq_year_to_date_cost",
        "gcp_bq_quarter_to_date_cost",
        "gcp_bq_month_to_date_cost",
        "gcp_areg_year_to_date_cost",
        "gcp_areg_quarter_to_date_cost",
        "gcp_areg_month_to_date_cost",
        "gcp_ce_month_to_date_cost",
        "gcp_ce_quarter_to_date_cost",
        "gcp_ce_year_to_date_cost",
        "gcp_kms_month_to_date_cost",
        "gcp_kms_quarter_to_date_cost",
        "gcp_kms_year_to_date_cost",
        "gcp_logging_month_to_date_cost",
        "gcp_logging_quarter_to_date_cost",
        "gcp_logging_year_to_date_cost",
        "gcp_networking_month_to_date_cost",
        "gcp_networking_quarter_to_date_cost",
        "gcp_networking_year_to_date_cost",
        "gcp_kubengine_month_to_date_cost",
        "gcp_kubengine_quarter_to_date_cost",
        "gcp_kubengine_year_to_date_cost",
        "gcp_ce_cost_by_time",
        "gcp_csql_cost_by_time",
        "gcp_ce_cost_by_region",
        "gcp_cstorage_cost_by_time",
        "gcp_areg_cost_by_time",
        "gcp_bq_cost_by_time",
        "gcp_kms_cost_by_time",
        "gcp_kubengine_cost_by_time",
        "gcp_networking_cost_by_time",
        "gcp_logging_cost_by_time",
        "gcp_cstorage_cost_by_region",
        "gcp_areg_cost_by_region",
        "gcp_bq_cost_by_region",
        "gcp_kms_cost_by_region",
        "gcp_kubengine_cost_by_region",
        "gcp_networking_cost_by_region",
        "gcp_logging_cost_by_region",
        "gcp_cstorage_cost_by_instance",
        "gcp_areg_cost_by_instance",
        "gcp_bq_cost_by_instance",
        "gcp_kms_cost_by_instance",
        "gcp_kubengine_cost_by_instance",
        "gcp_networking_cost_by_instance",
        "gcp_logging_cost_by_instance",
        "gcp_cost_by_service_category",
        "gcp_csql_cost_by_region",
        "gcp_ce_cost_by_instance",
        "gcp_csql_cost_by_instance",
        "azure_charge_period_dates",
        "gcp_cost_by_instance_cloud_sql",
        "gcp_cost_by_instance_kubernetes_engine",
        "gcp_cost_by_instance_compute_engine",
        "gcp_cost_by_instance_KMS",
        "gcp_cost_by_instance_networking",
        "tags_yearly_budget",
        "tags_azure_budgets_data",
        "gcp_dynamic_services_cost",
        "gcp_services_cost_by_instance",
        "gcp_services_year_to_date_cost",
        "gcp_services_month_to_date_cost",
        "gcp_services_quarter_to_date_cost",
        "gcp_services_cost_by_time",
        "gcp_services_cost_by_region",
        # consolidated
        "consolidated_budgets_data",
        "consolidated_cost_trends_over_time",
        "consolidated_service_name_cost",
        "consolidated_charge_period_dates",
        "consolidated_cost_by_service_category",
        "consolidated_cost_by_region",
        "consolidated_all_services_cost",
        "azure_sku_id_cost",
        "aws_tags_cost_by_service_category",
        "aws_cost_trends_over_time_tags",
        "tags_aws_budgets_data",
        "aws_tags_cost_by_region",
        "aws_resource_id_cost_tags",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid query_type. Must be 'percentage_tagged_resources' or 'untagged_resources'.",
        )

    resource_names = payload.resource_names
    # Convert resource_names to a list, if provided
    resource_list = resource_names.split(",") if resource_names else []

    # if resource name is not provided and tag id is provided, fetch resource list by querying the table
    if not resource_list:
        if payload.tag_id:
            try:
                # Fetch resources associated with the tag
                resource_tags = await ResourceTag.filter(
                    tag_id=payload.tag_id
                ).prefetch_related("resource")

                if not resource_tags:
                    pass
                    # raise HTTPException(status_code=404, detail="No resources found for this tag")

                # Extract the resource names from the related resources
                resource_list = [
                    resource_tag.resource.resource_name
                    for resource_tag in resource_tags
                ]
            except Exception as ex:
                print(ex)
    print("resource_list", resource_list)

    service_names = payload.service_names
    service_name = service_names.split(",") if service_names else []

    try:
        if query_type == "storage_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.storage_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "ecs_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.ecs_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "tags_yearly_budget":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.tags_yearly_budget"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "tags_azure_budgets_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.tags_yearly_budget",
                        "azure_fact_cost.tags_quarterly_budget",
                        "azure_fact_cost.tags_monthly_budget",
                        "azure_fact_cost.tags_monthly_budget_drift_percentage",
                        "azure_fact_cost.tags_quarterly_budget_drift_percentage",
                        "azure_fact_cost.tags_yearly_budget_drift_percentage",
                        "azure_fact_cost.tags_monthly_budget_utilization_actual_value",
                        "azure_fact_cost.tags_yearly_budget_utilization_actual_value",
                        "azure_fact_cost.tags_quarterly_budget_utilization_actual_value",
                        "azure_fact_cost.forecast_next_quarter_cost",
                        "azure_fact_cost.forecast_next_month_cost",
                        "azure_fact_cost.forecast_next_year_cost",
                        "azure_fact_cost.tags_yearly_budget_drift_value",
                        "azure_fact_cost.tags_quarterly_budget_drift_value",
                        "azure_fact_cost.tags_monthly_budget_drift_value",
                        "azure_fact_cost.tags_quarterly_budget_utilization_percentage",
                        "azure_fact_cost.tags_monthly_budget_utilization_percentage",
                        "azure_fact_cost.tags_yearly_budget_utilization_percentage",
                    ],
                    "filters": [],
                    "timeDimensions": [],
                }
            }

            # Add filters for resource names if provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.tags_monthly_budget", 0
                        )
                        or 0,
                        "quarterly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.tags_quarterly_budget", 0
                        )
                        or 0,
                        "yearly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.tags_yearly_budget", 0
                        )
                        or 0,
                        "monthly_budget_drift": sum(
                            item.get(
                                "azure_fact_cost.tags_monthly_budget_drift_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_drift": sum(
                            item.get(
                                "azure_fact_cost.tags_quarterly_budget_drift_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_drift": sum(
                            item.get(
                                "azure_fact_cost.tags_yearly_budget_drift_percentage", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "monthly_budget_actual_utilization": sum(
                            item.get(
                                "azure_fact_cost.tags_monthly_budget_utilization_actual_value",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_actual_utilization": sum(
                            item.get(
                                "azure_fact_cost.tags_quarterly_budget_utilization_actual_value",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_actual_utilization": sum(
                            item.get(
                                "azure_fact_cost.tags_yearly_budget_utilization_actual_value",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "forecast_next_month_cost": sum(
                            item.get("azure_fact_cost.forecast_next_month_cost", 0) or 0
                            for item in data.get("data", [])
                        ),
                        "forecast_next_quarter_cost": sum(
                            item.get("azure_fact_cost.forecast_next_quarter_cost", 0)
                            or 0
                            for item in data.get("data", [])
                        ),
                        "forecast_next_year_cost": sum(
                            item.get("azure_fact_cost.forecast_next_year_cost", 0) or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_actual_drift": sum(
                            item.get(
                                "azure_fact_cost.tags_yearly_budget_drift_value", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_actual_drift": sum(
                            item.get(
                                "azure_fact_cost.tags_quarterly_budget_drift_value", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "monthly_budget_actual_drift": sum(
                            item.get(
                                "azure_fact_cost.tags_monthly_budget_drift_value", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "monthly_budget_utilization": sum(
                            item.get(
                                "azure_fact_cost.tags_monthly_budget_utilization_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_utilization": sum(
                            item.get(
                                "azure_fact_cost.tags_yearly_budget_utilization_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_utilization": sum(
                            item.get(
                                "azure_fact_cost.tags_quarterly_budget_utilization_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "tags_aws_budgets_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.tags_yearly_budget",
                        "aws_fact_focus.tags_quarterly_budget",
                        "aws_fact_focus.tags_monthly_budget",
                        "aws_fact_focus.tags_monthly_budget_drift_percentage",
                        "aws_fact_focus.tags_quarterly_budget_drift_percentage",
                        "aws_fact_focus.tags_yearly_budget_drift_percentage",
                        "aws_fact_focus.tags_monthly_budget_utilization_actual_value",
                        "aws_fact_focus.tags_yearly_budget_utilization_actual_value",
                        "aws_fact_focus.tags_quarterly_budget_utilization_actual_value",
                        "aws_fact_focus.forecast_next_quarter_cost",
                        "aws_fact_focus.forecast_next_month_cost",
                        "aws_fact_focus.forecast_next_year_cost",
                        "aws_fact_focus.tags_yearly_budget_drift_value",
                        "aws_fact_focus.tags_quarterly_budget_drift_value",
                        "aws_fact_focus.tags_monthly_budget_drift_value",
                        "aws_fact_focus.tags_quarterly_budget_utilization_percentage",
                        "aws_fact_focus.tags_monthly_budget_utilization_percentage",
                        "aws_fact_focus.tags_yearly_budget_utilization_percentage",
                    ],
                    "filters": [],
                }
            }

            # Add filters for resource names if provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "aws_fact_focus.resource_id",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            print(query)

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": data.get("data", [{}])[0].get(
                            "aws_fact_focus.tags_monthly_budget", 0
                        )
                        or 0,
                        "quarterly_budget": data.get("data", [{}])[0].get(
                            "aws_fact_focus.tags_quarterly_budget", 0
                        )
                        or 0,
                        "yearly_budget": data.get("data", [{}])[0].get(
                            "aws_fact_focus.tags_yearly_budget", 0
                        )
                        or 0,
                        "monthly_budget_drift": sum(
                            item.get(
                                "aws_fact_focus.tags_monthly_budget_drift_percentage", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_drift": sum(
                            item.get(
                                "aws_fact_focus.tags_quarterly_budget_drift_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_drift": sum(
                            item.get(
                                "aws_fact_focus.tags_yearly_budget_drift_percentage", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "monthly_budget_actual_utilization": sum(
                            item.get(
                                "aws_fact_focus.tags_monthly_budget_utilization_actual_value",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_actual_utilization": sum(
                            item.get(
                                "aws_fact_focus.tags_quarterly_budget_utilization_actual_value",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_actual_utilization": sum(
                            item.get(
                                "aws_fact_focus.tags_yearly_budget_utilization_actual_value",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "forecast_next_month_cost": sum(
                            item.get("aws_fact_focus.forecast_next_month_cost", 0) or 0
                            for item in data.get("data", [])
                        ),
                        "forecast_next_quarter_cost": sum(
                            item.get("aws_fact_focus.forecast_next_quarter_cost", 0)
                            or 0
                            for item in data.get("data", [])
                        ),
                        "forecast_next_year_cost": sum(
                            item.get("aws_fact_focus.forecast_next_year_cost", 0) or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_actual_drift": sum(
                            item.get("aws_fact_focus.tags_yearly_budget_drift_value", 0)
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_actual_drift": sum(
                            item.get(
                                "aws_fact_focus.tags_quarterly_budget_drift_value", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "monthly_budget_actual_drift": sum(
                            item.get(
                                "aws_fact_focus.tags_monthly_budget_drift_value", 0
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "monthly_budget_utilization": sum(
                            item.get(
                                "aws_fact_focus.tags_monthly_budget_utilization_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "yearly_budget_utilization": sum(
                            item.get(
                                "aws_fact_focus.tags_yearly_budget_utilization_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                        "quarterly_budget_utilization": sum(
                            item.get(
                                "aws_fact_focus.tags_quarterly_budget_utilization_percentage",
                                0,
                            )
                            or 0
                            for item in data.get("data", [])
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "load_balancing_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.load_balancing_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_cost_by_instance_secret_manager":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSSecretsManager"],
                        }
                    ],
                }
            }

        elif query_type == "vpc_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.vpc_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "cloud_watch_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.cloud_watch_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "kms_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.kms_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "cost_explorer_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.cost_explorer_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "ecr_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.ecr_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "secret_manager_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.secret_manager_month_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "ecs_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.ecs_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "load_balancing_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.load_balancing_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "vpc_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.vpc_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "cloud_watch_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.cloud_watch_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "kms_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.kms_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "cost_explorer_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.cost_explorer_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "ecr_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.ecr_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "secret_manager_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.secret_manager_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "ecs_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.ecs_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "load_balancing_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.load_balancing_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "vpc_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.vpc_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "cloud_watch_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.cloud_watch_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "kms_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.kms_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "cost_explorer_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.cost_explorer_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "ecr_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.ecr_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "secret_manager_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.secret_manager_year_to_date_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_total_list_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.total_list_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_total_list_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.total_list_cost"

            # Create the query to fetch the measure
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_effective_list_cost_resource":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "dimensions": [
                        "azure_resource_dim.resource_name",
                        "azure_fact_cost.effective_cost",
                        "azure_fact_cost.list_cost",
                    ],
                }
            }

        elif query_type == "azure_effective_list_cost_service_category":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "dimensions": [
                        "azure_resource_dim.service_category",
                        "azure_fact_cost.effective_cost",
                        "azure_fact_cost.list_cost",
                    ],
                }
            }

        elif query_type == "azure_cost_trends_over_time":
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.total_billed_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",  # or use the appropriate time dimension
                            "granularity": granularity,  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                }
            }

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "gcp_cost_trends_over_time":
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.total_list_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",  # or use the appropriate time dimension
                            "granularity": "month",  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                }
            }

        elif query_type == "gcp_list_cost_trends_over_month":
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.total_list_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",  # or use the appropriate time dimension
                            "granularity": "month",  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                }
            }

        elif query_type == "gcp_list_cost_trends_over_year":
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.total_list_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",  # or use the appropriate time dimension
                            "granularity": "year",  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                }
            }

        elif query_type == "gcp_list_cost_trends_over_quarter":
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.total_list_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",  # or use the appropriate time dimension
                            "granularity": "quarter",  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                }
            }

        elif query_type == "azure_dq_demo_workspace_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.dq_demo_workspace_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.dq_demo_workspace_month_to_date_cost"
                    ],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["dq-demo-workspace"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_gateway_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_gateway_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_gateway_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["nat-gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_sigmoid_devops_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.sigmoid_devops_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.sigmoid_devops_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["Sigmoid-DevOps"],
                        }
                    ],
                }
            }

        elif query_type == "azure_dq_demo_workspace_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.dq_demo_workspace_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.dq_demo_workspace_quarter_to_date_cost"
                    ],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["dq-demo-workspace"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_gateway_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_gateway_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_gateway_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["nat-gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_sigmoid_devops_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.sigmoid_devops_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.sigmoid_devops_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["Sigmoid-DevOps"],
                        }
                    ],
                }
            }

        elif query_type == "azure_dq_demo_workspace_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.dq_demo_workspace_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.dq_demo_workspace_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["dq-demo-workspace"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_gateway_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_gateway_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_gateway_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["nat-gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_sigmoid_devops_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.sigmoid_devops_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.sigmoid_devops_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["Sigmoid-DevOps"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_resource_name_cost":
            # Query for ResourceType and BilledCost
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                }
            }

        elif query_type == "gcp_resource_name_list_cost":
            # Query for ResourceType and BilledCost
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                }
            }

        elif query_type == "gcp_provider_name_cost":
            # Query for ResourceType and BilledCost
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.provider_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                }
            }

        elif query_type == "gcp_provider_name_list_cost":
            # Query for ResourceType and BilledCost
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.provider_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                }
            }

        elif query_type == "gcp_service_name_cost":
            # Query for ResourceType and BilledCost
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                }
            }

        elif query_type == "gcp_service_name_list_cost":
            # Query for ResourceType and BilledCost
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                }
            }

        elif query_type == "azure_resource_name_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_resource_id_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )

        elif query_type == "azure_service_name_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "timeDimensions": [],
                }
            }

            # Add time dimension with granularity if provided
            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_service_name_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )

        elif query_type == "azure_sku_meter_name_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                }
            }

        elif query_type == "aws_service_category_name_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.service_category"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )

        elif query_type == "azure_total_billed_cost":
            measure = "azure_fact_cost.total_billed_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_monthly_budget_utilization":
            measure = "azure_fact_cost.monthly_budget_utilization"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_quarterly_budget_utilization":
            measure = "azure_fact_cost.quarterly_budget_utilization"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_yearly_budget_utilization":
            measure = "azure_fact_cost.yearly_budget_utilization"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_forecast_next_month_cost":
            measure = "azure_fact_cost.forecast_next_month_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_forecast_next_quarter_cost":
            measure = "azure_fact_cost.forecast_next_quarter_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_forecast_next_year_cost":
            measure = "azure_fact_cost.forecast_next_year_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_cost_variance":
            measure = "azure_fact_cost.cost_variance"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_idle_resource_cost":
            measure = "azure_fact_cost.idle_resource_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_effective_cost_per_unit":
            measure = "azure_fact_cost.effective_cost_per_unit"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_configuration_drift":
            measure = "azure_fact_cost.configuration_drift"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_critical_resource_alert":
            measure = "azure_fact_cost.critical_resource_alert"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_total_consumed_quantity":
            measure = "azure_fact_cost.total_consumed_quantity"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_max_monthly_budget":
            measure = "azure_fact_cost.max_monthly_budget"
            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_quarter_to_date_cost":
            measure = "azure_fact_cost.quarter_to_date_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_quarter_to_date_cost":
            measure = "gcp_fact_dim.quarter_to_date_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_quarter_to_date_list_cost":
            measure = "gcp_fact_dim.quarter_to_date_list_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "storage_quarter_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_monthly_budget":
            # Query for untagged resources
            measure = "aws_fact_focus.max_monthly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_monthly_budget":
            # Query for untagged resources
            measure = "azure_fact_cost.max_monthly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_monthly_budget":
            # Query for untagged resources
            measure = "gcp_fact_dim.max_monthly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_monthly_budget_drift":
            # Query for untagged resources
            measure = "aws_fact_focus.monthly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_monthly_budget_drift":
            # Query for untagged resources
            measure = "gcp_fact_dim.monthly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_monthly_budget_drift":
            # Query for untagged resources
            measure = "azure_fact_cost.monthly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_quarterly_budget_drift":
            # Query for untagged resources
            measure = "aws_fact_focus.quarterly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_quarterly_budget_drift":
            # Query for untagged resources
            measure = "azure_fact_cost.quarterly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_quarterly_budget_drift":
            # Query for untagged resources
            measure = "gcp_fact_dim.quarterly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_yearly_budget_drift":
            # Query for untagged resources
            measure = "aws_fact_focus.yearly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_yearly_budget_drift":
            # Query for untagged resources
            measure = "azure_fact_cost.yearly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_yearly_budget_drift":
            # Query for untagged resources
            measure = "gcp_fact_dim.yearly_budget_drift"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_quarterly_budget":
            # Query for untagged resources
            measure = "aws_fact_focus.quarterly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_quarterly_budget":
            # Query for untagged resources
            measure = "azure_fact_cost.quarterly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_quarterly_budget":
            # Query for untagged resources
            measure = "gcp_fact_dim.quarterly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_yearly_budget":
            # Query for untagged resources
            measure = "aws_fact_focus.yearly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_yearly_budget":
            # Query for untagged resources
            measure = "azure_fact_cost.yearly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_yearly_budget":
            # Query for untagged resources
            measure = "gcp_fact_dim.yearly_budget"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_total_list_cost_card":
            # Query for untagged resources
            measure = "aws_fact_focus.total_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_total_list_cost_card":
            # Query for untagged resources
            measure = "gcp_fact_dim.total_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_total_cost_card":
            # Query for untagged resources
            measure = "azure_fact_cost.total_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_total_cost_card":
            # Query for untagged resources
            measure = "gcp_fact_dim.total_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_month_to_date_list_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.month_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_month_to_date_cost":
            # Query for untagged resources
            measure = "azure_fact_cost.month_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_month_to_date_list_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.month_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_quarter_to_date_list_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.quarter_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_year_to_date_list_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.year_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_month_to_date_cost":
            # Query for untagged resources
            measure = "gcp_fact_dim.month_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_month_to_date_list_cost":
            # Query for untagged resources
            measure = "gcp_fact_dim.month_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_year_to_date_list_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.year_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "azure_year_to_date_cost":
            # Query for untagged resources
            measure = "azure_fact_cost.year_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_year_to_date_cost":
            # Query for untagged resources
            measure = "gcp_fact_dim.year_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "gcp_year_to_date_list_cost":
            # Query for untagged resources
            measure = "gcp_fact_dim.year_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_quarter_to_date_list_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.quarter_to_date_list_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "ecc_month_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.ecc_month_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "ecc_quarter_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.ecc_quarter_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "ecc_year_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.ecc_year_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "rds_month_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.rds_month_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "storage_year_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.storage_year_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "rds_year_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.rds_year_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "rds_quarter_to_date_cost":
            # Query for untagged resources
            measure = "aws_fact_focus.rds_quarter_to_date_cost"

            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_cost_by_bucket":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_cost"],
                    "dimensions": ["aws_fact_focus.resource_name"],
                    "order": {"aws_fact_focus.total_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.product_code",
                            "operator": "equals",
                            "values": ["AmazonS3"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_service_category":
            # Query for untagged resources

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.service_category"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "timeDimensions": [],
                }
            }

        elif query_type == "gcp_cost_by_service_category":
            # Query for untagged resources

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.service_category"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "timeDimensions": [],
                }
            }

        elif query_type == "aws_cost_by_service_name":
            # Query for untagged resources

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "timeDimensions": [],
                }
            }

        elif query_type == "aws_cost_by_region_rds":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonRDS"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_kms":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["awskms"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_ecs":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonEKS"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_s3":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonS3"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_secret_manager":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSSecretsManager"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_vpc":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonVPC"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_ecr":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonECR"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_load_balancing":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSELB"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_cloud_watch":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonCloudWatch"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_cost_explorer":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSCostExplorer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_vm":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_vs":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_db_for_postgres":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_container_registry":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_vnet":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_ml":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_vmss":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_monitor":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_load_balancer":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_databricks":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region_nat":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_region_ecc":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonEC2"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_instance_ecc":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Amazon Elastic Compute Cloud"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_cost_by_instance_ecr":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonECR"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_instance_ecs":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonEKS"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_instance_kms":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["awskms"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_instance_rds":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonRDS"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_instance_cloud_watch":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["AmazonCloudWatch"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_cost_by_instance_load_balancing":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Elastic Load Balancing"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_cost_by_instance_cost_explorer":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSCostExplorer"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_instance_vpc":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Amazon Virtual Private Cloud"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_cost_by_storageclass":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonS3"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_rds":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonRDS"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                }
            }

        elif query_type == "gcp_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_ecc":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonEC2"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_storage":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonS3"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_ecs":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonEKS"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_kms":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["awskms"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_secret_manager":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSSecretsManager"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_vpc":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonVPC"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_ecr":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonECR"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_load_balancing":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSELB"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_cost_explorer":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AWSCostExplorer"],
                        }
                    ],
                }
            }

        elif query_type == "aws_cost_by_time_cloud_watch":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.x_service_code",
                            "operator": "equals",
                            "values": ["AmazonCloudWatch"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_nat":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_services":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                }
            }

        elif query_type == "azure_cost_by_time_vs":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_db_for_postgres":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_container_registry":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_vnet":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_ml":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_vmss":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_monitor":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_load_balancer":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_databricks":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_time_vm":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_region":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_cost_by_region":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )

        elif query_type == "gcp_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.region_name"],
                }
            }

        elif query_type == "azure_cost_by_service_category":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.service_category"],
                    "timeDimensions": [],
                }
            }

            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )

        elif query_type == "azure_nat_month_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_month_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.databricks_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vm_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vm_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_quarter_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.databricks_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vm_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vm_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vs_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vs_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vs_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                }
            }

        elif query_type == "azure_db_for_postgres_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.db_for_postgres_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.db_for_postgres_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_container_registry_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.container_registry_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.container_registry_year_to_date_cost"
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vnet_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vnet_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vnet_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_ml_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.ml_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.ml_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vmss_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vmss_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vmss_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_monitor_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.monitor_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.monitor_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_load_balancer_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.load_balancer_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.load_balancer_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_year_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.databricks_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vm_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vm_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_month_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_month_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.databricks_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vm_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vm_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_quarter_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.databricks_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vm_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vm_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.nat_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.nat_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_year_to_date_cost":
            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.databricks_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vm_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vm_year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_databricks":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_vnet":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_vmss":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_monitor":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_resource.resourcename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_load_balancer":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_vm":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.resource_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_nat":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_csql_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.csql_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.csql_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_csql_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.csql_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.csql_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_csql_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.csql_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.csql_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_networking_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.networking_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.networking_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_logging_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.logging_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.logging_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Logging"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kubengine_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.kubengine_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.kubengine_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kms_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.kms_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.kms_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_ce_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.ce_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.ce_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_ce_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.ce_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.ce_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_networking_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.networking_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.networking_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_logging_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.logging_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.logging_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Logging"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kubengine_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.kubengine_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.kubengine_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kms_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.kms_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.kms_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_ce_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.ce_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.ce_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_networking_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.networking_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.networking_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_logging_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.logging_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.logging_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Logging"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kubengine_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.kubengine_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.kubengine_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kms_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.kms_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.kms_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_ce_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_csql_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_bq_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["BigQuery"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cstorage_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Storage"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_areg_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Artifact Registry"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_logging_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Logging"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_networking_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kubengine_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kms_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": "day",
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_ce_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_csql_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cstorage_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Storage"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kubengine_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_areg_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Artifact Registry"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_networking_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_bq_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["BigQuery"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kms_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_logging_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_location_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Logging"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_ce_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_csql_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_bq_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["BigQuery"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cstorage_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Storage"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kubengine_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_logging_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Logging"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_areg_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Artifact Registry"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_networking_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_kms_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["CCloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_bq_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.bq_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.bq_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["BigQuery"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_bq_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.bq_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.bq_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["BigQuery"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_bq_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.bq_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.bq_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["BigQuery"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_areg_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.areg_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.areg_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Artifact Registry"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_areg_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.areg_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.areg_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Artifact Registry"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_areg_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.areg_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.areg_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Artifact Registry"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cstorage_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.cstorage_year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.cstorage_year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Storage"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cstorage_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.cstorage_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.cstorage_month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Storage"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cstorage_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.cstorage_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.cstorage_quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Storage"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vs_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vs_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vs_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                }
            }

        elif query_type == "azure_db_for_postgres_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.db_for_postgres_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.db_for_postgres_quarter_to_date_cost"
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_container_registry_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.container_registry_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.container_registry_quarter_to_date_cost"
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vnet_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vnet_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vnet_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_ml_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.ml_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.ml_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vmss_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vmss_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vmss_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_monitor_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.monitor_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.monitor_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_load_balancer_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.load_balancer_quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.load_balancer_quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vs_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vs_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vs_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                }
            }

        elif query_type == "azure_db_for_postgres_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.db_for_postgres_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.db_for_postgres_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_container_registry_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.container_registry_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.container_registry_month_to_date_cost"
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vnet_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vnet_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vnet_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_ml_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.ml_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.ml_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vmss_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.vmss_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.vmss_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_monitor_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.monitor_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.monitor_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_load_balancer_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.load_balancer_month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.load_balancer_month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_nat_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.nat_month_to_date_cost",
                        "azure_fact_cost.nat_quarter_to_date_cost",
                        "azure_fact_cost.nat_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_databricks_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.databricks_month_to_date_cost",
                        "azure_fact_cost.databricks_quarter_to_date_cost",
                        "azure_fact_cost.databricks_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vm_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.vm_month_to_date_cost",
                        "azure_fact_cost.vm_quarter_to_date_cost",
                        "azure_fact_cost.vm_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_db_for_postgres_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.db_for_postgres_month_to_date_cost",
                        "azure_fact_cost.db_for_postgres_quarter_to_date_cost",
                        "azure_fact_cost.db_for_postgres_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_container_registry_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.container_registry_month_to_date_cost",
                        "azure_fact_cost.container_registry_quarter_to_date_cost",
                        "azure_fact_cost.container_registry_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_ml_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.ml_month_to_date_cost",
                        "azure_fact_cost.ml_quarter_to_date_cost",
                        "azure_fact_cost.ml_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vmss_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.vmss_month_to_date_cost",
                        "azure_fact_cost.vmss_quarter_to_date_cost",
                        "azure_fact_cost.vmss_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_vnet_combined_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.vnet_month_to_date_cost",
                        "azure_fact_cost.vnet_quarter_to_date_cost",
                        "azure_fact_cost.vnet_year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_all_services_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.month_to_date_cost",
                        "azure_fact_cost.quarter_to_date_cost",
                        "azure_fact_cost.year_to_date_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "azure_resource_dim.service_name", ""
                            ),
                            "costs": {
                                "month_to_date": entry.get(
                                    "azure_fact_cost.month_to_date_cost", 0
                                ),
                                "quarter_to_date": entry.get(
                                    "azure_fact_cost.quarter_to_date_cost", 0
                                ),
                                "year_to_date": entry.get(
                                    "azure_fact_cost.year_to_date_cost", 0
                                ),
                            },
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_dynamic_services_cost":
            # Query for total billed cost for all services
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.total_billed_cost"
                    ],  # Only fetching the total billed cost
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "azure_resource_dim.service_name", ""
                            ),
                            "cost": entry.get("azure_fact_cost.total_billed_cost", 0),
                        }
                        formatted_data.append(formatted_entry)

                    # Sort by cost in descending order
                    formatted_data.sort(key=lambda x: x["cost"], reverse=True)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "aws_dynamic_services_cost":
            # Query for total billed cost for all services
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.total_list_cost"
                    ],  # Only fetching the total billed cost
                    "dimensions": ["aws_fact_focus.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "aws_fact_focus.service_name", ""
                            ),
                            "cost": entry.get("aws_fact_focus.total_list_cost", 0),
                        }
                        formatted_data.append(formatted_entry)

                    # Sort by cost in descending order
                    formatted_data.sort(key=lambda x: x["cost"], reverse=True)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "gcp_dynamic_services_cost":
            # Query for total billed cost for all services
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.total_list_cost"
                    ],  # Only fetching the total billed cost
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        # Ensure 'cost' is always a float, even if it's None or a string
                        total_cost = entry.get("gcp_fact_dim.total_list_cost")
                        formatted_entry = {
                            "service_name": entry.get("gcp_fact_dim.service_name", ""),
                            "cost": (
                                float(total_cost) if total_cost is not None else 0.0
                            ),  # Convert to float
                        }
                        formatted_data.append(formatted_entry)

                    # Sort by cost in descending order (ensuring all values are numbers)
                    formatted_data.sort(key=lambda x: x["cost"], reverse=True)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_services_billed_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.total_billed_cost",
                    ],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "azure_resource_dim.service_name", ""
                            ),
                            "total_billed_cost": entry.get(
                                "azure_fact_cost.total_billed_cost", 0
                            ),
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "gcp_services_billed_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.total_billed_cost",
                    ],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get("gcp_fact_dim.service_name", ""),
                            "total_billed_cost": entry.get(
                                "gcp_fact_dim.total_billed_cost", 0
                            ),
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "aws_all_services_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.month_to_date_list_cost",  # Correct measure name
                        "aws_fact_focus.quarter_to_date_list_cost",  # Correct measure name
                        "aws_fact_focus.year_to_date_list_cost",  # Correct measure name
                    ],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")

                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "aws_fact_focus.service_name", ""
                            ),
                            "costs": {
                                "month_to_date": entry.get(
                                    "aws_fact_focus.month_to_date_list_cost", 0
                                ),
                                "quarter_to_date": entry.get(
                                    "aws_fact_focus.quarter_to_date_list_cost", 0
                                ),
                                "year_to_date": entry.get(
                                    "aws_fact_focus.year_to_date_list_cost", 0
                                ),
                            },
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "gcp_all_services_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.month_to_date_list_cost",  # Correct measure name
                        "gcp_fact_dim.quarter_to_date_list_cost",  # Correct measure name
                        "gcp_fact_dim.year_to_date_list_cost",  # Correct measure name
                    ],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get("gcp_fact_dim.service_name", ""),
                            "costs": {
                                "month_to_date": entry.get(
                                    "gcp_fact_dim.month_to_date_list_cost", 0
                                ),
                                "quarter_to_date": entry.get(
                                    "gcp_fact_dim.quarter_to_date_list_cost", 0
                                ),
                                "year_to_date": entry.get(
                                    "gcp_fact_dim.year_to_date_list_cost", 0
                                ),
                            },
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "aws_services_list_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.total_list_cost",
                    ],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "filters": [],  # No filters to include all services
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "aws_fact_focus.service_name", ""
                            ),
                            "total_billed_cost": entry.get(
                                "aws_fact_focus.total_list_cost", 0
                            ),
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_budgets":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.max_monthly_budget",
                        "azure_fact_cost.quarterly_budget",
                        "azure_fact_cost.yearly_budget",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.max_monthly_budget", 0
                        ),
                        "quarterly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget", 0
                        ),
                        "yearly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget", 0
                        ),
                    }

                return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_budgets_drifts":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.monthly_budget_drift",
                        "azure_fact_cost.quarterly_budget_drift",
                        "azure_fact_cost.yearly_budget_drift",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.monthly_budget_drift", 0
                        ),
                        "quarterly_budget_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget_drift", 0
                        ),
                        "yearly_budget_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget_drift", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_budgets_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.max_monthly_budget",
                        "azure_fact_cost.quarterly_budget",
                        "azure_fact_cost.yearly_budget",
                        "azure_fact_cost.monthly_budget_drift_percentage",
                        "azure_fact_cost.quarterly_budget_utilization_percentage_value",
                        "azure_fact_cost.quarterly_budget_drift_percentage",
                        "azure_fact_cost.yearly_budget_drift_percentage",
                        "azure_fact_cost.monthly_budget_utilization_actual_value",
                        "azure_fact_cost.yearly_budget_utilization_actual_value",
                        "azure_fact_cost.monthly_budget_utilization_percentage_value",
                        "azure_fact_cost.quarterly_budget_utilization_actual_value",
                        "azure_fact_cost.yearly_budget_utilization_percentage_value",
                        "azure_fact_cost.forecast_next_quarter_cost",
                        "azure_fact_cost.forecast_next_month_cost",
                        "azure_fact_cost.forecast_next_year_cost",
                        "azure_fact_cost.yearly_budget_drift_value",
                        "azure_fact_cost.quarterly_budget_drift_value",
                        "azure_fact_cost.monthly_budget_drift_value",
                    ],
                    "timeDimensions": [],
                }
            }

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.max_monthly_budget", 0
                        ),
                        "quarterly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget", 0
                        ),
                        "yearly_budget": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget", 0
                        ),
                        "monthly_budget_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.monthly_budget_drift_percentage", 0
                        ),
                        "quarterly_budget_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget_drift_percentage", 0
                        ),
                        "yearly_budget_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget_drift_percentage", 0
                        ),
                        "monthly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "azure_fact_cost.monthly_budget_utilization_actual_value", 0
                        ),
                        "quarterly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "azure_fact_cost.quarterly_budget_utilization_actual_value",
                            0,
                        ),
                        "yearly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "azure_fact_cost.yearly_budget_utilization_actual_value", 0
                        ),
                        "monthly_budget_utilization": data.get("data", [{}])[0].get(
                            "azure_fact_cost.monthly_budget_utilization_percentage_value",
                            0,
                        ),
                        "quarterly_budget_utilization": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget_utilization_percentage_value",
                            0,
                        ),
                        "yearly_budget_utilization": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget_utilization_percentage_value",
                            0,
                        ),
                        "forecast_next_month_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.forecast_next_month_cost", 0
                        ),
                        "forecast_next_quarter_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.forecast_next_quarter_cost", 0
                        ),
                        "forecast_next_year_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.forecast_next_year_cost", 0
                        ),
                        "yearly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget_drift_value", 0
                        ),
                        "quarterly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget_drift_value", 0
                        ),
                        "monthly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "azure_fact_cost.monthly_budget_drift_value", 0
                        ),
                    }

                    return {
                        "message": "Success",
                        "data": formatted_data,
                        "description": "Azure budget information including monthly, quarterly, and yearly budgets, utilization metrics, drift percentages, and cost forecasts",
                    }

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_charge_period_dates":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.earliest_charge_period_date",
                        "azure_fact_cost.latest_charge_period_date",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "earliest_charge_period_date": data.get("data", [{}])[0].get(
                            "azure_fact_cost.earliest_charge_period_date", 0
                        ),
                        "latest_charge_period_date": data.get("data", [{}])[0].get(
                            "azure_fact_cost.latest_charge_period_date", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "aws_charge_period_dates":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.earliest_charge_period_date",
                        "aws_fact_focus.latest_charge_period_date",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "earliest_charge_period_date": data.get("data", [{}])[0].get(
                            "aws_fact_focus.earliest_charge_period_date", 0
                        ),
                        "latest_charge_period_date": data.get("data", [{}])[0].get(
                            "aws_fact_focus.latest_charge_period_date", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "gcp_charge_period_dates":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.earliest_charge_period_date",
                        "gcp_fact_dim.latest_charge_period_date",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "earliest_charge_period_date": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.earliest_charge_period_date", 0
                        ),
                        "latest_charge_period_date": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.latest_charge_period_date", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "aws_budgets_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.max_monthly_budget",
                        "aws_fact_focus.quarterly_budget",
                        "aws_fact_focus.yearly_budget",
                        "aws_fact_focus.monthly_budget_drift_percentage",
                        "aws_fact_focus.quarterly_budget_utilization_percentage_value",
                        "aws_fact_focus.quarterly_budget_drift_percentage",
                        "aws_fact_focus.yearly_budget_drift_percentage",
                        "aws_fact_focus.monthly_budget_utilization_actual_value",
                        "aws_fact_focus.yearly_budget_utilization_actual_value",
                        "aws_fact_focus.monthly_budget_utilization_percentage_value",
                        "aws_fact_focus.quarterly_budget_utilization_actual_value",
                        "aws_fact_focus.yearly_budget_utilization_percentage_value",
                        "aws_fact_focus.forecast_next_quarter_cost",
                        "aws_fact_focus.forecast_next_month_cost",
                        "aws_fact_focus.forecast_next_year_cost",
                        "aws_fact_focus.yearly_budget_drift_value",
                        "aws_fact_focus.quarterly_budget_drift_value",
                        "aws_fact_focus.monthly_budget_drift_value",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": data.get("data", [{}])[0].get(
                            "aws_fact_focus.max_monthly_budget", 0
                        ),
                        "quarterly_budget": data.get("data", [{}])[0].get(
                            "aws_fact_focus.quarterly_budget", 0
                        ),
                        "yearly_budget": data.get("data", [{}])[0].get(
                            "aws_fact_focus.yearly_budget", 0
                        ),
                        "monthly_budget_drift": data.get("data", [{}])[0].get(
                            "aws_fact_focus.monthly_budget_drift_percentage", 0
                        ),
                        "quarterly_budget_drift": data.get("data", [{}])[0].get(
                            "aws_fact_focus.quarterly_budget_drift_percentage", 0
                        ),
                        "yearly_budget_drift": data.get("data", [{}])[0].get(
                            "aws_fact_focus.yearly_budget_drift_percentage", 0
                        ),
                        "monthly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "aws_fact_focus.monthly_budget_utilization_actual_value", 0
                        ),
                        "quarterly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "aws_fact_focus.quarterly_budget_utilization_actual_value",
                            0,
                        ),
                        "yearly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "aws_fact_focus.yearly_budget_utilization_actual_value", 0
                        ),
                        "monthly_budget_utilization": data.get("data", [{}])[0].get(
                            "aws_fact_focus.monthly_budget_utilization_percentage_value",
                            0,
                        ),
                        "quarterly_budget_utilization": data.get("data", [{}])[0].get(
                            "aws_fact_focus.quarterly_budget_utilization_percentage_value",
                            0,
                        ),
                        "yearly_budget_utilization": data.get("data", [{}])[0].get(
                            "aws_fact_focus.yearly_budget_utilization_percentage_value",
                            0,
                        ),
                        "forecast_next_month_cost": data.get("data", [{}])[0].get(
                            "aws_fact_focus.forecast_next_month_cost", 0
                        ),
                        "forecast_next_quarter_cost": data.get("data", [{}])[0].get(
                            "aws_fact_focus.forecast_next_quarter_cost", 0
                        ),
                        "forecast_next_year_cost": data.get("data", [{}])[0].get(
                            "aws_fact_focus.forecast_next_year_cost", 0
                        ),
                        "yearly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "aws_fact_focus.yearly_budget_drift_value", 0
                        ),
                        "quarterly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "aws_fact_focus.quarterly_budget_drift_value", 0
                        ),
                        "monthly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "aws_fact_focus.monthly_budget_drift_value", 0
                        ),
                    }

                    return {
                        "message": "Success",
                        "data": formatted_data,
                        "description": "AWS budget information including monthly, quarterly, and yearly budgets, utilization metrics, drift percentages, and cost forecasts",
                    }

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "gcp_budgets_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "gcp_fact_dim.max_monthly_budget",
                        "gcp_fact_dim.quarterly_budget",
                        "gcp_fact_dim.yearly_budget",
                        "gcp_fact_dim.monthly_budget_drift_percentage",
                        "gcp_fact_dim.quarterly_budget_utilization_percentage_value",
                        "gcp_fact_dim.quarterly_budget_drift_percentage",
                        "gcp_fact_dim.yearly_budget_drift_percentage",
                        "gcp_fact_dim.monthly_budget_utilization_actual_value",
                        "gcp_fact_dim.yearly_budget_utilization_actual_value",
                        "gcp_fact_dim.monthly_budget_utilization_percentage_value",
                        "gcp_fact_dim.quarterly_budget_utilization_actual_value",
                        "gcp_fact_dim.yearly_budget_utilization_percentage_value",
                        "gcp_fact_dim.forecast_next_quarter_cost",
                        "gcp_fact_dim.forecast_next_month_cost",
                        "gcp_fact_dim.forecast_next_year_cost",
                        "gcp_fact_dim.yearly_budget_drift_value",
                        "gcp_fact_dim.quarterly_budget_drift_value",
                        "gcp_fact_dim.monthly_budget_drift_value",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.max_monthly_budget", 0
                        ),
                        "quarterly_budget": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.quarterly_budget", 0
                        ),
                        "yearly_budget": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.yearly_budget", 0
                        ),
                        "monthly_budget_drift": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.monthly_budget_drift_percentage", 0
                        ),
                        "quarterly_budget_drift": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.quarterly_budget_drift_percentage", 0
                        ),
                        "yearly_budget_drift": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.yearly_budget_drift_percentage", 0
                        ),
                        "monthly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "gcp_fact_dim.monthly_budget_utilization_actual_value", 0
                        ),
                        "quarterly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get(
                            "gcp_fact_dim.quarterly_budget_utilization_actual_value", 0
                        ),
                        "yearly_budget_actual_utilization": data.get("data", [{}])[
                            0
                        ].get("gcp_fact_dim.yearly_budget_utilization_actual_value", 0),
                        "monthly_budget_utilization": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.monthly_budget_utilization_percentage_value",
                            0,
                        ),
                        "quarterly_budget_utilization": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.quarterly_budget_utilization_percentage_value",
                            0,
                        ),
                        "yearly_budget_utilization": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.yearly_budget_utilization_percentage_value", 0
                        ),
                        "forecast_next_month_cost": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.forecast_next_month_cost", 0
                        ),
                        "forecast_next_quarter_cost": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.forecast_next_quarter_cost", 0
                        ),
                        "forecast_next_year_cost": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.forecast_next_year_cost", 0
                        ),
                        "yearly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.yearly_budget_drift_value", 0
                        ),
                        "quarterly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.quarterly_budget_drift_value", 0
                        ),
                        "monthly_budget_actual_drift": data.get("data", [{}])[0].get(
                            "gcp_fact_dim.monthly_budget_drift_value", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_budgets_utilizations":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.monthly_budget_utilization",
                        "azure_fact_cost.quarterly_budget_utilization",
                        "azure_fact_cost.yearly_budget_utilization",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget_utilization": data.get("data", [{}])[0].get(
                            "azure_fact_cost.monthly_budget_utilization", 0
                        ),
                        "quarterly_budget_utilization": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarterly_budget_utilization", 0
                        ),
                        "yearly_budget_utilization": data.get("data", [{}])[0].get(
                            "azure_fact_cost.yearly_budget_utilization", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_costs_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.month_to_date_cost",
                        "azure_fact_cost.quarter_to_date_cost",
                        "azure_fact_cost.year_to_date_cost",
                        "azure_fact_cost.total_billed_cost",
                    ]
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "month_to_date_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.month_to_date_cost", 0
                        ),
                        "quarter_to_date_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.quarter_to_date_cost", 0
                        ),
                        "year_to_date_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.year_to_date_cost", 0
                        ),
                        "total_billed_cost": data.get("data", [{}])[0].get(
                            "azure_fact_cost.total_billed_cost", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "aws_monthly_budget_utilization":
            measure = "aws_fact_focus.monthly_budget_utilization"
            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_quarterly_budget_utilization":
            measure = "aws_fact_focus.quarterly_budget_utilization"
            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_yearly_budget_utilization":
            measure = "aws_fact_focus.yearly_budget_utilization"
            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_forecast_next_month_cost":
            measure = "aws_fact_focus.forecast_next_month_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_forecast_next_quarter_cost":
            measure = "aws_fact_focus.forecast_next_quarter_cost"
            query = {"query": {"measures": [measure]}}

        elif query_type == "aws_forecast_next_year_cost":
            measure = "aws_fact_focus.forecast_next_year_cost"
            query = {"query": {"measures": [measure]}}

        # aditya
        elif query_type == "gcp_cost_by_instance_cloud_sql":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Cloud SQL"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "gcp_cost_by_instance_kubernetes_engine":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Kubernetes Engine"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "gcp_cost_by_instance_compute_engine":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Compute Engine"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "gcp_cost_by_instance_KMS":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": ["Cloud Key Management Service (KMS)"],
                        }
                    ],
                }
            }

        elif query_type == "gcp_cost_by_instance_networking":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Networking"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_cost_by_sku_vm":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machines"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_nat":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure NAT Gateway"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_databricks":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Databricks"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_vs":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_ml":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_vmss":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Machine Scale Sets"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_vnet":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_monitor":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Monitor"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_container_registry":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_sku_db_for_postgres":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_resources_quarter_to_date_cost":
            # Create the query to fetch the measure for multiple resources
            query = {
                "query": {
                    "measures": ["azure_fact_cost.quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "filters": [],
                }
            }

            # Add filters for resource names if provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_cost_by_sku_load_balancer":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Load Balancer"],
                        }
                    ],
                }
            }

        elif query_type == "azure_budgets_data_tags":
            # Query for monthly, quarterly, and yearly budgets with resource filter
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.max_monthly_budget",
                        "azure_fact_cost.quarterly_budget",
                        "azure_fact_cost.yearly_budget",
                        "azure_fact_cost.monthly_budget_drift_percentage",
                        "azure_fact_cost.quarterly_budget_utilization_percentage_value",
                        "azure_fact_cost.quarterly_budget_drift_percentage",
                        "azure_fact_cost.yearly_budget_drift_percentage",
                        "azure_fact_cost.monthly_budget_utilization_actual_value",
                        "azure_fact_cost.yearly_budget_utilization_actual_value",
                        "azure_fact_cost.monthly_budget_utilization_percentage_value",
                        "azure_fact_cost.quarterly_budget_utilization_actual_value",
                        "azure_fact_cost.yearly_budget_utilization_percentage_value",
                        "azure_fact_cost.forecast_next_quarter_cost",
                        "azure_fact_cost.forecast_next_month_cost",
                        "azure_fact_cost.forecast_next_year_cost",
                        "azure_fact_cost.yearly_budget_drift_value",
                        "azure_fact_cost.quarterly_budget_drift_value",
                        "azure_fact_cost.monthly_budget_drift_value",
                    ],
                    "filters": [],
                }
            }

            # Add filters for resource names if provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    total_monthly_budget = sum(
                        item.get("azure_fact_cost.max_monthly_budget", 0)
                        for item in data.get("data", [])
                    )
                    total_quarterly_budget = sum(
                        item.get("azure_fact_cost.quarterly_budget", 0)
                        for item in data.get("data", [])
                    )
                    total_yearly_budget = sum(
                        item.get("azure_fact_cost.yearly_budget", 0)
                        for item in data.get("data", [])
                    )
                    total_monthly_budget_drift = sum(
                        item.get("azure_fact_cost.monthly_budget_drift_percentage", 0)
                        for item in data.get("data", [])
                    )
                    total_quarterly_budget_drift = sum(
                        item.get("azure_fact_cost.quarterly_budget_drift_percentage", 0)
                        for item in data.get("data", [])
                    )
                    total_yearly_budget_drift = sum(
                        item.get("azure_fact_cost.yearly_budget_drift_percentage", 0)
                        for item in data.get("data", [])
                    )
                    total_monthly_actual_utilization = sum(
                        item.get(
                            "azure_fact_cost.monthly_budget_utilization_actual_value", 0
                        )
                        for item in data.get("data", [])
                    )
                    total_quarterly_actual_utilization = sum(
                        item.get(
                            "azure_fact_cost.quarterly_budget_utilization_actual_value",
                            0,
                        )
                        for item in data.get("data", [])
                    )
                    total_yearly_actual_utilization = sum(
                        item.get(
                            "azure_fact_cost.yearly_budget_utilization_actual_value", 0
                        )
                        for item in data.get("data", [])
                    )
                    total_monthly_utilization = sum(
                        item.get(
                            "azure_fact_cost.monthly_budget_utilization_percentage_value",
                            0,
                        )
                        for item in data.get("data", [])
                    )
                    total_quarterly_utilization = sum(
                        item.get(
                            "azure_fact_cost.quarterly_budget_utilization_percentage_value",
                            0,
                        )
                        for item in data.get("data", [])
                    )
                    total_yearly_utilization = sum(
                        item.get(
                            "azure_fact_cost.yearly_budget_utilization_percentage_value",
                            0,
                        )
                        for item in data.get("data", [])
                    )
                    total_forecast_next_month_cost = sum(
                        item.get("azure_fact_cost.forecast_next_month_cost", 0)
                        for item in data.get("data", [])
                    )
                    total_forecast_next_quarter_cost = sum(
                        item.get("azure_fact_cost.forecast_next_quarter_cost", 0)
                        for item in data.get("data", [])
                    )
                    total_forecast_next_year_cost = sum(
                        item.get("azure_fact_cost.forecast_next_year_cost", 0)
                        for item in data.get("data", [])
                    )
                    total_yearly_budget_drift_value = sum(
                        item.get("azure_fact_cost.yearly_budget_drift_value", 0)
                        for item in data.get("data", [])
                    )
                    total_quarterly_budget_drift_value = sum(
                        item.get("azure_fact_cost.quarterly_budget_drift_value", 0)
                        for item in data.get("data", [])
                    )
                    total_monthly_budget_drift_value = sum(
                        item.get("azure_fact_cost.monthly_budget_drift_value", 0)
                        for item in data.get("data", [])
                    )

                    # Format the final response with summed up values
                    formatted_data = {
                        "monthly_budget": total_monthly_budget,
                        "quarterly_budget": total_quarterly_budget,
                        "yearly_budget": total_yearly_budget,
                        "monthly_budget_drift": total_monthly_budget_drift,
                        "quarterly_budget_drift": total_quarterly_budget_drift,
                        "yearly_budget_drift": total_yearly_budget_drift,
                        "monthly_budget_actual_utilization": total_monthly_actual_utilization,
                        "quarterly_budget_actual_utilization": total_quarterly_actual_utilization,
                        "yearly_budget_actual_utilization": total_yearly_actual_utilization,
                        "monthly_budget_utilization": total_monthly_utilization,
                        "quarterly_budget_utilization": total_quarterly_utilization,
                        "yearly_budget_utilization": total_yearly_utilization,
                        "forecast_next_month_cost": total_forecast_next_month_cost,
                        "forecast_next_quarter_cost": total_forecast_next_quarter_cost,
                        "forecast_next_year_cost": total_forecast_next_year_cost,
                        "yearly_budget_actual_drift": total_yearly_budget_drift_value,
                        "quarterly_budget_actual_drift": total_quarterly_budget_drift_value,
                        "monthly_budget_actual_drift": total_monthly_budget_drift_value,
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_resource_name_cost_tags":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [],
                    "timeDimensions": [],
                }
            }

            # Add filters for resource names if provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_service_name_cost_tags":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "timeDimensions": [],
                    "filters": [],
                }
            }

            # Add time dimension with granularity if provided
            if granularity:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "granularity": granularity,
                    }
                )

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_resource_id_cost_tags":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [],
                }
            }

            # Add filters for resource names if provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "aws_fact_focus.resource_id",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_tags_cost_by_region":
            # Create the base query
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [],
                    "timeDimensions": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_tags_cost_by_region":
            # Create the base query
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "aws_fact_focus.resource_id",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_cost_trends_over_time_tags":
            query = {
                "query": {
                    "measures": [
                        "azure_fact_cost.total_billed_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",  # or use the appropriate time dimension
                            "granularity": granularity,  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                    "filters": [],
                }
            }
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_cost_trends_over_time_tags":
            print(resource_list)
            query = {
                "query": {
                    "measures": [
                        "aws_fact_focus.total_list_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",  # or use the appropriate time dimension
                            "granularity": granularity,  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                    "filters": [],
                }
            }
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "aws_fact_focus.resource_id",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_tags_cost_by_service_category":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.service_category"],
                    "filters": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "aws_tags_cost_by_service_category":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.service_category"],
                    "filters": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "aws_fact_focus.resource_id",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_tags_cost_by_resource_group_name":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.resource_group_name"],
                    "filters": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_tags_sku_meter_name_cost":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

        elif query_type == "azure_services_cost_by_region":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.region_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                    "timeDimensions": [],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_services_cost_by_sku":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_fact_cost.sku_meter_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                    "timeDimensions": [],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_services_cost_by_time":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "azure_fact_cost.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_services_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.month_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                    "timeDimensions": [],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_services_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.quarter_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                    "timeDimensions": [],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_services_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "azure_fact_cost.year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["azure_fact_cost.year_to_date_cost"],
                    "dimensions": ["azure_resource_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                    "timeDimensions": [],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_services_cost_by_instance":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                    "timeDimensions": [],
                }
            }

            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "azure_resource_dim.resource_name",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "azure_fact_cost.charge_period_start",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "aws_services_cost_by_region":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.region_name"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "aws_services_cost_by_time":
            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "aws_fact_focus.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "aws_fact_focus.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "aws_services_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["aws_fact_focus.month_to_date_cost"],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "filters": [
                        {
                            "dimension": "aws_fact_focus.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "aws_services_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["aws_fact_focus.quarter_to_date_cost"],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "filters": [
                        {
                            "dimension": "aws_fact_focus.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "aws_services_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "aws_fact_focus.year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["aws_fact_focus.year_to_date_cost"],
                    "dimensions": ["aws_fact_focus.service_name"],
                    "filters": [
                        {
                            "dimension": "aws_fact_focus.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "aws_services_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["aws_fact_focus.total_list_cost"],
                    "dimensions": ["aws_fact_focus.resource_id"],
                    "order": {"aws_fact_focus.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "aws_fact_focus.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "gcp_services_cost_by_region":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.region_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "gcp_services_cost_by_time":
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "timeDimensions": [
                        {
                            "dimension": "gcp_fact_dim.charge_period_start",
                            "granularity": granularity,
                        }
                    ],
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "gcp_services_month_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.month_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.month_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "gcp_services_quarter_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.quarter_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.quarter_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "gcp_services_year_to_date_cost":
            # Query for percentage tagged resources
            measure = "gcp_fact_dim.year_to_date_cost"

            # Create the query to fetch the measure
            query = {
                "query": {
                    "measures": ["gcp_fact_dim.year_to_date_cost"],
                    "dimensions": ["gcp_fact_dim.service_name"],
                    "filters": [
                        {
                            "dimension": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "gcp_services_cost_by_instance":
            # Query for untagged resources
            # measure = "aws_fact_focus.storage_quarter_to_date_cost"

            query = {
                "query": {
                    "measures": ["gcp_fact_dim.total_list_cost"],
                    "dimensions": ["gcp_fact_dim.resource_name"],
                    "order": {"gcp_fact_dim.total_list_cost": "desc"},
                    "filters": [
                        {
                            "member": "gcp_fact_dim.service_name",
                            "operator": "equals",
                            "values": service_name,
                        }
                    ],
                }
            }

        elif query_type == "consolidated_budgets_data":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "view_fact_billing.max_monthly_budget",
                        "view_fact_billing.quarterly_budget",
                        "view_fact_billing.yearly_budget",
                        "view_fact_billing.monthly_budget_drift_percentage",
                        "view_fact_billing.quarterly_budget_utilization_percentage",
                        "view_fact_billing.quarterly_budget_drift_percentage",
                        "view_fact_billing.yearly_budget_drift_percentage",
                        "view_fact_billing.monthly_budget_utilization_actual_value",
                        "view_fact_billing.yearly_budget_utilization_actual_value",
                        "view_fact_billing.monthly_budget_utilization_percentage",
                        "view_fact_billing.quarterly_budget_utilization_actual_value",
                        "view_fact_billing.yearly_budget_utilization_percentage",
                        "view_fact_billing.forecast_next_quarter_cost",
                        "view_fact_billing.forecast_next_month_cost",
                        "view_fact_billing.forecast_next_year_cost",
                        "view_fact_billing.yearly_budget_drift_value",
                        "view_fact_billing.quarterly_budget_drift_value",
                        "view_fact_billing.monthly_budget_drift_value",
                    ],
                    "filters": [],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "monthly_budget": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.max_monthly_budget", 0
                            )
                        ),
                        "quarterly_budget": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.quarterly_budget", 0
                            )
                        ),
                        "yearly_budget": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.yearly_budget", 0
                            )
                        ),
                        "monthly_budget_drift": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.monthly_budget_drift_percentage", 0
                            )
                        ),
                        "quarterly_budget_drift": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.quarterly_budget_drift_percentage", 0
                            )
                        ),
                        "yearly_budget_drift": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.yearly_budget_drift_percentage", 0
                            )
                        ),
                        "monthly_budget_actual_utilization": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.monthly_budget_utilization_actual_value",
                                0,
                            )
                        ),
                        "quarterly_budget_actual_utilization": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.quarterly_budget_utilization_actual_value",
                                0,
                            )
                        ),
                        "yearly_budget_actual_utilization": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.yearly_budget_utilization_actual_value",
                                0,
                            )
                        ),
                        "monthly_budget_utilization": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.monthly_budget_utilization_percentage",
                                0,
                            )
                        ),
                        "quarterly_budget_utilization": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.quarterly_budget_utilization_percentage",
                                0,
                            )
                        ),
                        "yearly_budget_utilization": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.yearly_budget_utilization_percentage",
                                0,
                            )
                        ),
                        "forecast_next_month_cost": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.forecast_next_month_cost", 0
                            )
                        ),
                        "forecast_next_quarter_cost": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.forecast_next_quarter_cost", 0
                            )
                        ),
                        "forecast_next_year_cost": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.forecast_next_year_cost", 0
                            )
                        ),
                        "yearly_budget_actual_drift": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.yearly_budget_drift_value", 0
                            )
                        ),
                        "quarterly_budget_actual_drift": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.quarterly_budget_drift_value", 0
                            )
                        ),
                        "monthly_budget_actual_drift": float(
                            data.get("data", [{}])[0].get(
                                "view_fact_billing.monthly_budget_drift_value", 0
                            )
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "consolidated_cost_trends_over_time":
            query = {
                "query": {
                    "measures": [
                        "view_fact_billing.total_billed_cost"
                    ],  # or use the appropriate measure
                    "timeDimensions": [
                        {
                            "dimension": "view_dim_time.chargeperiodstart",  # or use the appropriate time dimension
                            "granularity": granularity,  # or "day", "quarter", "year", depending on your granularity
                        }
                    ],
                    "filters": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "consolidated_service_name_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_fact_billing.servicename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "consolidated_charge_period_dates":
            # Query for monthly, quarterly, and yearly budgets
            query = {
                "query": {
                    "measures": [
                        "view_dim_time.earliest_charge_period_date",
                        "view_dim_time.latest_charge_period_date",
                    ],
                    "filters": [],
                    "timeDimensions": [],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    formatted_data = {
                        "earliest_charge_period_date": data.get("data", [{}])[0].get(
                            "view_dim_time.earliest_charge_period_date", 0
                        ),
                        "latest_charge_period_date": data.get("data", [{}])[0].get(
                            "view_dim_time.latest_charge_period_date", 0
                        ),
                    }

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "consolidated_cost_by_service_category":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_service.servicecategory"],
                    "filters": [],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "consolidated_cost_by_region":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_region.regionname"],
                    "filters": [],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "consolidated_all_services_cost":
            # Combined query for month-to-date, quarter-to-date, and year-to-date costs for all services
            query = {
                "query": {
                    "measures": [
                        "view_fact_billing.month_to_date_cost",
                        "view_fact_billing.quarter_to_date_cost",
                        "view_fact_billing.year_to_date_cost",
                    ],
                    "dimensions": ["view_fact_billing.servicename"],
                    "filters": [],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CUBEJS_API_URL}/load", json=query, headers=headers
                    )
                    print(f"Status Code: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    print("Response received successfully.")
                    # Process data to the desired format
                    formatted_data = []
                    for entry in data.get("data", []):
                        formatted_entry = {
                            "service_name": entry.get(
                                "view_fact_billing.servicename", ""
                            ),
                            "costs": {
                                "month_to_date": entry.get(
                                    "view_fact_billing.month_to_date_cost", 0
                                ),
                                "quarter_to_date": entry.get(
                                    "view_fact_billing.quarter_to_date_cost", 0
                                ),
                                "year_to_date": entry.get(
                                    "view_fact_billing.year_to_date_cost", 0
                                ),
                            },
                        }
                        formatted_data.append(formatted_entry)

                    return {"message": "Success", "data": formatted_data}

            except httpx.HTTPStatusError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response Content: {http_err.response.text}")
            except httpx.RequestError as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        elif query_type == "azure_sku_id_cost":
            # Query for Resource Type and Total Billed Cost
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_service.skuid"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )
        
            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_cost_by_instance_vs":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_resource.resourcename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Visual Studio"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_cost_by_instance_db_for_postgres":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_resource.resourcename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Azure DB for PostgreSQL"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_container_registry":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_resource.resourcename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Azure Container Registry"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        elif query_type == "azure_cost_by_instance_vnet":
            query = {
                "query": {
                    "measures": ["azure_fact_cost.total_billed_cost"],
                    "dimensions": ["azure_resource_dim.resource_name"],
                    "order": {"azure_fact_cost.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "azure_resource_dim.service_name",
                            "operator": "equals",
                            "values": ["Virtual Network"],
                        }
                    ],
                }
            }

        elif query_type == "azure_cost_by_instance_ml":
            query = {
                "query": {
                    "measures": ["view_fact_billing.total_billed_cost"],
                    "dimensions": ["view_dim_resource.resourcename"],
                    "order": {"view_fact_billing.total_billed_cost": "desc"},
                    "filters": [
                        {
                            "member": "view_fact_billing.servicename",
                            "operator": "equals",
                            "values": ["Azure Machine Learning"],
                        }
                    ],
                    "timeDimensions":[],
                }
            }

            # Add filters for resource names if resource_list is provided
            if resource_list:
                query["query"]["filters"].append(
                    {
                        "dimension": "view_dim_resource.resourceid",
                        "operator": "in",
                        "values": resource_list,
                    }
                )

            if start_date:
                query["query"]["timeDimensions"].append(
                    {
                        "dimension": "view_dim_time.chargeperiodstart",
                        "dateRange": [start_date_str, end_date_str],
                    }
                )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{CUBEJS_API_URL}/load", json=query, headers=headers
                )
                print(f"Status Code: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                print("Response received successfully.")
                return {"message": "Success", "data": data}

        except httpx.HTTPStatusError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response Content: {http_err.response.text}")
        except httpx.RequestError as req_err:
            print(f"Request error occurred: {req_err}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    except httpx.RequestError as e:
        print(f"Request error occurred while requesting {e.request.url!r}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching data")
    except httpx.HTTPStatusError as e:
        print(f"HTTP status error occurred: {e}")
        print(f"Response status code: {e.response.status_code}")
        print(f"Response content: {e.response.content}")
        raise HTTPException(status_code=500, detail="Error fetching data")
    except Exception as e:
        print(f"Unexpected error occurred while fetching data: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error fetching data")
