import os
import json
import datetime
import asyncpg
import asyncio
from .celery_app import celery_app
from app.ingestion.aws.main import aws_create_focus_export, aws_run_ingestion
from app.ingestion.aws.aws_ce.main import aws_ce_main
from app.ingestion.aws.aws_cur.exports.scripts.create_export import enable_and_create_export
from app.ingestion.aws.aws_cur.exports.scripts.exports_ops import delete_export
from app.ingestion.aws.aws_cur.postgres.main import aws_cur_main, delete_s3_bucket
from app.ingestion.aws.aws_cur.postgres.scripts.postgres_operations import drop_schema
from app.ingestion.gcp.main import fetch_data_from_bigquery_to_postgres
from app.ingestion.gcp.bigquery_view import create_view
from app.ingestion.azure.main import azure_main
from app.ingestion.azure.azure_ops import AzFunctions
from app.ingestion.dashboard.main import create_dashboard_view
from app.core.misc import execute_query
from app.core.encryption import decrypt_data
from app.models.project import Project
from app.models.alert_integration import Integration
from app.models.alert import Alert
from app.core.misc import build_query, init_tortoise_connection, close_tortoise_connection, send_message

DB_HOST_NAME = os.getenv("DB_HOST_NAME")
DB_NAME = os.getenv("DB_NAME")
DB_USER_NAME = os.getenv("DB_USER_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


@celery_app.task(name="run_daily_alerts")
def run_daily_alerts_sync():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_daily_alerts())


async def run_daily_alerts():
    await init_tortoise_connection()
    alerts = await Alert.filter(schedule__iexact='Daily').all()
    for alert in alerts:
        await process_alert(alert)
    await close_tortoise_connection()


@celery_app.task(name="run_weekly_alerts")
def run_weekly_alerts_sync():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_weekly_alerts())


async def run_weekly_alerts():
    await init_tortoise_connection()
    alerts = await Alert.filter(schedule__iexact='weekly').all()
    for alert in alerts:
        await process_alert(alert)
    await close_tortoise_connection()


@celery_app.task(name="run_monthly_alerts")
def run_monthly_alerts_sync():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_monthly_alerts())


async def run_monthly_alerts():
    await init_tortoise_connection()
    alerts = await Alert.filter(schedule__iexact='monthly').all()
    for alert in alerts:
        await process_alert(alert)
    await close_tortoise_connection()


async def process_alert(alert):
    await init_tortoise_connection()

    # Ensure you're accessing the 'id' correctly (alert.id if it's a model instance)
    alert_instance = alert if isinstance(alert, Alert) else await Alert.get(
        id=alert['id'])  # Handle if it's a model instance or dict

    # Check if status is True
    if not alert.status:  # Use alert.status if it's a model instance
        print(f"Skipping Alert ID {alert.id} - status is False")  # Use alert.id for model instance
        return  # Skip further processing for this alert

    # Get all tag_ids and project_ids
    tag_ids = alert.get('tag_ids', []) if isinstance(alert, dict) else alert.tag_ids
    project_ids = alert.get('project_ids', []) if isinstance(alert, dict) else alert.project_ids

    # If no tag_ids or project_ids specified, process with None values
    if not tag_ids:
        tag_ids = [None]
    if not project_ids:
        project_ids = [None]

    # Example PostgreSQL connection using asyncpg
    conn = await asyncpg.connect(user=DB_USER_NAME, password=DB_PASSWORD, database=DB_NAME, host=DB_HOST_NAME)

    # Process each combination of tag_id and project_id
    for tag_id in tag_ids:
        for project_id in project_ids:
            # Create a copy of the alert data for this combination
            alert_data = alert.copy() if isinstance(alert, dict) else alert.__dict__.copy()
            alert_data['tag_id'] = tag_id

            # Fetch project details if project_id exists
            if project_id:
                project = await Project.get(id=project_id).values('name', 'cloud_platform')
                if not project:
                    print(f"Project not found for Alert ID {alert.id}, Project ID {project_id}")  # Use alert.id
                    continue
                schema_name = project['name']
                cloud_platform = project['cloud_platform']
            else:
                # Use default project if no project_id specified
                schema_name = alert.get('default_schema', 'public') if isinstance(alert, dict) else alert.default_schema

            # Build and execute query for this combination
            query = await build_query(alert_data, schema_name, cloud_platform)
            print(f"Alert Data: {alert_data}")
            print(f"Executing query for Alert ID {alert.id}, Tag ID {tag_id}, Project ID {project_id}")  # Use alert.id
            print(f"Query: {query}")

            # Execute the query and fetch the full result row
            result = await conn.fetchrow(query)
            print(f"Result for Alert ID {alert.id}, Tag ID {tag_id}, Project ID {project_id}: {result}")  # Use alert.id

            # Only proceed with notification if 'trigger' is True
            if result and result['trigger']:
                # if Trigger is True, set state to firing
                if alert_instance.state.get("status") != "firing":
                    await alert_instance.update_state("firing")
                    print(f"Alert ID {alert.id} state set to 'firing'.")  # Use alert.id

                # Fetch integration details
                integration_id = alert.get('integration_id') if isinstance(alert, dict) else alert.integration_id
                if integration_id:
                    integration = await Integration.get(id=integration_id).values()

                    if integration:
                        # Prepare the message based on the notification template
                        alert_details = [
                            f"Cloud Platform: {cloud_platform}" if cloud_platform else "Cloud Platform: N/A",
                            f"Connection: {schema_name}" if schema_name else "Connection: N/A",
                            f"Alert Name: {alert.name}" if hasattr(alert, 'name') else "Alert Name: N/A",
                            f"Alert Type: {alert.type}" if hasattr(alert, 'type') else "Alert Type: N/A",
                            f"Alert Format: {alert.alert_type}" if hasattr(alert,
                                                                           'alert_type') else "Alert Format: N/A",
                            f"Tag ID: {tag_id}" if tag_id else "Tag ID: N/A",
                            f"Project ID: {project_id}" if project_id else "Project ID: N/A",
                            f"Resources: {result.get('resources', 'N/A')}",
                            f"Total Billed Cost: {result['total_billed_cost']}" if result.get(
                                'total_billed_cost') else "Total Billed Cost: N/A",
                            f"Threshold Value: {result['threshold_value']}" if result.get(
                                'threshold_value') else "Threshold Value: N/A",
                            f"Operation: {alert.operation}" if hasattr(alert, 'operation') else "Operation: N/A",
                            f"Condition: {alert.condition}" if hasattr(alert, 'condition') else "Condition: N/A",
                            f"Difference: {result['difference']}" if result.get('difference') else "Difference: N/A"
                        ]

                        # Join the alert details with newline characters
                        alert_details_str = "\n".join(alert_details)

                        # Add alert details to the message text
                        message = {
                            "text": f"Alert ID {alert.id} has triggered.\nDetails:\n{alert_details_str}"  # Use alert.id
                        }

                        # Determine the integration type and send the message
                        integration_type = integration.get('integration_type', 'slack')
                        webhook_url = integration['url']
                        await send_message(webhook_url, message, integration_type)

                        print(
                            f"Notification sent for Alert ID {alert.id}, Tag ID {tag_id}, Project ID {project_id}")  # Use alert.id
                    else:
                        print(f"No valid integration found for Alert ID {alert.id}")  # Use alert.id
            else:
                if alert_instance.state.get("status") == "firing":
                    await alert_instance.update_state("resolved")
                    print(f"Alert ID {alert.id} state set to 'resolved'.")  # Use alert.id
                print(
                    f"No notification needed for Alert ID {alert.id}, Tag ID {tag_id}, Project ID {project_id} - trigger is False or no results")  # Use alert.id

    # Close the database connection
    await conn.close()
    await close_tortoise_connection()


# This is dummy task for testing
@celery_app.task(name="task_sample")
def task_sample(payload):
    print("start dummy task")
    # task_run_daily_dashboard_ingestion()
    task_run_daily_ingestion()
    print("end dummy task")
    return True


@celery_app.task(name='task_run_periodic')
def task_run_periodic():
    print('printing running periodic task')
    return "Task executed"


@celery_app.task(name="task_create_aws_ce")
def task_create_aws_ce(payload):
    print("aws_ingestion ce start...")
    aws_ce_main(
        project_name=payload["project_name"],
        access_key=payload["aws_access_key"],
        secret_key=payload["aws_secret_key"],
        start_date=payload["date"]
    )
    print("aws_ingestion ce end...")
    return True


@celery_app.task(name="task_create_aws_export")
def task_create_aws_export(payload):
    print("create_aws_task_export start...")
    aws_create_focus_export(
        aws_access_key=payload["aws_access_key"],
        aws_secret_key=payload["aws_secret_key"],
        aws_region=payload["aws_region"],
        s3_bucket=payload["s3_bucket"],
        s3_prefix=payload["s3_prefix"],
        export_name=payload["export_name"]
    )
    # enable_and_create_export(
    #     aws_access_key=payload["aws_access_key"],
    #     aws_secret_key=payload["aws_secret_key"],
    #     aws_region=payload["aws_region"],
    #     s3_bucket=payload["s3_bucket"],
    #     s3_prefix=payload["s3_prefix"],
    #     export_name=payload["export_name"]
    # )

    query = f"""
    update awsconnection set export = true where id = {payload["aws_connection_id"]};
    """
    execute_query(query=query, fetch=False)

    print("create_aws_task_export end...")
    return True


@celery_app.task(name="task_run_ingestion_aws")
def task_run_ingestion_aws(payload):
    print("task_run_ingestion_aws start...")
    aws_run_ingestion(
        project_name=payload["project_name"],
        monthly_budget=str(payload["monthly_budget"]),
        aws_access_key=payload["aws_access_key"],
        aws_secret_key=payload["aws_secret_key"],
        aws_region=payload["aws_region"],
        s3_bucket=payload["s3_bucket"],
        s3_prefix=payload["s3_prefix"],
        export_name=payload["export_name"],
        billing_period=payload["billing_period"]
    )
    # aws_cur_main(
    #     project_name=payload["project_name"],
    #     monthly_budget=str(payload["monthly_budget"]),
    #     aws_access_key=payload["aws_access_key"],
    #     aws_secret_key=payload["aws_secret_key"],
    #     aws_region=payload["aws_region"],
    #     s3_bucket=payload["s3_bucket"],
    #     s3_prefix=payload["s3_prefix"],
    #     export_name=payload["export_name"],
    #     billing_period=payload["billing_period"]
    # )

    query = f"""
    update project set status = true where id = {payload["project_id"]};
    """
    execute_query(query=query, fetch=False)

    query = f"""
    update awsconnection set status = true where id = {payload["aws_connection_id"]};
    """
    execute_query(query=query, fetch=False)

    print("task_run_ingestion_aws end...")

    return True


@celery_app.task(name="task_delete_aws_project")
def task_delete_aws_project(payload):
    # drop project schema
    drop_schema(schema_name=payload["project_name"])

    if payload["delete_export"]:
        delete_export(
            aws_access_key=payload["aws_access_key"],
            aws_secret_key=payload["aws_secret_key"],
            aws_region=payload["aws_region"],
            export_name=payload["export_name"],
        )
    if payload["delete_s3"]:
        delete_s3_bucket(
            aws_access_key=payload["aws_access_key"],
            aws_secret_key=payload["aws_secret_key"],
            aws_region=payload["aws_region"],
            s3_bucket=payload["s3_bucket"],
        )
    return True


@celery_app.task(name="task_delete_aws_s3_bucket")
def task_delete_aws_s3_bucket(payload):
    delete_s3_bucket(
        aws_access_key=payload["aws_access_key"],
        aws_secret_key=payload["aws_secret_key"],
        aws_region=payload["aws_region"],
        s3_bucket=payload["s3_bucket"],
    )
    return True


@celery_app.task(name="task_delete_aws_export")
def task_delete_aws_export(payload):
    delete_export(
        aws_access_key=payload["aws_access_key"],
        aws_secret_key=payload["aws_secret_key"],
        aws_region=payload["aws_region"],
        export_name=payload["export_name"],
    )
    return True


@celery_app.task(name="task_drop_schema")
def task_drop_schema(schema_name):
    # drop project schema
    drop_schema(schema_name=schema_name)


@celery_app.task(name="task_run_ingestion_gcp")
def task_run_ingestion_gcp(payload):
    print("task_run_ingestion_gcp start...")

    create_view(credentials=payload["credentials"],
                project_id=payload["gcp_project_id"],
                dataset_id=payload["dataset_id"],
                billing_account_id=payload["billing_account_id"],
                date=payload["date"],
                view_id="focus_format_temp")

    fetch_data_from_bigquery_to_postgres(project_id=payload["gcp_project_id"],
                                         dataset_id=payload["dataset_id"],
                                         view_id="focus_format_temp",
                                         credentials=payload["credentials"],
                                         schema=payload["project_name"],
                                         table_name="bronze_focus_gcp_data",
                                         monthly_budget=str(payload["monthly_budget"]))

    query = f"""
    update project set status = true where id = {payload["project_id"]};
    """
    execute_query(query=query, fetch=False)

    query = f"""
    update gcpconnection set status = true where id = {payload["gcp_connection_id"]};
    """
    execute_query(query=query, fetch=False)

    print("task_run_ingestion_gcp end...")

    return True


@celery_app.task(name="task_delete_gcp_project")
def task_delete_gcp_project(payload):
    # drop project schema
    drop_schema(schema_name=payload["project_name"])

    return True


@celery_app.task(name="task_create_azure_export")
def task_create_azure_export(payload):
    start_date = payload["date"]
    subscription_id = payload["subscription_info"]["subscription_id"]
    subscription_name = payload["subscription_info"]["display_name"]
    resource_group = payload["resource_group_name"]
    storage_account_name = payload["storage_account_name"]
    container_name = payload["container_name"]
    export_name = container_name
    time_frame = 'MonthToDate'
    storage_account_resource_id = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Storage/storageAccounts/{storage_account_name}"
    print(start_date)

    # Initialize AzFunctions class
    az_funcs = AzFunctions(azure_tenant_id=payload["azure_tenant_id"],
                           azure_client_id=payload["azure_client_id"],
                           azure_client_secret=payload["azure_client_secret"])
    # az_funcs.register_cost_management_export(provider_namespace="Microsoft.CostManagement",
    #                                          subscription_id=subscription_id)
    # az_funcs.register_cost_management_export(provider_namespace="Microsoft.CostManagementExports",
    #                                          subscription_id=subscription_id)
    az_funcs.create_blob_container(storage_account_name, container_name)
    az_funcs.create_focus_export(storage_account_resource_id=storage_account_resource_id,
                                 storage_account_container=container_name,
                                 subscription_id=subscription_id,
                                 export_name=export_name,
                                 time_frame=time_frame,
                                 start_date=start_date)

    query = f"""
    update azureconnection set export = true where id = {payload["azure_connection_id"]};
    """
    execute_query(query=query, fetch=False)

    return True


@celery_app.task(name="task_run_ingestion_azure")
def task_run_ingestion_azure(payload):
    print("task_run_ingestion_azure start...")

    azure_main(project_name=payload["project_name"],
               budget=str(payload["monthly_budget"]),
               tenant_id=payload["azure_tenant_id"],
               client_id=payload["azure_client_id"],
               client_secret=payload["azure_client_secret"],
               storage_account_name=payload["storage_account_name"],
               container_name=payload["container_name"],
               subscription_id = payload["subscription_info"]["subscription_id"]
               )

    query = f"""
    update project set status = true where id = {payload["project_id"]};
    """
    execute_query(query=query, fetch=False)

    query = f"""
    update azureconnection set status = true where id = {payload["azure_connection_id"]};
    """
    execute_query(query=query, fetch=False)

    print("task_run_ingestion_azure end...")

    return True


@celery_app.task(name="task_delete_azure_project")
def task_delete_azure_project(payload):
    # drop project schema
    drop_schema(schema_name=payload["project_name"])

    az_funcs = AzFunctions(azure_tenant_id=payload["azure_tenant_id"],
                           azure_client_id=payload["azure_client_id"],
                           azure_client_secret=payload["azure_client_secret"])
    if payload["delete_export"]:
        az_funcs.delete_export(subscription_id=payload["subscription_info"]["subscription_id"],
                               export_name=payload["export_name"])
    if payload["delete_container"]:
        az_funcs.delete_blob_container(storage_account_name=payload["storage_account_name"],
                                       container_name=payload["export_name"])

    return True


@celery_app.task(name='task_run_daily_ingestion')
def task_run_daily_ingestion(input={}):
    print('task_run_daily_ingestion')

    # Ensure encryption key is fetched correctly
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        raise ValueError("Encryption key not found in environment variables")

    # Convert encryption key from string to bytes
    encryption_key = bytes.fromhex(encryption_key)

    query = f"""select * from project order by id desc;"""
    projects = execute_query(query=query)
    for p in projects:
        print("project: ", p)

        # check if project_id is provided in input.
        # If yes, run ingestion only for that project, else run for all
        print("input", input)
        if input:
            if int(input["project_id"]) != int(p[0]):
                continue

        try:
            if p[4] == "aws":
                query = f"""select id, aws_access_key, aws_secret_key, monthly_budget, date, export_location
                from awsconnection 
                where project_id = {p[0]};"""
                connection = execute_query(query=query)
                for c in connection:
                    print("aws conn: ", c)

                    s3_bucket_split = c[5].split("/")
                    print(c[4], type(c[4]))
                    billing_period = datetime.datetime.utcnow().strftime("%Y-%m")
                    payload = {
                        "project_id": p[0],
                        "project_name": p[1],
                        "aws_connection_id": c[0],
                        "aws_access_key": decrypt_data(encrypted_data=c[1], key=encryption_key),
                        "aws_secret_key": decrypt_data(encrypted_data=c[2], key=encryption_key),
                        "aws_region": 'us-east-1',
                        "monthly_budget": c[3],
                        "s3_bucket": s3_bucket_split[2],
                        "s3_prefix": s3_bucket_split[3],
                        "export_name": s3_bucket_split[4],
                        "billing_period": billing_period
                    }
                    # task_run_ingestion_aws(payload)
                    aws_run_ingestion(
                        project_name=payload["project_name"],
                        monthly_budget=str(payload["monthly_budget"]),
                        aws_access_key=payload["aws_access_key"],
                        aws_secret_key=payload["aws_secret_key"],
                        aws_region=payload["aws_region"],
                        s3_bucket=payload["s3_bucket"],
                        s3_prefix=payload["s3_prefix"],
                        export_name=payload["export_name"],
                        billing_period=payload["billing_period"]
                    )
                    query = f"""
                    update project set status = true where id = {payload["project_id"]};
                    """
                    execute_query(query=query, fetch=False)

                    query = f"""
                    update awsconnection set status = true where id = {payload["aws_connection_id"]};
                    """
                    execute_query(query=query, fetch=False)

            elif p[4] == "azure":
                query = f"""select id, azure_tenant_id, azure_client_id, azure_client_secret, monthly_budget, storage_account_name, container_name,subscription_info
                 from azureconnection 
                 where project_id = {p[0]};"""
                connection = execute_query(query=query)
                for c in connection:
                    print("azure conn: ", c)
                    subscription_info = c[7]
                    if isinstance(subscription_info, str):
                        subscription_info = json.loads(subscription_info)
                        print("Parsed subscription_info: ", subscription_info)

                    payload = {
                        "project_id": p[0],
                        "project_name": p[1],
                        "azure_connection_id": c[0],
                        "azure_tenant_id": decrypt_data(encrypted_data=c[1], key=encryption_key),
                        "azure_client_id": decrypt_data(encrypted_data=c[2], key=encryption_key),
                        "azure_client_secret": decrypt_data(encrypted_data=c[3], key=encryption_key),
                        "monthly_budget": c[4],
                        "storage_account_name": c[5],
                        "container_name": c[6],
                        "subscription_info": subscription_info
                    }
                    # task_run_ingestion_azure(payload)
                    azure_main(project_name=payload["project_name"],
                               budget=str(payload["monthly_budget"]),
                               tenant_id=payload["azure_tenant_id"],
                               client_id=payload["azure_client_id"],
                               client_secret=payload["azure_client_secret"],
                               storage_account_name=payload["storage_account_name"],
                               container_name=payload["container_name"],
                               subscription_id = payload["subscription_info"]["subscription_id"]
                               )

                    query = f"""
                    update project set status = true where id = {payload["project_id"]};
                    """
                    execute_query(query=query, fetch=False)

                    query = f"""
                    update azureconnection set status = true where id = {payload["azure_connection_id"]};
                    """
                    execute_query(query=query, fetch=False)

            elif p[4] == "gcp":
                query = f"""select id, credentials, project_info, date, monthly_budget, dataset_id, billing_account_id
                from gcpconnection 
                where project_id = {p[0]};"""
                connection = execute_query(query=query)
                for c in connection:
                    print("gcp conn: ", c)

                    credentials = json.loads(
                        decrypt_data(encrypted_data=c[1]["encrypted_credentials"], key=encryption_key))

                    payload = {
                        "project_id": p[0],
                        "project_name": p[1],
                        "gcp_connection_id": c[0],
                        "credentials": credentials,
                        "gcp_project_id": c[2]["project_id"],
                        "date": c[3],
                        "monthly_budget": c[4],
                        "dataset_id": c[5],
                        "billing_account_id": c[6],
                    }
                    # task_run_ingestion_gcp(payload)
                    create_view(credentials=payload["credentials"],
                                project_id=payload["gcp_project_id"],
                                dataset_id=payload["dataset_id"],
                                billing_account_id=payload["billing_account_id"],
                                date=payload["date"],
                                view_id="focus_format_temp")
                    fetch_data_from_bigquery_to_postgres(project_id=payload["gcp_project_id"],
                                                         dataset_id=payload["dataset_id"],
                                                         view_id="focus_format_temp",
                                                         credentials=payload["credentials"],
                                                         schema=payload["project_name"],
                                                         table_name="bronze_focus_gcp_data",
                                                         monthly_budget=str(payload["monthly_budget"]))

                    query = f"""
                    update project set status = true where id = {payload["project_id"]};
                    """
                    execute_query(query=query, fetch=False)

                    query = f"""
                    update gcpconnection set status = true where id = {payload["gcp_connection_id"]};
                    """
                    execute_query(query=query, fetch=False)

        except Exception as ex:
            print(ex)

    return True


@celery_app.task(name="task_run_daily_dashboard_ingestion")
def task_run_daily_dashboard_ingestion(payload={}):
    query = f"""select * from dashboard order by id desc;"""
    dashboards = execute_query(query=query)
    for d in dashboards:
        print("dashboard: ", d)

        project_ids = d[5]
        project_names = []

        # Validate if the project exists
        for project_id in d[5]:
            try:
                query = f"""select name from project where id = {project_id};"""
                result = execute_query(query=query)
                if result:
                    # take first object from list, as execute query returns List
                    result = result[0]
                    print("project", result)
                    project_names.append(result[0])
            except Exception as ex:
                print(ex)

        # run task to refresh views
        payload = {
            "cloud_platforms": d[6],
            "project_ids": project_ids,
            "project_names": project_names,
            "dashboard_id": d[0],
            "dashboard_name": d[1],
        }
        task_create_dashboard_view(payload=payload)
    return True


@celery_app.task(name="task_create_dashboard_view")
def task_create_dashboard_view(payload):
    try:
        # Run the async dashboard creation logic
        result = asyncio.run(create_dashboard_view(
            project_ids=payload["project_ids"],
            project_names=payload["project_names"],
            cloud_platforms=payload["cloud_platforms"],
            dashboard_name=payload["dashboard_name"]
        ))

        if result:
            # Update status for all dashboards with the same name
            update_query = f"""
                UPDATE dashboard 
                SET status = TRUE 
                WHERE name = '{payload["dashboard_name"]}';
            """
            execute_query(query=update_query, fetch=False)
            print(f"Updated status for all dashboards with name: {payload['dashboard_name']}")

        return result

    except Exception as e:
        print(f"Error in task_create_dashboard_view: {e}")
        return False


@celery_app.task(name="task_delete_dashboard")
def task_delete_dashboard(payload):
    # drop schema
    # drop_schema(schema_name=payload["dashboard_name"])
    return True
