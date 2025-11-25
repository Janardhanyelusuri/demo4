import os
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException
from tortoise.exceptions import DoesNotExist
from celery.result import AsyncResult
from app.core.encryption import decrypt_data
from app.schemas.connection import (
    CheckProjectNameRequest, CheckProjectNameResponse, TableColumnsResponse,
    DeleteAwsProjectConfirmation, DeleteAwsS3Bucket, DeleteAwsExport
)
from app.models.project import Project, ProjectIn_Pydantic, Project_Pydantic
from app.models.aws import AwsConnection, AwsConnection_Pydantic
from app.models.azure import AzureConnection, AzureConnection_Pydantic
from app.models.gcp import GCPConnection, GCPConnection_Pydantic
from app.models.snowflake import SnowflakeConnection, SnowflakeConnection_Pydantic
from app.models.project_access import ProjectAccess, ProjectAccess_Pydantic
from app.models.alert import Alert, Alert_Pydantic
from app.models.database import Database_Pydantic, Database, DataResponse
from app.models.alert_integration import Integration, Integration_Pydantic, IntegrationIn_Pydantic
from app.models.dashboard_request import DashboardRequest, DashboardRequest_Pydantic, DashboardRequestIn_Pydantic
from app.worker.celery_worker import (task_sample, task_delete_aws_project, task_delete_aws_s3_bucket,
                                      task_delete_aws_export, task_drop_schema, task_delete_gcp_project,
                                      task_delete_azure_project, task_run_daily_ingestion,
                                      task_create_aws_export, task_create_azure_export)
from app.core.misc import create_project_and_database, fetch_data, fetch_data_from_database

router = APIRouter()

@router.post('/check_name', response_model=CheckProjectNameResponse, tags=["project"])
async def check_project_name(project: CheckProjectNameRequest):
    try:
        project_name = project.name.lower()  

        # check if project with same name already exists
        obj = await Project.filter(name=project_name).first()
        if obj:
            return CheckProjectNameResponse(status=False,
                                            message="Project with same name already exists. Please try another name.")
        print(f"Project name '{project_name}' is available.")
    except Exception as e:
        print(f"Error: {e}")
        return CheckProjectNameResponse(status=False,
                                        message="Something went wrong")
    return CheckProjectNameResponse(status=True,
                                    message="Success")


@router.post('/', response_model=Project_Pydantic, tags=["project"])
async def add_project(project: ProjectIn_Pydantic):
    try:
        project_data = project.dict()
        # Convert project name to lowercase
        if 'name' in project_data:
            project_data['name'] = project_data['name'].lower()
        
        project_obj = await Project.create(**project_data)
        return await Project_Pydantic.from_tortoise_orm(project_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{project_id}', response_model=Project_Pydantic, tags=["project"])
async def get_project(project_id: int):
    try:
        project = await Project.get(id=project_id)
        return await Project_Pydantic.from_tortoise_orm(project)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/aws', response_model=AwsConnection_Pydantic, tags=["project"])
async def get_project_aws_connection(project_id: int):
    try:
        obj = await AwsConnection.get(project_id=project_id)
        return await AwsConnection_Pydantic.from_tortoise_orm(obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/azure', response_model=AzureConnection_Pydantic, tags=["project"])
async def get_project_azure_connection(project_id: int):
    try:
        obj = await AzureConnection.get(project_id=project_id)
        return await AzureConnection_Pydantic.from_tortoise_orm(obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/snowflake', response_model=SnowflakeConnection_Pydantic, tags=["project"])
async def get_project_snowflake_connection(project_id: int):
    try:
        obj = await SnowflakeConnection.get(project_id=project_id)
        return await SnowflakeConnection_Pydantic.from_tortoise_orm(obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/gcp', response_model=GCPConnection_Pydantic, tags=["project"])
async def get_project_gcp_connection(project_id: int):
    try:
        obj = await GCPConnection.get(project_id=project_id)
        return await GCPConnection_Pydantic.from_tortoise_orm(obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/integrations', response_model=List[Integration_Pydantic], tags=["project"])
async def get_project_integrations(project_id: int):
    try:
        return await Integration_Pydantic.from_queryset(Integration.filter(project_id=project_id).all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/users', response_model=List[ProjectAccess_Pydantic], tags=["project"])
async def get_project_users(project_id: int):
    try:
        return await ProjectAccess_Pydantic.from_queryset(ProjectAccess.filter(project_id=project_id).all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/alerts', response_model=List[Alert_Pydantic], tags=["project"])
async def get_project_alerts(project_id: int):
    try:
        # Filter alerts where project_id is present in project_ids JSON field
        return await Alert_Pydantic.from_queryset(Alert.filter(project_ids__contains=[project_id]).all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{project_id}/service/{service_id}/dashboard_requests',
            response_model=List[DashboardRequest_Pydantic],
            tags=["project"])
async def get_project_dashboard_requests(project_id: int, service_id: int):
    try:
        return await DashboardRequest_Pydantic.from_queryset(DashboardRequest.filter(project_id=project_id,
                                                                                     service_id=service_id).all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[Project_Pydantic], tags=["project"])
async def get_all_project():
    try:
        return await Project_Pydantic.from_queryset(Project.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{project_id}", response_model=Project_Pydantic, tags=["project"])
async def update_project(project_id: int, project: ProjectIn_Pydantic):
    project_data = project.model_dump(exclude_unset=True)
    # Convert project name to lowercase if it's being updated
    if 'name' in project_data:
        project_data['name'] = project_data['name'].lower()
    
    await Project.filter(id=project_id).update(**project_data)
    return await Project_Pydantic.from_queryset_single(Project.get(id=project_id))


@router.delete("/{project_id}", tags=["project"])
async def delete_project(project_id: int, project: DeleteAwsProjectConfirmation):
    project_obj = await Project.filter(id=project_id).first()

    if project_obj.cloud_platform.lower() == "aws":
        aws_connection_obj = await AwsConnection.filter(project_id=project_id).first()
        s3_bucket_split = aws_connection_obj.export_location.split("/")

        # Ensure encryption key is fetched correctly
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        payload = {
            "project_name": project_obj.name,
            "aws_access_key": decrypt_data(encrypted_data=aws_connection_obj.aws_access_key, key=encryption_key),
            "aws_secret_key": decrypt_data(encrypted_data=aws_connection_obj.aws_secret_key, key=encryption_key),
            "aws_region": 'us-east-1',
            "s3_bucket": s3_bucket_split[2],
            "s3_prefix": s3_bucket_split[3],
            "export_name": s3_bucket_split[4],
            "delete_s3": project.delete_s3,
            "delete_export": project.delete_export
        }
        task = task_delete_aws_project.delay(payload)
        print({"task_id": task.id})

    elif project_obj.cloud_platform.lower() == "gcp":
        payload = {"project_name": project_obj.name}
        task = task_delete_gcp_project.delay(payload)
        print({"task_id": task.id})

    elif project_obj.cloud_platform.lower() == "azure":
        # Ensure encryption key is fetched correctly
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        connection_obj = await AzureConnection.filter(project_id=project_id).first()
        payload = {
            "azure_tenant_id": decrypt_data(encrypted_data=connection_obj.azure_tenant_id, key=encryption_key),
            "azure_client_id": decrypt_data(encrypted_data=connection_obj.azure_client_id, key=encryption_key),
            "azure_client_secret": decrypt_data(encrypted_data=connection_obj.azure_client_secret, key=encryption_key),
            "project_name": project_obj.name,
            "subscription_info": connection_obj.subscription_info,
            "storage_account_name": connection_obj.storage_account_name,
            "export_name": connection_obj.container_name,
            "delete_export": project.delete_export,
            "delete_container": project.delete_container,
        }
        task = task_delete_azure_project.delay(payload)
        print({"task_id": task.id})

    elif project_obj.cloud_platform.lower() == "snowflake":
        pass

    await Project.filter(id=project_id).delete()
    return {"status": True, "message": "Successfully deleted project"}


@router.get("/{project_id}/database", response_model=Database_Pydantic, tags=["project"])
async def get_project_database(project_id: int):
    try:
        # Fetch the database linked to the project directly from the Database model
        database = await Database.filter(project_id=project_id).first()
        if not database:
            raise HTTPException(status_code=404, detail="Database information not found")

        # Serialize and return the database information using Database_Pydantic
        return await Database_Pydantic.from_tortoise_orm(database)

    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Database information not found")

    except Exception as e:
        print(f"Error fetching project database: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get('/{project_id}/database-data', response_model=DataResponse, tags=["project"])
async def get_project_database_data(project_id: int, table_name: str):
    try:
        # Fetch the database linked to the project directly from the Database model
        database = await Database.filter(project_id=project_id).first()
        if not database:
            raise HTTPException(status_code=404, detail="Database information not found")

        # Fetch connection string
        connection_string = database.connection_string

        # Define the query to execute on the actual database
        query = f"SELECT * FROM {table_name};"

        # Fetch data from the actual database using the connection string
        data = await fetch_data_from_database(connection_string, query)

        # Serialize and return the fetched data
        return DataResponse(data=data)

    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Database information not found")

    except Exception as e:
        print(f"Error fetching project database data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get('/{project_id}/database-tables-columns', response_model=List[TableColumnsResponse], tags=["project"])
async def get_project_tables_and_columns(project_id: int):
    try:
        # # Fetch the database linked to the project directly from the Database model
        # database = await Database.filter(project_id=project_id).first()
        # if not database:
        #     raise HTTPException(status_code=404, detail="Database information not found")

        # get project object to fetch project name
        project_obj = await Project.filter(id=project_id).first()

        # Fetch connection string
        connection_string = os.getenv('DB_CONNECTION_STRING')

        # Define the query to get the list of tables in the project schema(this example is for PostgreSQL)
        # filter only gold tables/views
        table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{project_obj.name}' and table_name like 'gold%';
        """

        # Fetch table names from the database
        tables = await fetch_data(connection_string, table_query)

        # Initialize the response list
        response = []

        # Iterate through each table and fetch its columns
        for table in tables:
            table_name = table  # Since tables is a list of strings
            column_query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{project_obj.name}' AND table_name = '{table_name}';
            """
            columns = await fetch_data(connection_string, column_query)

            # Append the table and its columns to the response
            response.append(TableColumnsResponse(table=table_name, columns=columns))

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/create-project-database/{project_name}", tags=["project"])
async def test_create_project_database(project_name: str, cloud_platform: str, aws_access_key: str,
                                       aws_secret_key: str):
    try:
        # Run the function
        database = await create_project_and_database(project_name, cloud_platform, aws_access_key, aws_secret_key)
        return {"message": f"Project '{project_name}' and database created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project and database: {str(e)}")


@router.post("/tasks", status_code=201, tags=["sample_celery_tasks"])
def run_task(payload: dict):
    task = task_sample.delay(payload)
    return {"task_id": task.id}


@router.post("/tasks/delete_aws_s3_bucket", status_code=201, tags=["sample_celery_tasks"])
def run_task_delete_aws_s3_bucket(payload: DeleteAwsS3Bucket):
    task = task_delete_aws_s3_bucket.delay(payload.dict())
    return {"task_id": task.id}


@router.post("/tasks/delete_aws_export", status_code=201, tags=["sample_celery_tasks"])
def run_task_delete_aws_export(payload: DeleteAwsExport):
    task = task_delete_aws_export.delay(payload.dict())
    return {"task_id": task.id}


@router.post("/tasks/drop_schema", status_code=201, tags=["sample_celery_tasks"])
def run_task_drop_schema(schema_name: str):
    task = task_drop_schema.delay(schema_name)
    return {"task_id": task.id}


@router.get("/tasks/{task_id}", tags=["sample_celery_tasks"])
def get_task_status(task_id):
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return result


@router.get('/{project_id}/run_ingestion', tags=["project"])
async def rerun_ingestion(project_id: int):
    try:
        project_obj = await Project.get(id=project_id)

        # Retrieve encryption key from environment variable
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        # check if export is enabled. If not run create export Else run ingestion task
        if project_obj.cloud_platform.lower() == "aws":
            aws_connection_obj = await AwsConnection.filter(project_id=project_id).first()
            if aws_connection_obj.export is False and aws_connection_obj.status is False:
                s3_bucket_split = aws_connection_obj.export_location.split("/")
                task = task_create_aws_export.delay({
                    "project_id": project_obj.id,
                    "project_name": project_obj.name,
                    "aws_connection_id": aws_connection_obj.id,
                    # "sync_status_id": sync_status_obj.id,
                    "aws_access_key": decrypt_data(encrypted_data=aws_connection_obj.aws_access_key, key=encryption_key),
                    "aws_secret_key": decrypt_data(encrypted_data=aws_connection_obj.aws_secret_key, key=encryption_key),
                    "aws_region": 'us-east-1',  # todo: take input from UI
                    "s3_bucket": s3_bucket_split[2],
                    "s3_prefix": s3_bucket_split[3],
                    "export_name": s3_bucket_split[4],
                    "billing_period": datetime.datetime.utcnow().strftime("%Y-%m")
                })
                print({"task_id": task.id})
                return {"status": True, "message": "Export successfully scheduled"}

        elif project_obj.cloud_platform.lower() == "azure":
            azure_connection_obj = await AzureConnection.filter(project_id=project_id).first()
            if azure_connection_obj.export is False and azure_connection_obj.status is False:
                task = task_create_azure_export.delay({
                    "project_id": project_obj.id,
                    "project_name": project_obj.name,
                    "azure_connection_id": azure_connection_obj.id,
                    # "sync_status_id": sync_status_obj.id,
                    "azure_tenant_id": decrypt_data(encrypted_data=azure_connection_obj.azure_tenant_id, key=encryption_key),
                    "azure_client_id": decrypt_data(encrypted_data=azure_connection_obj.azure_client_id, key=encryption_key),
                    "azure_client_secret": decrypt_data(encrypted_data=azure_connection_obj.azure_client_secret, key=encryption_key),
                    "subscription_info": azure_connection_obj.subscription_info,
                    "date": str(azure_connection_obj.date),
                    "storage_account_name": azure_connection_obj.storage_account_name,
                    "resource_group_name": azure_connection_obj.resource_group_name,
                    "container_name": azure_connection_obj.container_name
                })
                print({"task_id": task.id})
                return {"status": True, "message": "Export successfully scheduled"}

        elif project_obj.cloud_platform.lower() == "gcp":
            obj = await GCPConnection.filter(project_id=project_id).first()
            if obj.export is False and obj.status is False:
                return {"status": True, "message": "Export is not enabled"}

        task = task_run_daily_ingestion.delay({"project_id": project_obj.id})
        print({"task_id": task.id})
        return {"status": True, "message": "Data Ingestion successfully scheduled"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
