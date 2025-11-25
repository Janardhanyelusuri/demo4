import datetime
import os
from typing import List
from app.core.encryption import decrypt_data
from fastapi import APIRouter, HTTPException
from app.schemas.connection import AWSConnectionRequest, AWSConnectionResponse
from app.core.aws import aws_validate_creds
from app.models.aws import AwsConnection, AwsConnection_Pydantic, AwsConnectionIn_Pydantic
from app.models.project import Project
from app.models.sync_status import SyncStatus
from app.worker.celery_worker import task_create_aws_ce, task_create_aws_export, task_run_ingestion_aws

router = APIRouter()


@router.post('/', response_model=AwsConnection_Pydantic, tags=["aws_connection"])
async def add_aws_connection(aws_connection: AwsConnectionIn_Pydantic):
    try:
        # Test connection
        if not aws_validate_creds(aws_connection.aws_access_key, aws_connection.aws_secret_key):
            raise Exception('Invalid credentials')

        # change monthly_budget and quarterly_budget using yearly_budget
        aws_connection.monthly_budget = int(aws_connection.yearly_budget / 12)
        aws_connection.quarterly_budget = int(aws_connection.yearly_budget / 3)

        # update the table if export is false > s3 path is not provided
        if not aws_connection.export:
            s3_bucket = f'cloud-meter-{aws_connection.project_id}-{datetime.datetime.utcnow().strftime("%s")}'
            s3_prefix = "cm"
            export_name = s3_bucket
            export_location = f's3://{s3_bucket}/{s3_prefix}/{export_name}/'
            # await AwsConnection.filter(aws_connection_obj.id).update(export_location=export_location)
            # print("update")
            aws_connection.export_location = export_location
        aws_connection_obj = await AwsConnection.create(**aws_connection.dict())

        # add record in sync status table to keep track of sync status
        # sync_status_dict = {
        #     "project_id": aws_connection.project_id,
        #     "module": "AWS CE",
        #     "status": "In Progress",
        # }
        # sync_status_obj = await SyncStatus.create(**sync_status_dict)

        project_obj = await Project.filter(id=aws_connection.project_id).first()

        # # not in use
        # # AWS CE
        # task = task_create_aws_ce.delay({
        #     "project_id": project_obj.id,
        #     "project_name": project_obj.name,
        #     "aws_connection_id": aws_connection_obj.id,
        #     # "sync_status_id": sync_status_obj.id,
        #     "aws_access_key": aws_connection.aws_access_key,
        #     "aws_secret_key": aws_connection.aws_secret_key,
        #     "date": str(aws_connection.date)
        # })
        # print({"task_id": task.id})

        # if export is true, use s3 path
        if aws_connection.export:
            # sample export_location
            # aws_connection.export_location = "s3://cloud-meter1613/cm/testing/"
            s3_bucket_split = aws_connection.export_location.split("/")

            # AWS CUR
            task = task_run_ingestion_aws.delay({
                "project_id": project_obj.id,
                "project_name": project_obj.name,
                "aws_connection_id": aws_connection_obj.id,
                # "sync_status_id": sync_status_obj.id,
                "monthly_budget": aws_connection.monthly_budget,
                "aws_access_key": aws_connection.aws_access_key,
                "aws_secret_key": aws_connection.aws_secret_key,
                "aws_region": 'us-east-1',  # todo: take input from UI
                "s3_bucket": s3_bucket_split[2],
                "s3_prefix": s3_bucket_split[3],
                "export_name": s3_bucket_split[4],
                "billing_period": datetime.datetime.utcnow().strftime("%Y-%m")
            })
            print({"task_id": task.id})

        else:
            # CREATE/ENABLE EXPORT
            task = task_create_aws_export.delay({
                "project_id": project_obj.id,
                "project_name": project_obj.name,
                "aws_connection_id": aws_connection_obj.id,
                # "sync_status_id": sync_status_obj.id,
                "aws_access_key": aws_connection.aws_access_key,
                "aws_secret_key": aws_connection.aws_secret_key,
                "aws_region": 'us-east-1',  # todo: take input from UI
                "s3_bucket": s3_bucket,
                "s3_prefix": s3_prefix,
                "export_name": export_name,
                "billing_period": datetime.datetime.utcnow().strftime("%Y-%m")
            })
            print({"task_id": task.id})

        return await AwsConnection_Pydantic.from_tortoise_orm(aws_connection_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{aws_connection_id}', response_model=AwsConnection_Pydantic, tags=["aws_connection"])
async def get_aws_connection(aws_connection_id: int):
    try:
        aws_connection = await AwsConnection.get(id=aws_connection_id)
        return await AwsConnection_Pydantic.from_tortoise_orm(aws_connection)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[AwsConnection_Pydantic], tags=["aws_connection"])
async def get_all_aws_connection():
    try:
        # Fetch all encrypted Snowflake connections
        connections = await AwsConnection.all()

        # Convert to Pydantic models
        return connections
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{aws_connection_id}", response_model=AwsConnection_Pydantic, tags=["aws_connection"])
async def update_aws_connection(aws_connection_id: int, aws_connection: AwsConnectionIn_Pydantic):
    await AwsConnection.filter(id=aws_connection_id).update(**aws_connection.model_dump(exclude_unset=True))
    return await AwsConnection_Pydantic.from_queryset_single(AwsConnection.get(id=aws_connection_id))


@router.post("/test_connection", response_model=AWSConnectionResponse, tags=["aws_connection"])
async def test_aws_connection(request: AWSConnectionRequest):
    aws_access_key = request.aws_access_key
    aws_secret_key = request.aws_secret_key

    # Test connection
    if aws_validate_creds(aws_access_key, aws_secret_key):
        return AWSConnectionResponse(status=True, message='Connection successful')
    else:
        return AWSConnectionResponse(status=False, message='Connection failed')


# @router.get('/{aws_connection_id}/enable_export', response_model=AwsConnection_Pydantic, tags=["aws_connection"])
# async def aws_enable_export(aws_connection_id: int):
#     try:
#         aws_connection = await AwsConnection.get(id=aws_connection_id)
#         return AWSConnectionResponse(status=True, message='Success')
#     except Exception as e:
#         print(f"Error: {e}")
#         return AWSConnectionResponse(status=False, message='Failed')
#
#
# @router.get('/{aws_connection_id}/sync', response_model=AwsConnection_Pydantic, tags=["aws_connection"])
# async def aws_sync_data(aws_connection_id: int):
#     try:
#         aws_connection = await AwsConnection.get(id=aws_connection_id)
#         return AWSConnectionResponse(status=True, message='Success')
#     except Exception as e:
#         print(f"Error: {e}")
#         return AWSConnectionResponse(status=False, message='Failed')
