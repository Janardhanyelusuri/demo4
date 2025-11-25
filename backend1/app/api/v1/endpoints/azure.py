from typing import List
import os
from fastapi import APIRouter, HTTPException
from app.core.azure import azure_validate_creds, get_azure_subscriptions
from app.core.encryption import decrypt_data
from app.schemas.connection import AzureConnectionRequest, AzureConfirmRequest, AzureConnectionResponse, SubscriptionInfo,SubscriptionsResponse
from app.models.azure import AzureConnection, AzureConnection_Pydantic, AzureConnectionIn_Pydantic
from app.models.project import Project
from app.worker.celery_worker import task_create_azure_export, task_run_ingestion_azure
router = APIRouter()



@router.post('/', response_model=AzureConnection_Pydantic, tags=["azure_connection"])
async def add_azure_connection(azure_connection: AzureConnectionIn_Pydantic):
    try:
        # Test connection
        if not azure_validate_creds(azure_connection.azure_client_id,
                                    azure_connection.azure_client_secret,
                                    azure_connection.azure_tenant_id):
            raise Exception('Invalid credentials')

        # change monthly_budget and quarterly_budget using yearly_budget
        azure_connection.monthly_budget = int(azure_connection.yearly_budget / 12)
        azure_connection.quarterly_budget = int(azure_connection.yearly_budget / 3)

        # update the table if export is false > container name is not given
        if not azure_connection.export:
            azure_connection.container_name = f'cloud-meter-{azure_connection.project_id}'

        azure_connection_obj = await AzureConnection.create(**azure_connection.dict())
        project_obj = await Project.filter(id=azure_connection.project_id).first()

        # update the table if export is false > container name is not given
        if not azure_connection.export:
            # create container, export
            task = task_create_azure_export.delay({
                "project_id": project_obj.id,
                "project_name": project_obj.name,
                "azure_connection_id": azure_connection_obj.id,
                # "sync_status_id": sync_status_obj.id,
                "azure_tenant_id": azure_connection.azure_tenant_id,
                "azure_client_id": azure_connection.azure_client_id,
                "azure_client_secret": azure_connection.azure_client_secret,
                "subscription_info": azure_connection.subscription_info,
                "date": str(azure_connection.date),
                "storage_account_name": azure_connection.storage_account_name,
                "resource_group_name": azure_connection.resource_group_name,
                "container_name": azure_connection.container_name
            })
            print({"task_id": task.id})

        else:
            task = task_run_ingestion_azure.delay({
                "project_id": project_obj.id,
                "project_name": project_obj.name,
                "azure_connection_id": azure_connection_obj.id,
                # "sync_status_id": sync_status_obj.id,
                "azure_tenant_id": azure_connection.azure_tenant_id,
                "azure_client_id": azure_connection.azure_client_id,
                "azure_client_secret": azure_connection.azure_client_secret,
                "subscription_info": azure_connection.subscription_info,
                "date": str(azure_connection.date),
                "storage_account_name": azure_connection.storage_account_name,
                "monthly_budget": azure_connection.monthly_budget,
                "container_name": azure_connection.container_name
            })
            print({"task_id": task.id})

        return await AzureConnection_Pydantic.from_tortoise_orm(azure_connection_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{azure_connection_id}', response_model=AzureConnection_Pydantic, tags=["azure_connection"])
async def get_azure_connection(azure_connection_id: int):
    try:
        azure_connection = await AzureConnection.get(id=azure_connection_id)
        return await AzureConnection_Pydantic.from_tortoise_orm(azure_connection)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[AzureConnection_Pydantic], tags=["azure_connection"])
async def get_all_azure_connection():
    try:
        # Fetch all encrypted connections
        connections = await AzureConnection.all()

        # Convert to Pydantic models
        return connections
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{azure_connection_id}", response_model=AzureConnection_Pydantic, tags=["azure_connection"])
async def update_azure_connection(azure_connection_id: int, azure_connection: AzureConnectionIn_Pydantic):
    await AzureConnection.filter(id=azure_connection_id).update(**azure_connection.model_dump(exclude_unset=True))
    return await AzureConnection_Pydantic.from_queryset_single(AzureConnection.get(id=azure_connection_id))


@router.post("/test_connection", response_model=AzureConnectionResponse, tags=["azure_connection"])
async def test_azure_connection(request: AzureConnectionRequest):
    azure_client_id = request.azure_client_id
    azure_client_secret = request.azure_client_secret
    azure_tenant_id = request.azure_tenant_id

    # Test connection
    if azure_validate_creds(azure_client_id, azure_client_secret, azure_tenant_id):
        return AzureConnectionResponse(status=True, message='Connection successful')
    else:
        return AzureConnectionResponse(status=False, message='Connection failed')


@router.post("/get_subscriptions", response_model=SubscriptionsResponse, tags=["azure_connection"])
async def get_subscriptions(request: AzureConnectionRequest):
    try:
        subscriptions_info = get_azure_subscriptions(
            client_id=request.azure_client_id,
            client_secret=request.azure_client_secret,
            tenant_id=request.azure_tenant_id,
        )
        return {"subscriptions": subscriptions_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



