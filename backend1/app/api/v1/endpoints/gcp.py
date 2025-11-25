import json
from typing import List
import os
from app.core.encryption import decrypt_data
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from app.schemas.connection import GCPConnectionResponse, GCPConfirmationResponse, ProjectInfo, ProjectsResponse, BillingAccountResponse
from app.core.gcp import gcp_validate_creds, list_projects, list_billing_account_ids, list_datasets
from app.worker.celery_worker import task_run_ingestion_gcp
from app.models.gcp import GCPConnection, GCPConnection_Pydantic, GCPConnectionIn_Pydantic
from app.models.project import Project

router = APIRouter()


@router.post("/", response_model=GCPConnection_Pydantic, tags=["gcp_connection"])
async def add_gcp_connection(
        file: UploadFile = File(...),
        date: str = Form(...),
        project_id: int = Form(...),
        gcp_project_id: str = Form(...),
        gcp_project_name: str = Form(...),
        status: bool = Form(...),
        export: bool = Form(...),
        billing_account_id: str = Form(...),
        dataset_id: str = Form(...),
        yearly_budget: int = Form(...),
):
    try:
        credentials = json.loads(file.file.read().decode("utf-8"))
        if credentials:
            if not gcp_validate_creds(credentials):
                raise Exception("Invalid credentials")

            # change monthly_budget and quarterly_budget using yearly_budget
            monthly_budget = int(yearly_budget / 12)
            quarterly_budget = int(yearly_budget / 3)

            gcp_connection_dict = {
                "credentials": credentials,
                "date": date,
                "project_info": {
                    "project_id": gcp_project_id,
                    "project_name": gcp_project_name
                },
                "project_id": project_id,
                "dataset_id": dataset_id,
                "billing_account_id": billing_account_id,
                "yearly_budget": yearly_budget,
                "quarterly_budget": quarterly_budget,
                "monthly_budget": monthly_budget,
                "status": status,
                "export": export,
            }
            gcp_connection_obj = await GCPConnection.create(**gcp_connection_dict)

            project_obj = await Project.filter(id=project_id).first()

            if export:
                # AWS CUR
                task = task_run_ingestion_gcp.delay({
                    "project_id": project_obj.id,
                    "project_name": project_obj.name,
                    "gcp_project_id": gcp_project_id,
                    "gcp_project_name": gcp_project_name,
                    "gcp_connection_id": gcp_connection_obj.id,
                    # "sync_status_id": sync_status_obj.id,
                    "monthly_budget": monthly_budget,
                    "credentials": credentials,  # gcp creds
                    "dataset_id": dataset_id,
                    "billing_account_id": billing_account_id,
                    "date": date
                })
                print({"task_id": task.id})

            return await GCPConnection_Pydantic.from_tortoise_orm(gcp_connection_obj)
        raise Exception('Failed')
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{gcp_connection_id}', response_model=GCPConnection_Pydantic, tags=["gcp_connection"])
async def get_gcp_connection(gcp_connection_id: int):
    try:
        gcp_connection = await GCPConnection.get(id=gcp_connection_id)
        return await GCPConnection_Pydantic.from_tortoise_orm(gcp_connection)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[GCPConnection_Pydantic], tags=["gcp_connection"])
async def get_all_gcp_connection():
    try:
        return await GCPConnection_Pydantic.from_queryset(GCPConnection.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{gcp_connection_id}", response_model=GCPConnection_Pydantic, tags=["gcp_connection"])
async def update_gcp_connection(gcp_connection_id: int, gcp_connection: GCPConnectionIn_Pydantic):
    await GCPConnection.filter(id=gcp_connection_id).update(**gcp_connection.model_dump(exclude_unset=True))
    return await GCPConnection_Pydantic.from_queryset_single(GCPConnection.get(id=gcp_connection_id))


@router.post("/test_connection", response_model=GCPConnectionResponse, tags=["gcp_connection"])
async def test_gcp_connection(file: UploadFile = File(...)):
    credentials = json.loads(file.file.read().decode("utf-8"))

    if credentials:
        if gcp_validate_creds(credentials):
            return GCPConnectionResponse(status=True, message='Connection successful')
        else:
            return GCPConnectionResponse(status=False, message='Connection Failed')
    else:
        return GCPConnectionResponse(status=False, message='Connection Failed')


@router.post("/get_projects", response_model=ProjectsResponse, tags=["gcp_connection"])
async def get_projects(file: UploadFile = File(...)):
    try:
        credentials = json.loads(file.file.read().decode("utf-8"))
        projects_info = list_projects(credentials)
        projects = [ProjectInfo(project_id=project['projectId'], project_name=project['name']) for project in projects_info]
        return ProjectsResponse(projects=projects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_billing_account_ids", response_model=List[BillingAccountResponse], tags=["gcp_connection"])
async def get_billing_account_ids(file: UploadFile = File(...)):
    try:
        credentials = json.loads(file.file.read().decode("utf-8"))
        return list_billing_account_ids(credentials)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_datasets", tags=["gcp_connection"])
async def get_datasets(file: UploadFile = File(...)):
    try:
        credentials = json.loads(file.file.read().decode("utf-8"))
        return list_datasets(credentials)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

