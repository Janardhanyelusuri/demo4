from fastapi import FastAPI, APIRouter, HTTPException
from tortoise.exceptions import DoesNotExist
from typing import List
from app.models.dashboard_request import DashboardRequest, DashboardRequestIn_Pydantic, DashboardRequest_Pydantic
from app.models.project import Project
from app.models.service import Service

router = APIRouter()


@router.post('/', response_model=DashboardRequest_Pydantic, tags=["dashboard_request"])
async def add_dashboard_request(dashboard_request: DashboardRequestIn_Pydantic):
    try:
        # Validate if the project exists
        if not await Project.filter(id=dashboard_request.project_id).exists():
            raise HTTPException(status_code=400, detail="Project not found")

        # Validate if the Service exists
        if not await Service.filter(id=dashboard_request.service_id).exists():
            raise HTTPException(status_code=400, detail="Service not found")

        # Create DashboardRequest record
        dashboard_request_obj = await DashboardRequest.create(**dashboard_request.dict())
        return await DashboardRequest_Pydantic.from_tortoise_orm(dashboard_request_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{id}', response_model=DashboardRequest_Pydantic, tags=["dashboard_request"])
async def get_dashboard_request(dashboard_request_id: int):
    try:
        dashboard_request_obj = await DashboardRequest.get(id=dashboard_request_id)
        return await DashboardRequest_Pydantic.from_tortoise_orm(dashboard_request_obj)
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="DashboardRequest not found")


@router.get('/', response_model=List[DashboardRequest_Pydantic], tags=["dashboard_request"])
async def get_all_dashboard_request():
    try:
        return await DashboardRequest_Pydantic.from_queryset(DashboardRequest.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{dashboard_request_id}", tags=["dashboard_request"])
async def delete_dashboard_request(dashboard_request_id: int):
    await DashboardRequest.filter(id=dashboard_request_id).delete()
    return {"status": True, "message": "Successfully deleted dashboard_request"}


@router.put("/{dashboard_request_id}", response_model=DashboardRequest_Pydantic, tags=["dashboard_request"])
async def update_dashboard_request(dashboard_request_id: int, dashboard_request: DashboardRequestIn_Pydantic):
    await DashboardRequest.filter(id=dashboard_request_id).update(**dashboard_request.model_dump(exclude_unset=True))
    return await DashboardRequest_Pydantic.from_queryset_single(DashboardRequest.get(id=dashboard_request_id))
