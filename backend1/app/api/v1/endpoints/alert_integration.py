from fastapi import FastAPI, APIRouter, HTTPException
from tortoise.exceptions import DoesNotExist
from typing import List
from app.models.alert_integration import Integration, IntegrationIn_Pydantic, Integration_Pydantic
from app.models.project import Project  # Ensure Project model is imported for validation if needed

router = APIRouter()


@router.post('/', response_model=Integration_Pydantic, tags=["integration"])
async def add_integration(integration: IntegrationIn_Pydantic):
    try:
        # Validate that the project exists
        if not await Project.filter(id=integration.project_id).exists():
            raise HTTPException(status_code=400, detail="Project not found")

        # Create Integration record
        integration_obj = await Integration.create(**integration.dict())
        return await Integration_Pydantic.from_tortoise_orm(integration_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{id}', response_model=Integration_Pydantic, tags=["integration"])
async def get_integration(integration_id: int):
    try:
        integration_obj = await Integration.get(id=integration_id)
        return await Integration_Pydantic.from_tortoise_orm(integration_obj)
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Integration not found")


@router.get('/', response_model=List[Integration_Pydantic], tags=["integration"])
async def get_all_integration():
    try:
        return await Integration_Pydantic.from_queryset(Integration.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{integration_id}", tags=["integration"])
async def delete_integration(integration_id: int):
    await Integration.filter(id=integration_id).delete()
    return {"status": True, "message": "Successfully deleted integration"}


@router.put("/{integration_id}", response_model=Integration_Pydantic, tags=["integration"])
async def update_integration(integration_id: int, integration: IntegrationIn_Pydantic):
    await Integration.filter(id=integration_id).update(**integration.model_dump(exclude_unset=True))
    return await Integration_Pydantic.from_queryset_single(Integration.get(id=integration_id))
