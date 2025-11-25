from typing import List
from fastapi import APIRouter, HTTPException
from app.models.project_access import ProjectAccess, ProjectAccess_Pydantic, ProjectAccessIn_Pydantic

router = APIRouter()


@router.post('/', response_model=ProjectAccess_Pydantic, tags=["project_access"])
async def add_project_access(project_access: ProjectAccessIn_Pydantic):
    try:
        project_access_obj = await ProjectAccess.create(**project_access.dict())
        return await ProjectAccess_Pydantic.from_tortoise_orm(project_access_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{project_access_id}', response_model=ProjectAccess_Pydantic, tags=["project_access"])
async def get_project_access(project_access_id: int):
    try:
        project_access = await ProjectAccess.get(id=project_access_id)
        return await ProjectAccess_Pydantic.from_tortoise_orm(project_access)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[ProjectAccess_Pydantic], tags=["project_access"])
async def get_all_project_access():
    try:
        return await ProjectAccess_Pydantic.from_queryset(ProjectAccess.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{project_access_id}", response_model=ProjectAccess_Pydantic, tags=["project_access"])
async def update_project_access(project_access_id: int, project_access: ProjectAccessIn_Pydantic):
    await ProjectAccess.filter(id=project_access_id).update(**project_access.model_dump(exclude_unset=True))
    return await ProjectAccess_Pydantic.from_queryset_single(ProjectAccess.get(id=project_access_id))


@router.delete("/{project_access_id}", tags=["project_access"])
async def delete_project_access(project_access_id: int):
    await ProjectAccess.filter(id=project_access_id).delete()
    return {"status": True, "message": "Successfully deleted"}
