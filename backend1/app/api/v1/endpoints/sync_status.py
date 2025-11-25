from typing import List
from fastapi import APIRouter, HTTPException
from app.models.sync_status import SyncStatus, SyncStatus_Pydantic, SyncStatusIn_Pydantic

router = APIRouter()


@router.post('/', response_model=SyncStatus_Pydantic, tags=["sync_status"])
async def add_sync_status(sync_status: SyncStatusIn_Pydantic):
    try:
        sync_status_obj = await SyncStatus.create(**sync_status.dict())
        return await SyncStatus_Pydantic.from_tortoise_orm(sync_status_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{sync_status_id}', response_model=SyncStatus_Pydantic, tags=["sync_status"])
async def get_sync_status(sync_status_id: int):
    try:
        sync_status = await SyncStatus.get(id=sync_status_id)
        return await SyncStatus_Pydantic.from_tortoise_orm(sync_status)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[SyncStatus_Pydantic], tags=["sync_status"])
async def get_all_sync_status():
    try:
        return await SyncStatus_Pydantic.from_queryset(SyncStatus.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{sync_status_id}", response_model=SyncStatus_Pydantic, tags=["sync_status"])
async def update_sync_status(sync_status_id: int, sync_status: SyncStatusIn_Pydantic):
    await SyncStatus.filter(id=sync_status_id).update(**sync_status.model_dump(exclude_unset=True))
    return await SyncStatus_Pydantic.from_queryset_single(SyncStatus.get(id=sync_status_id))


@router.delete("/{sync_status_id}", tags=["sync_status"])
async def delete_sync_status(sync_status_id: int):
    await SyncStatus.filter(id=sync_status_id).delete()
    return {"status": True, "message": "Successfully deleted"}
