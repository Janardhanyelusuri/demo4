from typing import List
from fastapi import APIRouter, HTTPException
from app.models.database import Database, Database_Pydantic, DatabaseIn_Pydantic


router = APIRouter()


@router.post('/', response_model=Database_Pydantic, tags=["database"])
async def add_database(database: DatabaseIn_Pydantic):
    try:
        # check if project with same name already exists
        obj = await Database.filter(name=database.name).first()
        if obj:
            raise Exception('Database with same name already exists')
        database_obj = await Database.create(**database.dict())
        return await Database_Pydantic.from_tortoise_orm(database_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{database_id}', response_model=Database_Pydantic, tags=["database"])
async def get_database(database_id: int):
    try:
        database = await Database.get(id=database_id)
        return await Database_Pydantic.from_tortoise_orm(database)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[Database_Pydantic], tags=["database"])
async def get_all_database():
    try:
        return await Database_Pydantic.from_queryset(Database.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{database_id}", response_model=Database_Pydantic, tags=["database"])
async def update_database(database_id: int, database: Database_Pydantic):
    await Database.filter(id=database_id).update(**database.model_dump(exclude_unset=True))
    return await Database_Pydantic.from_queryset_single(Database.get(id=database_id))


@router.delete("/{database_id}", tags=["database"])
async def delete_database(database_id: int):
    await Database.filter(id=database_id).delete()
    return {"status": True, "message": "Successfully deleted"}
