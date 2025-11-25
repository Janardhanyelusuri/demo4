from typing import List
import os
from app.core.encryption import decrypt_data
from fastapi import APIRouter, HTTPException
from app.schemas.connection import SnowflakeConnectionRequest, SnowflakeConnectionResponse, WarehousesResponse
from app.core.snowflake import snowflake_validate_creds, list_warehouses
from app.models.snowflake import SnowflakeConnection, SnowflakeConnection_Pydantic, SnowflakeConnectionIn_Pydantic

router = APIRouter()


@router.post('/', response_model=SnowflakeConnection_Pydantic, tags=["snowflake_connection"])
async def add_snowflake_connection(snowflake_connection: SnowflakeConnectionIn_Pydantic):
    try:
        # Test connection
        if not snowflake_validate_creds(snowflake_connection.account_name, snowflake_connection.user_name, snowflake_connection.password):
            raise Exception('Invalid credentials')
        snowflake_connection_obj = await SnowflakeConnection.create(**snowflake_connection.dict())
        return await SnowflakeConnection_Pydantic.from_tortoise_orm(snowflake_connection_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{snowflake_connection_id}', response_model=SnowflakeConnection_Pydantic, tags=["snowflake_connection"])
async def get_snowflake_connection(snowflake_connection_id: int):
    try:
        snowflake_connection = await SnowflakeConnection.get(id=snowflake_connection_id)
        return await SnowflakeConnection_Pydantic.from_tortoise_orm(snowflake_connection)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[SnowflakeConnection_Pydantic], tags=["snowflake_connection"])
async def get_all_snowflake_connection():
    try:
        # Fetch all encrypted connections
        connections = await SnowflakeConnection.all()

        # Convert to Pydantic models
        return connections
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{snowflake_connection_id}", response_model=SnowflakeConnection_Pydantic, tags=["snowflake_connection"])
async def update_snowflake_connection(snowflake_connection_id: int, snowflake_connection: SnowflakeConnectionIn_Pydantic):
    await SnowflakeConnection.filter(id=snowflake_connection_id).update(**snowflake_connection.model_dump(exclude_unset=True))
    return await SnowflakeConnection_Pydantic.from_queryset_single(SnowflakeConnection.get(id=snowflake_connection_id))


@router.post("/test_connection", response_model=SnowflakeConnectionResponse, tags=["snowflake_connection"])
async def test_aws_connection(request: SnowflakeConnectionRequest):
    account_name = request.account_name
    user_name = request.user_name
    password = request.password

    # Test connection
    if snowflake_validate_creds(account_name, user_name, password):
        return SnowflakeConnectionResponse(status=True, message='Connection successful')
    else:
        return SnowflakeConnectionResponse(status=False, message='Connection failed')
    

@router.post("/get_warehouses", response_model=WarehousesResponse, tags=["snowflake_connection"])
async def get_warehouses(request: SnowflakeConnectionRequest):
    try:
        warehouses_info = list_warehouses(
            account_name=request.account_name,
            user_name=request.user_name,
            password=request.password,
        )
        return {"warehouses": warehouses_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))