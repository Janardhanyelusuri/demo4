from typing import List
from fastapi import APIRouter, HTTPException
from app.models.user import Users, UserIn_Pydantic, User_Pydantic

router = APIRouter()


# Endpoint to add a user
@router.post('/user', response_model=User_Pydantic)
async def add_user(user: UserIn_Pydantic):
    try:
        user_obj = await Users.create(**user.dict())
        return await User_Pydantic.from_tortoise_orm(user_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/user/{user_id}', response_model=User_Pydantic)
async def get_user(user_id: int):
    try:
        user = await Users.get(id=user_id)
        return await User_Pydantic.from_tortoise_orm(user)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/user', response_model=List[User_Pydantic])
async def get_all_user():
    try:
        return await User_Pydantic.from_queryset(Users.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/user/{user_id}", response_model=User_Pydantic)
async def update_user(user_id: int, user: UserIn_Pydantic):
    await Users.filter(id=user_id).update(**user.model_dump(exclude_unset=True))
    return await User_Pydantic.from_queryset_single(Users.get(id=user_id))
