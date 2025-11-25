from fastapi import APIRouter, Security
# from app.api.v1.endpoints import user
from app.api.v1.dependencies.auth import azure_scheme

# api_router = APIRouter()
# api_router.include_router(user.router, prefix="/users", tags=["users"], dependencies=[Security(azure_scheme)])
