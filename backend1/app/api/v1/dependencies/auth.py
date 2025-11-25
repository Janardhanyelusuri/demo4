from fastapi_azure_auth import SingleTenantAzureAuthorizationCodeBearer
from app.core.config import settings

azure_scheme = SingleTenantAzureAuthorizationCodeBearer(
    app_client_id=settings.APP_CLIENT_ID,
    tenant_id=settings.TENANT_ID,
    scopes=settings.SCOPES,
)