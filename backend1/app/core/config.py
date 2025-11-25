import os
from pydantic_settings import BaseSettings
from fastapi_azure_auth import SingleTenantAzureAuthorizationCodeBearer
from pydantic import AnyHttpUrl, computed_field


class Settings(BaseSettings):
    PROJECT_NAME: str = "CloudMeter"
    API_V1_STR: str = "/api/v1"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 30 minutes
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # authentication settings
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://cm.sigmoid.io"
    ]
    OPENAPI_CLIENT_ID: str = os.getenv("OPENAPI_CLIENT_ID")
    APP_CLIENT_ID: str = os.getenv("APP_CLIENT_ID")
    TENANT_ID: str = os.getenv("TENANT_ID")
    SCOPE_DESCRIPTION: str = os.getenv("SCOPE_DESCRIPTION")
    
    @computed_field
    @property
    def SCOPE_NAME(self) -> str:
        return f'api://{self.APP_CLIENT_ID}/{self.SCOPE_DESCRIPTION}'

    @computed_field
    @property
    def SCOPES(self) -> dict:
        return {
            self.SCOPE_NAME: self.SCOPE_DESCRIPTION,
        }
        
    class Config:
        case_sensitive = True


settings = Settings()
