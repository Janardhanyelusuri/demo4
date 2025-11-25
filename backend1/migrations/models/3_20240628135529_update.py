from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "azureconnection" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "azure_tenant_id" VARCHAR(100) NOT NULL,
    "azure_client_id" VARCHAR(100) NOT NULL,
    "azure_client_secret" VARCHAR(100) NOT NULL,
    "subscription_info" JSONB NOT NULL,
    "date" DATE NOT NULL,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE
);
        CREATE TABLE IF NOT EXISTS "projectaccess" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "access_granted_by" VARCHAR(100) NOT NULL,
    "access_granted_on" DATE NOT NULL,
    "status" BOOL NOT NULL  DEFAULT True,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "azureconnection";
        DROP TABLE IF EXISTS "projectaccess";"""
