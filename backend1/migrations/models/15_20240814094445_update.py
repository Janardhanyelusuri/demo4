from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "dashboard" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(50) NOT NULL,
    "status" BOOL NOT NULL  DEFAULT True,
    "date" DATE NOT NULL,
    "cloud_platform" VARCHAR(50),
    "persona" JSONB,
    "project_ids" JSONB
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "dashboard";"""
