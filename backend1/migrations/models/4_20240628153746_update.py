from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "gcpconnection" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "credentials" JSONB NOT NULL,
    "project_info" JSONB NOT NULL,
    "date" DATE NOT NULL,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "gcpconnection";"""
