from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "snowflakeconnection" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "account_name" VARCHAR(100) NOT NULL,
    "user_name" VARCHAR(100) NOT NULL,
    "password" VARCHAR(100) NOT NULL,
    "warehouse_name" VARCHAR(100) NOT NULL,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "snowflakeconnection";"""
