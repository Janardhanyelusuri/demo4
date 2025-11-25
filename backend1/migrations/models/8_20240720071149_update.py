from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "syncstatus" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "module" VARCHAR(100) NOT NULL,
    "status" VARCHAR(100) NOT NULL,
    "start_date" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "end_date" TIMESTAMPTZ,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "syncstatus";"""
