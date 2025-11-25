from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "alert" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(100) NOT NULL,
    "table" VARCHAR(100) NOT NULL,
    "column" JSONB NOT NULL,
    "recipient" VARCHAR(100) NOT NULL,
    "end_in" VARCHAR(100) NOT NULL,
    "schedule" VARCHAR(100) NOT NULL,
    "tag" JSONB,
    "health" VARCHAR(100),
    "status" BOOL NOT NULL  DEFAULT False,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "alert";"""
