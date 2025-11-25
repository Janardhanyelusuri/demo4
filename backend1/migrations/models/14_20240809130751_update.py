from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "service" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "cloud_platform" VARCHAR(100) NOT NULL,
    "label" VARCHAR(100),
    "name" VARCHAR(100) NOT NULL,
    "status" BOOL NOT NULL  DEFAULT False
);
        CREATE TABLE IF NOT EXISTS "dashboardrequest" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "requested_on" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "requested_by" VARCHAR(100) NOT NULL,
    "status" BOOL NOT NULL  DEFAULT False,
    "message" VARCHAR(100),
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE,
    "service_id" INT NOT NULL REFERENCES "service" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "dashboardrequest";
        DROP TABLE IF EXISTS "service";"""
