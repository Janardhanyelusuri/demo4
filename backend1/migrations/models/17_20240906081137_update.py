from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "resource_dim" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "resource_id" VARCHAR(1024) NOT NULL UNIQUE,
    "resource_name" VARCHAR(255) NOT NULL,
    "region_id" VARCHAR(100) NOT NULL,
    "region_name" VARCHAR(255) NOT NULL,
    "service_category" VARCHAR(255) NOT NULL,
    "service_name" VARCHAR(255) NOT NULL,
    "cloud_platform" VARCHAR(255),
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE,
    "tag_id" INT REFERENCES "tag" ("tag_id") ON DELETE CASCADE
);
COMMENT ON TABLE "resource_dim" IS 'ORM Model representing the Resource entity.';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "resource_dim";"""
