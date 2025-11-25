from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "resource_tag" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "resource_id" INT NOT NULL REFERENCES "resource_dim" ("id") ON DELETE CASCADE,
    "tag_id" INT REFERENCES "tag" ("tag_id") ON DELETE CASCADE,
    CONSTRAINT "uid_resource_ta_resourc_20c425" UNIQUE ("resource_id", "tag_id")
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "resource_tag";"""
