from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "tag" (
    "tag_id" SERIAL NOT NULL PRIMARY KEY,
    "key" VARCHAR(255) NOT NULL,
    "value" VARCHAR(255) NOT NULL,
    "budget" INT
);
COMMENT ON TABLE "tag" IS 'ORM Model representing the Tag entity.';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "tag";"""
