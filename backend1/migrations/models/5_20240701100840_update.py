from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "project" ADD "date" DATE NOT NULL;
        ALTER TABLE "project" ADD "cloud_platform" VARCHAR(50);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "project" DROP COLUMN "date";
        ALTER TABLE "project" DROP COLUMN "cloud_platform";"""
