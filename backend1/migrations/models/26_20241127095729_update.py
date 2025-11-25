from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "dashboard" ADD "cloud_platforms" JSONB;
        ALTER TABLE "dashboard" DROP COLUMN "cloud_platform";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "dashboard" ADD "cloud_platform" VARCHAR(50);
        ALTER TABLE "dashboard" DROP COLUMN "cloud_platforms";"""
