from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" ADD "integration_id" INT NOT NULL;
        ALTER TABLE "alert" ADD CONSTRAINT "fk_alert_integrat_599cd950" FOREIGN KEY ("integration_id") REFERENCES "integration" ("id") ON DELETE CASCADE;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" DROP CONSTRAINT "fk_alert_integrat_599cd950";
        ALTER TABLE "alert" DROP COLUMN "integration_id";"""
