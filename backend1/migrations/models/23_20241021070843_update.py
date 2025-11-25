from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" ADD "operation" VARCHAR(100) NOT NULL;
        ALTER TABLE "alert" ADD "tag_id" INT;
        ALTER TABLE "alert" ADD CONSTRAINT "fk_alert_tag_13342050" FOREIGN KEY ("tag_id") REFERENCES "tag" ("tag_id") ON DELETE CASCADE;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" DROP CONSTRAINT "fk_alert_tag_13342050";
        ALTER TABLE "alert" DROP COLUMN "operation";
        ALTER TABLE "alert" DROP COLUMN "tag_id";"""
