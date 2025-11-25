from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" ADD "state" JSONB;
        ALTER TABLE "alert" ADD "value_threshold" DOUBLE PRECISION;
        ALTER TABLE "alert" ADD "type" VARCHAR(100) NOT NULL;
        ALTER TABLE "alert" ADD "ends_on" DATE NOT NULL;
        ALTER TABLE "alert" ADD "percentage_threshold" DOUBLE PRECISION;
        ALTER TABLE "alert" ADD "condition" VARCHAR(100) NOT NULL;
        ALTER TABLE "alert" ADD "resource_list" JSONB;
        ALTER TABLE "alert" ADD "alert_type" VARCHAR(100) NOT NULL;
        ALTER TABLE "alert" DROP COLUMN "tag";
        ALTER TABLE "alert" DROP COLUMN "health";
        ALTER TABLE "alert" DROP COLUMN "column";
        ALTER TABLE "alert" DROP COLUMN "table";
        ALTER TABLE "alert" DROP COLUMN "end_in";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" ADD "tag" JSONB;
        ALTER TABLE "alert" ADD "health" VARCHAR(100);
        ALTER TABLE "alert" ADD "column" JSONB NOT NULL;
        ALTER TABLE "alert" ADD "table" VARCHAR(100) NOT NULL;
        ALTER TABLE "alert" ADD "end_in" VARCHAR(100) NOT NULL;
        ALTER TABLE "alert" DROP COLUMN "state";
        ALTER TABLE "alert" DROP COLUMN "value_threshold";
        ALTER TABLE "alert" DROP COLUMN "type";
        ALTER TABLE "alert" DROP COLUMN "ends_on";
        ALTER TABLE "alert" DROP COLUMN "percentage_threshold";
        ALTER TABLE "alert" DROP COLUMN "condition";
        ALTER TABLE "alert" DROP COLUMN "resource_list";
        ALTER TABLE "alert" DROP COLUMN "alert_type";"""
