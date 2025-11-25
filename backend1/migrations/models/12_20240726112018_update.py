from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "gcpconnection" ADD "dataset_id" VARCHAR(100);
        ALTER TABLE "gcpconnection" ADD "billing_account_id" VARCHAR(100);
        ALTER TABLE "gcpconnection" ADD "yearly_budget" INT;
        ALTER TABLE "gcpconnection" ADD "monthly_budget" INT;
        ALTER TABLE "gcpconnection" ADD "quarterly_budget" INT;
        ALTER TABLE "gcpconnection" ADD "status" BOOL NOT NULL  DEFAULT False;
        ALTER TABLE "gcpconnection" ADD "export" BOOL NOT NULL  DEFAULT False;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "gcpconnection" DROP COLUMN "dataset_id";
        ALTER TABLE "gcpconnection" DROP COLUMN "billing_account_id";
        ALTER TABLE "gcpconnection" DROP COLUMN "yearly_budget";
        ALTER TABLE "gcpconnection" DROP COLUMN "monthly_budget";
        ALTER TABLE "gcpconnection" DROP COLUMN "quarterly_budget";
        ALTER TABLE "gcpconnection" DROP COLUMN "status";
        ALTER TABLE "gcpconnection" DROP COLUMN "export";"""
