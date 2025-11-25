from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "azureconnection" ADD "monthly_budget" INT;
        ALTER TABLE "azureconnection" ADD "storage_account_name" VARCHAR(100) NOT NULL;
        ALTER TABLE "azureconnection" ADD "export" BOOL NOT NULL  DEFAULT False;
        ALTER TABLE "azureconnection" ADD "status" BOOL NOT NULL  DEFAULT False;
        ALTER TABLE "azureconnection" ADD "yearly_budget" INT;
        ALTER TABLE "azureconnection" ADD "resource_group_name" VARCHAR(100);
        ALTER TABLE "azureconnection" ADD "quarterly_budget" INT;
        ALTER TABLE "azureconnection" ADD "container_name" VARCHAR(100);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "azureconnection" DROP COLUMN "monthly_budget";
        ALTER TABLE "azureconnection" DROP COLUMN "storage_account_name";
        ALTER TABLE "azureconnection" DROP COLUMN "export";
        ALTER TABLE "azureconnection" DROP COLUMN "status";
        ALTER TABLE "azureconnection" DROP COLUMN "yearly_budget";
        ALTER TABLE "azureconnection" DROP COLUMN "resource_group_name";
        ALTER TABLE "azureconnection" DROP COLUMN "quarterly_budget";
        ALTER TABLE "azureconnection" DROP COLUMN "container_name";"""
