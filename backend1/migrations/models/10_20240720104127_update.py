from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "awsconnection" ADD "quarterly_budget" INT;
        ALTER TABLE "awsconnection" ADD "yearly_budget" INT;
        ALTER TABLE "awsconnection" ADD "export_location" VARCHAR(100);
        ALTER TABLE "awsconnection" ADD "monthly_budget" INT;
        ALTER TABLE "awsconnection" ADD "status" BOOL NOT NULL  DEFAULT True;
        ALTER TABLE "awsconnection" ADD "export" BOOL NOT NULL  DEFAULT False;"""

async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "awsconnection" DROP COLUMN "quarterly_budget";
        ALTER TABLE "awsconnection" DROP COLUMN "yearly_budget";
        ALTER TABLE "awsconnection" DROP COLUMN "export_location";
        ALTER TABLE "awsconnection" DROP COLUMN "monthly_budget";
        ALTER TABLE "awsconnection" DROP COLUMN "status";
        ALTER TABLE "awsconnection" DROP COLUMN "export";"""
