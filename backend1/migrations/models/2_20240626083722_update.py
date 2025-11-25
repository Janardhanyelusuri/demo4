from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "project" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(50) NOT NULL,
    "status" BOOL NOT NULL  DEFAULT True
);
        CREATE TABLE IF NOT EXISTS "awsconnection" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "aws_access_key" VARCHAR(100) NOT NULL,
    "aws_secret_key" VARCHAR(100) NOT NULL,
    "date" DATE NOT NULL,
    "project_id" INT NOT NULL REFERENCES "project" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "project";
        DROP TABLE IF EXISTS "awsconnection";"""
