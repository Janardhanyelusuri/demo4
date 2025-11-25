from tortoise import BaseDBAsyncClient

async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_resource_di_resourc_fe293f') THEN
                DROP INDEX "idx_resource_di_resourc_fe293f";
            END IF;
            IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'uid_resource_di_resourc_fe293f') THEN
                DROP INDEX "uid_resource_di_resourc_fe293f";
            END IF;
            IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_resource_dim_resource_id') THEN
                DROP INDEX "idx_resource_dim_resource_id";
            END IF;
        END $$;
        
        ALTER TABLE "resource_dim" ALTER COLUMN "resource_id" DROP NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "region_id" DROP NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "region_name" DROP NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "service_category" DROP NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "service_name" DROP NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "resource_name" DROP NOT NULL;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "resource_dim" ALTER COLUMN "resource_id" SET NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "service_name" SET NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "resource_name" SET NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "region_id" SET NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "region_name" SET NOT NULL;
        ALTER TABLE "resource_dim" ALTER COLUMN "service_category" SET NOT NULL;
        CREATE UNIQUE INDEX "uid_resource_di_resourc_fe293f" ON "resource_dim" ("resource_id");"""
