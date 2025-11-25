from tortoise import BaseDBAsyncClient

async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints 
                WHERE constraint_name = 'fk_alert_project_b10b443d'
            ) THEN
                ALTER TABLE "alert" DROP CONSTRAINT "fk_alert_project_b10b443d";
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints 
                WHERE constraint_name = 'fk_alert_tag_13342050'
            ) THEN
                ALTER TABLE "alert" DROP CONSTRAINT "fk_alert_tag_13342050";
            END IF;
        END $$;

        ALTER TABLE "alert" ADD "project_ids" JSONB;
        ALTER TABLE "alert" ADD "tag_ids" JSONB;
        ALTER TABLE "alert" DROP COLUMN "project_id";
        ALTER TABLE "alert" DROP COLUMN "tag_id";"""

async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "alert" ADD "project_id" INT NOT NULL;
        ALTER TABLE "alert" ADD "tag_id" INT;
        ALTER TABLE "alert" DROP COLUMN "project_ids";
        ALTER TABLE "alert" DROP COLUMN "tag_ids";

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints 
                WHERE constraint_name = 'fk_alert_tag_13342050'
            ) THEN
                ALTER TABLE "alert" ADD CONSTRAINT "fk_alert_tag_13342050" FOREIGN KEY ("tag_id") REFERENCES "tag" ("tag_id") ON DELETE CASCADE;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints 
                WHERE constraint_name = 'fk_alert_project_b10b443d'
            ) THEN
                ALTER TABLE "alert" ADD CONSTRAINT "fk_alert_project_b10b443d" FOREIGN KEY ("project_id") REFERENCES "project" ("id") ON DELETE CASCADE;
            END IF;
        END $$;"""
