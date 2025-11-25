from app.ingestion.aws.aws_cur.postgres.scripts.s3 import get_parquet_file_from_s3, delete_s3_bucket
from app.ingestion.aws.aws_cur.postgres.scripts.postgres_operations import dump_to_postgresql
from app.ingestion.aws.aws_cur.postgres.scripts.sql_read import run_sql_file
import os


def aws_cur_main(project_name,
                 monthly_budget,
                 aws_access_key,
                 aws_secret_key,
                 aws_region,
                 s3_bucket,
                 s3_prefix,
                 export_name,
                 billing_period):
    try:
        df = get_parquet_file_from_s3(aws_access_key,
                                      aws_secret_key,
                                      aws_region,
                                      s3_bucket,
                                      s3_prefix,
                                      export_name,
                                      billing_period)

        schema_name = project_name
        base_path = "app/ingestion/aws/aws_cur/postgres"

        run_sql_file(sql_file_path=f'{base_path}/scripts/sql/new_schema.sql',
                     schema_name=project_name,
                     budget=monthly_budget)

        table_name = "bronze_aws_cur_standard"
        dump_to_postgresql(df, schema_name, table_name)

        run_sql_file(sql_file_path=f'{base_path}/scripts/sql/silver.sql',
                     schema_name=project_name,
                     budget=monthly_budget)
        run_sql_file(sql_file_path=f'{base_path}/scripts/sql/gold_views.sql',
                     schema_name=project_name,
                     budget=monthly_budget)
    except Exception as ex:
        print(ex)
