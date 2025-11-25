import datetime
import os
from dotenv import load_dotenv
from .scripts.aws_ce.ce_ops import run_ce_operations
from .scripts.postgres_operations import dump_to_postgresql
from .scripts.sql_read import run_sql_file
import pandas as pd

# Load environment variables from .env file
load_dotenv()


def aws_ce_main(project_name,
                access_key,
                secret_key,
                start_date):
    print(os.getcwd())

    # Execute AWS CE operations
    ce_data = run_ce_operations(start_date, pd.Timestamp.today().strftime('%Y-%m-%d'), access_key, secret_key)

    # Dump AWS CE data to PostgreSQL
    schema_name = project_name
    table_name = "bronze_aws_ce"

    base_path = "app/ingestion/aws/aws_ce"
    run_sql_file(sql_file_path=f'{base_path}/scripts/sql/new_schema.sql',
                 schema_name=schema_name)
    dump_to_postgresql(ce_data, schema_name, table_name)

    run_sql_file(sql_file_path=f'{base_path}/scripts/sql/silver.sql',
                 schema_name=schema_name)
    run_sql_file(sql_file_path=f'{base_path}/scripts/sql/gold_views.sql',
                 schema_name=schema_name)


# if __name__ == '__main__':
#     aws_ce_main()
