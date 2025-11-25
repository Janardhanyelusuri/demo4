# from google.oauth2 import service_account
# from google.cloud import bigquery
# from app.ingestion.gcp.postgres_operations import dump_to_postgresql, run_sql_file


# def fetch_data_from_bigquery_to_postgres(project_id, dataset_id, view_id, credentials, schema, table_name, monthly_budget):
#     # Set the environment variable for authentication
#     credentials_obj = service_account.Credentials.from_service_account_info(credentials)

#     # Initialize the BigQuery client
#     client = bigquery.Client(credentials=credentials_obj, project=project_id)

#     # Construct the query
#     query = f"""
#     SELECT *
#     FROM `{project_id}.{dataset_id}.{view_id}`
#     """

#     # Execute the query and convert the results to a pandas DataFrame
#     dataframe = client.query(query).to_dataframe()

#     # Convert DataFrame columns to types that are compatible with PostgreSQL
#     for col in dataframe.columns:
#         if dataframe[col].dtype == 'object':
#             dataframe[col] = dataframe[col].astype(str)
#         elif dataframe[col].dtype == 'int64':
#             dataframe[col] = dataframe[col].astype(int)
#         elif dataframe[col].dtype == 'float64':
#             dataframe[col] = dataframe[col].astype(float)

#     base_path = "app/ingestion/gcp"

#     # Create new schema
#     run_sql_file(sql_file_path=f'{base_path}/sql/new_schema.sql',
#                  schema_name=schema,
#                  budget=monthly_budget
#                  )

#     # Write the DataFrame to the PostgreSQL table
#     dump_to_postgresql(df=dataframe, schema_name=schema, table_name=table_name)
#     print(f'Data successfully fetched from BigQuery and inserted into {schema}.{table_name} in PostgreSQL.')

#     # Run the bronze-to-silver SQL script
#     run_sql_file(sql_file_path=f'{base_path}/sql/silver.sql',
#                  schema_name=schema,
#                  budget=monthly_budget
#                  )

#     # Run the silver-to-gold SQL script
#     run_sql_file(sql_file_path=f'{base_path}/sql/gold.sql',
#                  schema_name=schema,
#                  budget=monthly_budget
#                  )




import hashlib
import os
import tempfile
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
from sqlalchemy import create_engine, inspect
from .postgres_operations import run_sql_file, dump_to_postgresql, connection
from sqlalchemy.exc import SQLAlchemyError

# project_id = "cloud-meter-dev"
# dataset_id = "cloud_dataset"
# view_id = "focus_format_temp"
# credentials_path = "/Users/adityakumarbharatdeshmukh/Downloads/cloud-meter-dev-2ee2c6735be0.json"
# schema = "test"
# table_name = "gcp_temp"

def generate_hash_key(row):
    """
    Generate an MD5 hash key for a given row by concatenating all column values.
    """
    # print("This is rowww-------",row)
    concatenated_values = ''.join(str(value) for value in row)
    return hashlib.md5(concatenated_values.encode('utf-8')).hexdigest()

@connection
def fetch_existing_hash_keys(connection, schema, table_name):
    try:
        cursor = connection.cursor()
        query = f"SELECT hash_key FROM {schema}.{table_name};"
        cursor.execute(query)
        rows = cursor.fetchall()
        existing_keys = set(row[0] for row in rows)
        print(f"Fetched {len(existing_keys)} existing hash keys.")
        return existing_keys

    except Exception as ex:
        print(f"Error fetching hash keys: {ex}")
        return set()


def fetch_data_from_bigquery_to_postgres(project_id, dataset_id, view_id, credentials, schema, table_name, monthly_budget):
    # Hardcoded path to the SQL script
    # sql_script_path = "/Users/adityakumarbharatdeshmukh/Desktop/project/bigquery_to_postgress/sql/create_table.sql"
    print("-------", credentials)
    print("Data type of credentials:", type(credentials))

    base_path = "app/ingestion/gcp"
    table_name = 'bronze_focus_gcp_data'
        # parent_folder = f'{s3_prefix}/{export_name}/data/'

    # Use from_service_account_info if credentials is a dict
    if isinstance(credentials, dict):
        credentials_obj = service_account.Credentials.from_service_account_info(credentials)
    else:
        # Fallback for file-based credentials
        credentials_obj = service_account.Credentials.from_service_account_file(credentials)

    # Initialize the BigQuery client
    client = bigquery.Client(credentials=credentials_obj, project=project_id)

    print("BigQuery client initialized successfully.")

    # Construct the query
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{view_id}`
    """

    # Execute the query and convert the results to a pandas DataFrame
    dataframe = client.query(query).to_dataframe()

    # Replace problematic None/NaN values
    dataframe = dataframe.fillna(value="")

    # Log the data fetched from BigQuery
    print(f"Fetched {len(dataframe)} rows from BigQuery.")
    # Create new schema
    run_sql_file(sql_file_path=f'{base_path}/sql/new_schema.sql',
                 schema_name=schema,
                 budget=monthly_budget
                 )
    print(f"Schema {schema} created...")

    # Execute the SQL script to ensure the table exists in PostgreSQL
    run_sql_file(sql_file_path=f'{base_path}/sql/create_table.sql',
                 schema_name=schema,
                 budget=monthly_budget
                 )
    print(f"Table {table_name} created...")

    print(f"Table {schema}.{table_name} ensured to exist.")

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as temp_csv_file:
        temp_csv_path = temp_csv_file.name
        # Save the DataFrame to the CSV file
        dataframe.to_csv(temp_csv_file, index=False)
        print(f"DataFrame saved to temporary CSV file: {temp_csv_path}")

    # Read the CSV file back into a DataFrame
    temp_dataframe = pd.read_csv(temp_csv_path)
    print(f"CSV file loaded into DataFrame with {len(temp_dataframe)} rows.")

    # Create hash keys for each row after re-loading the data from CSV
    temp_dataframe['hash_key'] = temp_dataframe.apply(generate_hash_key, axis=1)

    # Use the function
    existing_keys = fetch_existing_hash_keys(schema, table_name)
    print(f"Existing keys: {len(existing_keys)}")


    # # Filter out rows that already exist in the PostgreSQL table
    new_data = temp_dataframe[~temp_dataframe['hash_key'].isin(existing_keys)]
    
    # Log the number of new rows before appending
    print(f"New rows to append: {len(new_data)}")

    # Only dump new data into PostgreSQL
    if not new_data.empty:
        dump_to_postgresql(new_data, schema, table_name)
        print(f"Appended {len(new_data)} new rows to {schema}.{table_name}.")
    # Run the bronze-to-silver SQL script
        run_sql_file(sql_file_path=f'{base_path}/sql/silver.sql',
                    schema_name=schema,
                    budget=monthly_budget
                    )
        # Run the silver-to-gold SQL script
        run_sql_file(sql_file_path=f'{base_path}/sql/gold.sql',
                    schema_name=schema,
                    budget=monthly_budget
                    )
    else:
        print("No new rows to append. All data already exists in the PostgreSQL table.")

    # Remove the temporary CSV file after the operation
    os.remove(temp_csv_path)
    print(f"Temporary CSV file removed: {temp_csv_path}")

# Call the function
# fetch_data_from_bigquery_to_postgres(project_id, dataset_id, view_id, credentials_path, schema, table_name)

