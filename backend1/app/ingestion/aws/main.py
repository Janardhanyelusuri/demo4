
import hashlib
import pandas as pd
from .export_ops import create_export, update_export, create_boto3_client
from .s3 import *
from .postgres_operations import execute_sql_files, dump_to_postgresql, fetch_existing_hash_keys,connection
import pandas as pd
from app.ingestion.aws.export_ops import create_export, update_export, create_boto3_client
from app.ingestion.aws.s3 import *
from app.ingestion.aws.postgres_operations import execute_sql_files, dump_to_postgresql
from .resource_metrics import fetch_and_store_cloudwatch_metrics
from app.ingestion.aws.metrics_s3 import metrics_dump as metrics_dump_s3
from app.ingestion.aws.metrics_ec2 import metrics_dump as metrics_dump_ec2
from .pricing import fetch_and_store_all_aws_pricing
import sys
import os

# Add path for recommendation pre-warming
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.core.recommendation_prewarm import prewarm_aws_recommendations



def generate_hash_key(df):
    """
    Generate a hash key for each row in the DataFrame by concatenating all column values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with an additional 'hash_key' column.
    """
    # Fill NaN/NULL values with an empty string
    df = df.fillna("")

    # Function to compute the hash key for a single row
    def hash_row(row):
        concatenated_values = "".join(map(str, row))
        return hashlib.md5(concatenated_values.encode('utf-8')).hexdigest()

    # Apply the hash function to each row
    df['hash_key'] = df.apply(hash_row, axis=1)
    return df


def filter_new_data(df, schema_name, table_name):
    """
    Filter out rows that already exist in the database based on the hash_key.

    Args:
        df (pd.DataFrame): The DataFrame containing the new data.
        schema_name (str): The schema name in the PostgreSQL database.
        table_name (str): The table name in the PostgreSQL database.

    Returns:
        pd.DataFrame: A DataFrame containing only the new rows.
    """
    # Fetch existing hash keys from the database
    existing_hash_keys = fetch_existing_hash_keys(schema_name, table_name)

    new_data = df[~df['hash_key'].isin(existing_hash_keys)]
    existing_hash_keys.update(new_data['hash_key'])  # update cache to avoid duplicates

    # # Filter the DataFrame to exclude rows with existing hash keys
    # new_data = df[~df['hash_key'].isin(existing_hash_keys)]
    print(f"New data rows: {len(new_data)}")
    return new_data


def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame based on all column values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with duplicate rows removed.
    """
    # Remove duplicate rows based on all column values
    return df.drop_duplicates()


def aws_create_focus_export(
        aws_region,
        aws_access_key,
        aws_secret_key,
        export_name,
        s3_bucket,
        s3_prefix):
    # check_and_create_bucket and bucket policy
    if check_and_create_bucket(bucket_name=s3_bucket, region=aws_region, aws_access_key=aws_access_key,
                               aws_secret_key=aws_secret_key):
        account_id = get_aws_account_id(aws_access_key=aws_access_key, aws_secret_key=aws_secret_key, region=aws_region)
        if account_id:
            add_bucket_policy(bucket_name=s3_bucket, account_id=account_id, aws_access_key=aws_access_key,
                              aws_secret_key=aws_secret_key, region=aws_region)

    # create export
    bucket(region=aws_region, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key)
    client = create_boto3_client(aws_access_key=aws_access_key, aws_secret_key=aws_secret_key, region=aws_region)
    try:
        create_export(client, export_name, export_name, s3_bucket, s3_prefix, aws_region)
    except:
        update_export(client, export_name, export_name, s3_bucket, s3_prefix, aws_region)


def aws_run_ingestion(project_name,
                      monthly_budget,
                      aws_access_key,
                      aws_secret_key,
                      aws_region,
                      s3_bucket,
                      s3_prefix,
                      export_name,
                      billing_period):
    try:
        # Connect to the PostgreSQL database
        with psycopg2.connect(
            host=DB_HOST_NAME,
            database=DB_NAME,
            user=DB_USER_NAME,
            password=DB_PASSWORD,
            port=DB_PORT,
            sslmode='require'
        ) as connection:
            print("Database connection established.")

            # Define paths and schema
            base_path = "app/ingestion/aws"
            schema_name = project_name
            table_name = 'silver_focus_aws'
            parent_folder = f'{s3_prefix}/{export_name}/data/'

            sql_file_paths = {
                'create_table': f'{base_path}/sql/create_table.sql',
                'new_schema': f'{base_path}/sql/new_schema.sql',
                'gz_gold_views': f'{base_path}/sql/gz_gold_views.sql',
                'parquet_silver': f'{base_path}/sql/parquet_silver.sql',
                'parquet_gold_views': f'{base_path}/sql/parquet_gold_views.sql'
            }

            # Execute SQL file to create a new schema
            execute_sql_files(sql_file_paths['new_schema'], schema_name, monthly_budget)
            print(f'Schema {schema_name} created....')
            execute_sql_files(sql_file_paths['create_table'], schema_name, monthly_budget)
            print(f'Table {table_name} created....')

            # Create pricing tables
            execute_sql_files(f'{base_path}/sql/pricing_tables.sql', schema_name, monthly_budget)
            print(f'Pricing tables created....')

            # Fetch and store AWS pricing data (early in pipeline for LLM use)
            try:
                fetch_and_store_all_aws_pricing(
                    schema_name=schema_name,
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    region=aws_region
                )
            except Exception as e:
                print(f'⚠️ Error fetching AWS pricing: {e}')
                # Continue even if pricing fetch fails

            # Create S3 client
            s3_client = get_s3_client(aws_access_key, aws_secret_key, aws_region)

            # List period folders in the S3 bucket
            period_folders = list_period_folders(s3_client, s3_bucket, parent_folder)
            all_dfs = []
            file_type = None  # Track the type of file being processed
        
            # # Process each period folder
            # for period_folder in period_folders.keys():
            #     latest_file = get_latest_file(s3_client, s3_bucket, period_folder)
            #     if latest_file:
            #         print(f"Downloading and processing file: {latest_file}")
            #         if latest_file.endswith('.csv.gz'):
            #             df = download_and_extract_csv(s3_client, s3_bucket, latest_file)
            #             file_type = 'csv'
            #         elif latest_file.endswith('.parquet'):
            #             df = download_and_read_parquet(s3_client, s3_bucket, latest_file)
            #             file_type = 'parquet'
            #         else:
            #             print(f"Unsupported file format: {latest_file}")
            #             continue
            #         all_dfs.append(df)

    #         if all_dfs:

    #             # Combine all DataFrames into a single DataFrame
    #             combined_df = pd.concat(all_dfs, ignore_index=True)

    #             print("Inspecting DataFrame before removing duplicates...")
    #             print(combined_df.dtypes)
    #             print(combined_df.head())

    #             # Convert unhashable columns to hashable types
    #             for col in combined_df.columns:
    #                 if combined_df[col].apply(lambda x: isinstance(x, list)).any():
    #                     print(f"Column '{col}' contains lists. Converting to strings.")
    #                     combined_df[col] = combined_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    #             # Remove duplicates
    #             combined_df = combined_df.drop_duplicates()
    #             print("Duplicates removed successfully.")
    #             # Generate hash key for the combined DataFrame
    #             combined_df = generate_hash_key(combined_df)

    #             # Fetch existing hash keys
    #             existing_hash_keys = fetch_existing_hash_keys(schema_name, table_name)

    #             # Filter out rows that already exist in the database
    #             new_data = combined_df[~combined_df['hash_key'].isin(existing_hash_keys)]

    #             if not new_data.empty:
    #                 # Append only new data to PostgreSQL
    #                 dump_to_postgresql( new_data, schema_name, table_name)
    #                 print(f"Appended {len(new_data)} new rows to the table '{table_name}'.")

    #                 # Run the appropriate SQL scripts based on file type
    #                 if file_type == 'csv':
    #                     execute_sql_files(sql_file_paths['gz_gold_views'], schema_name, monthly_budget)
    #                 elif file_type == 'parquet':
    #                     execute_sql_files( sql_file_paths['parquet_silver'], schema_name, monthly_budget)
    #                     print(f"Parquet silver file executed....")
    #                     execute_sql_files( sql_file_paths['parquet_gold_views'], schema_name, monthly_budget)
    #                     print(f"Parquet gold viws created....")
    #             else:
    #                 print("No new data to append.")
    #         else:
    #             print("No CSV or Parquet files found to process.")
    #         # Call CloudWatch ingestion after S3 + PostgreSQL logic
    #     fetch_and_store_cloudwatch_metrics(
    #         aws_access_key=aws_access_key,
    #         aws_secret_key=aws_secret_key,
    #         region=aws_region,
    #         db_host=DB_HOST_NAME,
    #         db_port=DB_PORT,
    #         db_user=DB_USER_NAME,
    #         db_password=DB_PASSWORD,
    #         db_name=DB_NAME,
    #         db_schema=schema_name,  # or any schema you want to use
    #         db_table='metrics_details'
    #     )

    # except Exception as ex:
    #     print(f"An error occurred: {ex}")


                    # Call CloudWatch ingestion after processing all files
        fetch_and_store_cloudwatch_metrics(
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                region=aws_region,
                db_host=DB_HOST_NAME,
                db_port=DB_PORT,
                db_user=DB_USER_NAME,
                db_password=DB_PASSWORD,
                db_name=DB_NAME,
                db_schema=schema_name,
                db_table='metrics_details'
            )
        for period_folder in period_folders.keys():
            latest_file = get_latest_file(s3_client, s3_bucket, period_folder)
            if latest_file:
                print(f"Downloading and processing file: {latest_file}")
                if latest_file.endswith('.csv.gz'):
                    df = download_and_extract_csv(s3_client, s3_bucket, latest_file)
                    file_type = 'csv'
                elif latest_file.endswith('.parquet'):
                    df = download_and_read_parquet(s3_client, s3_bucket, latest_file)
                    file_type = 'parquet'
                else:
                    print(f"Unsupported file format: {latest_file}")
                    continue

                # Convert unhashable columns (like lists) to strings
                # for col in df.columns:
                #     if df[col].apply(lambda x: isinstance(x, list)).any():
                #         print(f"Column '{col}' contains lists. Converting to strings.")
                #         df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                df = df.applymap(lambda x: str(x) if isinstance(x, list) else x)


                # Remove duplicates within the file
                # df = df.drop_duplicates()
                df = df.drop_duplicates(subset=df.columns.difference(['hash_key']))


                # Generate hash key
                df = generate_hash_key(df)

                # Filter new data only
                new_data = filter_new_data(df, schema_name, table_name)

                if not new_data.empty:
                    # Dump to PostgreSQL
                    # dump_to_postgresql(new_data, schema_name, table_name)
                    chunk_size = 100000
                    for start in range(0, len(new_data), chunk_size):
                        chunk = new_data.iloc[start:start + chunk_size]
                        dump_to_postgresql(chunk, schema_name, table_name)


                    print(f"Appended {len(new_data)} new rows from file '{latest_file}' to the table '{table_name}'.")

                    # Execute relevant SQLs based on file type
                    if file_type == 'csv':
                        execute_sql_files(sql_file_paths['gz_gold_views'], schema_name, monthly_budget)
                    elif file_type == 'parquet':
                        execute_sql_files(sql_file_paths['parquet_silver'], schema_name, monthly_budget)
                        print(f"Parquet silver file executed....")
                        execute_sql_files(sql_file_paths['parquet_gold_views'], schema_name, monthly_budget)
                        print(f"Parquet gold views created....")
                else:
                    print(f"No new data to append for file: {latest_file}")
        # S3 Metrics Ingestion
        execute_sql_files(f'{base_path}/sql/bronze_s3_metrics.sql', schema_name, monthly_budget)
        metrics_dump_s3(aws_access_key, aws_secret_key, aws_region, schema_name)
        execute_sql_files(f'{base_path}/sql/silver_s3_metrics.sql', schema_name, monthly_budget)
        execute_sql_files(f'{base_path}/sql/gold_s3_metrics.sql', schema_name, monthly_budget)

        # EC2 Metrics Ingestion
        execute_sql_files(f'{base_path}/sql/bronze_ec2_metrics.sql', schema_name, monthly_budget)
        metrics_dump_ec2(aws_access_key, aws_secret_key, aws_region, schema_name)
        execute_sql_files(f'{base_path}/sql/silver_ec2_metrics.sql', schema_name, monthly_budget)
        execute_sql_files(f'{base_path}/sql/gold_ec2_metrics.sql', schema_name, monthly_budget)

        # Pre-warm LLM recommendations cache for all resources and date ranges
        print(f"\n🔥 Starting recommendation cache pre-warming...")
        try:
            prewarm_aws_recommendations(schema_name, monthly_budget)
            print(f"✅ Recommendation cache pre-warming completed successfully")
        except Exception as e:
            print(f"⚠️ Error during recommendation pre-warming: {e}")
            # Don't fail the entire ingestion if pre-warming fails
            import traceback
            traceback.print_exc()

    except Exception as ex:
        print(f"An error occurred: {ex}")

