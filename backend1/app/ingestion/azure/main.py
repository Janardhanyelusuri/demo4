import hashlib
import pandas as pd
from .postgres_operation import dump_to_postgresql, run_sql_file,fetch_existing_hash_keys,create_hash_key
from .blob import get_df_from_blob
import psycopg2
from .metrics_vm import metrics_dump
from .metrics_storage_account import metrics_dump as storage_metrics_dump
from .metrics_public_ip import metrics_dump as public_ip_metrics_dump
from .pricing import fetch_and_store_all_azure_pricing
import json
import sys
import os

# Add path for recommendation pre-warming
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.core.recommendation_prewarm import prewarm_azure_recommendations


def azure_main(project_name,
               budget,
               tenant_id,
               client_id,
               client_secret,
               storage_account_name,
               container_name,
               subscription_id):
    base_path = base_path = "app/ingestion/azure"
    table_name = "bronze_azure_focus"
    schema_name = project_name.lower()
    print(f'Azure subscription id: {subscription_id}')
    # Get the dataframe from the blob storage
    df = get_df_from_blob(tenant_id, client_id, client_secret, storage_account_name, container_name)
    print(f'df created')

    # Create a hash key using all columns in the dataset
    df = create_hash_key(df)
    print(f'hash_key column added to dataframe')

    run_sql_file(f'{base_path}/sql/new_schema.sql', schema_name, budget)
    print(f'schema {schema_name} created')
    run_sql_file(f'{base_path}/sql/create_table.sql', schema_name, budget)
    print(f'Table {table_name} created')
    run_sql_file(f'{base_path}/sql/bronze_metrics.sql', schema_name, budget)
    run_sql_file(f'{base_path}/sql/genai_response.sql', schema_name, budget)

    # Create pricing tables
    run_sql_file(f'{base_path}/sql/pricing_tables.sql', schema_name, budget)
    print(f'Pricing tables created')

    # Fetch and store Azure pricing data (early in pipeline for LLM use)
    try:
        # Detect region from data or use default eastus
        fetch_and_store_all_azure_pricing(schema_name, region="eastus", currency="USD")
    except Exception as e:
        print(f'⚠️ Error fetching Azure pricing: {e}')
        # Continue even if pricing fetch fails


    # Check for existing hash keys in the PostgreSQL table
    existing_hash_keys = fetch_existing_hash_keys(schema_name, table_name)
    # print(f'Existing hash keys fetched: {len(existing_hash_keys)}')

     # Filter out rows with hash keys that already exist in PostgreSQL
    new_data = df[~df['hash_key'].isin(existing_hash_keys)]
    print(f'Number of new records to insert: {len(new_data)}')

    # If there is new data, dump it into PostgreSQL
    if not new_data.empty:
        dump_to_postgresql(new_data, schema_name, table_name)
        print(f'New records appended to PostgreSQL')

    # Run SQL files for silver and gold stages
    run_sql_file(f'{base_path}/sql/silver.sql', schema_name, budget)
    
    # Step 6: 🔁 Call dump_metrics() to fetch Azure VM metrics and dump
    metrics_dump(tenant_id, client_id, client_secret,subscription_id, schema_name,"bronze_azure_vm_metrics")
    
    run_sql_file(f'{base_path}/sql/silver_metrics.sql', schema_name, budget)
    run_sql_file(f'{base_path}/sql/gold.sql', schema_name, budget)

    # run_llm_vm(schema_name)
    print(f"LLM response generated")

    run_sql_file(f'{base_path}/sql/bronze_storage_metrics.sql', schema_name, budget)
    storage_metrics_dump(tenant_id, client_id, client_secret, subscription_id,
                        schema_name, "bronze_azure_storage_account_metrics")
    
    run_sql_file(f'{base_path}/sql/silver_storage_metrics.sql', schema_name, budget)

    run_sql_file(f'{base_path}/sql/gold_storage_metrics.sql', schema_name, budget)

    run_sql_file(f'{base_path}/sql/bronze_public_ip_metrics.sql', schema_name, budget)

    public_ip_metrics_dump(tenant_id, client_id, client_secret, subscription_id,
                           schema_name, "bronze_azure_public_ip_metrics")
    
    run_sql_file(f'{base_path}/sql/silver_public_ip_metrics.sql', schema_name, budget)

    run_sql_file(f'{base_path}/sql/gold_public_ip_metrics.sql', schema_name, budget)

    # Pre-warm LLM recommendations cache for all resources and date ranges
    print(f"\n🔥 Starting recommendation cache pre-warming...")
    try:
        prewarm_azure_recommendations(schema_name, budget)
        print(f"✅ Recommendation cache pre-warming completed successfully")
    except Exception as e:
        print(f"⚠️ Error during recommendation pre-warming: {e}")
        # Don't fail the entire ingestion if pre-warming fails
        import traceback
        traceback.print_exc()


# used to test in local---
#    print(f'Silver and Gold stages completed for schema {schema_name}')   
# # Run the main function
# azure_main('test', 300, '', '',
#            '', 'cloudmeterdev', 'cloud-meter-4')




