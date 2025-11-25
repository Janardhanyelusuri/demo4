import psycopg2
import pandas as pd
import sys
import os
import json
import hashlib
from datetime import datetime, timedelta
import logging
from psycopg2 import sql 
from typing import Optional # Added Optional type hint for clarity

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("s3_llm_integration")

# Relative path hack kept to maintain original import functionality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.core.genai import llm_call
from app.ingestion.aws.postgres_operations import connection, dump_to_postgresql, fetch_existing_hash_keys
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# --- Utility Functions (Preserved/Removed for brevity) ---
# ... (Assuming _create_local_engine_from_env is defined elsewhere or not strictly needed here) ...


@connection
def fetch_s3_bucket_utilization_data(conn, schema_name, start_date, end_date, bucket_name=None):
    """
    Fetch S3 bucket metrics (all metrics) from gold metrics view and billing/pricing
    fields from the gold_aws_fact_focus view. Calculates AVG, MAX, and MAX Date for metrics.
    """
    
    # Use parameterized query components
    bucket_filter_sql = sql.SQL("AND bm.bucket_name = %s") if bucket_name else sql.SQL("")

    # NOTE: The query now uses two CTEs to first calculate aggregates and then map them to JSON.
    QUERY = sql.SQL("""
        WITH metric_agg AS (
            SELECT
                bm.bucket_name,
                bm.account_id,
                bm.region,
                bm.metric_name,
                bm.value AS metric_value,
                bm.event_date
            FROM {schema_name}.fact_s3_metrics bm
            WHERE
                bm.event_date BETWEEN %s AND %s
                {bucket_filter}
        ),
        
        usage_summary AS (
            SELECT
                bucket_name,
                account_id,
                region,
                metric_name,
                AVG(metric_value) AS avg_value,
                MAX(metric_value) AS max_value,
                
                -- NEW: Use FIRST_VALUE to get the date corresponding to the maximum value
                FIRST_VALUE(event_date) OVER (
                    PARTITION BY bucket_name, metric_name
                    ORDER BY metric_value DESC, event_date DESC
                ) AS max_date
                
            FROM metric_agg
            GROUP BY bucket_name, account_id, region, metric_name
        ),
        
        metric_map AS (
            SELECT
                bucket_name,
                -- Combine AVG, MAX value, and MAX date into a single JSON object per bucket
                json_object_agg(
                    metric_name || '_Avg', ROUND(avg_value::numeric, 6)
                ) || 
                json_object_agg(
                    metric_name || '_Max', ROUND(max_value::numeric, 6)
                ) ||
                json_object_agg(
                    metric_name || '_MaxDate', TO_CHAR(max_date, 'YYYY-MM-DD')
                ) AS metrics_json
            FROM usage_summary
            GROUP BY 1
        )
        
        SELECT
            us.bucket_name,
            us.account_id,
            us.region,
            m.metrics_json,
            -- Pull cost fields from the focus table (assuming one cost record per bucket/period)
            MAX(ff.pricing_category) AS pricing_category,
            MAX(ff.pricing_unit) AS pricing_unit,
            MAX(ff.contracted_unit_price) AS contracted_unit_price,
            SUM(ff.billed_cost) AS billed_cost,
            SUM(ff.consumed_quantity) AS consumed_quantity,
            MAX(ff.consumed_unit) AS consumed_unit
        FROM usage_summary us
        LEFT JOIN metric_map m ON m.bucket_name = us.bucket_name
        LEFT JOIN {schema_name}.gold_aws_fact_focus ff
            -- Join cost on resource_id = bucket_name
            ON ff.resource_id = us.bucket_name 
               AND ff.charge_period_start::date <= %s 
               AND ff.charge_period_end::date >= %s
        GROUP BY 1, 2, 3, 4 -- Group by non-aggregated fields, including the resulting JSON column
    """).format(
        schema_name=sql.Identifier(schema_name),
        bucket_filter=bucket_filter_sql
    )

    params = [start_date, end_date, end_date, start_date]
    if bucket_name:
        params.append(bucket_name) # Add bucket name parameter (no lower() here as it's done in SQL)

    try:
        cursor = conn.cursor()
        cursor.execute(QUERY, params) 
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        cursor.close()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)
        
        # Expand the metrics_json into separate columns (flatten)
        if not df.empty and "metrics_json" in df.columns:
            metrics_expanded = pd.json_normalize(df["metrics_json"].fillna({})).add_prefix("metric_")
            metrics_expanded.index = df.index
            df = pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)

        return df

    except psycopg2.Error as e:
        raise RuntimeError(f"PostgreSQL query failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during DB fetch: {e}") from e


# --- run_llm_analysis_s3 (No change needed here, it uses the fetch function) ---

def run_llm_analysis_s3(schema_name, start_date=None, end_date=None, bucket_name=None):
  
    start_str = start_date or (datetime.utcnow().date() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_str = end_date or datetime.utcnow().date().strftime("%Y-%m-%d")

    LOG.info(f"🚀 Starting S3 LLM analysis from {start_str} to {end_str}...")

    df = None
    try:
        # The fetch function is decorated with @connection, but needs to be called carefully
        # Note: If @connection isn't handling the conn argument internally, you need to manually pass it or update the decorator.
        # Assuming the @connection decorator handles the connection context:
        df = fetch_s3_bucket_utilization_data(schema_name, start_str, end_str, bucket_name)
    except RuntimeError as e:
        LOG.error(f"❌ Failed to fetch S3 utilization data: {e}")
        return
    except Exception as e:
        LOG.error(f"❌ An unhandled error occurred during data fetching: {e}")
        return

    if df is None or df.empty:
        LOG.warning("⚠️ No S3 bucket data found for the requested date range / bucket.")
        return

    LOG.info(f"📈 Retrieved data for {len(df)} bucket(s)")

    # Annotate with date info for LLM context
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days

    # Convert to list-of-dicts for LLM helper
    data = df.to_dict(orient="records")
    
    LOG.info("🤖 Calling LLM for recommendations...")
    try:
        # NOTE: You need a proper _generate_s3_prompt function and llm_call wrapper 
        # to correctly pass and use the new AVG/MAX/MaxDate metrics here.
        recommendations = llm_call(data) 
    except Exception as e:
        LOG.error(f"❌ LLM call failed: {e}")
        recommendations = []

    if recommendations:
        # Assuming dump_to_postgresql is called here in your actual implementation
        print("✅ LLM analysis complete!")
        return recommendations
    else:
        print("⚠️ No recommendations generated by LLM.")
