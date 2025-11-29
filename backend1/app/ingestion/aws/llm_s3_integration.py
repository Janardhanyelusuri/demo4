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

    # NOTE: The query now uses three CTEs to properly calculate aggregates and max date
    QUERY = sql.SQL("""
        WITH metric_agg AS (
            SELECT
                bm.bucket_name,
                bm.account_id,
                db.region,
                bm.metric_name,
                bm.value AS metric_value,
                bm.event_date
            FROM {schema_name}.fact_s3_metrics bm
            LEFT JOIN {schema_name}.dim_s3_bucket db
                ON bm.bucket_name = db.bucket_name
            WHERE
                bm.event_date BETWEEN %s AND %s
                {bucket_filter}
        ),

        max_date_lookup AS (
            SELECT DISTINCT ON (bucket_name, metric_name)
                bucket_name,
                metric_name,
                event_date AS max_date
            FROM metric_agg
            ORDER BY bucket_name, metric_name, metric_value DESC, event_date DESC
        ),

        usage_summary AS (
            SELECT
                m.bucket_name,
                m.account_id,
                m.region,
                m.metric_name,
                -- Convert bytes to GB for BucketSizeBytes metric
                AVG(
                    CASE
                        WHEN m.metric_name = 'BucketSizeBytes'
                        THEN m.metric_value / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE m.metric_value
                    END
                ) AS avg_value,
                MAX(
                    CASE
                        WHEN m.metric_name = 'BucketSizeBytes'
                        THEN m.metric_value / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE m.metric_value
                    END
                ) AS max_value,
                MAX(md.max_date) AS max_date
            FROM metric_agg m
            LEFT JOIN max_date_lookup md
                ON md.bucket_name = m.bucket_name
                AND md.metric_name = m.metric_name
            GROUP BY m.bucket_name, m.account_id, m.region, m.metric_name
        ),
        
        metric_map AS (
            SELECT
                bucket_name,
                -- Combine AVG, MAX value, and MAX date into a single JSON object per bucket
                -- Cast to jsonb for concatenation operator to work
                (
                    json_object_agg(
                        metric_name || '_Avg', ROUND(avg_value::numeric, 6)
                    )::jsonb ||
                    json_object_agg(
                        metric_name || '_Max', ROUND(max_value::numeric, 6)
                    )::jsonb ||
                    json_object_agg(
                        metric_name || '_MaxDate', TO_CHAR(max_date, 'YYYY-MM-DD')
                    )::jsonb
                )::json AS metrics_json
            FROM usage_summary
            GROUP BY 1
        )
        
        SELECT
            us.bucket_name,
            us.account_id,
            us.region,
            MAX(m.metrics_json::text)::json AS metrics_json,  -- Cast to text for MAX(), then back to json
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
        GROUP BY 1, 2, 3 -- Group by bucket_name, account_id, region only
    """).format(
        schema_name=sql.Identifier(schema_name),
        bucket_filter=bucket_filter_sql
    )

    # Build params in the correct order to match the SQL placeholders
    params = [start_date, end_date]
    if bucket_name:
        params.append(bucket_name)  # Bucket filter comes after BETWEEN clause
    params.extend([end_date, start_date])  # For charge_period_start/end filters

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


def _format_s3_metrics_for_llm(bucket_data: dict) -> dict:
    """
    Format S3 metrics for LLM with proper units.
    Similar to Azure's metric formatting approach.
    """
    # Metrics that should be renamed to show GB units
    METRIC_DISPLAY_NAMES = {
        "BucketSizeBytes": "Bucket Size (GB)",
        "NumberOfObjects": "Number of Objects",
        "AllRequests": "All Requests",
        "GetRequests": "GET Requests",
        "PutRequests": "PUT Requests",
        "4xxErrors": "4xx Errors",
        "5xxErrors": "5xx Errors"
    }

    formatted_metrics = {}

    # Find all metric keys in the data
    for key in bucket_data.keys():
        if key.startswith('metric_') and key.endswith('_Avg'):
            # Extract metric name (e.g., "BucketSizeBytes" from "metric_BucketSizeBytes_Avg")
            metric_name = key.replace('metric_', '').replace('_Avg', '')

            # Get display name with units
            display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

            # Build metric entry
            entry = {
                "Avg": bucket_data.get(f'metric_{metric_name}_Avg'),
                "Max": bucket_data.get(f'metric_{metric_name}_Max'),
                "MaxDate": bucket_data.get(f'metric_{metric_name}_MaxDate')
            }

            # Only include if at least one value is present
            if any(v is not None for v in entry.values()):
                formatted_metrics[display_name] = entry

    return formatted_metrics


def generate_s3_prompt(bucket_data: dict) -> str:
    """
    Generate LLM prompt for S3 bucket optimization recommendations.

    Args:
        bucket_data: Dictionary containing bucket metrics and cost data

    Returns:
        Formatted prompt string for the LLM
    """
    bucket_name = bucket_data.get('bucket_name', 'Unknown')
    region = bucket_data.get('region', 'Unknown')
    account_id = bucket_data.get('account_id', 'Unknown')
    billed_cost = bucket_data.get('billed_cost', 0)

    start_date = bucket_data.get('start_date', 'N/A')
    end_date = bucket_data.get('end_date', 'N/A')
    duration_days = bucket_data.get('duration_days', 0)

    # Format metrics with proper units
    formatted_metrics = _format_s3_metrics_for_llm(bucket_data)

    prompt = f"""AWS S3 FinOps. Analyze metrics, output JSON only.

CONTEXT: {bucket_name} | Region: {region} | {start_date} to {end_date} ({duration_days}d) | Cost: ${billed_cost:.2f}

METRICS:
{json.dumps(formatted_metrics, indent=2)}

RULES:
1. Use exact values+units from METRICS (e.g., "512.5 GB", "1,247 requests")
2. State exact storage class names (e.g., "S3 Standard → S3 Glacier Flexible Retrieval")
3. Express savings as PERCENTAGES only (e.g., "Can reduce by 40%", "Save 75% with Glacier")
4. BANNED: "consider", "review", "optimize", "significant", "could", "should", "it is recommended", any dollar amounts in recommendations
5. Use action verbs: Move, Implement, Enable, Configure, Delete, Remove
6. Every recommendation needs explanation with actual metrics showing WHY
7. Always include units: GB, requests, objects, %

DECIDE: Primary optimization (storage class/lifecycle/versioning)? 2-3 additional optimizations? Which metrics drove decisions? 2-3 anomalies (spikes/drops)?

JSON (MUST: 2-3 additional_recommendation, 2-3 anomalies):
{{
  "recommendations": {{
    "effective_recommendation": {{"text": "[action with exact values]", "explanation": "[why with metrics]", "saving_pct": #}},
    "additional_recommendation": [
      {{"text": "[action with exact values]", "explanation": "[why with metrics]", "saving_pct": #}},
      {{"text": "[action with exact values]", "explanation": "[why with metrics]", "saving_pct": #}},
      {{"text": "[action with exact values]", "explanation": "[why with metrics]", "saving_pct": #}}
    ],
    "base_of_recommendations": ["[metric: value units]", "[metric: value units]"]
  }},
  "cost_forecasting": {{"monthly": <number>, "annually": <number>}},
  "anomalies": [
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #, "reason_short": "[why significant]"}},
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #, "reason_short": "[why anomalous]"}},
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #, "reason_short": "[why unusual]"}}
  ],
  "contract_deal": {{"assessment": "good"|"bad"|"unknown", "for sku": "S3 Standard", "reason": "...", "monthly_saving_pct": #, "annual_saving_pct": #}}
}}
"""
    return prompt


def get_s3_recommendation_single(bucket_data: dict) -> dict:
    """
    Get LLM recommendation for a single S3 bucket.

    Args:
        bucket_data: Dictionary containing bucket metrics

    Returns:
        Dictionary containing recommendations or None if error
    """
    try:
        prompt = generate_s3_prompt(bucket_data)
        llm_response = llm_call(prompt)

        if not llm_response:
            LOG.warning(f"Empty LLM response for bucket {bucket_data.get('bucket_name')}")
            return None

        # Try to parse as JSON directly
        try:
            # Remove markdown code blocks if present
            if '```json' in llm_response:
                llm_response = llm_response.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_response:
                llm_response = llm_response.split('```')[1].split('```')[0].strip()

            recommendation = json.loads(llm_response)
            recommendation['resource_id'] = bucket_data.get('bucket_name', 'Unknown')
            return recommendation
        except json.JSONDecodeError:
            LOG.warning(f"Failed to parse JSON for bucket {bucket_data.get('bucket_name')}")
            return None

    except Exception as e:
        LOG.error(f"Error getting S3 recommendation: {e}")
        return None


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
    buckets = df.to_dict(orient="records")

    LOG.info("🤖 Calling LLM for S3 recommendations...")
    recommendations = []

    for bucket_data in buckets:
        rec = get_s3_recommendation_single(bucket_data)
        if rec:
            recommendations.append(rec)

    if recommendations:
        LOG.info(f"✅ S3 analysis complete! Generated {len(recommendations)} recommendation(s).")
        return recommendations
    else:
        LOG.warning("⚠️ No recommendations generated by LLM.")
        return []
