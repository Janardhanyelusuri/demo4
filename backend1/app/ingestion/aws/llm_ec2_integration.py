"""
AWS EC2 LLM Integration
Provides cost optimization recommendations for EC2 instances
"""

import psycopg2
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta
import logging
from psycopg2 import sql
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ec2_llm_integration")

# Path adjustments for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.core.genai import llm_call
from app.ingestion.aws.postgres_operations import connection
from app.ingestion.azure.llm_json_extractor import extract_json
# Import pricing helpers for dynamic pricing context
from app.ingestion.aws.pricing_helpers import (
    get_ec2_current_pricing,
    get_ec2_alternative_pricing,
    format_ec2_pricing_for_llm
)

load_dotenv()


# ============================================================
# EC2 FUNCTIONS
# ============================================================

@connection
def fetch_ec2_utilization_data(conn, schema_name, start_date, end_date, instance_id=None):
    """
    Fetch EC2 instance metrics and cost data from fact tables and FOCUS billing.
    Calculates AVG, MAX, and MAX Date for each metric.

    Args:
        conn: Database connection
        schema_name: PostgreSQL schema name
        start_date: Start date for analysis
        end_date: End date for analysis
        instance_id: Optional specific instance ID to analyze

    Returns:
        DataFrame with EC2 instance utilization and cost data
    """

    instance_filter_sql = sql.SQL("AND LOWER(m.instance_id) = LOWER(%s)") if instance_id else sql.SQL("")

    QUERY = sql.SQL("""
        WITH metric_agg AS (
            SELECT
                m.instance_id,
                m.instance_name,
                i.instance_type,
                m.region,
                m.account_id,
                m.metric_name,
                m.value AS metric_value,
                m.timestamp
            FROM {schema_name}.fact_ec2_metrics m
            LEFT JOIN {schema_name}.dim_ec2_instance i ON m.instance_id = i.instance_id
            WHERE
                m.timestamp BETWEEN %s AND %s
                {instance_filter}
        ),

        usage_summary AS (
            SELECT
                instance_id,
                instance_name,
                instance_type,
                region,
                account_id,
                metric_name,
                -- Convert bytes to GB for disk and network metrics
                AVG(
                    CASE
                        WHEN metric_name IN ('DiskReadBytes', 'DiskWriteBytes', 'NetworkIn', 'NetworkOut')
                        THEN metric_value / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE metric_value
                    END
                ) AS avg_value,
                MAX(
                    CASE
                        WHEN metric_name IN ('DiskReadBytes', 'DiskWriteBytes', 'NetworkIn', 'NetworkOut')
                        THEN metric_value / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE metric_value
                    END
                ) AS max_value,

                -- Get the timestamp when max value occurred
                FIRST_VALUE(timestamp) OVER (
                    PARTITION BY instance_id, metric_name
                    ORDER BY metric_value DESC, timestamp DESC
                ) AS max_date

            FROM metric_agg
            GROUP BY instance_id, instance_name, instance_type, region, account_id, metric_name
        ),

        metric_map AS (
            SELECT
                instance_id,
                instance_name,
                instance_type,
                region,
                account_id,
                -- Combine AVG, MAX value, and MAX date into JSON objects
                json_object_agg(
                    metric_name || '_Avg', ROUND(avg_value::numeric, 6)
                ) ||
                json_object_agg(
                    metric_name || '_Max', ROUND(max_value::numeric, 6)
                ) ||
                json_object_agg(
                    metric_name || '_MaxDate', TO_CHAR(max_date, 'YYYY-MM-DD HH24:MI:SS')
                ) AS metrics_json
            FROM usage_summary
            GROUP BY 1, 2, 3, 4, 5
        )

        SELECT
            us.instance_id,
            us.instance_name,
            us.instance_type,
            us.region,
            us.account_id,
            m.metrics_json,
            -- Pull cost data from FOCUS billing table
            COALESCE(SUM(ff.billed_cost), 0) AS billed_cost,
            COALESCE(SUM(ff.consumed_quantity), 0) AS consumed_quantity,
            MAX(ff.consumed_unit) AS consumed_unit,
            MAX(ff.pricing_category) AS pricing_category,
            MAX(ff.pricing_unit) AS pricing_unit,
            MAX(ff.contracted_unit_price) AS contracted_unit_price
        FROM usage_summary us
        LEFT JOIN metric_map m ON m.instance_id = us.instance_id
        LEFT JOIN {schema_name}.gold_aws_fact_focus ff
            ON LOWER(ff.resource_id) LIKE '%%' || LOWER(us.instance_id) || '%%'
               AND ff.service_name = 'Amazon Elastic Compute Cloud - Compute'
               AND ff.charge_period_start::date <= %s
               AND ff.charge_period_end::date >= %s
        GROUP BY 1, 2, 3, 4, 5, 6
    """).format(
        schema_name=sql.Identifier(schema_name),
        instance_filter=instance_filter_sql
    )

    params = [start_date, end_date, end_date, start_date]
    if instance_id:
        params.insert(2, instance_id)

    try:
        cursor = conn.cursor()
        cursor.execute(QUERY, params)

        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        cursor.close()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)

        # Expand metrics_json into separate columns
        if not df.empty and "metrics_json" in df.columns:
            metrics_expanded = pd.json_normalize(df["metrics_json"].fillna({})).add_prefix("metric_")
            metrics_expanded.index = df.index
            df = pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)

        return df

    except psycopg2.Error as e:
        raise RuntimeError(f"PostgreSQL query failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during EC2 data fetch: {e}") from e


def _estimate_ec2_hourly_cost(instance_type: str) -> float:
    """
    Estimate EC2 hourly cost based on instance type when database pricing unavailable.
    Uses realistic AWS EC2 pricing patterns.
    """
    instance_lower = instance_type.lower()

    # Extract size from instance type (e.g., t3.large -> large)
    parts = instance_type.split('.')
    size = parts[1] if len(parts) > 1 else 'medium'
    family = parts[0] if len(parts) > 0 else 't3'

    # Size multipliers
    size_multipliers = {
        'nano': 0.125, 'micro': 0.25, 'small': 0.5, 'medium': 1.0,
        'large': 2.0, 'xlarge': 4.0, '2xlarge': 8.0, '4xlarge': 16.0,
        '8xlarge': 32.0, '12xlarge': 48.0, '16xlarge': 64.0, '24xlarge': 96.0
    }
    multiplier = size_multipliers.get(size, 1.0)

    # Base cost estimates by instance family
    if family.startswith('t'):
        base_cost = 0.0104  # T-series (burstable)
    elif family.startswith('m'):
        base_cost = 0.096   # M-series (general purpose)
    elif family.startswith('c'):
        base_cost = 0.085   # C-series (compute optimized)
    elif family.startswith('r'):
        base_cost = 0.126   # R-series (memory optimized)
    elif family.startswith('x'):
        base_cost = 0.333   # X-series (extreme memory)
    elif family.startswith('i') or family.startswith('d'):
        base_cost = 0.156   # I/D-series (storage optimized)
    elif family.startswith('g') or family.startswith('p'):
        base_cost = 0.526   # G/P-series (GPU instances)
    elif family.startswith('a'):
        base_cost = 0.0102  # A-series (ARM-based)
    else:
        base_cost = 0.096   # Default estimate

    # Apply size multiplier with economies of scale
    estimated = base_cost * (multiplier ** 0.95)

    return round(estimated, 4)


def generate_ec2_prompt(instance_data: Dict[str, Any]) -> str:
    """
    Generate LLM prompt for EC2 instance optimization recommendations with pricing context.

    Args:
        instance_data: Dictionary containing instance metrics and cost data

    Returns:
        Formatted prompt string for the LLM
    """

    instance_id = instance_data.get('instance_id', 'Unknown')
    instance_type = instance_data.get('instance_type', 'Unknown')
    region = instance_data.get('region', 'Unknown')
    billed_cost = instance_data.get('billed_cost', 0)

    # Extract metrics
    cpu_avg = instance_data.get('metric_CPUUtilization_Avg', 0)
    cpu_max = instance_data.get('metric_CPUUtilization_Max', 0)
    cpu_max_date = instance_data.get('metric_CPUUtilization_MaxDate', 'N/A')

    network_in_avg = instance_data.get('metric_NetworkIn_Avg', 0)
    network_in_max = instance_data.get('metric_NetworkIn_Max', 0)

    network_out_avg = instance_data.get('metric_NetworkOut_Avg', 0)
    network_out_max = instance_data.get('metric_NetworkOut_Max', 0)

    disk_read_avg = instance_data.get('metric_DiskReadOps_Avg', 0)
    disk_write_avg = instance_data.get('metric_DiskWriteOps_Avg', 0)

    start_date = instance_data.get('start_date', 'N/A')
    end_date = instance_data.get('end_date', 'N/A')
    duration_days = instance_data.get('duration_days', 0)

    # Fetch pricing data from database
    schema_name = instance_data.get('schema_name', '')

    pricing_context = ""
    if schema_name and instance_type and instance_type != 'Unknown':
        try:
            # Get current instance type pricing
            current_pricing = get_ec2_current_pricing(schema_name, instance_type, region)

            # Get alternative instance type pricing
            alternative_pricing = get_ec2_alternative_pricing(schema_name, instance_type, region, max_results=10)

            # Debug: Print fetched pricing
            print(f"\n{'='*60}")
            print(f"PRICING DEBUG - AWS EC2: {instance_type} in {region}")
            print(f"{'='*60}")
            if current_pricing:
                print(f"CURRENT INSTANCE PRICING:")
                print(f"  Instance Type: {current_pricing.get('instance_type')}")
                print(f"  Hourly: {current_pricing.get('price_per_hour')}")
                print(f"  Monthly: {current_pricing.get('monthly_cost')}")
                print(f"  Currency: {current_pricing.get('currency', 'N/A')}")

                if alternative_pricing:
                    print(f"\nALTERNATIVE INSTANCE TYPES (Top 5):")
                    for idx, alt in enumerate(alternative_pricing[:5], 1):
                        savings = current_pricing['monthly_cost'] - alt['monthly_cost']
                        savings_pct = (savings / current_pricing['monthly_cost']) * 100 if current_pricing['monthly_cost'] > 0 else 0
                        print(f"  {idx}. {alt['instance_type']}: {alt['price_per_hour']}/hr ({alt['monthly_cost']}/mo) - Save {savings_pct:.1f}%")
                else:
                    print(f"\nALTERNATIVE INSTANCE TYPES: Not found in database")
            else:
                print(f"CURRENT INSTANCE PRICING: Not found in database")
                print(f"Using fallback pricing estimates")

                # Fallback pricing when database pricing not available
                estimated_hourly = _estimate_ec2_hourly_cost(instance_type)
                current_pricing = {
                    'instance_type': instance_type,
                    'price_per_hour': estimated_hourly,
                    'monthly_cost': estimated_hourly * 730,
                    'currency': 'USD',
                    'vcpu': '2',
                    'memory': '4 GiB',
                    'network_performance': 'Moderate'
                }

                # Estimate alternatives
                alternative_pricing = [
                    {'instance_type': f'{instance_type.split(".")[0]}.small', 'price_per_hour': estimated_hourly * 0.5, 'monthly_cost': estimated_hourly * 0.5 * 730, 'vcpu': '1', 'memory': '2 GiB'},
                    {'instance_type': f'{instance_type.split(".")[0]}.medium', 'price_per_hour': estimated_hourly * 0.75, 'monthly_cost': estimated_hourly * 0.75 * 730, 'vcpu': '2', 'memory': '4 GiB'},
                    {'instance_type': f'{instance_type.split(".")[0]}.large', 'price_per_hour': estimated_hourly * 1.5, 'monthly_cost': estimated_hourly * 1.5 * 730, 'vcpu': '4', 'memory': '8 GiB'}
                ]

                print(f"  Estimated hourly: {estimated_hourly:.4f} USD")
                print(f"  Estimated monthly: {current_pricing['monthly_cost']:.2f} USD")
                print(f"  (Fallback estimates - actual pricing unavailable)")

            print(f"{'='*60}\n")

            # Format pricing for LLM
            pricing_context = "\n\n" + format_ec2_pricing_for_llm(current_pricing, alternative_pricing) + "\n"
        except Exception as e:
            LOG.warning(f"⚠️ Error fetching EC2 pricing data: {e}")
            import traceback
            traceback.print_exc()

            # Fallback pricing on error
            estimated_hourly = _estimate_ec2_hourly_cost(instance_type)
            pricing_context = f"\n\nPRICING DATA (ESTIMATED - Database unavailable):\nCurrent instance: {instance_type}\nEstimated cost: {estimated_hourly:.4f} USD/hour (~{estimated_hourly * 730:.2f} USD/month)\n"
    else:
        pricing_context = "\n\nPRICING DATA: Not available (schema or instance type not provided)\n"

    prompt = f"""
AWS EC2 FinOps. Analyze metrics, output JSON only.

**CONTEXT:**
- Instance ID: {instance_id}
- Instance Type: {instance_type}
- Region: {region}
- Analysis Period: {start_date} to {end_date} ({duration_days} days)
- Total Billed Cost: {billed_cost:.2f}

**METRICS:**
- CPU Utilization: Avg {cpu_avg:.2f}%, Max {cpu_max:.2f}% (on {cpu_max_date})
- Network In: Avg {network_in_avg:.2f} GB, Max {network_in_max:.2f} GB
- Network Out: Avg {network_out_avg:.2f} GB, Max {network_out_max:.2f} GB
- Disk Read Ops: Avg {disk_read_avg:.2f} ops/sec
- Disk Write Ops: Avg {disk_write_avg:.2f} ops/sec
{pricing_context}
**DECISION PROCESS (STRICT ORDER):**
1. **ANALYZE METRICS FIRST**: Examine CPU%, Network I/O, Disk ops from METRICS section above
2. **DETERMINE ACTION**: Based ONLY on metrics, decide: Downsize (low CPU/network), Upsize (high CPU), Reserved Instance (steady usage), or No change
3. **FIND ALTERNATIVES**: If action needed, use PRICING DATA to find exact instance types and costs
4. **CALCULATE SAVINGS**: Use actual pricing numbers from PRICING DATA to compute saving_pct

**CRITICAL RULES:**
- ALL metric values MUST come from METRICS section above - NEVER invent values
- ALL instance types and costs MUST come from PRICING DATA section - NEVER invent pricing
- BANNED WORDS: "consider", "review", "optimize", "significant", "could", "should", "it is recommended", "may", "might"
- USE DECISIVE LANGUAGE: "Downsize to t3.medium", "Purchase RI for current instance", "Switch to c5.large"
- Express savings as percentages: saving_pct = ((current_cost - new_cost) / current_cost) * 100
- Include units in ALL values: %, GB, ops/sec, vCPU
- For contract_deal: Compare On-Demand vs Reserved Instance pricing from PRICING DATA

**JSON OUTPUT (NO placeholders - use actual values from METRICS and PRICING DATA):**
{{
  "recommendations": {{
    "effective_recommendation": {{
      "text": "Action verb + target instance type from PRICING DATA with specs",
      "explanation": "Cite specific metrics (e.g., CPU 15% avg) and pricing from PRICING DATA",
      "saving_pct": 0
    }},
    "additional_recommendation": [
      {{
        "text": "Action verb + specific recommendation with pricing",
        "explanation": "Cite actual metrics and pricing",
        "saving_pct": 0
      }},
      {{
        "text": "Action verb + specific recommendation with pricing",
        "explanation": "Cite actual metrics and pricing",
        "saving_pct": 0
      }}
    ],
    "base_of_recommendations": []
  }},
  "cost_forecasting": {{
    "monthly": 0,
    "annually": 0
  }},
  "anomalies": [
    {{
      "metric_name": "Exact name from METRICS",
      "timestamp": "Exact timestamp from METRICS",
      "value": 0,
      "reason_short": "Explain using metric context"
    }},
    {{
      "metric_name": "Exact name from METRICS",
      "timestamp": "Exact timestamp from METRICS",
      "value": 0,
      "reason_short": "Explain using metric context"
    }}
  ],
  "contract_deal": {{
    "assessment": "good|bad|unknown",
    "for_sku": "{instance_type}",
    "reason": "Compare On-Demand vs RI pricing from PRICING DATA",
    "monthly_saving_pct": 0,
    "annual_saving_pct": 0
  }}
}}
"""

    return prompt


def get_ec2_recommendation_single(instance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get LLM recommendation for a single EC2 instance.

    Args:
        instance_data: Dictionary containing instance metrics

    Returns:
        Dictionary containing recommendations or None if error
    """
    try:
        prompt = generate_ec2_prompt(instance_data)
        llm_response = llm_call(prompt)

        if not llm_response:
            LOG.warning(f"Empty LLM response for instance {instance_data.get('instance_id')}")
            return None

        # Extract and parse JSON from LLM response
        recommendation = extract_json(llm_response)

        if recommendation:
            # Add resource_id to the recommendation
            recommendation['resource_id'] = instance_data.get('instance_id', 'Unknown')
            return recommendation
        else:
            LOG.warning(f"Failed to parse JSON for instance {instance_data.get('instance_id')}")
            return None

    except Exception as e:
        LOG.error(f"Error getting EC2 recommendation: {e}")
        return None


def run_llm_analysis_ec2(resource_type: str, schema_name: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         resource_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main entry point for EC2 LLM analysis.

    Args:
        resource_type: Should be 'ec2'
        schema_name: PostgreSQL schema name
        start_date: Start date for analysis
        end_date: End date for analysis
        resource_id: Optional specific instance ID

    Returns:
        List of recommendation dictionaries
    """

    start_str = start_date.strftime("%Y-%m-%d") if start_date else (datetime.utcnow().date() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d") if end_date else datetime.utcnow().date().strftime("%Y-%m-%d")

    LOG.info(f"🚀 Starting EC2 LLM analysis from {start_str} to {end_str}...")

    try:
        df = fetch_ec2_utilization_data(schema_name, start_str, end_str, resource_id)
    except RuntimeError as e:
        LOG.error(f"❌ Failed to fetch EC2 utilization data: {e}")
        return []
    except Exception as e:
        LOG.error(f"❌ Unhandled error during data fetching: {e}")
        return []

    if df is None or df.empty:
        LOG.warning("⚠️ No EC2 instance data found for the requested date range / instance.")
        return []

    LOG.info(f"📈 Retrieved data for {len(df)} EC2 instance(s)")

    # Annotate with date info for LLM context
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days

    # Convert to list of dicts
    instances = df.to_dict(orient="records")

    LOG.info("🤖 Calling LLM for EC2 recommendations...")
    recommendations = []

    for instance_data in instances:
        # Add schema_name and region for pricing lookups
        instance_data['schema_name'] = schema_name
        instance_data['region'] = instance_data.get('region', 'us-east-1')
        rec = get_ec2_recommendation_single(instance_data)
        if rec:
            recommendations.append(rec)

    if recommendations:
        LOG.info(f"✅ EC2 analysis complete! Generated {len(recommendations)} recommendation(s).")
        return recommendations
    else:
        LOG.warning("⚠️ No recommendations generated by LLM.")
        return []


# ============================================================

# Wrapper function for backward compatibility
def run_llm_analysis(resource_type: str, schema_name: str,
                     start_date: str, end_date: str,
                     resource_id: Optional[str] = None):
    """
    Main entry point for EC2 LLM analysis.
    Wrapper function that calls run_llm_analysis_ec2.
    """
    return run_llm_analysis_ec2(resource_type, schema_name, start_date, end_date, resource_id)
