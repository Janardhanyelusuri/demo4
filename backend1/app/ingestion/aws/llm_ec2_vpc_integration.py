"""
AWS EC2 and VPC LLM Integration
Provides cost optimization recommendations for EC2 instances and VPC resources
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
LOG = logging.getLogger("ec2_vpc_llm_integration")

# Path adjustments for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.core.genai import llm_call
from app.ingestion.aws.postgres_operations import connection
from app.ingestion.azure.llm_json_extractor import extract_json

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
                AVG(metric_value) AS avg_value,
                MAX(metric_value) AS max_value,

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


def generate_ec2_prompt(instance_data: Dict[str, Any]) -> str:
    """
    Generate LLM prompt for EC2 instance optimization recommendations.

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

    prompt = f"""
You are a cloud cost optimization expert for AWS. Analyze the following EC2 instance and provide optimization recommendations in strict JSON format.

**EC2 Instance Details:**
- Instance ID: {instance_id}
- Instance Type: {instance_type}
- Region: {region}
- Analysis Period: {start_date} to {end_date} ({duration_days} days)
- Total Billed Cost: ${billed_cost:.2f}

**Performance Metrics:**
- CPU Utilization: Avg {cpu_avg:.2f}%, Max {cpu_max:.2f}% (on {cpu_max_date})
- Network In: Avg {network_in_avg:.2f} bytes, Max {network_in_max:.2f} bytes
- Network Out: Avg {network_out_avg:.2f} bytes, Max {network_out_max:.2f} bytes
- Disk Read Ops: Avg {disk_read_avg:.2f} ops/sec
- Disk Write Ops: Avg {disk_write_avg:.2f} ops/sec

**Your Task:**
Based on the utilization metrics above, provide cost optimization recommendations. Consider:
1. Is the instance right-sized or should it be resized?
2. Could this workload run on Spot instances or use Reserved Instances?
3. Are there scheduling opportunities (e.g., stop during non-business hours)?
4. Any performance anomalies that indicate inefficient usage?

**Response Format (JSON only):**
{{
  "recommendations": {{
    "effective_recommendation": {{
      "text": "Primary recommendation (e.g., 'Downsize from t3.large to t3.medium')",
      "saving_pct": <percentage as number>
    }},
    "additional_recommendation": [
      {{
        "text": "Secondary recommendation",
        "saving_pct": <percentage as number>
      }}
    ],
    "base_of_recommendations": [
      "Reasoning point 1",
      "Reasoning point 2"
    ]
  }},
  "cost_forecasting": {{
    "monthly": <projected monthly cost as number>,
    "annually": <projected annual cost as number>
  }},
  "anomalies": [
    {{
      "metric_name": "CPUUtilization",
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "value": <anomaly value>,
      "reason_short": "Brief explanation"
    }}
  ],
  "contract_deal": {{
    "assessment": "good|bad|unknown",
    "for sku": "{instance_type}",
    "reason": "Explanation of pricing assessment",
    "monthly_saving_pct": <percentage as number>,
    "annual_saving_pct": <percentage as number>
  }}
}}

Return ONLY the JSON object, no additional text.
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
# VPC FUNCTIONS
# ============================================================

@connection
def fetch_vpc_utilization_data(conn, schema_name, start_date, end_date, resource_id=None):
    """
    Fetch VPC resource metrics and cost data.
    Includes VPC, NAT Gateway, VPN, and VPC Endpoint metrics.

    Args:
        conn: Database connection
        schema_name: PostgreSQL schema name
        start_date: Start date for analysis
        end_date: End date for analysis
        resource_id: Optional specific resource ID to analyze

    Returns:
        DataFrame with VPC resource utilization and cost data
    """

    resource_filter_sql = sql.SQL("AND LOWER(m.resource_id) = LOWER(%s)") if resource_id else sql.SQL("")

    QUERY = sql.SQL("""
        WITH metric_agg AS (
            SELECT
                m.resource_id,
                m.vpc_id,
                m.vpc_name,
                m.resource_type,
                r.region,
                r.account_id,
                m.metric_name,
                m.value AS metric_value,
                m.timestamp
            FROM {schema_name}.fact_vpc_metrics m
            LEFT JOIN {schema_name}.dim_vpc_resource r ON m.resource_id = r.resource_id
            WHERE
                m.timestamp BETWEEN %s AND %s
                {resource_filter}
        ),

        usage_summary AS (
            SELECT
                resource_id,
                vpc_id,
                vpc_name,
                resource_type,
                region,
                account_id,
                metric_name,
                AVG(metric_value) AS avg_value,
                MAX(metric_value) AS max_value,

                FIRST_VALUE(timestamp) OVER (
                    PARTITION BY resource_id, metric_name
                    ORDER BY metric_value DESC, timestamp DESC
                ) AS max_date

            FROM metric_agg
            GROUP BY resource_id, vpc_id, vpc_name, resource_type, region, account_id, metric_name
        ),

        metric_map AS (
            SELECT
                resource_id,
                vpc_id,
                vpc_name,
                resource_type,
                region,
                account_id,
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
            GROUP BY 1, 2, 3, 4, 5, 6
        )

        SELECT
            us.resource_id,
            us.vpc_id,
            us.vpc_name,
            us.resource_type,
            us.region,
            us.account_id,
            m.metrics_json,
            COALESCE(SUM(ff.billed_cost), 0) AS billed_cost,
            COALESCE(SUM(ff.consumed_quantity), 0) AS consumed_quantity,
            MAX(ff.consumed_unit) AS consumed_unit,
            MAX(ff.pricing_category) AS pricing_category
        FROM usage_summary us
        LEFT JOIN metric_map m ON m.resource_id = us.resource_id
        LEFT JOIN {schema_name}.gold_aws_fact_focus ff
            ON LOWER(ff.resource_id) LIKE '%%' || LOWER(us.resource_id) || '%%'
               AND ff.service_name LIKE 'Amazon Virtual Private Cloud%%'
               AND ff.charge_period_start::date <= %s
               AND ff.charge_period_end::date >= %s
        GROUP BY 1, 2, 3, 4, 5, 6, 7
    """).format(
        schema_name=sql.Identifier(schema_name),
        resource_filter=resource_filter_sql
    )

    params = [start_date, end_date, end_date, start_date]
    if resource_id:
        params.insert(2, resource_id)

    try:
        cursor = conn.cursor()
        cursor.execute(QUERY, params)

        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        cursor.close()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)

        # Expand metrics_json
        if not df.empty and "metrics_json" in df.columns:
            metrics_expanded = pd.json_normalize(df["metrics_json"].fillna({})).add_prefix("metric_")
            metrics_expanded.index = df.index
            df = pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)

        return df

    except psycopg2.Error as e:
        raise RuntimeError(f"PostgreSQL query failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during VPC data fetch: {e}") from e


def generate_vpc_prompt(resource_data: Dict[str, Any]) -> str:
    """
    Generate LLM prompt for VPC resource optimization recommendations.
    """

    resource_id = resource_data.get('resource_id', 'Unknown')
    resource_type = resource_data.get('resource_type', 'vpc')
    vpc_id = resource_data.get('vpc_id', 'Unknown')
    region = resource_data.get('region', 'Unknown')
    billed_cost = resource_data.get('billed_cost', 0)

    start_date = resource_data.get('start_date', 'N/A')
    end_date = resource_data.get('end_date', 'N/A')
    duration_days = resource_data.get('duration_days', 0)

    # Extract available metrics (varies by resource type)
    metrics_info = []
    for key, value in resource_data.items():
        if key.startswith('metric_') and '_Avg' in key:
            metric_name = key.replace('metric_', '').replace('_Avg', '')
            avg_val = value
            max_val = resource_data.get(f'metric_{metric_name}_Max', 'N/A')
            metrics_info.append(f"- {metric_name}: Avg {avg_val}, Max {max_val}")

    metrics_str = '\n'.join(metrics_info) if metrics_info else "No detailed metrics available"

    prompt = f"""
You are a cloud cost optimization expert for AWS networking and VPC resources. Analyze the following VPC resource and provide optimization recommendations in strict JSON format.

**VPC Resource Details:**
- Resource ID: {resource_id}
- Resource Type: {resource_type}
- VPC ID: {vpc_id}
- Region: {region}
- Analysis Period: {start_date} to {end_date} ({duration_days} days)
- Total Billed Cost: ${billed_cost:.2f}

**Performance Metrics:**
{metrics_str}

**Your Task:**
Based on the resource type and utilization, provide cost optimization recommendations. Consider:
1. For NAT Gateways: Can you use NAT instances or VPC endpoints instead?
2. For VPCs: Are there idle resources or unused subnets?
3. For VPN connections: Are they actively used or can they be consolidated?
4. For VPC Endpoints: Are they optimally configured?

**Response Format (JSON only):**
{{
  "recommendations": {{
    "effective_recommendation": {{
      "text": "Primary recommendation",
      "saving_pct": <percentage as number>
    }},
    "additional_recommendation": [
      {{
        "text": "Secondary recommendation",
        "saving_pct": <percentage as number>
      }}
    ],
    "base_of_recommendations": [
      "Reasoning point 1",
      "Reasoning point 2"
    ]
  }},
  "cost_forecasting": {{
    "monthly": <projected monthly cost as number>,
    "annually": <projected annual cost as number>
  }},
  "anomalies": [
    {{
      "metric_name": "BytesTransferred",
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "value": <anomaly value>,
      "reason_short": "Brief explanation"
    }}
  ],
  "contract_deal": {{
    "assessment": "good|bad|unknown",
    "for sku": "{resource_type}",
    "reason": "Explanation of pricing assessment",
    "monthly_saving_pct": <percentage as number>,
    "annual_saving_pct": <percentage as number>
  }}
}}

Return ONLY the JSON object, no additional text.
"""

    return prompt


def get_vpc_recommendation_single(resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get LLM recommendation for a single VPC resource.
    """
    try:
        prompt = generate_vpc_prompt(resource_data)
        llm_response = llm_call(prompt)

        if not llm_response:
            LOG.warning(f"Empty LLM response for resource {resource_data.get('resource_id')}")
            return None

        recommendation = extract_json(llm_response)

        if recommendation:
            recommendation['resource_id'] = resource_data.get('resource_id', 'Unknown')
            return recommendation
        else:
            LOG.warning(f"Failed to parse JSON for resource {resource_data.get('resource_id')}")
            return None

    except Exception as e:
        LOG.error(f"Error getting VPC recommendation: {e}")
        return None


def run_llm_analysis_vpc(resource_type: str, schema_name: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         resource_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main entry point for VPC LLM analysis.
    """

    start_str = start_date.strftime("%Y-%m-%d") if start_date else (datetime.utcnow().date() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d") if end_date else datetime.utcnow().date().strftime("%Y-%m-%d")

    LOG.info(f"🚀 Starting VPC LLM analysis from {start_str} to {end_str}...")

    try:
        df = fetch_vpc_utilization_data(schema_name, start_str, end_str, resource_id)
    except RuntimeError as e:
        LOG.error(f"❌ Failed to fetch VPC utilization data: {e}")
        return []
    except Exception as e:
        LOG.error(f"❌ Unhandled error during data fetching: {e}")
        return []

    if df is None or df.empty:
        LOG.warning("⚠️ No VPC resource data found for the requested date range / resource.")
        return []

    LOG.info(f"📈 Retrieved data for {len(df)} VPC resource(s)")

    # Annotate with date info
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days

    resources = df.to_dict(orient="records")

    LOG.info("🤖 Calling LLM for VPC recommendations...")
    recommendations = []

    for resource_data in resources:
        rec = get_vpc_recommendation_single(resource_data)
        if rec:
            recommendations.append(rec)

    if recommendations:
        LOG.info(f"✅ VPC analysis complete! Generated {len(recommendations)} recommendation(s).")
        return recommendations
    else:
        LOG.warning("⚠️ No recommendations generated by LLM.")
        return []


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_llm_analysis(resource_type: str, schema_name: str,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     resource_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main router for AWS EC2/VPC LLM analysis.

    Args:
        resource_type: 'ec2' or 'vpc'
        schema_name: PostgreSQL schema name
        start_date: Start date for analysis
        end_date: End date for analysis
        resource_id: Optional specific resource ID

    Returns:
        List of recommendation dictionaries
    """

    resource_type = resource_type.lower().strip()

    if resource_type == 'ec2':
        return run_llm_analysis_ec2(resource_type, schema_name, start_date, end_date, resource_id)
    elif resource_type == 'vpc':
        return run_llm_analysis_vpc(resource_type, schema_name, start_date, end_date, resource_id)
    else:
        LOG.error(f"❌ Unknown resource type: {resource_type}. Supported types: 'ec2', 'vpc'")
        return []