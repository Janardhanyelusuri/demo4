# app/ingestion/azure/llm_data_fetch.py

import psycopg2
import pandas as pd
import sys
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Import necessary functions from the same directory or core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.ingestion.azure.postgres_operation import connection
# Import LLM recommendation functions from the new analysis file
from app.ingestion.azure.llm_analysis import (
    _extrapolate_costs,
    get_compute_recommendation_single,
    get_storage_recommendation_single,
    get_public_ip_recommendation_single,
)

load_dotenv()

# --- Utility Functions ---

def _create_local_engine_from_env():
    """
    Create a SQLAlchemy engine using DB env vars.
    Uses quote_plus to safely escape the password.
    """
    user = os.getenv("DB_USER_NAME")
    password = os.getenv("DB_PASSWORD") or ""
    host = os.getenv("DB_HOST_NAME")
    port = os.getenv("DB_PORT") or "5432"
    db = os.getenv("DB_NAME")

    if not all([user, password, host, db]):
        raise RuntimeError("Missing DB env vars. Ensure DB_USER_NAME/DB_PASSWORD/DB_HOST_NAME/DB_NAME are set")

    pwd_esc = quote_plus(password)
    engine_url = f"postgresql+psycopg2://{user}:{pwd_esc}@{host}:{port}/{db}"
    print(f"[DEBUG] Creating local SQLAlchemy engine for {host}:{port}/{db} user={user}")
    engine = create_engine(engine_url, pool_pre_ping=True)
    return engine

def _is_resource_for_type(resource_type: str, resource_id: Optional[str]) -> bool:
    """
    Quick heuristic to confirm the supplied resource_id looks like the requested resource_type.
    Returns True if unknown or resource_id is None.
    """
    if not resource_id:
        return True
    rid = resource_id.lower()
    t = resource_type.strip().lower()
    if t in ("vm", "virtualmachine", "virtual_machine"):
        # ARM path fragment check for VMs
        return ("/virtualmachines/" in rid) or ("/compute/virtualmachines" in rid)
    if t in ("storage", "storageaccount", "storage_account"):
        return ("/storageaccounts/" in rid) or ("/storage/" in rid)
    # default: accept if unknown type
    return True


# --- VM: Dynamic Metrics + Spike Date (Data Fetching) ---

def fetch_vm_utilization_data(conn, schema_name, start_date, end_date, resource_id=None):
    """
    Fetch VM metrics including AVG, MAX value, and the MAX timestamp.
    Applies a filter to include only essential metrics for LLM analysis.
    
    ✅ MODIFICATION: Now includes 'vm_name' and 'instance_type' (SKU)
    """
    resource_filter_sql = ""
    resource_dim_filter_sql = ""
    params = {"start_date": start_date, "end_date": end_date}
    if resource_id:
        resource_filter_sql = "AND LOWER(resource_id) = LOWER(%(resource_id)s)"
        resource_dim_filter_sql = "WHERE LOWER(resource_id) = LOWER(%(resource_id)s)"
        params["resource_id"] = resource_id
    else:
        # Only include actual Virtual Machines (microsoft.compute/virtualmachines)
        # Exclude Databricks VMs and other resource types (Fabric, SQL, etc.)
        # Note: %% escapes % in parameterized queries
        resource_dim_filter_sql = """WHERE LOWER(resource_id) LIKE '%%/microsoft.compute/virtualmachines/%%'
                                      AND LOWER(resource_id) NOT LIKE '%%databricks%%'"""

    # Define the list of essential metrics
    essential_metrics = (
        'Percentage CPU', 
        'Available Memory Bytes',
        'Disk Read Operations/Sec', 
        'Disk Write Operations/Sec',
        'Network In', 
        'Network Out'
    )
    
    quoted_metrics_str = ", ".join([f"'{m}'" for m in essential_metrics])
    metrics_filter_sql = f"AND metric_name IN ({quoted_metrics_str})"
    
    # Build metrics filter for CTE queries
    if resource_id:
        metrics_cte_filter_sql = resource_filter_sql
    else:
        # Filter metrics to only include actual VMs, excluding Databricks
        metrics_cte_filter_sql = """AND LOWER(resource_id) LIKE '%%/microsoft.compute/virtualmachines/%%'
                                    AND LOWER(resource_id) NOT LIKE '%%databricks%%'"""

    query = f"""
       WITH metric_pivot AS (
            SELECT
                LOWER(resource_id) AS resource_id,
                metric_name,
                value::FLOAT AS metric_value,
                "timestamp"
            FROM {schema_name}.gold_azure_fact_vm_metrics
            WHERE resource_id IS NOT NULL
              AND "timestamp" >= %(start_date)s::timestamp
              AND "timestamp" <= (%(end_date)s::timestamp + INTERVAL '1 day' - INTERVAL '1 second')
              AND metric_name IS NOT NULL
              {metrics_cte_filter_sql}
              {metrics_filter_sql}
        ),

        -- ✅ NEW CTE: Get VM-specific details (name, instance type) from the metrics fact
        vm_details AS (
            SELECT DISTINCT ON (LOWER(resource_id))
                LOWER(resource_id) AS resource_id,
                vm_name,
                instance_type -- This is the VM SKU
            FROM {schema_name}.gold_azure_fact_vm_metrics
            WHERE resource_id IS NOT NULL
            {metrics_cte_filter_sql}
        ),

        -- NEW CTE 1: Calculate AVG and MAX metric values (with byte-to-GB conversion)
        metric_avg_max AS (
            SELECT
                resource_id,
                metric_name,
                -- Convert bytes to GB for memory and network metrics
                AVG(
                    CASE
                        WHEN metric_name IN ('Available Memory Bytes', 'Network In', 'Network Out', 'Network In Total', 'Network Out Total')
                        THEN metric_value / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE metric_value
                    END
                ) AS avg_value,
                MAX(
                    CASE
                        WHEN metric_name IN ('Available Memory Bytes', 'Network In', 'Network Out', 'Network In Total', 'Network Out Total')
                        THEN metric_value / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE metric_value
                    END
                ) AS max_value
            FROM metric_pivot
            GROUP BY resource_id, metric_name
        ),
        
        -- NEW CTE 2: Find the exact timestamp corresponding to the maximum value
        metric_max_timestamp AS (
            SELECT DISTINCT ON (resource_id, metric_name)
                resource_id,
                metric_name,
                "timestamp" AS max_timestamp
            FROM metric_pivot
            ORDER BY resource_id, metric_name, metric_value DESC, "timestamp" DESC
        ),

        -- NEW CTE 3: Combine AVG/MAX values with the MAX timestamp
        metric_agg AS (
            SELECT
                amm.resource_id,
                amm.metric_name,
                amm.avg_value,
                amm.max_value,
                amt.max_timestamp
            FROM metric_avg_max amm
            JOIN metric_max_timestamp amt 
                ON amm.resource_id = amt.resource_id 
                AND amm.metric_name = amt.metric_name
        ),

        metric_map AS (
            SELECT
                resource_id,
                (
                    json_object_agg(metric_name || '_Avg', ROUND(avg_value::NUMERIC, 6))::jsonb ||
                    json_object_agg(metric_name || '_Max', ROUND(max_value::NUMERIC, 6))::jsonb ||
                    json_object_agg(metric_name || '_MaxDate', TO_CHAR(max_timestamp, 'YYYY-MM-DD HH24:MI'))::jsonb
                )::json AS metrics_json
            FROM metric_agg
            GROUP BY resource_id
        ),
        
        cost_agg AS (
            SELECT
                LOWER(f.resource_id) AS resource_id,
                MAX(f.contracted_unit_price) AS contracted_unit_price,
                SUM(COALESCE(f.pricing_quantity,0)) AS pricing_quantity,
                SUM(COALESCE(f.billed_cost,0)) AS billed_cost,
                SUM(COALESCE(f.consumed_quantity,0)) AS consumed_quantity,
                MAX(COALESCE(f.consumed_unit, '')) AS consumed_unit,
                MAX(COALESCE(f.pricing_unit, '')) AS pricing_unit
            FROM {schema_name}.gold_azure_fact_cost f
            WHERE f.charge_period_start::date BETWEEN %(start_date)s::date AND %(end_date)s::date
              {metrics_cte_filter_sql}
            GROUP BY LOWER(f.resource_id)
        ),

        resource_dim AS (
            SELECT
                LOWER(resource_id) AS resource_id,
                resource_name,
                region_id,
                region_name,
                service_category,
                service_name
            FROM {schema_name}.gold_azure_resource_dim
            {resource_dim_filter_sql}
        )

        SELECT
            rd.resource_id,
            -- ADDED VM specific columns
            COALESCE(vd.vm_name, rd.resource_name) AS vm_name,
            vd.instance_type,
            rd.region_id,
            rd.region_name,
            rd.service_category,
            rd.service_name,
            COALESCE(c.contracted_unit_price, NULL) AS contracted_unit_price,
            COALESCE(c.pricing_quantity, 0) AS pricing_quantity,
            COALESCE(c.billed_cost, 0) AS billed_cost,
            COALESCE(c.consumed_quantity, 0) AS consumed_quantity,
            COALESCE(c.consumed_unit, '') AS consumed_unit,
            COALESCE(c.pricing_unit, '') AS pricing_unit,
            m.metrics_json
        FROM resource_dim rd
        --  JOIN with the new VM details CTE
        LEFT JOIN vm_details vd ON rd.resource_id = vd.resource_id
        LEFT JOIN metric_map m ON rd.resource_id = m.resource_id
        LEFT JOIN cost_agg c ON rd.resource_id = c.resource_id
        ORDER BY COALESCE(c.billed_cost, 0) DESC;
    """
    
    try:
        # Pass the query and standard parameters.
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"Error executing VM utilization query: {e}")
        return pd.DataFrame()

    # If resource_id provided, guarantee at most one row
    if resource_id and not df.empty:
        df = df.head(1).reset_index(drop=True)

    # Expand the metrics_json into separate columns (flatten)
    if not df.empty and "metrics_json" in df.columns:
        try:
            # Ensure JSON strings become dicts
            def _to_dict(x):
                if x is None:
                    return {}
                if isinstance(x, str):
                    try:
                        return json.loads(x)
                    except Exception:
                        return {}
                if isinstance(x, dict):
                    return x
                return {}

            metrics_series = df["metrics_json"].apply(_to_dict)
            metrics_expanded = pd.json_normalize(metrics_series).add_prefix("metric_")
            metrics_expanded.index = df.index
            # We rename rd.resource_name to 'resource_name' if it's not overridden by vm_name
            df = pd.concat([df.drop(columns=["metrics_json", "resource_name"]), metrics_expanded], axis=1)
            df.rename(columns={'vm_name': 'resource_name'}, inplace=True) # Ensure 'resource_name' remains
        except Exception as ex:
            print(f"Warning: failed to expand metrics_json: {ex}")
            # keep original df without expansion

    return df


@connection
def run_llm_vm(conn, schema_name, start_date=None, end_date=None, resource_id=None, task_id=None) -> Optional[Dict[str, Any]]:
    """
    Run LLM analysis for a single VM and return a single recommendation dict (or None).

    Args:
        task_id: Optional task ID for cancellation support
    """
    from app.core.task_manager import task_manager

    if end_date is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = pd.to_datetime(start_date).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # CRITICAL: Check if task was cancelled before starting
    if task_id:
        is_cancelled = task_manager.is_cancelled(task_id)
        if is_cancelled:
            print(f"🛑 Task {task_id} was cancelled before VM LLM could start. Returning None.")
            return None

    print(f"🔎 Running VM LLM for {schema_name} from {start_str} to {end_str} "
          f"{'(resource_id filter applied)' if resource_id else ''}")

    df = fetch_vm_utilization_data(conn, schema_name, start_str, end_str, resource_id=resource_id)
    if df is None or df.empty:
        print("⚠️ No VM data found for the requested date range / resource.")
        return None

    # annotate with date info for LLM context
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days or 1

    # convert to single record (we always process only one resource)
    if resource_id and df.shape[0] > 1:
        print(f"⚠️ WARNING: Resource ID was provided, but {df.shape[0]} records were fetched. Restricting to the first record for LLM analysis.")
    resource_row = df.head(1).to_dict(orient="records")[0]

    # Add schema_name and region to resource_row for pricing lookups
    resource_row['schema_name'] = schema_name
    resource_row['region'] = resource_row.get('location', 'eastus')  # Extract region from location or default

    # Check cancellation again before calling expensive LLM
    if task_id:
        is_cancelled = task_manager.is_cancelled(task_id)
        if is_cancelled:
            print(f"🛑 Task {task_id} was cancelled before LLM call. Returning None.")
            return None

    # Call the imported LLM analysis function
    recommendation = get_compute_recommendation_single(resource_row)
    
    if recommendation:
        print("✅ LLM analysis complete! Returning recommendation.")
        return recommendation
    else:
        print("⚠️ No recommendation generated by LLM.")
        return None


# --- Storage: Dynamic Metrics + Spike Date (Data Fetching) ---
def fetch_storage_account_utilization_data(
    conn,
    schema_name: str,
    start_date: str,
    end_date: str,
    resource_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch storage account metrics including AVG, MAX value, and the date of MAX (spike).
    Parameterized resource id and returns at most one row when resource_id provided.
    """
    params = {
        "start_date": start_date,
        "end_date": end_date,
    }

    resource_filter_sql = ""
    resource_filter_dim = ""
    if resource_id:
        params["resource_id"] = resource_id
        resource_filter_sql = "AND LOWER(resource_id) = LOWER(%(resource_id)s)"
        resource_filter_dim = "WHERE LOWER(resource_id) = LOWER(%(resource_id)s)"
    else:
        # Exclude Databricks resources when fetching all Storage Accounts
        # Note: %% escapes % in parameterized queries
        resource_filter_dim = "WHERE LOWER(resource_id) NOT LIKE '%%databricks%%'"


    query = f"""
            WITH fact_base AS (
        SELECT
            fsd.storage_account_key,
            LOWER(sa.resource_id) AS resource_id,
            dm.metric_name,
            fsd.date_key,
            fsd.daily_value_avg,
            fsd.daily_value_max,
            fsd.daily_cost_sum
        FROM {schema_name}.fact_storage_daily_usage fsd
        JOIN {schema_name}.dim_storage_account sa
            ON fsd.storage_account_key = sa.storage_account_key
        JOIN {schema_name}.dim_metric dm
            ON fsd.metric_key = dm.metric_key
        WHERE fsd.date_key IS NOT NULL
        AND to_date(fsd.date_key::text, 'YYYYMMDD')
                BETWEEN %(start_date)s::date AND %(end_date)s::date
        {resource_filter_sql}
    ),

    metric_avg_max AS (
        SELECT
            resource_id,
            metric_name,
            -- Convert bytes to GB for network and capacity metrics
            AVG(
                CASE
                    WHEN metric_name IN ('Ingress', 'Egress', 'UsedCapacity', 'BlobCapacity', 'FileCapacity', 'TableCapacity', 'QueueCapacity')
                    THEN daily_value_avg / 1073741824.0  -- Convert bytes to GB (1024^3)
                    ELSE daily_value_avg
                END
            ) AS avg_value,
            MAX(
                CASE
                    WHEN metric_name IN ('Ingress', 'Egress', 'UsedCapacity', 'BlobCapacity', 'FileCapacity', 'TableCapacity', 'QueueCapacity')
                    THEN daily_value_max / 1073741824.0  -- Convert bytes to GB (1024^3)
                    ELSE daily_value_max
                END
            ) AS max_value
        FROM fact_base
        GROUP BY resource_id, metric_name
    ),

    metric_max_date AS (
        SELECT DISTINCT ON (resource_id, metric_name)
            resource_id,
            metric_name,
            date_key AS max_date_key
        FROM fact_base
        ORDER BY resource_id, metric_name, daily_value_max DESC, date_key DESC
    ),

    metric_final AS (
        SELECT
            amm.resource_id,
            amm.metric_name,
            amm.avg_value,
            amm.max_value,
            mmd.max_date_key
        FROM metric_avg_max amm
        JOIN metric_max_date mmd 
            ON amm.resource_id = mmd.resource_id
        AND amm.metric_name = mmd.metric_name
    ),

    metric_map AS (
        SELECT
            resource_id,
            (
                json_object_agg(metric_name || '_Avg', ROUND(avg_value::NUMERIC, 6))::jsonb ||
                json_object_agg(metric_name || '_Max', ROUND(max_value::NUMERIC, 6))::jsonb ||
                json_object_agg(metric_name || '_MaxDate',
                    TO_CHAR(to_date(max_date_key::text, 'YYYYMMDD'), 'YYYY-MM-DD'))::jsonb
            )::json AS metrics_json
        FROM metric_final
        GROUP BY resource_id
    ),

    -- cost aggregated by exact resource_id (what's exported)
    cost_agg_exact AS (
        SELECT
            LOWER(f.resource_id) AS resource_id,
            MAX(f.contracted_unit_price) AS contracted_unit_price,
            SUM(COALESCE(f.pricing_quantity,0)) AS pricing_quantity,
            SUM(COALESCE(f.billed_cost,0)) AS billed_cost,
            SUM(COALESCE(f.consumed_quantity,0)) AS consumed_quantity,
            MAX(COALESCE(f.consumed_unit, '')) AS consumed_unit,
            MAX(COALESCE(f.pricing_unit, '')) AS pricing_unit
        FROM {schema_name}.gold_azure_fact_cost f
        WHERE f.charge_period_start::date BETWEEN %(start_date)s::date AND %(end_date)s::date
        {resource_filter_sql}
        GROUP BY LOWER(f.resource_id)
    ),

    -- cost aggregated by account root (strip subservice suffixes)
    cost_agg_root AS (
        SELECT
            LOWER(regexp_replace(f.resource_id,
                '/(blobServices|fileServices|queueServices|tableServices).*$', '', 'i')) AS resource_root,
            MAX(f.contracted_unit_price) AS contracted_unit_price_root,
            SUM(COALESCE(f.pricing_quantity,0)) AS pricing_quantity_root,
            SUM(COALESCE(f.billed_cost,0)) AS billed_cost_root,
            SUM(COALESCE(f.consumed_quantity,0)) AS consumed_quantity_root,
            MAX(COALESCE(f.consumed_unit, '')) AS consumed_unit_root,
            MAX(COALESCE(f.pricing_unit, '')) AS pricing_unit_root
        FROM {schema_name}.gold_azure_fact_cost f
        WHERE f.charge_period_start::date BETWEEN %(start_date)s::date AND %(end_date)s::date
        {resource_filter_sql}
        GROUP BY 1
    ),

    resource_dim AS (
        SELECT
            LOWER(resource_id) AS resource_id,
            storage_account_name,
            region,
            kind,
            sku,
            access_tier
        FROM {schema_name}.dim_storage_account
        {resource_filter_dim}
    )

    SELECT
        rd.resource_id,
        rd.storage_account_name,
        rd.region,
        rd.kind,
        rd.sku,
        rd.access_tier,

        -- prefer exact match values, otherwise fall back to root-aggregates
        COALESCE(ce.contracted_unit_price, cr.contracted_unit_price_root, NULL) AS contracted_unit_price,
        COALESCE(ce.pricing_quantity, cr.pricing_quantity_root, 0) AS pricing_quantity,
        COALESCE(ce.billed_cost, cr.billed_cost_root, 0) AS billed_cost,
        COALESCE(ce.consumed_quantity, cr.consumed_quantity_root, 0) AS consumed_quantity,
        COALESCE(ce.consumed_unit, cr.consumed_unit_root, '') AS consumed_unit,
        COALESCE(ce.pricing_unit, cr.pricing_unit_root, '') AS pricing_unit,

        m.metrics_json
    FROM resource_dim rd
    LEFT JOIN metric_map m ON rd.resource_id = m.resource_id

    -- exact match join (will be NULL if cost exported only to root)
    LEFT JOIN cost_agg_exact ce ON rd.resource_id = ce.resource_id

    -- root-level fallback: compute account root from rd.resource_id and join
    LEFT JOIN cost_agg_root cr
    ON regexp_replace(rd.resource_id,
            '/(blobServices|fileServices|queueServices|tableServices).*$', '', 'i') = cr.resource_root;



    """

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"Error executing Storage utilization query: {e}")
        return pd.DataFrame()

    # If resource_id provided, guarantee at most one row
    if resource_id and not df.empty:
        df = df.head(1).reset_index(drop=True)

    # Expand metrics_json into columns
    if not df.empty and "metrics_json" in df.columns:
        try:
            # Ensure JSON strings become dicts
            def _to_dict(x):
                if x is None:
                    return {}
                if isinstance(x, str):
                    try:
                        return json.loads(x)
                    except Exception:
                        return {}
                if isinstance(x, dict):
                    return x
                return {}

            metrics_series = df["metrics_json"].apply(_to_dict)
            metrics_expanded = pd.json_normalize(metrics_series).add_prefix("metric_")
            metrics_expanded.index = df.index
            df = pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)
        except Exception as ex:
            print(f"Warning: failed to expand storage metrics_json: {ex}")

    return df


@connection
def run_llm_storage(conn, schema_name, start_date=None, end_date=None, resource_id=None, task_id=None) -> Optional[Dict[str, Any]]:
    """
    Run LLM analysis for a single Storage Account and return a single recommendation dict (or None).

    Args:
        task_id: Optional task ID for cancellation support
    """
    from app.core.task_manager import task_manager

    if end_date is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=7)
    else:
        start_dt = pd.to_datetime(start_date).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # CRITICAL: Check if task was cancelled before starting
    if task_id:
        is_cancelled = task_manager.is_cancelled(task_id)
        if is_cancelled:
            print(f"🛑 Task {task_id} was cancelled before Storage LLM could start. Returning None.")
            return None

    print(f"🔎 Running Storage LLM for {schema_name} from {start_str} to {end_str} "
          f"{'(resource_id filter applied)' if resource_id else ''}")

    df = fetch_storage_account_utilization_data(conn, schema_name, start_str, end_str, resource_id=resource_id)
    if df is None or df.empty:
        print("⚠️ No storage account data found for the requested date range / resource.")
        return None

    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days or 1

    if resource_id and df.shape[0] > 1:
        print(f"⚠️ WARNING: Resource ID was provided, but {df.shape[0]} records were fetched. Restricting to the first record for LLM analysis.")

    resource_row = df.head(1).to_dict(orient="records")[0]

    # Add schema_name and region for pricing lookups
    resource_row['schema_name'] = schema_name
    resource_row['region'] = resource_row.get('location', 'eastus')

    # Check cancellation again before calling expensive LLM
    if task_id:
        is_cancelled = task_manager.is_cancelled(task_id)
        if is_cancelled:
            print(f"🛑 Task {task_id} was cancelled before LLM call. Returning None.")
            return None

    # Call the imported LLM analysis function
    recommendation = get_storage_recommendation_single(resource_row)

    if recommendation:
        print("✅ LLM analysis complete! Returning recommendation.")
        return recommendation
    else:
        print("⚠️ No recommendation generated by LLM.")
        return None


@connection
def run_llm_vm_all_resources(conn, schema_name, start_date=None, end_date=None, task_id=None) -> List[Dict[str, Any]]:
    """
    NEW FUNCTION: Fetch ALL distinct VMs and process each through LLM individually.

    Args:
        task_id: Optional task ID for cancellation support

    Returns:
        List of recommendation dictionaries, one per VM resource
    """
    from app.core.task_manager import task_manager
    if end_date is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = pd.to_datetime(start_date).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print(f"🔎 Running VM LLM for ALL distinct VMs in {schema_name} from {start_str} to {end_str}")

    # Fetch data for ALL VMs (no resource_id filter)
    df = fetch_vm_utilization_data(conn, schema_name, start_str, end_str, resource_id=None)

    if df is None or df.empty:
        print("⚠️ No VM data found for the requested date range.")
        return []

    # Annotate with date info
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days or 1

    recommendations = []
    total_resources = len(df)
    print(f"📊 Found {total_resources} distinct VM resources to analyze")

    # Process each row (resource) individually through LLM
    for idx, row in df.iterrows():
        # Check if task has been cancelled
        if task_id:
            is_cancelled = task_manager.is_cancelled(task_id)
            if is_cancelled:
                print(f"🛑 Task {task_id} was cancelled. Stopping VM analysis. Processed {len(recommendations)}/{total_resources}")
                break
            # Debug: Confirm task is still active
            if idx == 0 or idx % 10 == 0:  # Log every 10th iteration
                print(f"  🔍 Task {task_id[:8]}... still running, not cancelled (iteration {idx + 1})")

        resource_id = row.get('resource_id', 'Unknown')
        print(f"  [{idx + 1}/{total_resources}] Processing VM: {resource_id}")

        try:
            resource_dict = row.to_dict()
            # Add schema_name and region for pricing lookups
            resource_dict['schema_name'] = schema_name
            resource_dict['region'] = resource_dict.get('location', 'eastus')
            recommendation = get_compute_recommendation_single(resource_dict)

            if recommendation:
                recommendations.append(recommendation)
                print(f"    ✅ LLM analysis complete for {resource_id}")
            else:
                print(f"    ⚠️ No recommendation generated for {resource_id}")
        except Exception as e:
            print(f"    ❌ Error processing {resource_id}: {e}")
            continue

    print(f"✅ Completed processing {len(recommendations)}/{total_resources} VMs successfully")
    return recommendations


@connection
def run_llm_storage_all_resources(conn, schema_name, start_date=None, end_date=None, task_id=None) -> List[Dict[str, Any]]:
    """
    NEW FUNCTION: Fetch ALL distinct Storage Accounts and process each through LLM individually.

    Args:
        task_id: Optional task ID for cancellation support

    Returns:
        List of recommendation dictionaries, one per Storage Account
    """
    from app.core.task_manager import task_manager
    if end_date is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=7)
    else:
        start_dt = pd.to_datetime(start_date).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print(f"🔎 Running Storage LLM for ALL distinct Storage Accounts in {schema_name} from {start_str} to {end_str}")

    # Fetch data for ALL storage accounts (no resource_id filter)
    df = fetch_storage_account_utilization_data(conn, schema_name, start_str, end_str, resource_id=None)

    if df is None or df.empty:
        print("⚠️ No storage account data found for the requested date range.")
        return []

    # Annotate with date info
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days or 1

    recommendations = []
    total_resources = len(df)
    print(f"📊 Found {total_resources} distinct Storage Account resources to analyze")

    # Process each row (resource) individually through LLM
    for idx, row in df.iterrows():
        # Check if task has been cancelled
        if task_id:
            is_cancelled = task_manager.is_cancelled(task_id)
            if is_cancelled:
                print(f"🛑 Task {task_id} was cancelled. Stopping Storage analysis. Processed {len(recommendations)}/{total_resources}")
                break
            # Debug: Confirm task is still active
            if idx == 0 or idx % 10 == 0:  # Log every 10th iteration
                print(f"  🔍 Task {task_id[:8]}... still running, not cancelled (iteration {idx + 1})")

        resource_id = row.get('resource_id', 'Unknown')
        print(f"  [{idx + 1}/{total_resources}] Processing Storage Account: {resource_id}")

        try:
            resource_dict = row.to_dict()
            # Add schema_name and region for pricing lookups
            resource_dict['schema_name'] = schema_name
            resource_dict['region'] = resource_dict.get('location', 'eastus')
            recommendation = get_storage_recommendation_single(resource_dict)

            if recommendation:
                recommendations.append(recommendation)
                print(f"    ✅ LLM analysis complete for {resource_id}")
            else:
                print(f"    ⚠️ No recommendation generated for {resource_id}")
        except Exception as e:
            print(f"    ❌ Error processing {resource_id}: {e}")
            continue

    print(f"✅ Completed processing {len(recommendations)}/{total_resources} Storage Accounts successfully")
    return recommendations



# --- Public IP: Dynamic Metrics + Spike Date (Data Fetching) ---
def fetch_public_ip_utilization_data(
    conn,
    schema_name: str,
    start_date: str,
    end_date: str,
    resource_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch Public IP metrics including AVG, MAX value, and the date of MAX (spike).
    Parameterized resource id and returns at most one row when resource_id provided.
    """
    params = {
        "start_date": start_date,
        "end_date": end_date,
    }

    resource_filter_sql = ""
    resource_filter_dim = ""
    if resource_id:
        params["resource_id"] = resource_id
        resource_filter_sql = "AND LOWER(resource_id) = LOWER(%(resource_id)s)"
        resource_filter_dim = "WHERE LOWER(resource_id) = LOWER(%(resource_id)s)"
    else:
        # Exclude Databricks or other non-standard Public IPs
        resource_filter_dim = "WHERE LOWER(resource_id) NOT LIKE '%%databricks%%'"

    query = f"""
        WITH fact_base AS (
            SELECT
                fpi.public_ip_key,
                LOWER(pip.resource_id) AS resource_id,
                dm.metric_name,
                fpi.date_key,
                fpi.daily_value_avg,
                fpi.daily_value_max
            FROM {schema_name}.fact_public_ip_daily_usage fpi
            JOIN {schema_name}.dim_public_ip pip
                ON fpi.public_ip_key = pip.public_ip_key
            JOIN {schema_name}.dim_metric dm
                ON fpi.metric_key = dm.metric_key
            WHERE fpi.date_key IS NOT NULL
            AND to_date(fpi.date_key::text, 'YYYYMMDD')
                    BETWEEN %(start_date)s::date AND %(end_date)s::date
            {resource_filter_sql}
        ),

        metric_avg_max AS (
            SELECT
                resource_id,
                metric_name,
                -- Convert bytes to GB for ByteCount and DDoS byte metrics
                AVG(
                    CASE
                        WHEN metric_name IN ('ByteCount', 'TCPBytesForwardedDDoS', 'TCPBytesInDDoS', 'UDPBytesForwardedDDoS', 'UDPBytesInDDoS')
                        THEN daily_value_avg / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE daily_value_avg
                    END
                ) AS avg_value,
                MAX(
                    CASE
                        WHEN metric_name IN ('ByteCount', 'TCPBytesForwardedDDoS', 'TCPBytesInDDoS', 'UDPBytesForwardedDDoS', 'UDPBytesInDDoS')
                        THEN daily_value_max / 1073741824.0  -- Convert bytes to GB (1024^3)
                        ELSE daily_value_max
                    END
                ) AS max_value
            FROM fact_base
            GROUP BY resource_id, metric_name
        ),

        metric_max_date AS (
            SELECT DISTINCT ON (resource_id, metric_name)
                resource_id,
                metric_name,
                date_key AS max_date_key
            FROM fact_base
            ORDER BY resource_id, metric_name, daily_value_max DESC, date_key DESC
        ),

        metric_final AS (
            SELECT
                amm.resource_id,
                amm.metric_name,
                amm.avg_value,
                amm.max_value,
                mmd.max_date_key
            FROM metric_avg_max amm
            JOIN metric_max_date mmd 
                ON amm.resource_id = mmd.resource_id
            AND amm.metric_name = mmd.metric_name
        ),

        metric_map AS (
            SELECT
                resource_id,
                (
                    json_object_agg(metric_name || '_Avg', ROUND(avg_value::NUMERIC, 6))::jsonb ||
                    json_object_agg(metric_name || '_Max', ROUND(max_value::NUMERIC, 6))::jsonb ||
                    json_object_agg(metric_name || '_MaxDate',
                        TO_CHAR(to_date(max_date_key::text, 'YYYYMMDD'), 'YYYY-MM-DD'))::jsonb
                )::json AS metrics_json
            FROM metric_final
            GROUP BY resource_id
        ),

        -- cost aggregated by resource_id
        cost_agg AS (
            SELECT
                LOWER(f.resource_id) AS resource_id,
                MAX(f.contracted_unit_price) AS contracted_unit_price,
                SUM(COALESCE(f.pricing_quantity,0)) AS pricing_quantity,
                SUM(COALESCE(f.billed_cost,0)) AS billed_cost,
                SUM(COALESCE(f.consumed_quantity,0)) AS consumed_quantity,
                MAX(COALESCE(f.consumed_unit, '')) AS consumed_unit,
                MAX(COALESCE(f.pricing_unit, '')) AS pricing_unit
            FROM {schema_name}.gold_azure_fact_cost f
            WHERE f.charge_period_start::date BETWEEN %(start_date)s::date AND %(end_date)s::date
            {resource_filter_sql}
            GROUP BY LOWER(f.resource_id)
        ),

        resource_dim AS (
            SELECT
                LOWER(resource_id) AS resource_id,
                public_ip_name,
                region,
                ip_address,
                ip_version,
                sku,
                tier,
                allocation_method
            FROM {schema_name}.dim_public_ip
            {resource_filter_dim}
        )

        SELECT
            rd.resource_id,
            rd.public_ip_name,
            rd.region,
            rd.ip_address,
            rd.ip_version,
            rd.sku,
            rd.tier,
            rd.allocation_method,

            COALESCE(c.contracted_unit_price, NULL) AS contracted_unit_price,
            COALESCE(c.pricing_quantity, 0) AS pricing_quantity,
            COALESCE(c.billed_cost, 0) AS billed_cost,
            COALESCE(c.consumed_quantity, 0) AS consumed_quantity,
            COALESCE(c.consumed_unit, '') AS consumed_unit,
            COALESCE(c.pricing_unit, '') AS pricing_unit,

            m.metrics_json
        FROM resource_dim rd
        LEFT JOIN metric_map m ON rd.resource_id = m.resource_id
        LEFT JOIN cost_agg c ON rd.resource_id = c.resource_id
        ORDER BY COALESCE(c.billed_cost, 0) DESC;
    """

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"Error executing Public IP utilization query: {e}")
        return pd.DataFrame()

    # If resource_id provided, guarantee at most one row
    if resource_id and not df.empty:
        df = df.head(1).reset_index(drop=True)

    # Expand metrics_json into columns
    if not df.empty and "metrics_json" in df.columns:
        try:
            # Ensure JSON strings become dicts
            def _to_dict(x):
                if x is None:
                    return {}
                if isinstance(x, str):
                    try:
                        return json.loads(x)
                    except Exception:
                        return {}
                if isinstance(x, dict):
                    return x
                return {}

            metrics_series = df["metrics_json"].apply(_to_dict)
            metrics_expanded = pd.json_normalize(metrics_series).add_prefix("metric_")
            metrics_expanded.index = df.index
            df = pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)
        except Exception as ex:
            print(f"Warning: failed to expand public IP metrics_json: {ex}")

    return df


@connection
def run_llm_public_ip(conn, schema_name, start_date=None, end_date=None, resource_id=None, task_id=None) -> Optional[Dict[str, Any]]:
    """
    Run LLM analysis for a single Public IP and return a single recommendation dict (or None).

    Args:
        task_id: Optional task ID for cancellation support
    """
    from app.core.task_manager import task_manager

    if end_date is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = pd.to_datetime(start_date).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # CRITICAL: Check if task was cancelled before starting
    if task_id:
        is_cancelled = task_manager.is_cancelled(task_id)
        if is_cancelled:
            print(f"🛑 Task {task_id} was cancelled before Public IP LLM could start. Returning None.")
            return None

    print(f"🔎 Running Public IP LLM for {schema_name} from {start_str} to {end_str} "
          f"{'(resource_id filter applied)' if resource_id else ''}")

    df = fetch_public_ip_utilization_data(conn, schema_name, start_str, end_str, resource_id=resource_id)
    if df is None or df.empty:
        print("⚠️ No Public IP data found for the requested date range / resource.")
        return None

    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days or 1

    if resource_id and df.shape[0] > 1:
        print(f"⚠️ WARNING: Resource ID was provided, but {df.shape[0]} records were fetched. Restricting to the first record for LLM analysis.")

    resource_row = df.head(1).to_dict(orient="records")[0]

    # Add schema_name and region for pricing lookups
    resource_row['schema_name'] = schema_name
    resource_row['region'] = resource_row.get('location', 'eastus')

    # Check cancellation again before calling expensive LLM
    if task_id:
        is_cancelled = task_manager.is_cancelled(task_id)
        if is_cancelled:
            print(f"🛑 Task {task_id} was cancelled before LLM call. Returning None.")
            return None

    # Call the imported LLM analysis function
    recommendation = get_public_ip_recommendation_single(resource_row)

    if recommendation:
        print("✅ LLM analysis complete! Returning recommendation.")
        return recommendation
    else:
        print("⚠️ No recommendation generated by LLM.")
        return None


@connection
def run_llm_public_ip_all_resources(conn, schema_name, start_date=None, end_date=None, task_id=None) -> List[Dict[str, Any]]:
    """
    NEW FUNCTION: Fetch ALL distinct Public IPs and process each through LLM individually.

    Args:
        task_id: Optional task ID for cancellation support

    Returns:
        List of recommendation dictionaries, one per Public IP
    """
    from app.core.task_manager import task_manager
    if end_date is None:
        end_dt = datetime.utcnow().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = pd.to_datetime(start_date).date()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print(f"🔎 Running Public IP LLM for ALL distinct Public IPs in {schema_name} from {start_str} to {end_str}")

    # Fetch data for ALL Public IPs (no resource_id filter)
    df = fetch_public_ip_utilization_data(conn, schema_name, start_str, end_str, resource_id=None)

    if df is None or df.empty:
        print("⚠️ No Public IP data found for the requested date range.")
        return []

    # Annotate with date info
    df["start_date"] = start_str
    df["end_date"] = end_str
    df["duration_days"] = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days or 1

    recommendations = []
    total_resources = len(df)
    print(f"📊 Found {total_resources} distinct Public IP resources to analyze")

    # Process each row (resource) individually through LLM
    for idx, row in df.iterrows():
        # Check if task has been cancelled
        if task_id:
            is_cancelled = task_manager.is_cancelled(task_id)
            if is_cancelled:
                print(f"🛑 Task {task_id} was cancelled. Stopping Public IP analysis. Processed {len(recommendations)}/{total_resources}")
                break
            # Debug: Confirm task is still active
            if idx == 0 or idx % 10 == 0:  # Log every 10th iteration
                print(f"  🔍 Task {task_id[:8]}... still running, not cancelled (iteration {idx + 1})")

        resource_id = row.get('resource_id', 'Unknown')
        print(f"  [{idx + 1}/{total_resources}] Processing Public IP: {resource_id}")

        try:
            resource_dict = row.to_dict()
            # Add schema_name and region for pricing lookups
            resource_dict['schema_name'] = schema_name
            resource_dict['region'] = resource_dict.get('location', 'eastus')
            recommendation = get_public_ip_recommendation_single(resource_dict)

            if recommendation:
                recommendations.append(recommendation)
                print(f"    ✅ LLM analysis complete for {resource_id}")
            else:
                print(f"    ⚠️ No recommendation generated for {resource_id}")
        except Exception as e:
            print(f"    ❌ Error processing {resource_id}: {e}")
            continue

    print(f"✅ Completed processing {len(recommendations)}/{total_resources} Public IPs successfully")
    return recommendations
def run_llm_analysis(resource_type, schema_name, start_date=None, end_date=None, resource_id=None, task_id=None):
    """
    Unified entry point for running LLM cost optimization analyses.

    NEW BEHAVIOR:
    - If resource_id is provided: Returns a single recommendation dict for that resource
    - If resource_id is None: Fetches ALL distinct resources and returns a list of recommendations

    Args:
        task_id: Optional task ID for cancellation support

    Returns:
        - Single dict if resource_id is provided
        - List of dicts if resource_id is None (multiple resources analyzed)
    """
    # Input normalization
    rtype = resource_type.strip().lower()
    start_date = start_date or (datetime.utcnow().date().replace(day=1).strftime("%Y-%m-%d"))
    end_date = end_date or datetime.utcnow().date().strftime("%Y-%m-%d")

    if rtype in ["vm", "virtualmachine", "virtual_machine"]:
        # If resource_id is provided, return single result
        if resource_id:
            final_response = run_llm_vm(schema_name, start_date=start_date, end_date=end_date, resource_id=resource_id, task_id=task_id)
            print(f'Final response (single VM): {final_response}')
            return final_response
        else:
            # Fetch all distinct VMs and process each
            final_response = run_llm_vm_all_resources(schema_name, start_date=start_date, end_date=end_date, task_id=task_id)
            print(f'Final response (all VMs): {len(final_response)} resources processed')
            return final_response

    elif rtype in ["storage", "storageaccount", "storage_account"]:
        # If resource_id is provided, return single result
        if resource_id:
            final_response = run_llm_storage(schema_name, start_date=start_date, end_date=end_date, resource_id=resource_id, task_id=task_id)
            print(f'Final response (single Storage): {final_response}')
            return final_response
        else:
            # Fetch all distinct storage accounts and process each
            final_response = run_llm_storage_all_resources(schema_name, start_date=start_date, end_date=end_date, task_id=task_id)
            print(f'Final response (all Storage): {len(final_response)} resources processed')
            return final_response

    elif rtype in ["publicip", "public_ip", "pip"]:
        # If resource_id is provided, return single result
        if resource_id:
            final_response = run_llm_public_ip(schema_name, start_date=start_date, end_date=end_date, resource_id=resource_id, task_id=task_id)
            print(f'Final response (single Public IP): {final_response}')
            return final_response
        else:
            # Fetch all distinct Public IPs and process each
            final_response = run_llm_public_ip_all_resources(schema_name, start_date=start_date, end_date=end_date, task_id=task_id)
            print(f'Final response (all Public IPs): {len(final_response)} resources processed')
            return final_response

    else:
        raise ValueError(f"Unsupported resource_type: {resource_type}")

# The original get_storage_recommendation and get_compute_recommendation wrappers 
# are moved to llm_analysis.py, as they wrap the single-resource LLM call.