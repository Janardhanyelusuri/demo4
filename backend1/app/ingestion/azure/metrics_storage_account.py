import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import psycopg2
from psycopg2.extras import execute_values
from app.ingestion.azure.postgres_operation import  dump_to_postgresql, fetch_existing_hash_keys

# ---------------- Config ----------------
API_VERSION_LIST_STORAGE = "2023-01-01"
API_VERSION_METRIC_DEFS = "2023-10-01"
API_VERSION_METRICS = "2023-10-01"

DESIRED_METRICS = [
    "UsedCapacity",
    "BlobCapacity",
    "FileCapacity",
    "TableCapacity",
    "QueueCapacity",
    "BlobCount",
    "Transactions",
    "Egress",
    "Ingress",
    "SuccessE2ELatency",
    "SuccessServerLatency",
    "Availability",
]

PREFERRED_AGG = {
    "UsedCapacity": "Total",
    "BlobCapacity": "Total",
    "FileCapacity": "Total",
    "TableCapacity": "Total",
    "QueueCapacity": "Total",
    "BlobCount": "Total",
    "Transactions": "Total",
    "Egress": "Total",
    "Ingress": "Total",
    "SuccessE2ELatency": "Average",
    "SuccessServerLatency": "Average",
    "Availability": "Average",
}

SERVICE_RESOURCE_SUFFIX = {
    "BlobCapacity": "/blobServices/default",
    "BlobCount": "/blobServices/default",
    "FileCapacity": "/fileServices/default",
    "QueueCapacity": "/queueServices/default",
    "TableCapacity": "/tableServices/default",
}

INTERVAL = os.getenv("INTERVAL", "PT1H")
DAYS_BACK = int(os.getenv("DAYS_BACK", "90"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.35"))
BATCH_INSERT_SIZE = int(os.getenv("BATCH_INSERT_SIZE", "5000"))

# ---------------- Helpers ----------------
def get_access_token(tenant_id, client_id, client_secret):
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://management.azure.com/.default",
        "grant_type": "client_credentials",
    }
    r = requests.post(token_url, data=data, timeout=30)
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("No access token received from Azure")
    return token

def list_storage_accounts(subscription_id, headers):
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.Storage/storageAccounts"
        f"?api-version={API_VERSION_LIST_STORAGE}"
    )
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])

# metric definitions cache
_METRIC_DEFS_CACHE = {}

def get_metric_definitions(resource_id, headers):
    if resource_id in _METRIC_DEFS_CACHE:
        return _METRIC_DEFS_CACHE[resource_id]
    url = f"https://management.azure.com{resource_id}/providers/microsoft.insights/metricDefinitions?api-version={API_VERSION_METRIC_DEFS}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"⚠️ metricDefinitions failed for {resource_id}: {r.status_code}")
            _METRIC_DEFS_CACHE[resource_id] = []
            return []
        defs = r.json().get("value", [])
        _METRIC_DEFS_CACHE[resource_id] = defs
        return defs
    except Exception as e:
        print(f"⚠️ metricDefinitions exception for {resource_id}: {e}")
        _METRIC_DEFS_CACHE[resource_id] = []
        return []

def inspect_metric_definition(defs, metric_name):
    for m in defs:
        if (m.get("name") or {}).get("value") == metric_name:
            dims = m.get("dimensions", []) or []
            need_service = False
            allowed_vals = []
            for d in dims:
                if d.get("name", "").lower() == "service":
                    need_service = True
                    allowed_vals = d.get("allowedValues", []) or []
            supported_aggs = m.get("supportedAggregationTypes") or []
            return {
                "metric_def": m,
                "needs_service_filter": need_service,
                "allowed_values": allowed_vals,
                "supported_aggs": supported_aggs,
            }
    return None

def pick_aggregation(preferred, supported):
    if not supported:
        return preferred
    if preferred in supported:
        return preferred
    for cand in ["Total", "Average", "Minimum", "Maximum", "Count"]:
        if cand in supported:
            return cand
    return supported[0]

def service_resource_id_for_metric(storage_account_resource_id, metric_name):
    suffix = SERVICE_RESOURCE_SUFFIX.get(metric_name)
    if not suffix:
        return storage_account_resource_id
    if storage_account_resource_id.endswith(suffix):
        return storage_account_resource_id
    return storage_account_resource_id.rstrip("/") + suffix

def fetch_metric_response(resource_to_query, storage_account_name, headers, timespan, interval, metric_name, agg_to_use, needs_service_filter=False, allowed_values=None):
    filter_param = ""
    if needs_service_filter:
        chosen_val = None
        if allowed_values:
            for candidate in allowed_values:
                if candidate and candidate.lower().startswith("blob"):
                    chosen_val = candidate
                    break
            if not chosen_val:
                chosen_val = allowed_values[0]
        else:
            chosen_val = "blobs"
        filter_param = f"&$filter=Service eq '{chosen_val}'"
    metric_q = quote_plus(metric_name)
    metrics_url = (
        f"https://management.azure.com{resource_to_query}/providers/microsoft.insights/metrics"
        f"?api-version={API_VERSION_METRICS}"
        f"&metricnames={metric_q}"
        f"&timespan={timespan}"
        f"&interval={interval}"
        f"&aggregation={agg_to_use}"
        f"{filter_param}"
    )
    try:
        r = requests.get(metrics_url, headers=headers, timeout=30)
    except Exception as e:
        print(f"❌ REQUEST ERROR for {storage_account_name} metric {metric_name}: {e}")
        return None
    if r.status_code == 400:
        # metric not available for this resource -> skip
        print(f"⊘ Metric not available for '{storage_account_name}' -> {metric_name} (agg={agg_to_use}). 400")
        return None
    if r.status_code != 200:
        print(f"⚠️ Unexpected status for '{storage_account_name}' -> {metric_name} (agg={agg_to_use}): {r.status_code}")
        return r
    return r

def get_storage_account_details(resource_id, headers):
    url = f"https://management.azure.com{resource_id}?api-version={API_VERSION_LIST_STORAGE}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return {}
        data = r.json()
        props = data.get("properties", {}) or {}
        sku = data.get("sku", {}) or {}
        sku_name = sku.get("name", "")
        replication_type = sku_name.split("_")[-1] if "_" in sku_name else "unknown"
        return {
            "sku": sku_name or "unknown",
            "access_tier": props.get("accessTier", "unknown"),
            "replication": replication_type,
            "location": data.get("location", "unknown"),
            "kind": data.get("kind", "unknown"),
            "creation_time": props.get("creationTime", ""),
            "status": props.get("statusOfPrimary", "unknown"),
        }
    except Exception as e:
        print(f"⚠️ get_storage_account_details error for {resource_id}: {e}")
        return {}

# ---------- Collection ----------
def collect_all_storage_metrics(storage_accounts, headers, timespan, interval, subscription_id):
    rows = []
    for storage_account in storage_accounts:
        name = storage_account.get("name")
        resource_id = storage_account.get("id")
        print(f"\n📦 Processing storage account: {name}")

        # For each desired metric, look up metricDefinitions on appropriate resource (service-level when applicable)
        per_metric_defs = {}
        available_union = set()
        for metric in DESIRED_METRICS:
            resource_to_check = service_resource_id_for_metric(resource_id, metric)
            defs = get_metric_definitions(resource_to_check, headers)
            available = set(((m.get("name") or {}).get("value")) for m in defs if (m.get("name") or {}).get("value"))
            per_metric_defs[metric] = {"resource": resource_to_check, "defs": defs, "available": available}
            available_union.update(available)

        metrics_to_query = [m for m in DESIRED_METRICS if m in available_union]
        print(f"   ℹ️ Metrics to query for this account: {metrics_to_query}")

        details = get_storage_account_details(resource_id, headers)

        for metric_name in metrics_to_query:
            meta_entry = per_metric_defs.get(metric_name)
            if meta_entry is None:
                resource_to_query = service_resource_id_for_metric(resource_id, metric_name)
                defs = get_metric_definitions(resource_to_query, headers)
                meta = inspect_metric_definition(defs, metric_name)
            else:
                resource_to_query = meta_entry["resource"]
                defs = meta_entry["defs"]
                meta = inspect_metric_definition(defs, metric_name)

            if meta is None:
                preferred_agg = PREFERRED_AGG.get(metric_name, "Average")
                agg_to_use = preferred_agg
                needs_service_filter = False
                allowed_values = []
            else:
                supported = meta.get("supported_aggs") or []
                preferred_agg = PREFERRED_AGG.get(metric_name, "Average")
                agg_to_use = pick_aggregation(preferred_agg, supported)
                needs_service_filter = meta.get("needs_service_filter", False)
                allowed_values = meta.get("allowed_values", [])

            resp = fetch_metric_response(resource_to_query, name, headers, timespan, interval, metric_name, agg_to_use, needs_service_filter, allowed_values)
            time.sleep(SLEEP_BETWEEN_CALLS)
            if resp is None:
                continue
            if resp.status_code != 200:
                continue

            payload = resp.json()
            namespace = payload.get("namespace", "")
            resourceregion = payload.get("resourceregion", "")

            for metric in payload.get("value", []):
                full_id = metric.get("id", "")
                resource_id_clean = (full_id.split("/providers/Microsoft.Insights/metrics")[0] if "/providers/Microsoft.Insights/metrics" in full_id else full_id)
                metric_unit = metric.get("unit", "")
                metric_name_actual = (metric.get("name") or {}).get("value", metric_name)
                display_desc = metric.get("displayDescription", "")

                for series in metric.get("timeseries", []):
                    for point in series.get("data", []):
                        if "total" in point:
                            value = point.get("total")
                        elif "average" in point:
                            value = point.get("average")
                        elif "count" in point:
                            value = point.get("count")
                        else:
                            value = None
                            for k in ["maximum", "minimum", "sum"]:
                                if k in point:
                                    value = point.get(k)
                                    break
                            if value is None:
                                value = 0.0

                        row = {
                            "storage_account_name": name,
                            "resource_group": resource_id.split("/")[4] if resource_id and "/" in resource_id else "unknown",
                            "subscription_id": subscription_id,
                            "timestamp": point.get("timeStamp"),
                            "value": value,
                            "metric_name": metric_name_actual,
                            "unit": metric_unit,
                            "displaydescription": display_desc,
                            "namespace": namespace,
                            "resourceregion": resourceregion,
                            "resource_id": resource_id_clean,
                            "sku": details.get("sku", "unknown"),
                            "access_tier": details.get("access_tier", "unknown"),
                            "replication": details.get("replication", "unknown"),
                            "location": details.get("location", "unknown"),
                            "kind": details.get("kind", "unknown"),
                            "storage_account_status": details.get("status", "unknown"),
                            "cost": None,
                        }
                        rows.append(row)
    return pd.DataFrame(rows)

# ---------- Postgres helpers ----------

def fetch_existing_hash_keys(schema_name, table_name):
    """Return set of existing hash_key values from target table."""
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(f"SELECT hash_key FROM {schema_name}.{table_name}")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return set(r[0] for r in rows if r and r[0] is not None)
    except Exception as e:
        print(f"⚠️ fetch_existing_hash_keys failed: {e} -> returning empty set")
        return set()


# ---------- Main ----------
def metrics_dump(tenant_id, client_id, client_secret, subscription_id, schema_name, table_name):
    print("🔄 Starting Storage Account metrics dump...")
    try:
        token = get_access_token(tenant_id, client_id, client_secret)
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_BACK)
    timespan = f"{start_time.isoformat()}Z/{end_time.isoformat()}Z"
    print(f"📅 Time range: {start_time.date()} to {end_time.date()} ({DAYS_BACK} days)")

    try:
        storage_accounts = list_storage_accounts(subscription_id, headers)
        print(f"📦 Found {len(storage_accounts)} storage account(s)")
    except Exception as e:
        print(f"❌ Failed to list storage accounts: {e}")
        return

    if not storage_accounts:
        print("No storage accounts found. Exiting.")
        return

    df = collect_all_storage_metrics(storage_accounts, headers, timespan, INTERVAL, subscription_id)
    if df.empty:
        print("No metrics collected. Exiting.")
        return

    print("============================================================")
    print(f"📊 Total records collected: {len(df)}")

    # create hash_key
    df = df.copy()
    df['hash_key'] = df[['storage_account_name', 'timestamp', 'metric_name', 'resource_id', 'subscription_id']].astype(str).sum(axis=1).apply(lambda x: hash(x) % (10**12))

    existing = fetch_existing_hash_keys(schema_name, table_name)
    new_df = df[~df['hash_key'].isin(existing)].copy()
    print(f"🔎 New unique records to insert (after dedupe): {len(new_df)}")

    if new_df.empty:
        print("No new unique records to insert.")
        return

    # ensure columns order
    column_order = [
        "storage_account_name", "resource_group", "subscription_id", "timestamp", "value",
        "metric_name", "unit", "displaydescription", "namespace", "resourceregion",
        "resource_id", "sku", "access_tier", "replication", "kind",
        "storage_account_status", "cost", "hash_key"
    ]
    for c in column_order:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[column_order]

    try:
        dump_to_postgresql(new_df, schema_name, table_name)
        print(f"✅ Storage account metrics dumped successfully to {schema_name}.{table_name}!")
    except Exception as e:
        print(f"❌ Failed to dump to PostgreSQL: {e}")
