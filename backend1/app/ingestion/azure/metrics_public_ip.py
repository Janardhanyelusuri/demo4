import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from app.ingestion.azure.postgres_operation import dump_to_postgresql, fetch_existing_hash_keys

# ---------------- Config ----------------
API_VERSION_PUBLIC_IP = "2023-05-01"
API_VERSION_METRICS = "2023-10-01"
API_VERSION_METRIC_DEFS = "2023-10-01"

# Default preferred aggregations for known metrics
# This is used as a fallback if metric definitions don't specify aggregations
DEFAULT_PREFERRED_AGG = {
    "PacketCount": "Total",
    "ByteCount": "Total",
    "VipAvailability": "Average",
    "DDoSTriggered": "Total",
    "SynCount": "Total",
    "TCPBytesForwardedDDoS": "Total",
    "TCPBytesInDDoS": "Total",
    "UDPBytesForwardedDDoS": "Total",
    "UDPBytesInDDoS": "Total",
}

INTERVAL = os.getenv("INTERVAL", "PT1H")
DAYS_BACK = int(os.getenv("DAYS_BACK", "30")) # Default matching your test script
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.25"))

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

def list_public_ips(subscription_id, headers):
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.Network/publicIPAddresses"
        f"?api-version={API_VERSION_PUBLIC_IP}"
    )
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])

def get_public_ip_details(data):
    """Extracts relevant dimension properties from the raw resource dict"""
    props = data.get("properties", {}) or {}
    sku = data.get("sku", {}) or {}

    return {
        "sku": sku.get("name", "unknown"),
        "tier": sku.get("tier", "unknown"),
        "location": data.get("location", "unknown"),
        "ip_address": props.get("ipAddress", "unknown"),
        "ip_allocation_method": props.get("publicIPAllocationMethod", "unknown"),
        "ip_version": props.get("publicIPAddressVersion", "IPv4"),
        "provisioning_state": props.get("provisioningState", "unknown"),
    }

def get_available_metrics(resource_id, headers):
    """
    Discover all available metrics for a given Public IP resource.
    Returns a list of tuples: (metric_name, supported_aggregations)
    """
    url = (
        f"https://management.azure.com{resource_id}/providers/microsoft.insights/metricDefinitions"
        f"?api-version={API_VERSION_METRIC_DEFS}"
    )
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"⚠️  Failed to fetch metric definitions: {r.status_code}")
            return []

        definitions = r.json().get("value", [])
        metrics_info = []

        for metric_def in definitions:
            metric_name = metric_def.get("name", {}).get("value", "")
            if not metric_name:
                continue

            # Get supported aggregation types
            supported_aggs = metric_def.get("supportedAggregationTypes", [])

            # Pick the best aggregation method
            if "Total" in supported_aggs:
                preferred_agg = "Total"
            elif "Average" in supported_aggs:
                preferred_agg = "Average"
            elif "Maximum" in supported_aggs:
                preferred_agg = "Maximum"
            elif supported_aggs:
                preferred_agg = supported_aggs[0]
            else:
                # Fallback to defaults
                preferred_agg = DEFAULT_PREFERRED_AGG.get(metric_name, "Total")

            metrics_info.append((metric_name, preferred_agg))

        return metrics_info

    except Exception as e:
        print(f"❌ Error fetching metric definitions: {e}")
        return []

def fetch_metric_response(resource_id, metric_name, headers, timespan, interval, agg_to_use):
    metric_q = quote_plus(metric_name)
    metrics_url = (
        f"https://management.azure.com{resource_id}/providers/microsoft.insights/metrics"
        f"?api-version={API_VERSION_METRICS}"
        f"&metricnames={metric_q}"
        f"&timespan={timespan}"
        f"&interval={interval}"
        f"&aggregation={agg_to_use}"
    )
    try:
        r = requests.get(metrics_url, headers=headers, timeout=30)
    except Exception as e:
        print(f"❌ REQUEST ERROR for {resource_id} metric {metric_name}: {e}")
        return None
    
    if r.status_code == 400:
        print(f"⊘ Metric not available -> {metric_name}. 400")
        return None
    if r.status_code != 200:
        print(f"⚠️ Unexpected status -> {metric_name}: {r.status_code}")
        return r
    return r

# ---------- Collection ----------
def collect_all_public_ip_metrics(public_ips, headers, timespan, interval, subscription_id):
    rows = []
    for pip in public_ips:
        name = pip.get("name")
        resource_id = pip.get("id")
        print(f"\n📦 Processing Public IP: {name}")

        details = get_public_ip_details(pip)

        # Discover available metrics for this Public IP
        available_metrics = get_available_metrics(resource_id, headers)
        if not available_metrics:
            print(f"⚠️  No metrics available for {name}, skipping...")
            continue

        print(f"✅ Found {len(available_metrics)} available metric(s)")

        for metric_name, agg_to_use in available_metrics:
            
            resp = fetch_metric_response(resource_id, metric_name, headers, timespan, interval, agg_to_use)
            time.sleep(SLEEP_BETWEEN_CALLS)
            
            if resp is None or resp.status_code != 200:
                continue

            payload = resp.json()
            namespace = payload.get("namespace", "")
            resourceregion = payload.get("resourceregion", "")

            for metric in payload.get("value", []):
                metric_unit = metric.get("unit", "")
                metric_name_actual = (metric.get("name") or {}).get("value", metric_name)
                display_desc = metric.get("displayDescription", "")

                for series in metric.get("timeseries", []):
                    for point in series.get("data", []):
                        # Extract value based on aggregation used
                        if "total" in point: value = point.get("total")
                        elif "average" in point: value = point.get("average")
                        elif "count" in point: value = point.get("count")
                        elif "maximum" in point: value = point.get("maximum")
                        else: value = 0.0

                        if value is None: value = 0.0

                        row = {
                            "public_ip_name": name,
                            "resource_group": resource_id.split("/")[4] if resource_id and "/" in resource_id else "unknown",
                            "subscription_id": subscription_id,
                            "timestamp": point.get("timeStamp"),
                            "value": value,
                            "metric_name": metric_name_actual,
                            "unit": metric_unit,
                            "displaydescription": display_desc,
                            "namespace": namespace,
                            "resourceregion": resourceregion,
                            "resource_id": resource_id,
                            "sku": details.get("sku"),
                            "tier": details.get("tier"),
                            "ip_address": details.get("ip_address"),
                            "ip_version": details.get("ip_version"),
                            "ip_allocation_method": details.get("ip_allocation_method"),
                            "location": details.get("location"),
                            "provisioning_state": details.get("provisioning_state"),
                        }
                        rows.append(row)
    return pd.DataFrame(rows)

# ---------- Main ----------
def metrics_dump(tenant_id, client_id, client_secret, subscription_id, schema_name, table_name):
    print("🔄 Starting Public IP metrics dump...")
    try:
        token = get_access_token(tenant_id, client_id, client_secret)
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        return
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_BACK)
    timespan = f"{start_time.isoformat()}Z/{end_time.isoformat()}Z"
    
    try:
        public_ips = list_public_ips(subscription_id, headers)
        print(f"📦 Found {len(public_ips)} Public IP(s)")
    except Exception as e:
        print(f"❌ Failed to list Public IPs: {e}")
        return

    if not public_ips:
        print("No Public IPs found. Exiting.")
        return

    df = collect_all_public_ip_metrics(public_ips, headers, timespan, INTERVAL, subscription_id)
    if df.empty:
        print("No metrics collected. Exiting.")
        return

    print("============================================================")
    print(f"📊 Total records collected: {len(df)}")

    # create hash_key
    df = df.copy()
    # Hash based on resource + timestamp + metric
    df['hash_key'] = df[['public_ip_name', 'timestamp', 'metric_name', 'resource_id', 'subscription_id']].astype(str).sum(axis=1).apply(lambda x: hash(x) % (10**12))

    existing = fetch_existing_hash_keys(schema_name, table_name)
    new_df = df[~df['hash_key'].isin(existing)].copy()
    print(f"🔎 New unique records to insert: {len(new_df)}")

    if new_df.empty:
        print("No new unique records to insert.")
        return

    # Ensure column order matches Bronze table
    column_order = [
        "public_ip_name", "resource_group", "subscription_id", "timestamp", "value",
        "metric_name", "unit", "displaydescription", "namespace", "resourceregion",
        "resource_id", "sku", "tier", "ip_address", "ip_version", 
        "ip_allocation_method", "location", "provisioning_state", "hash_key"
    ]
    
    for c in column_order:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[column_order]

    try:
        dump_to_postgresql(new_df, schema_name, table_name)
        print(f"✅ Public IP metrics dumped successfully to {schema_name}.{table_name}!")
    except Exception as e:
        print(f"❌ Failed to dump to PostgreSQL: {e}")

if __name__ == "__main__":
    # Example usage for testing standalone
    from dotenv import load_dotenv
    load_dotenv()
    metrics_dump(
        os.getenv("TENANT_ID"),
        os.getenv("CLIENT_ID"),
        os.getenv("CLIENT_SECRET"),
        os.getenv("SUBSCRIPTION_ID"),
        "public", # Example Schema
        "bronze_azure_public_ip_metrics"
    )