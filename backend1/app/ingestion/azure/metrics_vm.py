import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os 
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.ingestion.azure.postgres_operation import dump_to_postgresql

AGGREGATION_METHODS = {
    "Percentage CPU": "Average",
    "CPU Percentage": "Average",
    "Network In Total": "Total",
    "Network Out Total": "Total",
    # Add more mappings...
}
# Map instance types to vCPU core counts
VCPU_MAP = {
    "Standard_DS3_v2": 4,
    "Standard_DS4_v2": 8,
    "Standard_D2s_v3": 2,
    "Standard_B2s": 2,
    "Standard_B1s": 1,
    # ➕ Add more instance types here
}

# ------------------ CONFIG ------------------ #
interval = "P1D"
days_back = 90




def create_hash_key(df, columns):
    # Generate hash key using MD5 for better collision resistance
    import hashlib
    def generate_hash(row):
        row_string = ''.join(str(row[col]) for col in columns)
        return int(hashlib.md5(row_string.encode('utf-8')).hexdigest(), 16) % (10**18)
    df['hash_key'] = df.apply(generate_hash, axis=1)
    return df

def fetch_existing_hash_keys(schema_name, table_name):
    # Fetch existing hash keys from the database for deduplication
    from app.ingestion.azure.postgres_operation import fetch_existing_hash_keys as fetch_keys
    return fetch_keys(schema_name, table_name)

def get_available_metrics(vm_id, headers):
    url = (
        f"https://management.azure.com{vm_id}/providers/microsoft.insights/metricDefinitions"
        f"?api-version=2023-10-01"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Failed to fetch available metrics for {vm_id}: {response.status_code}")
        return []

    definitions = response.json().get("value", [])
    return [metric["name"]["value"] for metric in definitions if "name" in metric]

def get_access_token(tenant_id, client_id, client_secret):
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://management.azure.com/.default",
        "grant_type": "client_credentials"
    }
    response = requests.post(token_url, data=token_data)
    access_token = response.json().get("access_token")
    if not access_token:
        raise Exception("❌ Failed to retrieve access token")
    return access_token

def list_vms(subscription_id, headers):
    url = f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.Compute/virtualMachines?api-version=2023-07-01"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"❌ Failed to list VMs: {response.status_code}")
    return response.json().get("value", [])

def fetch_vm_metrics(vm, headers, timespan, interval, metric_name, aggregation):
    vm_id = vm["id"]
    metrics_url = (
        f"https://management.azure.com{vm_id}/providers/microsoft.insights/metrics"
        f"?api-version=2023-10-01"
        f"&metricnames={metric_name.replace(' ', '%20')}"
        f"&timespan={timespan}"
        f"&interval={interval}"
        f"&aggregation={aggregation}" # ✅ FIXED: Use the correct aggregation dynamically
    )
    return requests.get(metrics_url, headers=headers)

def collect_all_vm_metrics(vms, headers, timespan, interval, metric_names, subscription_id):
    rows = []
    for vm in vms:
        vm_name = vm["name"]
        vm_id = vm["id"]
        resource_group = vm_id.split("/")[4]
        instance_type = vm.get("properties", {}).get("hardwareProfile", {}).get("vmSize", "unknown")

        for metric_name in metric_names:
            aggregation = AGGREGATION_METHODS.get(metric_name, "Average")
            response = fetch_vm_metrics(vm, headers, timespan, interval, metric_name, aggregation) # Pass aggregation
            time.sleep(0.5)
            if response.status_code != 200:
                print(f"❌ Failed for VM '{vm_name}' on metric '{metric_name}': {response.status_code}")
                continue

            data = response.json()
            namespace = data.get("namespace", "")
            resourceregion = data.get("resourceregion", "")

            for metric in data.get("value", []):
                full_id = metric.get("id", "")
                resource_id = full_id.split("/providers/Microsoft.Insights/metrics")[0] if "/providers/Microsoft.Insights/metrics" in full_id else full_id
                metric_unit = metric.get("unit", "")
                metric_name_actual = metric["name"]["value"]
                display_desc = metric.get("displayDescription", "")

                for series in metric.get("timeseries", []):
                    for point in series.get("data", []):
                        
                        # ✅ FIXED: Extract value based on the correct aggregation type
                        if aggregation == "Total":
                            raw_value = point.get("total", 0.0)
                        else:
                            # Default to Average
                            raw_value = point.get("average", 0.0)
                            
                        # Normalize only if the metric is 'Percentage CPU'
                        if metric_name_actual.lower() in ["percentage cpu", "cpu percentage"]:
                            # Convert to 0-100% scale (same as Azure UI)
                            value = min(raw_value, 100)  # Cap at 100%
                        else:
                            value = raw_value

                        row = {
                            "vm_name": vm_name,
                            "resource_group": resource_group,
                            "subscription_id": subscription_id,
                            "timestamp": point.get("timeStamp"),
                            "value": value,
                            "metric_name": metric_name_actual,
                            "unit": metric_unit,
                            "displaydescription": display_desc,
                            "namespace": namespace,
                            "resourceregion": resourceregion,
                            "resource_id": resource_id,
                            "instance_type": instance_type,
                            "cost": "",  # To be filled from separate billing export later
                        }
                        rows.append(row)
    return pd.DataFrame(rows)

def metrics_dump(tenant_id, client_id, client_secret,subscription_id,schema_name,table_name):
    access_token = get_access_token(tenant_id, client_id, client_secret)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        vms = list_vms(subscription_id, headers)
    except Exception as e:
        print(f"❌ Failed to list VMs. Halting ingestion. Error: {e}")
        return
    
    print(f"📦 Found {len(vms)} VM(s)\n")

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    timespan = f"{start_time.isoformat()}Z/{end_time.isoformat()}Z"

    all_metrics_df = pd.DataFrame()

    for vm in vms:
        vm_name = vm["name"]
        # ✅ ADDED: Robust try/except block for account-level failures 
        try:
            vm_id = vm["id"]
            available_metrics = get_available_metrics(vm_id, headers)
            
            if not available_metrics:
                print(f"⚠️ No metrics available for VM: {vm['name']}")
                continue

            print(f"Processing VM: {vm_name}")
            vm_metrics_df = collect_all_vm_metrics(
                vms=[vm],  # Send one VM at a time
                headers=headers,
                timespan=timespan,
                interval=interval,
                metric_names=available_metrics,
                subscription_id=subscription_id
            )
            
            all_metrics_df = pd.concat([all_metrics_df, vm_metrics_df], ignore_index=True)
            print(f"✅ Successfully processed metrics for VM: {vm_name}")

        except requests.exceptions.RequestException as e:
            # Catch network errors (ConnectionError, Timeout, etc.)
            print(f"❌ FATAL REQUEST ERROR for VM '{vm_name}'. Skipping to next VM. Error: {e}")
            continue 
        
        except Exception as e:
            # Catch any other unexpected error 
            print(f"❌ UNEXPECTED ERROR during processing of VM '{vm_name}'. Skipping. Error: {e}")
            continue

    print(f"\n📊 Total records collected: {len(all_metrics_df)}")
    print(f"\n📊 Total records collected: {len(all_metrics_df)}")
    if not all_metrics_df.empty:
        # ✅ FIX: Define the stable, unique set of columns for hashing
        # These columns define a unique metric reading at a specific time for a resource.
        key_columns = ["resource_id", "timestamp", "metric_name", "subscription_id"]

        # ✅ FIX: Call the function with the required 'columns' argument
        all_metrics_df = create_hash_key(all_metrics_df, key_columns)

        # Fetch existing hash keys and filter out duplicates
        print(f"🔍 Checking for existing records in {schema_name}.{table_name}...")
        existing_hash_keys = fetch_existing_hash_keys(schema_name, table_name)
        new_metrics_df = all_metrics_df[~all_metrics_df['hash_key'].isin(existing_hash_keys)]

        if new_metrics_df.empty:
            print("⚠️ No new records to insert. All records already exist.")
        else:
            print(f"✅ Inserting {len(new_metrics_df)} new records (filtered {len(all_metrics_df) - len(new_metrics_df)} duplicates)")
            dump_to_postgresql(new_metrics_df, schema_name, table_name)
    else:
        print("⚠️ No data collected to dump.")