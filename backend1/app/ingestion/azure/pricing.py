"""
Azure Pricing Module

Fetches and stores Azure SKU pricing information including:
- VM instance types (sizes, vCPUs, memory, pricing)
- Storage account pricing (per GB/month, transaction costs)
- Public IP pricing
- Managed Disk pricing

Uses Azure Retail Prices API for real-time pricing data.
"""

import requests
import pandas as pd
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ingestion.azure.postgres_operation import connection, dump_to_postgresql


# Azure Retail Prices API endpoint
AZURE_PRICING_API = "https://prices.azure.com/api/retail/prices"


def convert_to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def fetch_azure_vm_pricing(region: str = "eastus", currency: str = "USD") -> pd.DataFrame:
    """
    Fetch Azure VM pricing from Retail Prices API.

    Args:
        region: Azure region (e.g., 'eastus', 'westus2')
        currency: Currency code (default: USD)

    Returns:
        DataFrame with VM pricing information
    """
    print(f"📊 Fetching Azure VM pricing for region: {region}")

    # Filter for Virtual Machines, Linux OS, pay-as-you-go
    filter_query = (
        f"serviceName eq 'Virtual Machines' "
        f"and armRegionName eq '{region}' "
        f"and priceType eq 'Consumption' "
        f"and currencyCode eq '{currency}'"
    )

    all_items = []
    url = f"{AZURE_PRICING_API}?$filter={filter_query}"

    while url:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get('Items', [])
            all_items.extend(items)

            # Get next page URL
            url = data.get('NextPageLink')

            print(f"  Fetched {len(items)} VM pricing records (total: {len(all_items)})")

            # Limit to prevent excessive API calls
            if len(all_items) >= 5000:
                print(f"  ⚠️ Reached limit of 5000 records, stopping pagination")
                break

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching VM pricing: {e}")
            break

    if not all_items:
        print(f"  ⚠️ No VM pricing data found for region {region}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # Extract relevant columns
    columns_to_keep = [
        'skuName', 'productName', 'armSkuName', 'armRegionName',
        'retailPrice', 'unitPrice', 'currencyCode', 'unitOfMeasure',
        'meterName', 'type', 'isPrimaryMeterRegion', 'effectiveStartDate'
    ]

    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Add metadata
    df['last_updated'] = datetime.utcnow()
    df['pricing_tier'] = 'consumption'

    # Convert column names from camelCase to snake_case for PostgreSQL
    df.columns = [convert_to_snake_case(col) for col in df.columns]

    print(f"✅ Successfully fetched {len(df)} VM pricing records")
    return df


def fetch_azure_storage_pricing(region: str = "eastus", currency: str = "USD") -> pd.DataFrame:
    """
    Fetch Azure Storage pricing from Retail Prices API.

    Args:
        region: Azure region
        currency: Currency code

    Returns:
        DataFrame with storage pricing information
    """
    print(f"📊 Fetching Azure Storage pricing for region: {region}")

    # Filter for Storage accounts
    filter_query = (
        f"serviceName eq 'Storage' "
        f"and armRegionName eq '{region}' "
        f"and priceType eq 'Consumption' "
        f"and currencyCode eq '{currency}'"
    )

    all_items = []
    url = f"{AZURE_PRICING_API}?$filter={filter_query}"

    while url:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get('Items', [])
            all_items.extend(items)

            url = data.get('NextPageLink')

            print(f"  Fetched {len(items)} storage pricing records (total: {len(all_items)})")

            if len(all_items) >= 3000:
                print(f"  ⚠️ Reached limit of 3000 records, stopping pagination")
                break

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching storage pricing: {e}")
            break

    if not all_items:
        print(f"  ⚠️ No storage pricing data found for region {region}")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # Extract relevant columns
    columns_to_keep = [
        'skuName', 'productName', 'armRegionName',
        'retailPrice', 'unitPrice', 'currencyCode', 'unitOfMeasure',
        'meterName', 'type', 'effectiveStartDate'
    ]

    df = df[[col for col in columns_to_keep if col in df.columns]]
    df['last_updated'] = datetime.utcnow()

    # Convert column names from camelCase to snake_case for PostgreSQL
    df.columns = [convert_to_snake_case(col) for col in df.columns]

    print(f"✅ Successfully fetched {len(df)} storage pricing records")
    return df


def fetch_azure_disk_pricing(region: str = "eastus", currency: str = "USD") -> pd.DataFrame:
    """
    Fetch Azure Managed Disk pricing from Retail Prices API.

    Args:
        region: Azure region
        currency: Currency code

    Returns:
        DataFrame with disk pricing information
    """
    print(f"📊 Fetching Azure Managed Disk pricing for region: {region}")

    # Filter for Managed Disks
    # Note: Using 'Disk' instead of 'Managed Disks' for broader match
    filter_query = (
        f"serviceName eq 'Storage' "
        f"and armRegionName eq '{region}' "
        f"and priceType eq 'Consumption' "
        f"and currencyCode eq '{currency}'"
    )

    all_items = []
    url = f"{AZURE_PRICING_API}?$filter={filter_query}"

    while url:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get('Items', [])
            all_items.extend(items)

            url = data.get('NextPageLink')

            print(f"  Fetched {len(items)} disk pricing records (total: {len(all_items)})")

            if len(all_items) >= 2000:
                break

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching disk pricing: {e}")
            break

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # Filter for Managed Disk products only
    if 'productName' in df.columns:
        df = df[df['productName'].str.contains('Disk', case=False, na=False)]

    columns_to_keep = [
        'skuName', 'productName', 'armRegionName',
        'retailPrice', 'unitPrice', 'currencyCode', 'unitOfMeasure',
        'meterName', 'effectiveStartDate'
    ]

    df = df[[col for col in columns_to_keep if col in df.columns]]
    df['last_updated'] = datetime.utcnow()

    # Convert column names from camelCase to snake_case for PostgreSQL
    df.columns = [convert_to_snake_case(col) for col in df.columns]

    print(f"✅ Successfully fetched {len(df)} disk pricing records")
    return df


def fetch_azure_ip_pricing(region: str = "eastus", currency: str = "USD") -> pd.DataFrame:
    """
    Fetch Azure Public IP pricing from Retail Prices API.

    Args:
        region: Azure region
        currency: Currency code

    Returns:
        DataFrame with public IP pricing information
    """
    print(f"📊 Fetching Azure Public IP pricing for region: {region}")

    # Filter for Public IP addresses
    # Using broader filter and will filter productName after fetching
    filter_query = (
        f"serviceName eq 'Virtual Network' "
        f"and armRegionName eq '{region}' "
        f"and priceType eq 'Consumption' "
        f"and currencyCode eq '{currency}'"
    )

    all_items = []
    url = f"{AZURE_PRICING_API}?$filter={filter_query}"

    while url:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get('Items', [])
            all_items.extend(items)

            url = data.get('NextPageLink')

            if len(all_items) >= 500:
                break

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching IP pricing: {e}")
            break

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # Filter for Public IP products only
    if 'productName' in df.columns:
        df = df[df['productName'].str.contains('IP', case=False, na=False)]

    columns_to_keep = [
        'skuName', 'productName', 'armRegionName',
        'retailPrice', 'unitPrice', 'currencyCode', 'unitOfMeasure',
        'meterName', 'effectiveStartDate'
    ]

    df = df[[col for col in columns_to_keep if col in df.columns]]
    df['last_updated'] = datetime.utcnow()

    # Convert column names from camelCase to snake_case for PostgreSQL
    df.columns = [convert_to_snake_case(col) for col in df.columns]

    print(f"✅ Successfully fetched {len(df)} IP pricing records")
    return df


@connection
def store_azure_pricing(conn, schema_name: str, pricing_df: pd.DataFrame, pricing_type: str):
    """
    Store Azure pricing data in PostgreSQL.

    Args:
        conn: Database connection
        schema_name: Schema name
        pricing_df: DataFrame with pricing data
        pricing_type: Type of pricing (vm, storage, disk, ip)
    """
    if pricing_df.empty:
        print(f"  ⚠️ No {pricing_type} pricing data to store")
        return

    table_name = f"azure_pricing_{pricing_type}"

    # Truncate and reload (pricing data should be refreshed completely)
    truncate_query = f"TRUNCATE TABLE {schema_name}.{table_name}"

    try:
        cursor = conn.cursor()
        cursor.execute(truncate_query)
        conn.commit()
        print(f"  🗑️ Cleared existing {pricing_type} pricing data")
    except Exception as e:
        print(f"  ⚠️ Table {table_name} may not exist yet: {e}")
        conn.rollback()

    # Insert new data
    dump_to_postgresql(pricing_df, schema_name, table_name)
    print(f"  💾 Stored {len(pricing_df)} {pricing_type} pricing records in {schema_name}.{table_name}")


def fetch_and_store_all_azure_pricing(schema_name: str, region: str = "eastus", currency: str = "USD"):
    """
    Main function to fetch and store all Azure pricing data.

    Args:
        schema_name: PostgreSQL schema name
        region: Azure region
        currency: Currency code
    """
    print(f"\n{'='*70}")
    print(f"💰 FETCHING AZURE PRICING DATA")
    print(f"   Schema: {schema_name}")
    print(f"   Region: {region}")
    print(f"   Currency: {currency}")
    print(f"{'='*70}\n")

    try:
        # Fetch VM pricing
        vm_pricing = fetch_azure_vm_pricing(region, currency)
        if not vm_pricing.empty:
            store_azure_pricing(schema_name, vm_pricing, "vm")

        # Fetch Storage pricing
        storage_pricing = fetch_azure_storage_pricing(region, currency)
        if not storage_pricing.empty:
            store_azure_pricing(schema_name, storage_pricing, "storage")

        # Fetch Disk pricing
        disk_pricing = fetch_azure_disk_pricing(region, currency)
        if not disk_pricing.empty:
            store_azure_pricing(schema_name, disk_pricing, "disk")

        # Fetch IP pricing
        ip_pricing = fetch_azure_ip_pricing(region, currency)
        if not ip_pricing.empty:
            store_azure_pricing(schema_name, ip_pricing, "ip")

        print(f"\n{'='*70}")
        print(f"✅ AZURE PRICING DATA FETCH COMPLETE")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Error fetching Azure pricing: {e}")
        import traceback
        traceback.print_exc()


def get_vm_sku_pricing(schema_name: str, sku_name: str, region: str = "eastus") -> Optional[Dict]:
    """
    Get pricing for a specific VM SKU.

    Args:
        schema_name: Schema name
        sku_name: VM SKU name (e.g., 'Standard_D4s_v3')
        region: Azure region

    Returns:
        Dictionary with pricing information or None
    """
    from app.ingestion.azure.postgres_operation import connection

    @connection
    def _get_pricing(conn, schema, sku, rgn):
        query = f"""
            SELECT
                sku_name,
                product_name,
                retail_price,
                unit_price,
                currency_code,
                unit_of_measure,
                meter_name
            FROM {schema}.azure_pricing_vm
            WHERE LOWER(sku_name) = LOWER(%s)
              AND LOWER(arm_region_name) = LOWER(%s)
            LIMIT 1
        """

        cursor = conn.cursor()
        cursor.execute(query, (sku, rgn))
        result = cursor.fetchone()

        if result:
            return {
                'sku_name': result[0],
                'product_name': result[1],
                'retail_price': result[2],
                'unit_price': result[3],
                'currency_code': result[4],
                'unit_of_measure': result[5],
                'meter_name': result[6]
            }
        return None

    return _get_pricing(schema_name, sku_name, region)
