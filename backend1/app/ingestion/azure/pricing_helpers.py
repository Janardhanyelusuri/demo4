"""
Azure Pricing Query Helpers

Provides functions to query pricing data and generate alternative SKU recommendations
based on resource utilization patterns.
"""

import pandas as pd
import sys
import os
from typing import Dict, List, Optional

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ingestion.azure.postgres_operation import connection


@connection
def get_vm_current_pricing(conn, schema_name: str, sku_name: str, region: str = "eastus") -> Optional[Dict]:
    """
    Get pricing for the current VM SKU.

    Args:
        conn: Database connection
        schema_name: Schema name
        sku_name: VM SKU (e.g., 'Standard_D4s_v3')
        region: Azure region

    Returns:
        Dict with pricing info or None
    """
    if not sku_name:
        return None

    query = f"""
        SELECT
            sku_name,
            product_name,
            retail_price,
            unit_price,
            currency_code,
            unit_of_measure,
            meter_name
        FROM {schema_name}.azure_pricing_vm
        WHERE LOWER(sku_name) = LOWER(%s)
          AND LOWER(arm_region_name) = LOWER(%s)
          AND meter_name LIKE '%%Compute%%'
        ORDER BY retail_price ASC
        LIMIT 1
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query, (sku_name, region))
        result = cursor.fetchone()

        if result:
            return {
                'sku_name': result[0],
                'product_name': result[1],
                'retail_price': float(result[2]) if result[2] else 0.0,
                'unit_price': float(result[3]) if result[3] else 0.0,
                'currency_code': result[4],
                'unit_of_measure': result[5],
                'meter_name': result[6],
                'monthly_cost': float(result[2]) * 730 if result[2] else 0.0  # 730 hours/month
            }
        return None

    except Exception as e:
        print(f"Error fetching VM pricing: {e}")
        return None


@connection
def get_vm_alternative_pricing(conn, schema_name: str, current_sku: str, region: str = "eastus", max_results: int = 10) -> List[Dict]:
    """
    Get alternative VM SKUs with pricing for comparison.

    Args:
        conn: Database connection
        schema_name: Schema name
        current_sku: Current VM SKU
        region: Azure region
        max_results: Maximum number of alternatives

    Returns:
        List of alternative SKU pricing dicts
    """
    # Extract SKU family (e.g., 'D' from 'Standard_D4s_v3')
    sku_family = None
    if current_sku and '_' in current_sku:
        parts = current_sku.split('_')
        if len(parts) >= 2:
            # Extract letter from size (e.g., 'D' from 'D4s')
            size_part = parts[1]
            if size_part:
                sku_family = size_part[0]

    query = f"""
        SELECT DISTINCT
            sku_name,
            product_name,
            retail_price,
            currency_code,
            unit_of_measure,
            meter_name
        FROM {schema_name}.azure_pricing_vm
        WHERE LOWER(arm_region_name) = LOWER(%s)
          AND meter_name LIKE '%%Compute%%'
          AND sku_name != %s
          AND sku_name IS NOT NULL
        ORDER BY retail_price ASC
        LIMIT %s
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query, (region, current_sku, max_results))
        results = cursor.fetchall()

        alternatives = []
        for row in results:
            alternatives.append({
                'sku_name': row[0],
                'product_name': row[1],
                'retail_price': float(row[2]) if row[2] else 0.0,
                'currency_code': row[3],
                'unit_of_measure': row[4],
                'meter_name': row[5],
                'monthly_cost': float(row[2]) * 730 if row[2] else 0.0
            })

        return alternatives

    except Exception as e:
        print(f"Error fetching alternative VM pricing: {e}")
        return []


@connection
def get_storage_pricing_context(conn, schema_name: str, region: str = "eastus") -> Dict:
    """
    Get storage pricing context for different tiers and operations.

    Args:
        conn: Database connection
        schema_name: Schema name
        region: Azure region

    Returns:
        Dict with storage pricing by tier
    """
    query = f"""
        SELECT
            sku_name,
            product_name,
            meter_name,
            retail_price,
            unit_of_measure
        FROM {schema_name}.azure_pricing_storage
        WHERE LOWER(arm_region_name) = LOWER(%s)
          AND (
              meter_name LIKE '%%Data Stored%%'
              OR meter_name LIKE '%%Capacity%%'
          )
        ORDER BY retail_price ASC
        LIMIT 20
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query, (region,))
        results = cursor.fetchall()

        pricing_by_tier = {}
        for row in results:
            tier_key = row[0] or 'Unknown'
            pricing_by_tier[tier_key] = {
                'sku_name': row[0],
                'product_name': row[1],
                'meter_name': row[2],
                'retail_price': float(row[3]) if row[3] else 0.0,
                'unit_of_measure': row[4]
            }

        return pricing_by_tier

    except Exception as e:
        print(f"Error fetching storage pricing: {e}")
        return {}


@connection
def get_public_ip_pricing_context(conn, schema_name: str, region: str = "eastus") -> Dict:
    """
    Get public IP pricing context.

    Args:
        conn: Database connection
        schema_name: Schema name
        region: Azure region

    Returns:
        Dict with IP pricing info
    """
    query = f"""
        SELECT
            sku_name,
            product_name,
            meter_name,
            retail_price,
            unit_of_measure
        FROM {schema_name}.azure_pricing_ip
        WHERE LOWER(arm_region_name) = LOWER(%s)
        ORDER BY retail_price ASC
        LIMIT 5
    """

    try:
        cursor = conn.cursor()
        cursor.execute(query, (region,))
        results = cursor.fetchall()

        pricing_options = []
        for row in results:
            pricing_options.append({
                'sku_name': row[0],
                'product_name': row[1],
                'meter_name': row[2],
                'retail_price': float(row[3]) if row[3] else 0.0,
                'unit_of_measure': row[4],
                'monthly_cost': float(row[3]) * 730 if row[3] else 0.0
            })

        return {
            'options': pricing_options,
            'available_tiers': len(pricing_options)
        }

    except Exception as e:
        print(f"Error fetching IP pricing: {e}")
        return {'options': [], 'available_tiers': 0}


def format_vm_pricing_for_llm(current_pricing: Optional[Dict], alternatives: List[Dict]) -> str:
    """
    Format VM pricing data for LLM context.

    Args:
        current_pricing: Current SKU pricing dict
        alternatives: List of alternative SKU pricing dicts

    Returns:
        Formatted string for LLM prompt
    """
    output = []

    if current_pricing:
        output.append("CURRENT VM PRICING:")
        output.append(f"- SKU: {current_pricing['sku_name']}")
        output.append(f"- Hourly Rate: {current_pricing['retail_price']:.4f} {current_pricing['currency_code']}")
        output.append(f"- Monthly Cost: {current_pricing['monthly_cost']:.2f} {current_pricing['currency_code']}")
        output.append(f"- Unit: {current_pricing['unit_of_measure']}")
    else:
        output.append("CURRENT VM PRICING: Not available in pricing database")

    if alternatives:
        output.append("\nALTERNATIVE VM SKUs (sorted by price):")
        curr_code = current_pricing.get('currency_code', '') if current_pricing else ''
        for i, alt in enumerate(alternatives[:5], 1):
            output.append(f"{i}. {alt['sku_name']}: {alt['retail_price']:.4f}/hr ({alt['monthly_cost']:.2f}/month {curr_code})")

            if current_pricing and current_pricing['monthly_cost'] > 0:
                savings = current_pricing['monthly_cost'] - alt['monthly_cost']
                savings_pct = (savings / current_pricing['monthly_cost']) * 100

                if savings > 0:
                    output.append(f"   💰 Save {savings:.2f}/month ({savings_pct:.1f}% reduction)")
                elif savings < 0:
                    output.append(f"   💸 Cost increase {abs(savings):.2f}/month ({abs(savings_pct):.1f}% more)")
    else:
        output.append("\nALTERNATIVE VM SKUs: Not available in pricing database")

    return "\n".join(output)


def format_storage_pricing_for_llm(storage_pricing: Dict) -> str:
    """
    Format storage pricing data for LLM context.

    Args:
        storage_pricing: Storage pricing dict by tier

    Returns:
        Formatted string for LLM prompt
    """
    if not storage_pricing:
        return "STORAGE PRICING: Not available in pricing database"

    output = ["STORAGE PRICING OPTIONS:"]

    for tier, info in list(storage_pricing.items())[:5]:
        output.append(f"- {info['meter_name']}: {info['retail_price']:.6f} per {info['unit_of_measure']}")

    return "\n".join(output)


def format_ip_pricing_for_llm(ip_pricing: Dict) -> str:
    """
    Format public IP pricing data for LLM context.

    Args:
        ip_pricing: IP pricing dict

    Returns:
        Formatted string for LLM prompt
    """
    options = ip_pricing.get('options', [])

    if not options:
        return "PUBLIC IP PRICING: Not available in pricing database"

    output = ["PUBLIC IP PRICING:"]

    for opt in options:
        output.append(f"- {opt['meter_name']}: {opt['retail_price']:.6f}/hr ({opt['monthly_cost']:.2f}/month)")

    return "\n".join(output)
