# app/ingestion/azure/llm_analysis.py

import json
import logging
from typing import Optional, List, Dict, Any
import sys
import os

# Set up basic logging configuration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming app.core.genai and app.ingestion.azure.llm_json_extractor are available
# Ensure this path manipulation is correct for your environment structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.core.genai import llm_call
from app.ingestion.azure.llm_json_extractor import extract_json_str
# from app.ingestion.azure.llm_json_extractor import extract_json_str # Assuming this is the correct import

# --- Utility Functions ---

def _extrapolate_costs(billed_cost: float, duration_days: int) -> Dict[str, float]:
    """Helper to calculate monthly/annual forecasts."""
    if duration_days == 0:
        return {"monthly": 0.0, "annually": 0.0}
        
    avg_daily_cost = billed_cost / duration_days
    print(f"Avg daily cost calculated: {avg_daily_cost}")
    # Use 30.4375 for average days in a month (365.25 / 12)
    monthly = avg_daily_cost * 30.4375 
    annually = avg_daily_cost * 365 
    print(f"Extrapolated monthly: {monthly}, annually: {annually}")
    return {
        "monthly": round(monthly, 2),
        "annually": round(annually, 2)
    }

def _format_metrics_for_llm(resource_data: dict, resource_type: str = "vm") -> Dict[str, Any]:
    """
    Groups and formats RELEVANT metric data (AVG, MAX, MaxDate) for the LLM.
    Filters to only cost-relevant metrics to reduce token usage.

    Args:
        resource_data: Dict containing metric data with keys like "metric_X_Avg"
        resource_type: "vm" or "storage" to determine which metrics to include
    """
    # Define relevant metrics for each resource type (to reduce token usage)
    RELEVANT_METRICS = {
        "vm": [
            "Percentage CPU",
            "Available Memory Bytes",
            "Available Memory",
            "Disk Read Operations/Sec",
            "Disk Write Operations/Sec",
            "Network In Total",
            "Network Out Total",
            "OS Disk IOPS Consumed Percentage",
            "Data Disk IOPS Consumed Percentage"
        ],
        "storage": [
            "UsedCapacity",
            "Transactions",
            "Ingress",
            "Egress",
            "Availability",
            "SuccessServerLatency",
            "BlobCapacity",
            "FileCapacity"
        ]
    }

    relevant_metric_names = RELEVANT_METRICS.get(resource_type, [])
    formatted_metrics = {}

    # Identify unique metric names (e.g., "Percentage CPU" from "metric_Percentage CPU_Avg")
    unique_metric_names = set(
        k.replace("metric_", "").rsplit('_', 1)[0]
        for k in resource_data.keys()
        if k.startswith("metric_") and len(k.split('_')) > 2
    )

    for metric_name in unique_metric_names:
        # Only include if in relevant list OR if we don't have a filter for this resource type
        if relevant_metric_names and metric_name not in relevant_metric_names:
            continue

        # Reconstruct the full keys
        avg_key = f"metric_{metric_name}_Avg"
        max_key = f"metric_{metric_name}_Max"
        date_key = f"metric_{metric_name}_MaxDate"

        # Build the structured entry for the LLM
        entry = {
            "Avg": resource_data.get(avg_key),
            "Max": resource_data.get(max_key),
            "MaxDate": resource_data.get(date_key)
        }

        # Only include if at least one value is present and not None
        if any(v is not None for v in entry.values()):
            formatted_metrics[metric_name] = entry

    return formatted_metrics

# --- PROMPT GENERATION FUNCTIONS (Updated for dynamic metric inclusion) ---

def _generate_storage_prompt(resource_data: dict, start_date: str, end_date: str, monthly_forecast: float, annual_forecast: float) -> str:
    """Generates the structured prompt for Storage LLM analysis with dynamically included metrics."""

    # Prepare the structured metrics for the prompt (only storage-relevant metrics)
    formatted_metrics = _format_metrics_for_llm(resource_data, resource_type="storage")
    current_sku = resource_data.get("sku", "N/A")
    current_tier = resource_data.get("access_tier", "N/A")
    billed_cost = resource_data.get("billed_cost", 0.0)

    # Use f-string for better readability and variable injection
    return f"""Azure Storage FinOps. Analyze metrics, output JSON only.

CONTEXT: {resource_data.get("resource_id", "N/A")} | {current_sku} {current_tier} | {start_date} to {end_date} ({resource_data.get("duration_days", 30)}d) | Cost: ${billed_cost:.2f} (Est: ${monthly_forecast:.2f}/mo, ${annual_forecast:.2f}/yr)

METRICS:
{json.dumps(formatted_metrics, indent=2)}

RULES:
1. Use exact values+units from METRICS (e.g., "478.3 GB", "1,247 tx/day")
2. State exact SKU/tier names+pricing (e.g., "Hot $0.0184/GB → Cool $0.01/GB")
3. Show calculations (e.g., "478 GB × $0.0084 = $4.02/mo")
4. BANNED: "consider", "review", "optimize", "significant", "could", "should", "it is recommended"
5. Use action verbs: Move, Change, Configure, Enable, Disable, Purchase
6. Every recommendation needs explanation with actual metrics showing WHY
7. Always include units: GB, GB/mo, tx/day, %, $, $/mo, $/yr

DECIDE: Primary optimization? 2-3 additional optimizations? Which metrics drove decisions? 2-3 anomalies (spikes/drops)?

JSON (MUST: 2-3 additional_recommendation, 2-3 anomalies):
{{
  "recommendations": {{
    "effective_recommendation": {{"text": "[action with exact values+savings]", "explanation": "[why with metrics+math]", "saving_pct": #}},
    "additional_recommendation": [
      {{"text": "[action with exact values+savings]", "explanation": "[why with metrics]", "saving_pct": #}},
      {{"text": "[action with exact values+savings]", "explanation": "[why with metrics]", "saving_pct": #}},
      {{"text": "[action with exact values+savings]", "explanation": "[why with metrics]", "saving_pct": #}}
    ],
    "base_of_recommendations": ["[metric: value units]", "[metric: value units]"]
  }},
  "cost_forecasting": {{"monthly": {monthly_forecast:.2f}, "annually": {annual_forecast:.2f}}},
  "anomalies": [
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #, "reason_short": "[why significant]"}},
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #, "reason_short": "[why anomalous]"}},
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #, "reason_short": "[why unusual]"}}
  ],
  "contract_deal": {{"assessment": "good"|"bad"|"unknown", "for sku": "{current_sku} {current_tier}", "reason": "...", "monthly_saving_pct": #, "annual_saving_pct": #}}
}}
"""

def _generate_compute_prompt(resource_data: dict, start_date: str, end_date: str, monthly_forecast: float, annual_forecast: float) -> str:
    """Generates the structured prompt for Compute/VM LLM analysis with dynamically included metrics."""

    # Prepare the structured metrics for the prompt (only VM-relevant metrics)
    formatted_metrics = _format_metrics_for_llm(resource_data, resource_type="vm")
    current_sku = resource_data.get("instance_type", "N/A")
    billed_cost = resource_data.get("billed_cost", 0.0)

    return f"""Azure VM FinOps. Analyze metrics, output JSON only.

CONTEXT: {resource_data.get("resource_id", "N/A")} | SKU: {current_sku} | {start_date} to {end_date} ({resource_data.get("duration_days", 30)}d) | Cost: ${billed_cost:.2f} (Est: ${monthly_forecast:.2f}/mo, ${annual_forecast:.2f}/yr)

METRICS:
{json.dumps(formatted_metrics, indent=2)}

RULES:
1. Use exact values+units from METRICS (e.g., "CPU: 12.3% avg, 35.7% max", "Memory: 8.2 GB")
2. State exact SKU names+specs (e.g., "Standard_D4s_v3 (4 vCPU, 16 GB RAM) → Standard_B2s (2 vCPU, 4 GB RAM)")
3. Show calculations (e.g., "$237.50/mo → $95.20/mo = Save $142.30/mo")
4. BANNED: "consider", "review", "optimize", "significant", "could", "should", "it is recommended", "smaller instance"
5. Use action verbs: Downsize, Upsize, Purchase, Enable, Disable, Change, Configure, Migrate
6. Every recommendation needs explanation with actual metrics showing WHY
7. Always include units: %, GB, vCPU, $, %/mo, %/yr, IOPS, ops/sec

DECIDE: Primary optimization (downsize/upsize/RI/auto-shutdown/disk/other)? VM size change (to what SKU, why)? 2-3 additional optimizations? Which metrics drove decisions? 2-3 anomalies?

JSON (MUST: 2-3 additional_recommendation,  must include units for values of all 2-3 anomalies metrics):
{{
  "recommendations": {{
    "effective_recommendation": {{"text": "[action with exact SKU specs+savings]", "explanation": "[why with metrics+math]", "saving_pct": #}},
    "additional_recommendation": [
      {{"text": "[action with exact details+savings]", "explanation": "[why with metrics]", "saving_pct": #}},
      {{"text": "[action with exact details+savings]", "explanation": "[why with metrics]", "saving_pct": #}},    ],
    "base_of_recommendations": ["[metric: value units]", "[metric: value units]"]
  }},
  "cost_forecasting": {{"monthly": {monthly_forecast:.2f}, "annually": {annual_forecast:.2f}}},
  "anomalies": [
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": # with units, "reason_short": "[why significant]"}},
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": # with units, "reason_short": "[why anomalous]"}},
    {{"metric_name": "[from METRICS]", "timestamp": "[MaxDate]", "value": #with units, "reason_short": "[why unusual]"}}
  ],
  "contract_deal": {{"assessment": "good"|"bad"|"unknown", "for sku": "{current_sku}", "reason": "...", "monthly_saving_pct": #, "annual_saving_pct": #}}
}}
"""

# --- EXPORTED LLM CALL FUNCTIONS (with logging) ---

def get_storage_recommendation_single(resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generates cost recommendations for a single Azure Storage Account.
    """
    if not resource_data:
        logging.warning("Received empty resource_data for storage.")
        return None

    billed_cost = resource_data.get("billed_cost", 0.0)
    duration_days = int(resource_data.get("duration_days", 30) or 30)
    start_date = resource_data.get("start_date", "N/A")
    end_date = resource_data.get("end_date", "N/A")
    resource_id = resource_data.get('resource_id', 'Unknown')
    
    forecast = _extrapolate_costs(billed_cost, duration_days)
    prompt = _generate_storage_prompt(resource_data, start_date, end_date, forecast['monthly'], forecast['annually'])
    
    raw = llm_call(prompt)
    if not raw:
        logging.error(f"Empty LLM response for storage resource {resource_id}")
        return None

    # NOTE: Assuming extract_json_str is available and correctly imported
    json_str = extract_json_str(raw)
    if not json_str:
        logging.error(f"Could not extract JSON from LLM output for storage resource {resource_id}. Raw output:\n{raw[:200]}...")
        return None

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            logging.error(f"LLM storage response parsed to non-dict: {type(parsed)} for {resource_id}")
            return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON (after extraction) for storage resource {resource_id}. Extracted string:\n{json_str[:200]}...")
        return None

    parsed['resource_id'] = resource_id
    parsed['_forecast_monthly'] = forecast['monthly']
    parsed['_forecast_annual'] = forecast['annually']
    return parsed


def get_compute_recommendation_single(resource_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generates cost recommendations for a single VM resource.
    """
    if not resource_data:
        logging.warning("Received empty resource_data for compute.")
        return None

    billed_cost = resource_data.get("billed_cost", 0.0)
    duration_days = int(resource_data.get("duration_days", 30) or 30)
    start_date = resource_data.get("start_date", "N/A")
    end_date = resource_data.get("end_date", "N/A")
    resource_id = resource_data.get('resource_id', 'Unknown')

    forecast = _extrapolate_costs(billed_cost, duration_days)
    prompt = _generate_compute_prompt(resource_data, start_date, end_date, forecast['monthly'], forecast['annually'])
    
    raw = llm_call(prompt)
    if not raw:
        logging.error(f"Empty LLM response for compute resource {resource_id}")
        return None

    # NOTE: Assuming extract_json_str is available and correctly imported
    json_str = extract_json_str(raw)
    if not json_str:
        logging.error(f"Could not extract JSON from LLM output for compute resource {resource_id}. Raw output:\n{raw[:200]}...")
        return None

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            logging.error(f"LLM compute response parsed to non-dict: {type(parsed)} for {resource_id}")
            return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON (after extraction) for compute resource {resource_id}. Extracted string:\n{json_str[:200]}...")
        return None

    parsed['resource_id'] = resource_id
    parsed['_forecast_monthly'] = forecast['monthly']
    parsed['_forecast_annual'] = forecast['annually']
    return parsed


# Backwards-compatible wrappers (process lists but only the first element)
def get_storage_recommendation(data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Wrapper for backward compatibility, processes only the first resource."""
    if not data:
        return None
    # Only process first resource (single-resource flow)
    single = get_storage_recommendation_single(data[0])
    return [single] if single else None

def get_compute_recommendation(data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Wrapper for backward compatibility, processes only the first resource."""
    if not data:
        return None
    single = get_compute_recommendation_single(data[0])
    return [single] if single else None