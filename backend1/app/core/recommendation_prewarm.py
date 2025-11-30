"""
Recommendation Pre-warming Module

This module provides functions to pre-generate LLM recommendations for all resources
across all standard date ranges, storing them in Redis cache for instant user access.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Dict, Any

# Setup path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.llm_cache_utils import generate_cache_hash_key, save_to_cache, get_cached_result
from app.ingestion.azure.llm_data_fetch import run_llm_analysis
from app.ingestion.aws.llm_s3_integration import run_llm_analysis_s3
from app.ingestion.aws.llm_ec2_integration import run_llm_analysis as run_llm_analysis_ec2


# ============================================================
# DATE RANGE CALCULATION (Matching Frontend Logic)
# ============================================================

def calculate_date_ranges() -> Dict[str, Tuple[datetime, datetime]]:
    """
    Calculate all standard date ranges matching frontend date-fns logic.

    Returns:
        Dictionary mapping preset names to (start_date, end_date) tuples
    """
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    date_ranges = {
        'today': (today, today),
        'yesterday': (today - timedelta(days=1), today - timedelta(days=1)),
        'last_week': (today - timedelta(days=7), today),
        'last_month': (today - relativedelta(months=1), today),
        'last_6_months': (today - relativedelta(months=6), today),
        'last_year': (today - relativedelta(years=1), today),
    }

    return date_ranges


# ============================================================
# AZURE PRE-WARMING
# ============================================================

async def prewarm_azure_recommendations_async(schema_name: str, budget: float = None):
    """
    Pre-generate Azure recommendations for all resources and date ranges.
    Stores results in Redis cache for instant user access.

    Args:
        schema_name: PostgreSQL schema name for the Azure project
        budget: Monthly budget (optional, for display purposes)
    """
    print(f"\n{'='*70}")
    print(f"🔥 STARTING AZURE RECOMMENDATION PRE-WARMING")
    print(f"   Schema: {schema_name}")
    print(f"{'='*70}\n")

    # Resource types to pre-warm
    resource_types = [
        ("vm", "VM"),
        ("storage", "Storage Account"),
        ("publicip", "Public IP")
    ]

    # Get all date ranges
    date_ranges = calculate_date_ranges()

    total_tasks = len(resource_types) * len(date_ranges)
    completed_tasks = 0
    cached_tasks = 0
    generated_tasks = 0

    print(f"📋 Total pre-warming tasks: {total_tasks}")
    print(f"   Resource types: {len(resource_types)}")
    print(f"   Date ranges: {len(date_ranges)}")
    print("")

    # Pre-warm each resource type × date range combination
    for resource_type, display_name in resource_types:
        print(f"\n{'─'*70}")
        print(f"🔧 Processing Resource Type: {display_name} ({resource_type})")
        print(f"{'─'*70}")

        for range_name, (start_date, end_date) in date_ranges.items():
            completed_tasks += 1

            # Generate cache key
            hash_key = generate_cache_hash_key(
                cloud_platform="azure",
                schema_name=schema_name,
                resource_type=resource_type,
                start_date=start_date.date(),
                end_date=end_date.date(),
                resource_id=None  # None = all resources
            )

            # Check if already cached
            cached_result = await get_cached_result(hash_key)

            if cached_result:
                print(f"  [{completed_tasks}/{total_tasks}] ✅ {range_name:15} - Already cached ({len(cached_result)} resources)")
                cached_tasks += 1
                continue

            # Generate recommendations
            print(f"  [{completed_tasks}/{total_tasks}] 🔄 {range_name:15} - Generating recommendations...")

            try:
                # Call LLM analysis for ALL resources of this type (resource_id=None)
                result = run_llm_analysis(
                    resource_type=resource_type,
                    schema_name=schema_name,
                    start_date=start_date,
                    end_date=end_date,
                    resource_id=None,  # None = fetch all resources
                    task_id=None
                )

                # Convert to list if needed
                result_list = [result] if isinstance(result, dict) else result if result else []

                if result_list:
                    # Save to cache
                    await save_to_cache(
                        hash_key=hash_key,
                        cloud_platform="azure",
                        schema_name=schema_name,
                        resource_type=resource_type,
                        start_date=start_date.date(),
                        end_date=end_date.date(),
                        resource_id=None,
                        output_json=result_list
                    )
                    print(f"       ✅ Generated and cached {len(result_list)} recommendations")
                    generated_tasks += 1
                else:
                    print(f"       ⚠️  No data found for this date range")

            except Exception as e:
                print(f"       ❌ Error generating recommendations: {e}")
                continue

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ AZURE PRE-WARMING COMPLETE")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Already cached: {cached_tasks}")
    print(f"   Newly generated: {generated_tasks}")
    print(f"   Failed/Skipped: {total_tasks - cached_tasks - generated_tasks}")
    print(f"{'='*70}\n")


def prewarm_azure_recommendations(schema_name: str, budget: float = None):
    """
    Synchronous wrapper for Azure pre-warming (for Celery compatibility).

    Args:
        schema_name: PostgreSQL schema name for the Azure project
        budget: Monthly budget (optional)
    """
    try:
        asyncio.run(prewarm_azure_recommendations_async(schema_name, budget))
    except Exception as e:
        print(f"❌ Error in Azure pre-warming: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# AWS PRE-WARMING
# ============================================================

async def prewarm_aws_recommendations_async(schema_name: str, monthly_budget: float = None):
    """
    Pre-generate AWS recommendations for all resources and date ranges.
    Stores results in Redis cache for instant user access.

    Args:
        schema_name: PostgreSQL schema name for the AWS project
        monthly_budget: Monthly budget (optional, for display purposes)
    """
    print(f"\n{'='*70}")
    print(f"🔥 STARTING AWS RECOMMENDATION PRE-WARMING")
    print(f"   Schema: {schema_name}")
    print(f"{'='*70}\n")

    # Resource types to pre-warm
    resource_types = [
        ("ec2", "EC2 Instance"),
        ("s3", "S3 Bucket")
    ]

    # Get all date ranges
    date_ranges = calculate_date_ranges()

    total_tasks = len(resource_types) * len(date_ranges)
    completed_tasks = 0
    cached_tasks = 0
    generated_tasks = 0

    print(f"📋 Total pre-warming tasks: {total_tasks}")
    print(f"   Resource types: {len(resource_types)}")
    print(f"   Date ranges: {len(date_ranges)}")
    print("")

    # Pre-warm each resource type × date range combination
    for resource_type, display_name in resource_types:
        print(f"\n{'─'*70}")
        print(f"🔧 Processing Resource Type: {display_name} ({resource_type})")
        print(f"{'─'*70}")

        for range_name, (start_date, end_date) in date_ranges.items():
            completed_tasks += 1

            # Generate cache key
            hash_key = generate_cache_hash_key(
                cloud_platform="aws",
                schema_name=schema_name,
                resource_type=resource_type,
                start_date=start_date.date(),
                end_date=end_date.date(),
                resource_id=None  # None = all resources
            )

            # Check if already cached
            cached_result = await get_cached_result(hash_key)

            if cached_result:
                print(f"  [{completed_tasks}/{total_tasks}] ✅ {range_name:15} - Already cached ({len(cached_result)} resources)")
                cached_tasks += 1
                continue

            # Generate recommendations
            print(f"  [{completed_tasks}/{total_tasks}] 🔄 {range_name:15} - Generating recommendations...")

            try:
                # Call appropriate LLM analysis function based on resource type
                if resource_type == "ec2":
                    result = run_llm_analysis_ec2(
                        resource_type="ec2",
                        schema_name=schema_name,
                        start_date=start_date,
                        end_date=end_date,
                        resource_id=None  # None = fetch all instances
                    )
                elif resource_type == "s3":
                    result = run_llm_analysis_s3(
                        schema_name=schema_name,
                        start_date=start_date,
                        end_date=end_date,
                        bucket_name=None  # None = fetch all buckets
                    )
                else:
                    print(f"       ⚠️  Unsupported resource type: {resource_type}")
                    continue

                # Convert to list if needed
                result_list = [result] if isinstance(result, dict) else result if result else []

                if result_list:
                    # Save to cache
                    await save_to_cache(
                        hash_key=hash_key,
                        cloud_platform="aws",
                        schema_name=schema_name,
                        resource_type=resource_type,
                        start_date=start_date.date(),
                        end_date=end_date.date(),
                        resource_id=None,
                        output_json=result_list
                    )
                    print(f"       ✅ Generated and cached {len(result_list)} recommendations")
                    generated_tasks += 1
                else:
                    print(f"       ⚠️  No data found for this date range")

            except Exception as e:
                print(f"       ❌ Error generating recommendations: {e}")
                continue

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ AWS PRE-WARMING COMPLETE")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Already cached: {cached_tasks}")
    print(f"   Newly generated: {generated_tasks}")
    print(f"   Failed/Skipped: {total_tasks - cached_tasks - generated_tasks}")
    print(f"{'='*70}\n")


def prewarm_aws_recommendations(schema_name: str, monthly_budget: float = None):
    """
    Synchronous wrapper for AWS pre-warming (for Celery compatibility).

    Args:
        schema_name: PostgreSQL schema name for the AWS project
        monthly_budget: Monthly budget (optional)
    """
    try:
        asyncio.run(prewarm_aws_recommendations_async(schema_name, monthly_budget))
    except Exception as e:
        print(f"❌ Error in AWS pre-warming: {e}")
        import traceback
        traceback.print_exc()
