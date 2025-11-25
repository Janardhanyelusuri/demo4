# app/core/llm_cache_utils.py

import hashlib
import json
from datetime import date
from typing import Optional, List, Dict, Any
from app.models.llm_cache import LLMCache


def generate_cache_hash_key(
    cloud_platform: str,
    schema_name: str,
    resource_type: str,
    start_date: Optional[date],
    end_date: Optional[date],
    resource_id: Optional[str] = None
) -> str:
    """
    Generate a unique MD5 hash key for caching based on input parameters.

    Args:
        cloud_platform: Cloud platform (aws, azure, gcp)
        schema_name: Schema/project name
        resource_type: Resource type (vm, storage, ec2, etc.)
        start_date: Analysis start date
        end_date: Analysis end date
        resource_id: Specific resource ID (optional)

    Returns:
        MD5 hash string (32 characters)
    """
    # Normalize inputs
    cloud = cloud_platform.lower().strip()
    schema = schema_name.lower().strip()
    rtype = resource_type.lower().strip()
    rid = resource_id.lower().strip() if resource_id else ""

    # Convert dates to string format
    start_str = start_date.isoformat() if start_date else ""
    end_str = end_date.isoformat() if end_date else ""

    # Concatenate all parameters in a consistent order
    cache_string = f"{cloud}|{schema}|{rtype}|{start_str}|{end_str}|{rid}"

    # Generate MD5 hash
    hash_object = hashlib.md5(cache_string.encode('utf-8'))
    hash_key = hash_object.hexdigest()

    return hash_key


async def get_cached_result(hash_key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve cached LLM result by hash key.

    Args:
        hash_key: The MD5 hash key

    Returns:
        Cached output as a list of dictionaries, or None if not found
    """
    try:
        cache_entry = await LLMCache.get_or_none(hash_key=hash_key)
        if cache_entry:
            print(f"✅ Cache HIT for hash_key: {hash_key[:8]}...")
            return cache_entry.output_json
        else:
            print(f"❌ Cache MISS for hash_key: {hash_key[:8]}...")
            return None
    except Exception as e:
        print(f"⚠️ Error retrieving from cache: {e}")
        return None


async def save_to_cache(
    hash_key: str,
    cloud_platform: str,
    schema_name: str,
    resource_type: str,
    start_date: Optional[date],
    end_date: Optional[date],
    resource_id: Optional[str],
    output_json: List[Dict[str, Any]]
) -> bool:
    """
    Save LLM result to cache.

    Args:
        hash_key: The MD5 hash key
        cloud_platform: Cloud platform
        schema_name: Schema/project name
        resource_type: Resource type
        start_date: Analysis start date
        end_date: Analysis end date
        resource_id: Specific resource ID (optional)
        output_json: The LLM output to cache

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Check if entry already exists
        existing_entry = await LLMCache.get_or_none(hash_key=hash_key)

        if existing_entry:
            # Update existing entry (updated_at handled automatically by auto_now=True)
            existing_entry.output_json = output_json
            await existing_entry.save()
            print(f"🔄 Cache UPDATED for hash_key: {hash_key[:8]}...")
        else:
            # Create new entry
            await LLMCache.create(
                hash_key=hash_key,
                cloud_platform=cloud_platform.lower(),
                schema_name=schema_name.lower(),
                resource_type=resource_type.lower(),
                resource_id=resource_id,
                start_date=start_date,
                end_date=end_date,
                output_json=output_json
            )
            print(f"💾 Cache SAVED for hash_key: {hash_key[:8]}...")

        return True
    except Exception as e:
        print(f"⚠️ Error saving to cache: {e}")
        return False
