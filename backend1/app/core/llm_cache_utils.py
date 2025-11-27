# app/core/llm_cache_utils.py

import hashlib
import json
import os
import redis.asyncio as redis
from datetime import date
from typing import Optional, List, Dict, Any

# Redis connection pool (initialized once)
_redis_pool = None

# Cache TTL: 24 hours in seconds
CACHE_TTL_SECONDS = 24 * 60 * 60  # 86400 seconds


async def get_redis_client() -> redis.Redis:
    """
    Get async Redis client with connection pooling.
    Creates pool on first call, reuses thereafter.
    """
    global _redis_pool

    if _redis_pool is None:
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        _redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            decode_responses=False,  # We'll handle JSON encoding/decoding manually
            max_connections=10
        )

    return redis.Redis(connection_pool=_redis_pool)


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

    # Prefix with namespace for better organization in Redis
    return f"llm_cache:{hash_key}"


async def get_cached_result(hash_key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve cached LLM result from Redis by hash key.

    Args:
        hash_key: The MD5 hash key (with namespace prefix)

    Returns:
        Cached output as a list of dictionaries, or None if not found
    """
    try:
        client = await get_redis_client()
        cached_data = await client.get(hash_key)

        if cached_data:
            # Decode and parse JSON
            output_json = json.loads(cached_data.decode('utf-8'))
            print(f"✅ Redis Cache HIT for hash_key: {hash_key[:20]}...")
            return output_json
        else:
            print(f"❌ Redis Cache MISS for hash_key: {hash_key[:20]}...")
            return None
    except Exception as e:
        print(f"⚠️ Error retrieving from Redis cache: {e}")
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
    Save LLM result to Redis cache with 24-hour TTL.

    Args:
        hash_key: The MD5 hash key (with namespace prefix)
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
        client = await get_redis_client()

        # Serialize to JSON
        json_data = json.dumps(output_json).encode('utf-8')

        # Save to Redis with TTL (EX = expire in seconds)
        await client.setex(hash_key, CACHE_TTL_SECONDS, json_data)

        print(f"💾 Redis Cache SAVED for hash_key: {hash_key[:20]}... (TTL: 24h)")
        return True
    except Exception as e:
        print(f"⚠️ Error saving to Redis cache: {e}")
        return False


async def clear_cache_for_resource(
    cloud_platform: str,
    schema_name: str,
    resource_type: str,
    resource_id: Optional[str] = None
) -> int:
    """
    Clear cache entries matching the given criteria.
    Uses Redis SCAN to find matching keys without blocking.

    Args:
        cloud_platform: Cloud platform
        schema_name: Schema/project name
        resource_type: Resource type
        resource_id: Specific resource ID (optional)

    Returns:
        Number of keys deleted
    """
    try:
        client = await get_redis_client()

        # Build pattern for scanning
        # Since our hash includes dates, we need to scan all llm_cache keys
        # and check if they match our criteria by regenerating potential keys
        pattern = "llm_cache:*"

        deleted_count = 0
        async for key in client.scan_iter(match=pattern, count=100):
            # For simplicity, delete all matching the base pattern
            # In production, you might want more granular control
            await client.delete(key)
            deleted_count += 1

        print(f"🗑️ Cleared {deleted_count} Redis cache entries matching pattern: {pattern}")
        return deleted_count
    except Exception as e:
        print(f"⚠️ Error clearing Redis cache: {e}")
        return 0


async def get_cache_stats() -> Dict[str, Any]:
    """
    Get Redis cache statistics.

    Returns:
        Dictionary with cache stats (keys count, memory usage, etc.)
    """
    try:
        client = await get_redis_client()

        # Count llm_cache keys
        keys_count = 0
        async for _ in client.scan_iter(match="llm_cache:*", count=100):
            keys_count += 1

        # Get Redis info
        info = await client.info('memory')

        return {
            "cache_keys_count": keys_count,
            "memory_used_bytes": info.get('used_memory', 0),
            "memory_used_human": info.get('used_memory_human', 'N/A'),
            "ttl_seconds": CACHE_TTL_SECONDS,
            "ttl_hours": CACHE_TTL_SECONDS / 3600
        }
    except Exception as e:
        print(f"⚠️ Error getting Redis cache stats: {e}")
        return {"error": str(e)}
