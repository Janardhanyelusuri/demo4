# app/ingestion/aws/metrics_s3.py
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import hashlib

# Configuration for the target table
S3_BRONZE_TABLE_NAME = "bronze_s3_bucket_metrics" # Consistent name

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.ingestion.aws.postgres_operations import dump_to_postgresql, fetch_existing_hash_keys

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("s3_metrics_scraper")

# Config
MAX_OBJECT_SAMPLE = 100
THREADS = 10

def session_for_region(access_key, secret_key, region_name=None):
    return boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name or "us-east-1",
    )

def discover_all_buckets(aws_access_key, aws_secret_key, default_region="us-east-1"):
    LOG.info("🔍 Discovering all S3 buckets...")
    
    account_id = None
    try:
        # Use STS to get the Account ID once
        sess_sts = session_for_region(aws_access_key, aws_secret_key, "us-east-1")
        sts = sess_sts.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        LOG.info("Account ID retrieved: %s", account_id)
    except (ClientError, NoCredentialsError) as e:
        LOG.warning("Could not retrieve Account ID via STS: %s", e)
        account_id = "UNKNOWN"
    except Exception as e:
        LOG.warning("An unexpected error occurred while retrieving Account ID: %s", e)
        account_id = "UNKNOWN"

    try:
        sess = session_for_region(aws_access_key, aws_secret_key, default_region)
        s3 = sess.client("s3")
        resp = s3.list_buckets()
        buckets = []
        for b in resp.get("Buckets", []):
            name = b["Name"]
            try:
                # Determine region
                loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
                region = loc if loc else "us-east-1"
            except ClientError as e:
                LOG.warning("Could not get location for bucket %s: %s (skipping)", name, e)
                continue

            arn = f"arn:aws:s3:::{name}"
            # Sample storage classes
            storage_classes = sample_bucket_storage_classes(aws_access_key, aws_secret_key, name, region, max_keys=MAX_OBJECT_SAMPLE)
            buckets.append({
                "Name": name, 
                "Region": region, 
                "ARN": arn, 
                "StorageClassesSample": storage_classes,
                "AccountId": account_id # Include the account ID here
            })
        LOG.info("✅ Discovered %d bucket(s)", len(buckets))
        return buckets
    except Exception as e:
        LOG.error("Failed to discover buckets: %s", e)
        return []

def sample_bucket_storage_classes(aws_access_key, aws_secret_key, bucket_name, region, max_keys=100):
    try:
        sess = session_for_region(aws_access_key, aws_secret_key, region)
        s3 = sess.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, PaginationConfig={"MaxItems": max_keys, "PageSize": 100})
        counts = {}
        retrieved = 0
        for page in pages:
            for obj in page.get("Contents", []):
                sc = obj.get("StorageClass", "STANDARD")
                counts[sc] = counts.get(sc, 0) + 1
                retrieved += 1
                if retrieved >= max_keys:
                    break
            if retrieved >= max_keys:
                break
        return counts
    except ClientError as e:
        LOG.debug("Sampling objects for bucket %s failed: %s", bucket_name, e)
        return {}

def list_bucket_metrics_for_bucket(aws_access_key, aws_secret_key, bucket_name, region):
    try:
        sess = session_for_region(aws_access_key, aws_secret_key, region)
        cw = sess.client("cloudwatch", region_name=region)
        paginator = cw.get_paginator("list_metrics")
        metrics = []
        page_iter = paginator.paginate(Namespace="AWS/S3", Dimensions=[{"Name": "BucketName", "Value": bucket_name}])
        for page in page_iter:
            for m in page.get("Metrics", []):
                metrics.append(m)
        return metrics
    except ClientError as e:
        LOG.warning("Failed to list metrics for bucket %s in %s: %s", bucket_name, region, e)
        return []

def fetch_latest_datapoint(aws_access_key, aws_secret_key, bucket_name, region, metric):
    try:
        sess = session_for_region(aws_access_key, aws_secret_key, region)
        cw = sess.client("cloudwatch", region_name=region)
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=3)
        period = 3600

        dimensions = metric.get("Dimensions", [])
        if not any(d.get("Name") == "BucketName" for d in dimensions):
            dimensions = [{"Name": "BucketName", "Value": bucket_name}] + dimensions

        resp = cw.get_metric_statistics(
            Namespace="AWS/S3",
            MetricName=metric["MetricName"],
            Dimensions=dimensions,
            StartTime=start,
            EndTime=now,
            Period=period,
            Statistics=["Average", "Sum", "Maximum", "Minimum"],
        )
    except ClientError as e:
        LOG.debug("CloudWatch query failed for %s %s: %s", bucket_name, metric.get("MetricName"), e)
        return None

    datapoints = resp.get("Datapoints", [])
    if not datapoints:
        return None

    latest = max(datapoints, key=lambda d: d["Timestamp"])
    value = latest.get("Average")
    if value is None:
        value = latest.get("Sum") or latest.get("Maximum") or latest.get("Minimum")

    # Determine StorageType. Default to None/empty string if not a dimension.
    storage_type = None
    for dim in dimensions:
        if dim.get("Name") == "StorageType":
            storage_type = dim.get("Value")
            break
            
    record = {
        "bucket_name": bucket_name,
        "region": region,
        "metric_name": metric.get("MetricName"),
        "storage_type": storage_type,
        "dimensions_json": json.dumps(dimensions),
        "timestamp": latest["Timestamp"].astimezone(timezone.utc).replace(tzinfo=None),  # naive UTC
        "value": value,
        "unit": latest.get("Unit"),
        # account_id and storage_class will be added in scrape_bucket_metrics/metrics_dump
    }
    return record

def scrape_bucket_metrics(aws_access_key, aws_secret_key, bucket):
    name = bucket["Name"]
    region = bucket["Region"]
    account_id = bucket["AccountId"] # Retrieve account ID
    metrics = list_bucket_metrics_for_bucket(aws_access_key, aws_secret_key, name, region)
    if not metrics:
        LOG.info("No CloudWatch metrics for bucket %s", name)
        return []
    records = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_latest_datapoint, aws_access_key, aws_secret_key, name, region, m): m for m in metrics}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                # Add ARN, Account ID, and Storage Classes Sample from bucket discovery
                res.update({
                    "arn": bucket["ARN"], 
                    "storage_classes_sample_json": json.dumps(bucket["StorageClassesSample"]),
                    "account_id": account_id # Add account_id
                })
                records.append(res)
    return records

def collect_all_s3_metrics(aws_access_key, aws_secret_key, default_region="us-east-1"):
    LOG.info("=" * 60)
    LOG.info("🚀 Starting S3 metrics collection...")
    LOG.info("=" * 60)

    buckets = discover_all_buckets(aws_access_key, aws_secret_key, default_region)
    if not buckets:
        LOG.warning("No buckets found")
        return pd.DataFrame()

    all_records = []
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(scrape_bucket_metrics, aws_access_key, aws_secret_key, b): b for b in buckets}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                recs = fut.result()
                LOG.info("Collected %d metrics for bucket %s", len(recs), b["Name"])
                all_records.extend(recs)
            except Exception as e:
                LOG.exception("Error scraping bucket %s: %s", b["Name"], e)

    if all_records:
        df = pd.DataFrame(all_records)
        LOG.info("Total records collected: %d", len(df))
        return df
    else:
        LOG.warning("No metrics collected")
        return pd.DataFrame()

# ----------------- dedupe helpers -----------------
def _compute_s3_hash_key_for_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure columns exist
    if 'bucket_name' not in df.columns:
        df['bucket_name'] = ''
    df['bucket_name'] = df['bucket_name'].astype(str).fillna('').str.lower()
    # normalize timestamp to 'YYYY-MM-DD HH:MM:SS' (second precision)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # value normalization
    df['value_norm'] = df['value'].astype(float).round(6).astype(str)
    def row_hash(r):
        s = f"{r['bucket_name']}|{r['timestamp_str']}|{r.get('metric_name','') or ''}|{r.get('value_norm','')}"
        return hashlib.md5(s.encode('utf-8')).hexdigest()
    df['hash_key'] = df.apply(row_hash, axis=1)
    df = df.drop(columns=['timestamp_str','value_norm'], errors='ignore')
    return df

def metrics_dump(aws_access_key, aws_secret_key, region, schema_name):
    # Use the consistent table name
    table_name = S3_BRONZE_TABLE_NAME
    LOG.info("🔄 Starting S3 metrics dump...")
    LOG.info("Schema: %s, Table: %s", schema_name, table_name)

    all_metrics_df = collect_all_s3_metrics(aws_access_key, aws_secret_key, region)

    LOG.info("=" * 60)
    if all_metrics_df is None or all_metrics_df.empty:
        LOG.warning("No data collected to dump.")
        LOG.info("=" * 60)
        return

    # 🚨 FIX: Ensure all required columns from the SQL schema exist in the DataFrame 
    # and fill missing ones with appropriate defaults (e.g., None, blank string)
    REQUIRED_COLUMNS = [
        "bucket_name", "region", "account_id", "timestamp", "metric_name", 
        "value", "unit", "storage_class", "storage_type", "dimensions_json", 
        "arn", "storage_classes_sample_json"
    ]
    
    for col in REQUIRED_COLUMNS:
        if col not in all_metrics_df.columns:
            # storage_class is not populated by metrics
            all_metrics_df[col] = None 
            LOG.debug("Added missing column: %s", col)

    # Reorder columns to match the SQL schema for safer insertion
    all_metrics_df = all_metrics_df[[c for c in REQUIRED_COLUMNS if c in all_metrics_df.columns]]

    # compute hash keys
    all_metrics_df = _compute_s3_hash_key_for_df(all_metrics_df)

    # fetch existing hash keys from bronze to avoid duplicate insert
    try:
        # Pass the corrected table name
        existing = fetch_existing_hash_keys(schema_name, table_name)
    except Exception as e:
        # The original code logged a warning; we'll keep that but log the specific error
        LOG.warning("Could not fetch existing hash keys: %s", e)
        existing = set()

    # filter only new unique rows
    new_df = all_metrics_df[~all_metrics_df['hash_key'].isin(existing)].copy()
    LOG.info("Total collected: %d  New unique: %d", len(all_metrics_df), len(new_df))

    if new_df.empty:
        LOG.info("No new unique S3 metric rows to insert. Skipping DB dump.")
        LOG.info("=" * 60)
        return

    # dump in chunks
    try:
        chunk_size = 100000
        num_chunks = (len(new_df) + chunk_size - 1) // chunk_size
        LOG.info("Inserting %d rows in %d chunk(s)...", len(new_df), num_chunks)
        for start in range(0, len(new_df), chunk_size):
            chunk = new_df.iloc[start:start + chunk_size]
            # Pass the corrected table name
            dump_to_postgresql(chunk, schema_name, table_name)
        LOG.info("✅ S3 metrics dumped successfully to %s.%s!", schema_name, table_name)
    except Exception as e:
        LOG.error("Failed to dump S3 metrics: %s", e)