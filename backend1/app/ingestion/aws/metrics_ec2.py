# app/ingestion/aws/metrics_ec2.py
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
EC2_BRONZE_TABLE_NAME = "bronze_ec2_instance_metrics"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.ingestion.aws.postgres_operations import dump_to_postgresql, fetch_existing_hash_keys

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ec2_metrics_scraper")

# Config
THREADS = 10
LOOKBACK_DAYS = 14  # How many days of metrics to fetch

# EC2 metrics we want to collect
EC2_METRICS = [
    "CPUUtilization",
    "DiskReadOps",
    "DiskWriteOps",
    "DiskReadBytes",
    "DiskWriteBytes",
    "NetworkIn",
    "NetworkOut",
    "NetworkPacketsIn",
    "NetworkPacketsOut",
    "StatusCheckFailed",
    "StatusCheckFailed_Instance",
    "StatusCheckFailed_System",
    "CPUCreditUsage",
    "CPUCreditBalance",
    "CPUSurplusCreditBalance",
    "CPUSurplusCreditsCharged",
]


def session_for_region(access_key, secret_key, region_name=None):
    return boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name or "us-east-1",
    )


def get_account_id(aws_access_key, aws_secret_key):
    """Get AWS Account ID using STS"""
    try:
        sess = session_for_region(aws_access_key, aws_secret_key, "us-east-1")
        sts = sess.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        LOG.info("Account ID retrieved: %s", account_id)
        return account_id
    except Exception as e:
        LOG.warning("Could not retrieve Account ID: %s", e)
        return "UNKNOWN"


def discover_all_ec2_instances(aws_access_key, aws_secret_key):
    """
    Discover all EC2 instances across all regions.
    Returns a list of instance dicts with metadata.
    """
    LOG.info("🔍 Discovering all EC2 instances...")

    account_id = get_account_id(aws_access_key, aws_secret_key)

    # Get all regions
    sess = session_for_region(aws_access_key, aws_secret_key, "us-east-1")
    ec2_client = sess.client("ec2")

    try:
        regions_response = ec2_client.describe_regions()
        regions = [r["RegionName"] for r in regions_response["Regions"]]
        LOG.info("Found %d AWS regions", len(regions))
    except Exception as e:
        LOG.error("Failed to describe regions: %s", e)
        return []

    all_instances = []

    for region in regions:
        try:
            sess_region = session_for_region(aws_access_key, aws_secret_key, region)
            ec2 = sess_region.client("ec2")

            paginator = ec2.get_paginator("describe_instances")
            page_iterator = paginator.paginate()

            for page in page_iterator:
                for reservation in page["Reservations"]:
                    for instance in reservation["Instances"]:
                        instance_id = instance["InstanceId"]
                        instance_type = instance.get("InstanceType", "unknown")
                        state = instance.get("State", {}).get("Name", "unknown")
                        az = instance.get("Placement", {}).get("AvailabilityZone", "unknown")

                        # Get instance name from tags
                        instance_name = ""
                        for tag in instance.get("Tags", []):
                            if tag["Key"] == "Name":
                                instance_name = tag["Value"]
                                break

                        all_instances.append({
                            "InstanceId": instance_id,
                            "InstanceName": instance_name,
                            "InstanceType": instance_type,
                            "Region": region,
                            "AvailabilityZone": az,
                            "State": state,
                            "AccountId": account_id,
                        })

            LOG.info("Found %d instance(s) in %s",
                    len([i for i in all_instances if i["Region"] == region]), region)

        except Exception as e:
            LOG.warning("Failed to list instances in %s: %s", region, e)
            continue

    LOG.info("✅ Discovered %d total EC2 instance(s)", len(all_instances))
    return all_instances


def fetch_ec2_metric_data(aws_access_key, aws_secret_key, instance, metric_name, start_time, end_time):
    """
    Fetch CloudWatch metric data for a single EC2 instance and metric.
    Returns list of records.
    """
    instance_id = instance["InstanceId"]
    region = instance["Region"]

    try:
        sess = session_for_region(aws_access_key, aws_secret_key, region)
        cw = sess.client("cloudwatch", region_name=region)

        dimensions = [{"Name": "InstanceId", "Value": instance_id}]

        response = cw.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=["Average", "Sum", "Maximum", "Minimum"],
        )

        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return []

        records = []
        for dp in datapoints:
            # Prefer Average, then Sum, then Maximum, then Minimum
            value = dp.get("Average")
            if value is None:
                value = dp.get("Sum")
            if value is None:
                value = dp.get("Maximum")
            if value is None:
                value = dp.get("Minimum")

            record = {
                "instance_id": instance_id,
                "instance_name": instance["InstanceName"],
                "instance_type": instance["InstanceType"],
                "region": region,
                "account_id": instance["AccountId"],
                "timestamp": dp["Timestamp"].astimezone(timezone.utc).replace(tzinfo=None),
                "metric_name": metric_name,
                "value": value,
                "unit": dp.get("Unit", ""),
                "availability_zone": instance["AvailabilityZone"],
                "dimensions_json": json.dumps(dimensions),
            }
            records.append(record)

        return records

    except ClientError as e:
        LOG.debug("CloudWatch query failed for %s %s: %s", instance_id, metric_name, e)
        return []


def scrape_instance_metrics(aws_access_key, aws_secret_key, instance, start_time, end_time):
    """
    Scrape all metrics for a single EC2 instance.
    """
    instance_id = instance["InstanceId"]

    # Skip instances that are not running (optional - comment out to collect all states)
    if instance["State"] not in ["running", "stopped"]:
        LOG.debug("Skipping instance %s (state: %s)", instance_id, instance["State"])
        return []

    all_records = []

    for metric_name in EC2_METRICS:
        records = fetch_ec2_metric_data(
            aws_access_key, aws_secret_key, instance, metric_name, start_time, end_time
        )
        all_records.extend(records)

    LOG.info("Collected %d metric datapoints for instance %s", len(all_records), instance_id)
    return all_records


def collect_all_ec2_metrics(aws_access_key, aws_secret_key, lookback_days=LOOKBACK_DAYS):
    """
    Main collection function - discovers all EC2 instances and collects metrics.
    """
    LOG.info("=" * 60)
    LOG.info("🚀 Starting EC2 metrics collection...")
    LOG.info("=" * 60)

    instances = discover_all_ec2_instances(aws_access_key, aws_secret_key)
    if not instances:
        LOG.warning("No EC2 instances found")
        return pd.DataFrame()

    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    LOG.info("Fetching metrics from %s to %s", start_time.isoformat(), end_time.isoformat())

    all_records = []

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {
            pool.submit(scrape_instance_metrics, aws_access_key, aws_secret_key, inst, start_time, end_time): inst
            for inst in instances
        }

        for fut in as_completed(futures):
            inst = futures[fut]
            try:
                records = fut.result()
                all_records.extend(records)
            except Exception as e:
                LOG.exception("Error scraping instance %s: %s", inst["InstanceId"], e)

    if all_records:
        df = pd.DataFrame(all_records)
        LOG.info("Total records collected: %d", len(df))
        return df
    else:
        LOG.warning("No metrics collected")
        return pd.DataFrame()


def _compute_ec2_hash_key_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate deterministic hash keys for EC2 metrics.
    Hash based on: instance_id + timestamp (hour precision) + metric_name
    """
    df = df.copy()

    # Ensure required columns exist
    if 'instance_id' not in df.columns:
        df['instance_id'] = ''
    df['instance_id'] = df['instance_id'].astype(str).fillna('').str.lower()

    # Normalize timestamp to hour precision
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')

    # Value normalization
    df['value_norm'] = df['value'].astype(float).round(6).astype(str)

    def row_hash(r):
        s = f"{r['instance_id']}|{r['timestamp_str']}|{r.get('metric_name','') or ''}|{r.get('value_norm','')}"
        return hashlib.md5(s.encode('utf-8')).hexdigest()

    df['hash_key'] = df.apply(row_hash, axis=1)
    df = df.drop(columns=['timestamp_str', 'value_norm'], errors='ignore')

    return df


def metrics_dump(aws_access_key, aws_secret_key, region, schema_name):
    """
    Main entry point for EC2 metrics ingestion.
    Collects metrics and dumps to PostgreSQL bronze table.
    """
    table_name = EC2_BRONZE_TABLE_NAME
    LOG.info("🔄 Starting EC2 metrics dump...")
    LOG.info("Schema: %s, Table: %s", schema_name, table_name)

    all_metrics_df = collect_all_ec2_metrics(aws_access_key, aws_secret_key)

    LOG.info("=" * 60)
    if all_metrics_df is None or all_metrics_df.empty:
        LOG.warning("No EC2 data collected to dump.")
        LOG.info("=" * 60)
        return

    # Ensure all required columns exist
    REQUIRED_COLUMNS = [
        "instance_id", "instance_name", "instance_type", "region", "account_id",
        "timestamp", "metric_name", "value", "unit", "availability_zone", "dimensions_json"
    ]

    for col in REQUIRED_COLUMNS:
        if col not in all_metrics_df.columns:
            all_metrics_df[col] = None
            LOG.debug("Added missing column: %s", col)

    # Reorder columns to match SQL schema
    all_metrics_df = all_metrics_df[[c for c in REQUIRED_COLUMNS if c in all_metrics_df.columns]]

    # Compute hash keys
    all_metrics_df = _compute_ec2_hash_key_for_df(all_metrics_df)

    # Fetch existing hash keys to avoid duplicates
    try:
        existing = fetch_existing_hash_keys(schema_name, table_name)
    except Exception as e:
        LOG.warning("Could not fetch existing hash keys: %s", e)
        existing = set()

    # Filter only new unique rows
    new_df = all_metrics_df[~all_metrics_df['hash_key'].isin(existing)].copy()
    LOG.info("Total collected: %d  New unique: %d", len(all_metrics_df), len(new_df))

    if new_df.empty:
        LOG.info("No new unique EC2 metric rows to insert. Skipping DB dump.")
        LOG.info("=" * 60)
        return

    # Dump in chunks
    try:
        chunk_size = 100000
        num_chunks = (len(new_df) + chunk_size - 1) // chunk_size
        LOG.info("Inserting %d rows in %d chunk(s)...", len(new_df), num_chunks)

        for start in range(0, len(new_df), chunk_size):
            chunk = new_df.iloc[start:start + chunk_size]
            dump_to_postgresql(chunk, schema_name, table_name)

        LOG.info("✅ EC2 metrics dumped successfully to %s.%s!", schema_name, table_name)
    except Exception as e:
        LOG.error("Failed to dump EC2 metrics: %s", e)

    LOG.info("=" * 60)


if __name__ == "__main__":
    # Example usage for testing
    import os
    from dotenv import load_dotenv
    load_dotenv()

    metrics_dump(
        aws_access_key=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        region="us-east-1",
        schema_name="test_schema"
    )
