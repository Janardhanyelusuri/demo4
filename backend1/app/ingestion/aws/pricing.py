"""
AWS Pricing Module

Fetches and stores AWS pricing information including:
- EC2 instance types (sizes, vCPUs, memory, pricing)
- S3 storage pricing (per GB/month, request costs)
- EBS volume pricing

Uses AWS Price List API for real-time pricing data.
"""

import boto3
import pandas as pd
import sys
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ingestion.aws.postgres_operations import connection, dump_to_postgresql


def fetch_ec2_pricing(aws_access_key: str, aws_secret_key: str, region: str = "us-east-1") -> pd.DataFrame:
    """
    Fetch EC2 instance pricing from AWS Price List API.

    Args:
        aws_access_key: AWS access key
        aws_secret_key: AWS secret key
        region: AWS region (e.g., 'us-east-1')

    Returns:
        DataFrame with EC2 pricing information
    """
    print(f"📊 Fetching AWS EC2 pricing for region: {region}")

    try:
        # Create pricing client (pricing API is only available in us-east-1 and ap-south-1)
        pricing_client = boto3.client(
            'pricing',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'  # Pricing API only available here
        )

        # Map region codes to region names
        region_name_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'EU (Ireland)',
            'eu-central-1': 'EU (Frankfurt)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)',
        }

        region_name = region_name_map.get(region, region)

        # Fetch pricing for Linux/Unix, on-demand instances
        filters = [
            {'Type': 'TERM_MATCH', 'Field': 'ServiceCode', 'Value': 'AmazonEC2'},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
            {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
            {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
        ]

        all_products = []
        next_token = None

        while True:
            if next_token:
                response = pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=filters,
                    NextToken=next_token,
                    MaxResults=100
                )
            else:
                response = pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=filters,
                    MaxResults=100
                )

            price_list = response.get('PriceList', [])

            for price_item in price_list:
                price_data = json.loads(price_item)
                product = price_data.get('product', {})
                attributes = product.get('attributes', {})

                # Extract on-demand pricing
                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})

                if on_demand:
                    for term_key, term_data in on_demand.items():
                        price_dimensions = term_data.get('priceDimensions', {})

                        for dim_key, dimension in price_dimensions.items():
                            price_per_unit = dimension.get('pricePerUnit', {})
                            usd_price = price_per_unit.get('USD', '0')

                            all_products.append({
                                'instance_type': attributes.get('instanceType'),
                                'vcpu': attributes.get('vcpu'),
                                'memory': attributes.get('memory'),
                                'storage': attributes.get('storage'),
                                'network_performance': attributes.get('networkPerformance'),
                                'instance_family': attributes.get('instanceFamily'),
                                'physical_processor': attributes.get('physicalProcessor'),
                                'clock_speed': attributes.get('clockSpeed'),
                                'price_per_hour': float(usd_price) if usd_price else 0.0,
                                'currency': 'USD',
                                'region': region,
                                'region_name': region_name,
                                'operating_system': 'Linux',
                                'tenancy': 'Shared',
                                'unit': dimension.get('unit', 'Hrs'),
                            })

            next_token = response.get('NextToken')

            print(f"  Fetched {len(price_list)} EC2 pricing records (total: {len(all_products)})")

            if not next_token or len(all_products) >= 1000:
                break

        df = pd.DataFrame(all_products)
        df['last_updated'] = datetime.utcnow()

        print(f"✅ Successfully fetched {len(df)} EC2 pricing records")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching EC2 pricing: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_s3_pricing(aws_access_key: str, aws_secret_key: str, region: str = "us-east-1") -> pd.DataFrame:
    """
    Fetch S3 storage pricing from AWS Price List API.

    Args:
        aws_access_key: AWS access key
        aws_secret_key: AWS secret key
        region: AWS region

    Returns:
        DataFrame with S3 pricing information
    """
    print(f"📊 Fetching AWS S3 pricing for region: {region}")

    try:
        pricing_client = boto3.client(
            'pricing',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )

        region_name_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'EU (Ireland)',
            'eu-central-1': 'EU (Frankfurt)',
        }

        region_name = region_name_map.get(region, region)

        filters = [
            {'Type': 'TERM_MATCH', 'Field': 'ServiceCode', 'Value': 'AmazonS3'},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
        ]

        all_products = []
        next_token = None

        while True:
            if next_token:
                response = pricing_client.get_products(
                    ServiceCode='AmazonS3',
                    Filters=filters,
                    NextToken=next_token,
                    MaxResults=100
                )
            else:
                response = pricing_client.get_products(
                    ServiceCode='AmazonS3',
                    Filters=filters,
                    MaxResults=100
                )

            price_list = response.get('PriceList', [])

            for price_item in price_list:
                price_data = json.loads(price_item)
                product = price_data.get('product', {})
                attributes = product.get('attributes', {})

                # Extract on-demand pricing
                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})

                if on_demand:
                    for term_key, term_data in on_demand.items():
                        price_dimensions = term_data.get('priceDimensions', {})

                        for dim_key, dimension in price_dimensions.items():
                            price_per_unit = dimension.get('pricePerUnit', {})
                            usd_price = price_per_unit.get('USD', '0')

                            all_products.append({
                                'storage_class': attributes.get('storageClass'),
                                'volume_type': attributes.get('volumeType'),
                                'usage_type': attributes.get('usagetype'),
                                'price_per_unit': float(usd_price) if usd_price else 0.0,
                                'currency': 'USD',
                                'region': region,
                                'region_name': region_name,
                                'unit': dimension.get('unit', 'GB-Month'),
                                'description': dimension.get('description', ''),
                            })

            next_token = response.get('NextToken')

            print(f"  Fetched {len(price_list)} S3 pricing records (total: {len(all_products)})")

            if not next_token or len(all_products) >= 500:
                break

        df = pd.DataFrame(all_products)
        df['last_updated'] = datetime.utcnow()

        print(f"✅ Successfully fetched {len(df)} S3 pricing records")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching S3 pricing: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_ebs_pricing(aws_access_key: str, aws_secret_key: str, region: str = "us-east-1") -> pd.DataFrame:
    """
    Fetch EBS volume pricing from AWS Price List API.

    Args:
        aws_access_key: AWS access key
        aws_secret_key: AWS secret key
        region: AWS region

    Returns:
        DataFrame with EBS pricing information
    """
    print(f"📊 Fetching AWS EBS pricing for region: {region}")

    try:
        pricing_client = boto3.client(
            'pricing',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )

        region_name_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
        }

        region_name = region_name_map.get(region, region)

        filters = [
            {'Type': 'TERM_MATCH', 'Field': 'ServiceCode', 'Value': 'AmazonEC2'},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
            {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
        ]

        all_products = []
        next_token = None

        while True:
            if next_token:
                response = pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=filters,
                    NextToken=next_token,
                    MaxResults=100
                )
            else:
                response = pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=filters,
                    MaxResults=100
                )

            price_list = response.get('PriceList', [])

            for price_item in price_list:
                price_data = json.loads(price_item)
                product = price_data.get('product', {})
                attributes = product.get('attributes', {})

                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})

                if on_demand:
                    for term_key, term_data in on_demand.items():
                        price_dimensions = term_data.get('priceDimensions', {})

                        for dim_key, dimension in price_dimensions.items():
                            price_per_unit = dimension.get('pricePerUnit', {})
                            usd_price = price_per_unit.get('USD', '0')

                            all_products.append({
                                'volume_type': attributes.get('volumeApiName'),
                                'storage_media': attributes.get('storageMedia'),
                                'max_iops_volume': attributes.get('maxIopsvolume'),
                                'max_throughput_volume': attributes.get('maxThroughputvolume'),
                                'price_per_gb_month': float(usd_price) if usd_price else 0.0,
                                'currency': 'USD',
                                'region': region,
                                'region_name': region_name,
                                'unit': dimension.get('unit', 'GB-Mo'),
                            })

            next_token = response.get('NextToken')

            if not next_token or len(all_products) >= 200:
                break

        df = pd.DataFrame(all_products)
        df['last_updated'] = datetime.utcnow()

        print(f"✅ Successfully fetched {len(df)} EBS pricing records")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching EBS pricing: {e}")
        return pd.DataFrame()


@connection
def store_aws_pricing(conn, schema_name: str, pricing_df: pd.DataFrame, pricing_type: str):
    """
    Store AWS pricing data in PostgreSQL.

    Args:
        conn: Database connection
        schema_name: Schema name
        pricing_df: DataFrame with pricing data
        pricing_type: Type of pricing (ec2, s3, ebs)
    """
    if pricing_df.empty:
        print(f"  ⚠️ No {pricing_type} pricing data to store")
        return

    table_name = f"aws_pricing_{pricing_type}"

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


def fetch_and_store_all_aws_pricing(
    schema_name: str,
    aws_access_key: str,
    aws_secret_key: str,
    region: str = "us-east-1"
):
    """
    Main function to fetch and store all AWS pricing data.

    Args:
        schema_name: PostgreSQL schema name
        aws_access_key: AWS access key
        aws_secret_key: AWS secret key
        region: AWS region
    """
    print(f"\n{'='*70}")
    print(f"💰 FETCHING AWS PRICING DATA")
    print(f"   Schema: {schema_name}")
    print(f"   Region: {region}")
    print(f"{'='*70}\n")

    try:
        # Fetch EC2 pricing
        ec2_pricing = fetch_ec2_pricing(aws_access_key, aws_secret_key, region)
        if not ec2_pricing.empty:
            store_aws_pricing(schema_name, ec2_pricing, "ec2")

        # Fetch S3 pricing
        s3_pricing = fetch_s3_pricing(aws_access_key, aws_secret_key, region)
        if not s3_pricing.empty:
            store_aws_pricing(schema_name, s3_pricing, "s3")

        # Fetch EBS pricing
        ebs_pricing = fetch_ebs_pricing(aws_access_key, aws_secret_key, region)
        if not ebs_pricing.empty:
            store_aws_pricing(schema_name, ebs_pricing, "ebs")

        print(f"\n{'='*70}")
        print(f"✅ AWS PRICING DATA FETCH COMPLETE")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Error fetching AWS pricing: {e}")
        import traceback
        traceback.print_exc()


def get_ec2_instance_pricing(schema_name: str, instance_type: str, region: str = "us-east-1") -> Optional[Dict]:
    """
    Get pricing for a specific EC2 instance type.

    Args:
        schema_name: Schema name
        instance_type: EC2 instance type (e.g., 't2.micro')
        region: AWS region

    Returns:
        Dictionary with pricing information or None
    """
    from app.ingestion.aws.postgres_operations import connection

    @connection
    def _get_pricing(conn, schema, inst_type, rgn):
        query = f"""
            SELECT
                instance_type,
                vcpu,
                memory,
                price_per_hour,
                currency,
                network_performance,
                physical_processor
            FROM {schema}.aws_pricing_ec2
            WHERE LOWER(instance_type) = LOWER(%s)
              AND LOWER(region) = LOWER(%s)
            LIMIT 1
        """

        cursor = conn.cursor()
        cursor.execute(query, (inst_type, rgn))
        result = cursor.fetchone()

        if result:
            return {
                'instance_type': result[0],
                'vcpu': result[1],
                'memory': result[2],
                'price_per_hour': result[3],
                'currency': result[4],
                'network_performance': result[5],
                'physical_processor': result[6]
            }
        return None

    return _get_pricing(schema_name, instance_type, region)
