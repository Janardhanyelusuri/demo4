import boto3
import botocore.exceptions
import json
import pandas as pd
import io
import gzip
from datetime import datetime
from app.ingestion.aws.postgres_operations import *


def bucket(region, aws_access_key, aws_secret_key):
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    s3_client = session.client('s3')
    return s3_client


def create_s3_bucket(bucket_name, region, aws_access_key, aws_secret_key):
    s3_client = bucket(region, aws_access_key, aws_secret_key)

    try:
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f'Bucket {bucket_name} created successfully.')
        return True
    except Exception as e:
        print(f'Error creating bucket: {e}')
        return False


def check_and_create_bucket(bucket_name, region, aws_access_key, aws_secret_key):
    s3_client = bucket(region, aws_access_key, aws_secret_key)
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} already exists.')
        return True
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response['ResponseMetadata']['HTTPStatusCode'])
        if error_code == 404:
            print(f'Bucket {bucket_name} does not exist. Creating bucket...')
            return create_s3_bucket(bucket_name, region, aws_access_key, aws_secret_key)
        else:
            print(f'Error occurred while checking bucket: {e}')
            return False


def get_aws_account_id(aws_access_key, aws_secret_key, region):
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    sts_client = session.client('sts')
    try:
        identity_info = sts_client.get_caller_identity()
        account_id = identity_info['Account']
        return account_id
    except Exception as e:
        print(f'Error retrieving account ID: {e}')
        return None


def add_bucket_policy(bucket_name, account_id, aws_access_key, aws_secret_key, region):
    s3_client = bucket(region, aws_access_key, aws_secret_key)

    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "EnableAWSFocusExportsToWriteToS3AndCheckPolicy",
                "Effect": "Allow",
                "Principal": {
                    "Service": [
                        "billingreports.amazonaws.com",
                        "bcm-data-exports.amazonaws.com"
                    ]
                },
                "Action": [
                    "s3:PutObject",
                    "s3:GetBucketPolicy"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "StringLike": {
                        "aws:SourceAccount": account_id,
                        "aws:SourceArn": [
                            f"arn:aws:cur:{region}:{account_id}:definition/*",
                            f"arn:aws:bcm-data-exports:{region}:{account_id}:export/*"
                        ]
                    }
                }
            }
        ]
    }

    bucket_policy_string = json.dumps(bucket_policy)

    try:
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=bucket_policy_string
        )
        print(f'Bucket policy added to {bucket_name} successfully.')
    except Exception as e:
        print(f'Error adding bucket policy: {e}')


def get_s3_client(aws_access_key, aws_secret_key, region):
    """
    Create a Boto3 S3 client.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    return s3_client


def list_period_folders(s3_client, bucket_name, parent_folder):
    paginator = s3_client.get_paginator('list_objects_v2')
    periods = {}

    for page in paginator.paginate(Bucket=bucket_name, Prefix=parent_folder, Delimiter='/'):
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                folder_name = prefix['Prefix']
                print(f"Found period folder: {folder_name}")

                # Check for both cases of "BILLING_PERIOD="
                if 'BILLING_PERIOD=' in folder_name or 'billing_period=' in folder_name:
                    period_str = (
                        folder_name.split('BILLING_PERIOD=')[-1].split('/')[0]
                        if 'BILLING_PERIOD=' in folder_name
                        else folder_name.split('billing_period=')[-1].split('/')[0]
                    )
                    try:
                        period = datetime.strptime(period_str, '%Y-%m')
                        periods[folder_name] = None  # Placeholder for the latest file
                    except ValueError:
                        continue

    return periods


def get_latest_file(s3_client, bucket_name, folder_name):
    paginator = s3_client.get_paginator('list_objects_v2')
    latest_file = None
    latest_time = None

    for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith(('.csv.gz', '.snappy.parquet')):
                    last_modified = obj['LastModified']
                    if latest_file is None or last_modified > latest_time:
                        latest_file = obj['Key']
                        latest_time = last_modified

    return latest_file


# def download_and_extract_csv(s3_client, bucket_name, key):
#     response = s3_client.get_object(Bucket=bucket_name, Key=key)
#     with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gz:
#         return pd.read_csv(gz)

import gzip
import pandas as pd

def download_and_extract_csv(s3_client, bucket_name, key, chunksize=None):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    
    # Use streaming directly from the response body
    with gzip.GzipFile(fileobj=response['Body'], mode='rb') as gz:
        if chunksize:
            return pd.read_csv(gz, chunksize=chunksize)
        else:
            return pd.read_csv(gz)

def download_and_read_parquet(s3_client, bucket_name, key):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return pd.read_parquet(io.BytesIO(response['Body'].read()), engine='pyarrow')
