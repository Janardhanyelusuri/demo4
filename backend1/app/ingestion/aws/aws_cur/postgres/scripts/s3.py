import boto3
import datetime
import json
import os
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()


# Use os.getenv to retrieve environment variables
# aws_access_key = os.getenv('aws_access_key')
# aws_secret_key = os.getenv('aws_secret_key')
# S3_BUCKET = os.getenv("s3_bucket_name")


def create_s3_client(aws_access_key, aws_secret_key, region):
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


def create_s3_resource(aws_access_key, aws_secret_key, region):
    """
    Create a Boto3 S3 client.
    """
    s3_client = boto3.resource(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    return s3_client


def get_latest_folder(s3_client, bucket_name, parent_folder=f'BILLING_PERIOD'):
    """
    Find the latest folder inside the specified parent folder in the S3 bucket.
    """
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=parent_folder, Delimiter='/')

    folders = []
    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            folder_name = prefix['Prefix']
            folders.append(folder_name)

    def parse_folder_name(folder_name):
        # Extract the timestamp part only
        return folder_name.rstrip('/').split('/')[-1]

    latest_folder = max(folders, key=parse_folder_name, default=None)

    if not latest_folder:
        raise FileNotFoundError(f"No folders found in {parent_folder}")

    return latest_folder


def get_parquet_file_from_s3(aws_access_key,
                             aws_secret_key,
                             aws_region,
                             s3_bucket,
                             s3_prefix,
                             export_name,
                             billing_period):
    """
    Get the Parquet file from the latest folder in the specified S3 bucket and return it as a Pandas DataFrame.
    """

    # create s3 client
    s3_client = create_s3_client(aws_access_key=aws_access_key,
                                 aws_secret_key=aws_secret_key,
                                 region=aws_region)
    parent_folder = f'{s3_prefix}/{export_name}/data/BILLING_PERIOD={billing_period}/'

    # Find the latest folder
    latest_folder = get_latest_folder(s3_client, s3_bucket, parent_folder)
    print("latest_folder", latest_folder)

    # List all objects in the latest folder
    objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=latest_folder)

    # Find the first Parquet file
    parquet_file = None
    for obj in objects.get('Contents', []):
        if obj['Key'].endswith('.snappy.parquet'):
            parquet_file = obj['Key']
            break

    if not parquet_file:
        raise FileNotFoundError(f"No Parquet file found in {latest_folder}")

    # Get the object
    obj = s3_client.get_object(Bucket=s3_bucket, Key=parquet_file)
    data = obj['Body'].read()

    df = pd.read_parquet(BytesIO(data))
    return df


def delete_s3_bucket(aws_access_key,
                     aws_secret_key,
                     aws_region,
                     s3_bucket):
    try:
        # create s3 client
        s3_client = create_s3_client(aws_access_key=aws_access_key,
                                     aws_secret_key=aws_secret_key,
                                     region=aws_region)
        objects = s3_client.list_objects_v2(Bucket=s3_bucket)
        fileCount = objects['KeyCount']
        if fileCount == 0:
            s3_client.delete_bucket(Bucket=s3_bucket)
        else:
            for obj in objects.get('Contents', []):
                print(obj)
                s3_client.delete_object(Bucket=s3_bucket, Key=obj["Key"])
            s3_client.delete_bucket(Bucket=s3_bucket)
    except Exception as ex:
        print(ex)
