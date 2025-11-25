import boto3
import datetime
import json
import os
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

current_date = datetime.datetime.now().strftime('%Y-%m-%d')


def create_s3_bucket(bucket_name, region, aws_access_key, aws_secret_key):
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    s3_client = session.client('s3', region_name=region)

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
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    s3_client = session.client('s3')

    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "EnableAWSDataExportsToWriteToS3AndCheckPolicy",
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
        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=bucket_policy_string)
        print(f'Policy added to bucket {bucket_name} successfully.')
    except Exception as e:
        print(f'Error adding bucket policy: {e}')


def bucket(S3_BUCKET, AWS_REGION, aws_access_key, aws_secret_key):
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=AWS_REGION
    )
    s3_client = session.client('s3')

    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        bucket_exists = True
    except:
        bucket_exists = False

    if not bucket_exists:
        create_s3_bucket(S3_BUCKET, AWS_REGION, aws_access_key, aws_secret_key)
        account_id = get_aws_account_id(aws_access_key, aws_secret_key, AWS_REGION)
        add_bucket_policy(S3_BUCKET, account_id, aws_access_key, aws_secret_key, AWS_REGION)
