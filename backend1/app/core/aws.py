import boto3
import json
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from azure.identity import ClientSecretCredential
from azure.core.exceptions import ClientAuthenticationError


def aws_validate_creds(aws_access_key: str, aws_secret_key: str) -> bool:
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        s3_client = session.client('s3')
        # Attempt to list buckets to verify credentials
        s3_client.list_buckets()
        return True
    except (NoCredentialsError, PartialCredentialsError):
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
