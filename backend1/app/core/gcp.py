import json
from typing import List
from fastapi import HTTPException
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from google.cloud import storage, billing_v1, bigquery
from google.auth.exceptions import GoogleAuthError


def get_gcp_credentials_from_file(credentials_file):
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
            return credentials
    except FileNotFoundError:
        print(f"File not found: {credentials_file}")
        return None
    except json.JSONDecodeError:
        print("Invalid JSON format in the credentials file.")
        return None


def gcp_validate_creds(credentials_file: str) -> bool:
    """
    Function to test GCP connection by obtaining an access token.
    """
    try:
        credentials_obj = service_account.Credentials.from_service_account_info(credentials_file)
        project_id = credentials_obj.project_id
        # commenting below code as all gcp account/credentials don't have storage list bucket access
        # client = storage.Client(credentials=credentials_obj, project=project_id)
        # # Try to list buckets to check the connection
        # list(client.list_buckets())
        return True
    except GoogleAuthError as e:
        print(f"Google authentication error: {e}")
        return False
    except Exception as e:
        print(f"Failed to connect to GCP: {e}")
        return False


def list_projects(credentials_file: dict) -> List[dict]:
    try:
        credentials_obj = service_account.Credentials.from_service_account_info(credentials_file)
        service = discovery.build('cloudresourcemanager', 'v1', credentials=credentials_obj)
        request = service.projects().list()
        response = request.execute()

        projects = response.get('projects', [])
        if not projects:
            raise HTTPException(status_code=404, detail="No projects found.")
        
        return projects
    except HttpError as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


def list_billing_account_ids(credentials_file: dict):
    items = []
    try:
        # Create a credentials object from the service account key
        credentials = service_account.Credentials.from_service_account_info(credentials_file)

        # Create a billing client
        client = billing_v1.CloudBillingClient(credentials=credentials)

        # List all billing accounts
        billing_accounts = client.list_billing_accounts()

        # Print out the billing account IDs
        for billing_account in billing_accounts:
            print(billing_account)
        return billing_accounts
    except Exception as ex:
        print(ex)
    return items


def list_datasets(credentials_file: dict):
    items = []
    try:
        # Create a credentials object from the service account key
        credentials = service_account.Credentials.from_service_account_info(credentials_file)

        # Create a BigQuery client
        client = bigquery.Client(credentials=credentials)

        # List datasets in the project
        datasets = client.list_datasets()

        for dataset in datasets:
            items.append(dataset.dataset_id)
    except Exception as ex:
        print(ex)
    return items
