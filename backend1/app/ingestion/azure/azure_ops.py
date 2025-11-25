import time

import requests, datetime, json, os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from azure.mgmt.resource import ResourceManagementClient


def print_object(obj):
    print(json.dumps(obj, indent=4))


class AutoAuthRequests():
    def __init__(self, azure_tenant_id, azure_client_id, azure_client_secret):
        self.azure_tenant_id = azure_tenant_id
        self.azure_client_id = azure_client_id
        self.azure_client_secret = azure_client_secret
        self.expires_in = datetime.datetime.now().timestamp() * 1000 - 1
        self.access_token, self.expires_in = self.get_access_token()

    def get_access_token(self):
        url = f'https://login.microsoftonline.com/{self.azure_tenant_id}/oauth2/v2.0/token'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'client_id': self.azure_client_id,
            'scope': 'https://management.azure.com/.default',
            'client_secret': self.azure_client_secret,
            'grant_type': 'client_credentials'
        }
        response = requests.post(url, headers=headers, data=data)
        access_token = response.json()['access_token']
        expires_in = response.json()['expires_in']
        return access_token, expires_in

    def get(self, *args, **kwargs):
        if self.expires_in < datetime.datetime.now().timestamp() * 1000:
            self.access_token, self.expires_in = self.get_access_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        kwargs['headers'] = headers
        return requests.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        if self.expires_in < datetime.datetime.now().timestamp() * 1000:
            self.access_token, self.expires_in = self.get_access_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        kwargs['headers'] = headers
        return requests.post(*args, **kwargs)

    def put(self, *args, **kwargs):
        if self.expires_in < datetime.datetime.now().timestamp() * 1000:
            self.access_token, self.expires_in = self.get_access_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        kwargs['headers'] = headers
        return requests.put(*args, **kwargs)

    def delete(self, *args, **kwargs):
        if self.expires_in < datetime.datetime.now().timestamp() * 1000:
            self.access_token, self.expires_in = self.get_access_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        kwargs['headers'] = headers
        return requests.delete(*args, **kwargs)


class AzFunctions():
    def __init__(self, azure_tenant_id, azure_client_id, azure_client_secret):
        self.requests = AutoAuthRequests(azure_tenant_id, azure_client_id, azure_client_secret)
        self.credential = ClientSecretCredential(azure_tenant_id, azure_client_id, azure_client_secret)

    def list_exports(self, subscription_id):
        url = f'https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports?api-version=2023-11-01'
        response = self.requests.get(url)
        response.raise_for_status()
        return response.json()['value']

    def register_cost_management_export(self, provider_namespace, subscription_id):
        resource_client = ResourceManagementClient(self.credential, subscription_id)

        # Check if Microsoft.CostManagementExports is registered
        provider = resource_client.providers.get(provider_namespace)
        print(provider.registration_state)

        if provider.registration_state.lower() == "registered":
            print(f"{provider_namespace} is already registered.")

        else:
            print(f"{provider_namespace} is not registered. Registering now...")
            resource_client.providers.register(provider_namespace)
            print(f"{provider_namespace} registration initiated.")

            status = False
            while not status:
                # Check the registration status again
                provider = resource_client.providers.get(provider_namespace)
                if provider.registration_state.lower() == "registered":
                    print(f"{provider_namespace} is now registered.")
                    break
                else:
                    print(f"{provider_namespace} registration is in progress. Current state: {provider.registration_state}")
                    time.sleep(5)
        return True

    def create_or_update_onetime_export(self, storage_account_resource_id, storage_account_container, subscription_id,
                                        subscription_name, export_name, time_frame, start_date, end_date):
        url = f'https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports/{export_name}?api-version=2023-11-01'
        definition = {
            "location": "global",
            "properties": {
                "schedule": {
                    "status": "Inactive"
                },
                "format": "Csv",
                "deliveryInfo": {
                    "destination": {
                        "resourceId": storage_account_resource_id,
                        "container": storage_account_container,
                        "rootFolderPath": subscription_name
                    }
                },
                "definition": {
                    "type": "ActualCost",
                    "timeframe": time_frame,
                    "timePeriod": {
                        "from": f"{start_date}T00:00:00Z",
                        "to": f"{end_date}T00:00:00Z"
                    },
                    "dataSet": {
                        "granularity": "Daily"
                    }
                }
            }
        }
        if time_frame != 'Custom':
            definition['properties']['definition'].pop('timePeriod')

        response = self.requests.put(url, json=definition)
        if response.status_code not in (200, 201):
            print(url)
            print_object(definition)
            print_object(response.json())
        response.raise_for_status()

    def create_or_update_scheduled_export(self, storage_account_resource_id, storage_account_container, subscription_id,
                                          export_name, status, recurrence, recurrence_start_date, recurrence_end_date,
                                          time_frame, start_date, end_date):
        url = f'https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports/{export_name}?api-version=2023-11-01'
        definition = {
            "location": "global",
            "properties": {
                "schedule": {
                    "status": status,
                    "recurrence": recurrence,
                    "recurrencePeriod": {
                        "from": f"{recurrence_start_date}00:00:00Z",
                        "to": f"{recurrence_end_date}T00:00:00Z"
                    }
                },
                "format": "Csv",
                "deliveryInfo": {
                    "destination": {
                        "resourceId": storage_account_resource_id,
                        "container": storage_account_container,
                        "rootFolderPath": subscription_id
                    }
                },
                "definition": {
                    "type": "ActualCost",
                    "timeframe": time_frame,
                    "timePeriod": {
                        "from": f"{start_date}T00:00:00Z",
                        "to": f"{end_date}T00:00:00Z"
                    },
                    "dataSet": {
                        "granularity": "Daily"
                    }
                }
            }
        }
        if time_frame == 'Custom':
            definition['properties']['definition'].remove('timePeriod')
        response = self.requests.put(url, json=definition)
        response.raise_for_status()

    def run_export(self, subscription_id, export_name):
        url = f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports/{export_name}/run?api-version=2023-11-01"
        response = self.requests.post(url)
        response.raise_for_status()

    def delete_export(self, subscription_id, export_name):
        url = f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports/{export_name}?api-version=2023-11-01"
        response = self.requests.delete(url)
        response.raise_for_status()

    def create_blob_container(self, storage_account_name, container_name):
        """
        Creates a container in the specified Azure Blob Storage account.
        """
        # Create a BlobServiceClient
        blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net",
                                                credential=self.credential)

        # Create the container
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
            print(f"Container '{container_name}' created successfully.")
        except Exception as e:
            print(f"Error creating container: {e}")

    def delete_blob_container(self, storage_account_name, container_name):
        """
        Deletes a container in the specified Azure Blob Storage account.
        """
        # Create a BlobServiceClient
        blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net",
                                                credential=self.credential)

        # Delete the container
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.delete_container()
            print(f"Container '{container_name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting container: {e}")

    def create_focus_export(self, storage_account_resource_id, storage_account_container, subscription_id,
                            export_name, time_frame, start_date):

        url = f'https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports/{export_name}?api-version=2023-07-01-preview'

        definition = {
            "properties": {
                "definition": {
                    "dataSet": {
                        "configuration": {
                            "dataVersion": "1.0"
                        },
                        "granularity": "Daily"
                    },
                    "timeframe": time_frame,  # TheLastBillingMonth, TheLastMonth MonthToDate
                    "type": "FocusCost"
                },
                "deliveryInfo": {
                    "destination": {
                        "container": storage_account_container,
                        "rootFolderPath": "focus",
                        "type": "AzureBlob",
                        "resourceId": storage_account_resource_id
                    }
                },
                "schedule": {
                    "recurrence": "Daily",
                    "recurrencePeriod": {
                        "from": f"{start_date}T00:00:00Z",
                        "to": "2050-02-01T00:00:00.000Z"
                    },
                    "status": "Active"
                },
                "format": "Csv",
                "partitionData": True,
                "dataOverwriteBehavior": "OverwritePreviousReport",
                "compressionMode": "None",
                "exportDescription": ""
            },
            "id": f"/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/exports/{export_name}",
            "name": export_name,
            "type": "Microsoft.CostManagement/reports"
        }

        response = self.requests.put(url, json=definition)
        if response.status_code not in (200, 201):
            print(url)
            print_object(definition)
            print_object(response.json())
        response.raise_for_status()
        print("azure focus export executed")
