from azure.identity import ClientSecretCredential
from azure.mgmt.resource import SubscriptionClient
from azure.core.exceptions import HttpResponseError, ClientAuthenticationError


def azure_validate_creds(azure_client_id: str, azure_client_secret: str, azure_tenant_id: str) -> bool:
    """
    Function to test Azure connection.
    """
    try:
        # Create a credential object using the provided client ID, client secret, and tenant ID
        credential = ClientSecretCredential(
            client_id=azure_client_id,
            client_secret=azure_client_secret,
            tenant_id=azure_tenant_id
        )

        # Attempt to get a token to verify credentials
        token = credential.get_token("https://management.azure.com/.default")

        return True

    except ClientAuthenticationError as e:
        print(f"Azure authentication error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def get_azure_subscriptions(client_id: str, client_secret: str,tenant_id: str):
    try:
        # Authenticate using service principal
        credentials = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        # Initialize the SubscriptionClient
        subscription_client = SubscriptionClient(credentials)
        # Fetch the list of subscriptions
        subscriptions = subscription_client.subscriptions.list()
        subscriptions_list = list(subscriptions)
        if not subscriptions_list:
            return []
        else:
            # Prepare the list of subscriptions
            subscriptions_info = [
                {"subscription_id": sub.subscription_id, "display_name": sub.display_name}
                for sub in subscriptions_list
            ]
            return subscriptions_info
    except HttpResponseError as e:
        raise Exception(f"HTTP response error: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
