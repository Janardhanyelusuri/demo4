from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import pandas as pd
from io import StringIO

load_dotenv()


def get_df_from_blob(tenant_id, client_id, client_secret, storage_account_name, container_name):
    """
    Combines all CSV files in the specified Azure Blob Storage container into a single DataFrame.

    Args:
        tenant_id (str): The Azure Active Directory tenant ID.
        client_id (str): The Azure Active Directory client ID.
        client_secret (str): The Azure Active Directory client secret.
        storage_account_name (str): The name of the Azure Storage account.
        container_name (str): The name of the container within the storage account.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all CSV files in the container.
    """
    # Authenticate using the ClientSecretCredential
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    # Create a BlobServiceClient and get a reference to the container object
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net",
                                            credential=credential)
    blob_container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container
    blob_list = blob_container_client.list_blobs()

    # Filter out only CSV files
    csv_blobs = [blob for blob in blob_list if blob.name.endswith('.csv')]

    # Initialize an empty list to hold the DataFrames
    dataframes = []

    # Download each CSV file and read it into a DataFrame
    for blob in csv_blobs:
        blob_client = blob_container_client.get_blob_client(blob)
        csv_content = blob_client.download_blob().readall().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        dataframes.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    # If the 'Tags' column is empty, its data type changes to double precision,
    # so we need to convert it to string to ensure consistent processing.
    if 'Tags' in combined_dataframe.columns:
        if combined_dataframe['Tags'].dtype != 'object':  # Not text
            combined_dataframe['Tags'] = combined_dataframe['Tags'].astype(str)  # Convert to string

        # Replace NaN with empty JSON object and ensure proper formatting
        combined_dataframe['Tags'] = combined_dataframe['Tags'].fillna('{}').astype(str)
        combined_dataframe['Tags'] = combined_dataframe['Tags'].replace('nan', '{}')

        # Validate and correct improper JSON formats (e.g., replacing single quotes with double quotes)
        combined_dataframe['Tags'] = combined_dataframe['Tags'].apply(lambda x: x if x == '{}' else x.replace("'", '"'))
    
    print("Data type of 'Tags' column:", combined_dataframe['Tags'].dtypes)


    return combined_dataframe
