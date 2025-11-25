import boto3
import datetime
import json

current_date = datetime.datetime.now().strftime('%Y-%m-%d')


def create_boto3_client(aws_access_key, aws_secret_key, region):
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    return session.client('bcm-data-exports')  # Ensure this is the correct service name for Focus Exports


def create_export(client, export_name, export_description, s3_bucket, s3_prefix, s3_region, start_date=None,
                  end_date=None, focus_columns=None):
    # Define the query statement
    query_statement = """
            SELECT AvailabilityZone, BilledCost, BillingAccountId, 
            BillingAccountName, BillingCurrency, BillingPeriodEnd, BillingPeriodStart, ChargeCategory, ChargeClass, 
            ChargeDescription, ChargeFrequency, ChargePeriodEnd, ChargePeriodStart, CommitmentDiscountCategory,
            CommitmentDiscountId, CommitmentDiscountName, CommitmentDiscountStatus, CommitmentDiscountType, ConsumedQuantity, 
            ConsumedUnit, ContractedCost, ContractedUnitPrice, EffectiveCost, InvoiceIssuerName, ListCost, ListUnitPrice, 
            PricingCategory, PricingQuantity, PricingUnit, ProviderName, PublisherName, RegionId, RegionName, ResourceId, 
            ResourceName, ResourceType, ServiceCategory, ServiceName, SkuId, SkuPriceId, SubAccountId, SubAccountName, 
            Tags, x_CostCategories, x_Discounts, x_Operation, x_ServiceCode, x_UsageType FROM FOCUS_1_0_AWS
        """

    response = client.create_export(
        Export={
            'Name': export_name,
            'Description': export_description,
            'DataQuery': {
                'QueryStatement': query_statement,
                'TableConfigurations': {
                    'FOCUS_1_0_AWS': {}
                }
            },
            'DestinationConfigurations': {
                'S3Destination': {
                    'S3Bucket': s3_bucket,
                    'S3Prefix': f'{s3_prefix}',
                    'S3Region': s3_region,
                    'S3OutputConfigurations': {
                        'Overwrite': 'CREATE_NEW_REPORT',
                        'Format': 'TEXT_OR_CSV',  # Updated to TEXT_OR_CSV from PARQUET
                        'Compression': 'GZIP',  # Updated to GZIP from PARQUET
                        'OutputType': 'CUSTOM'
                    }
                }
            },
            'RefreshCadence': {
                'Frequency': 'SYNCHRONOUS'
            }
        }
    )

    print(response)
    export_arn = response['ExportArn']
    with open('export_arn.json', 'w') as f:
        json.dump({'export_arn': export_arn}, f)
    print("Export ARN stored successfully.")


def get_export_arn_from_file():
    try:
        with open('export_arn.json', 'r') as f:
            data = json.load(f)
            return data.get('export_arn')
    except FileNotFoundError:
        print("Export ARN file not found.")
        return None


def update_export(client, export_name, export_description, s3_bucket, s3_prefix, s3_region, start_date=None,
                  end_date=None, focus_columns=None):
    query_statement = """
                SELECT AvailabilityZone, BilledCost, BillingAccountId, 
                BillingAccountName, BillingCurrency, BillingPeriodEnd, BillingPeriodStart, ChargeCategory, ChargeClass, 
                ChargeDescription, ChargeFrequency, ChargePeriodEnd, ChargePeriodStart, CommitmentDiscountCategory,
                CommitmentDiscountId, CommitmentDiscountName, CommitmentDiscountStatus, CommitmentDiscountType, ConsumedQuantity, 
                ConsumedUnit, ContractedCost, ContractedUnitPrice, EffectiveCost, InvoiceIssuerName, ListCost, ListUnitPrice, 
                PricingCategory, PricingQuantity, PricingUnit, ProviderName, PublisherName, RegionId, RegionName, ResourceId, 
                ResourceName, ResourceType, ServiceCategory, ServiceName, SkuId, SkuPriceId, SubAccountId, SubAccountName, 
                Tags, x_CostCategories, x_Discounts, x_Operation, x_ServiceCode, x_UsageType FROM FOCUS_1_0_AWS
            """

    export_arn = get_export_arn_from_file()
    if export_arn:
        response = client.update_export(
            ExportArn=export_arn,
            Export={
                'Name': export_name,
                'Description': export_description,
                'DataQuery': {
                    'QueryStatement': query_statement,
                    'TableConfigurations': {
                        'FOCUS_1_0_AWS': {}
                    }
                },
                'DestinationConfigurations': {
                    'S3Destination': {
                        'S3Bucket': s3_bucket,
                        'S3Prefix': f'{s3_prefix}',
                        'S3Region': s3_region,
                        'S3OutputConfigurations': {
                            'Overwrite': 'CREATE_NEW_REPORT',
                            'Format': 'TEXT_OR_CSV',  # Updated to TEXT_OR_CSV from PARQUET
                            'Compression': 'GZIP',  # Updated to GZIP from PARQUET
                            'OutputType': 'CUSTOM'
                        }
                    }
                },
                'RefreshCadence': {
                    'Frequency': 'SYNCHRONOUS'
                }
            }
        )
        print(response)
        print('Export updated successfully.')
    else:
        print("Export ARN not found. Cannot update export.")
