import boto3
import datetime
import json

# import os

# load_dotenv()

current_date = datetime.datetime.now().strftime('%Y-%m-%d')


def create_boto3_client(aws_access_key, aws_secret_key, region):
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    return session.client('bcm-data-exports')


def create_export(client, export_name, export_description, s3_bucket, s3_prefix, s3_region, query_columns,
                  start_date=None, end_date=None):
    columns = ', '.join(query_columns)
    query_statement = f"SELECT {columns} FROM COST_AND_USAGE_REPORT"

    response = client.create_export(
        Export={
            'Name': export_name,
            'Description': export_description,
            'DataQuery': {
                'QueryStatement': query_statement,
                'TableConfigurations': {
                    'COST_AND_USAGE_REPORT': {
                        'TIME_GRANULARITY': 'DAILY',  # dashbaord filter
                        'INCLUDE_RESOURCES': 'TRUE',
                        'INCLUDE_SPLIT_COST_ALLOCATION_DATA': 'TRUE'
                    }
                }
            },
            'DestinationConfigurations': {
                'S3Destination': {
                    'S3Bucket': s3_bucket,
                    'S3Prefix': f'{s3_prefix}',
                    'S3Region': s3_region,
                    'S3OutputConfigurations': {
                        'Overwrite': 'CREATE_NEW_REPORT',  # OVERWRITE_REPORT
                        'Format': 'PARQUET',
                        'Compression': 'PARQUET',
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
    return export_arn


def get_export_arn_from_file():
    try:
        with open('export_arn.json', 'r') as f:
            data = json.load(f)
            return data.get('export_arn')
    except FileNotFoundError:
        print("Export ARN file not found.")
        return None


def update_export(client, export_name, export_description, s3_bucket, s3_prefix, s3_region, query_columns,
                  start_date=None, end_date=None):
    columns = ', '.join(query_columns)

    query_statement = f"""
    SELECT {columns}
    FROM COST_AND_USAGE_REPORT
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
                        'COST_AND_USAGE_REPORT': {
                            'TIME_GRANULARITY': 'DAILY',
                            'INCLUDE_RESOURCES': 'TRUE',
                            'INCLUDE_SPLIT_COST_ALLOCATION_DATA': 'TRUE'
                        }
                    }
                },
                'DestinationConfigurations': {
                    'S3Destination': {
                        'S3Bucket': s3_bucket,
                        'S3Prefix': f'{s3_prefix}',
                        'S3Region': s3_region,
                        'S3OutputConfigurations': {
                            'Overwrite': 'CREATE_NEW_REPORT',
                            'Format': 'PARQUET',
                            'Compression': 'PARQUET',
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
        print('updated')
    else:
        print("Export ARN not found. Cannot update export.")


def get_export(aws_access_key,
               aws_secret_key,
               aws_region,
               export_name):
    try:
        client = create_boto3_client(aws_access_key, aws_secret_key, aws_region)

        response = client.get_export(
            get_export=export_name
        )
        print(response)
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    return None


def delete_export(aws_access_key,
                  aws_secret_key,
                  aws_region,
                  export_name):
    try:
        client = create_boto3_client(aws_access_key, aws_secret_key, aws_region)
        export_arn = ''
        NextToken = ''
        loop = True
        while loop:
            if NextToken:
                response = client.list_exports(
                    NextToken=NextToken
                )
            else:
                response = client.list_exports()
            print(list(response.keys()))

            if "NextToken" not in list(response.keys()):
                loop = False

            for i in response["Exports"]:
                print(i)
                if i["ExportName"] == export_name:
                    export_arn = i["ExportArn"]
                    loop = False
                    break

        response = client.delete_export(
            ExportArn=export_arn
        )
        print(response)
        print('Export deleted successfully.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
