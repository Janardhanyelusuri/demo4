import boto3
import pandas as pd
import os


def fetch_data_to_dataframe(dimensions, metrics, start_date, end_date, access_key, secret_key):
    """
    Fetch AWS Cost Explorer data and store in a Pandas DataFrame.
    """
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    ce_client = session.client('ce', region_name='us-east-1')

    response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=metrics,
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': dimension} for dimension in dimensions
        ]
    )

    data = []
    for result in response['ResultsByTime']:
        time_period = f"{result['TimePeriod']['Start']} - {result['TimePeriod']['End']}"
        for group in result['Groups']:
            dimension_values = group['Keys']
            metrics_data = group['Metrics']
            row = [time_period] + dimension_values + [metrics_data.get(metric, {}).get('Amount', '') for metric in metrics]
            data.append(row)

    header = ['TimePeriod'] + dimensions + metrics
    df = pd.DataFrame(data, columns=header)
    return df
