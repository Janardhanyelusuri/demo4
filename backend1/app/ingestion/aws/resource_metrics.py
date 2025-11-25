
import boto3
import datetime
import pandas as pd
from sqlalchemy import create_engine, text

def fetch_and_store_cloudwatch_metrics(aws_access_key, aws_secret_key, region,
                                       db_host, db_port, db_user, db_password,
                                       db_name, db_schema, db_table):
    cloudwatch = boto3.client(
        'cloudwatch',
        region_name=region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    services_metrics = {
        'AWS/EC2': ['CPUUtilization', 'DiskReadOps', 'DiskWriteOps', 'NetworkIn', 'NetworkOut', 'StatusCheckFailed'],
        'AWS/RDS': ['CPUUtilization', 'DatabaseConnections', 'FreeableMemory', 'ReadIOPS', 'WriteIOPS', 'FreeStorageSpace'],
        'AWS/S3': ['BucketSizeBytes', 'NumberOfObjects', 'AllRequests', 'GetRequests', 'PutRequests', '4xxErrors', '5xxErrors'],
        'AWS/Lambda': ['Invocations', 'Errors', 'Duration', 'Throttles', 'ProvisionedConcurrencyUtilization'],
        'AWS/ELB': ['RequestCount', 'HealthyHostCount', 'UnHealthyHostCount', 'Latency', 'HTTPCode_ELB_4XX', 'HTTPCode_ELB_5XX'],
        'AWS/DynamoDB': ['ConsumedReadCapacityUnits', 'ConsumedWriteCapacityUnits', 'ThrottledRequests'],
        'AWS/CloudFront': ['Requests', 'BytesDownloaded', 'BytesUploaded', 'TotalErrorRate'],
        'AWS/EBS': ['VolumeReadOps', 'VolumeWriteOps', 'VolumeReadBytes', 'VolumeWriteBytes', 'VolumeQueueLength', 'BurstBalance'],
        'AWS/ECS': ['CPUUtilization', 'MemoryUtilization', 'CPUReservation', 'MemoryReservation'],
        'AWS/ApiGateway': ['Count', '4xxError', '5xxError', 'Latency']
    }

    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(days=14)
    period = 3600  # 1 hour
    metrics_data = []

    for namespace, metrics in services_metrics.items():
        for metric_name in metrics:
            metric_id = metric_name.lower()
            if metric_id[0].isdigit():
                metric_id = "metric_" + metric_id
            try:
                response = cloudwatch.get_metric_data(
                    MetricDataQueries=[
                        {
                            'Id': metric_id,
                            'MetricStat': {
                                'Metric': {
                                    'Namespace': namespace,
                                    'MetricName': metric_name
                                },
                                'Period': period,
                                'Stat': 'Average'
                            },
                            'ReturnData': True
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time
                )
                for result in response['MetricDataResults']:
                    timestamps = result['Timestamps']
                    values = result['Values']
                    for i in range(len(timestamps)):
                        metrics_data.append({
                            'namespace': namespace,
                            'metric_name': metric_name,
                            'timestamp': timestamps[i],
                            'value': values[i]
                        })

            except Exception as e:
                print(f"Error fetching data for {namespace} - {metric_name}: {e}")

    if metrics_data:
        print("Saving CloudWatch data to PostgreSQL...")
        df = pd.DataFrame(metrics_data)

        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE SCHEMA IF NOT EXISTS {db_schema};
                CREATE TABLE IF NOT EXISTS {db_schema}.{db_table} (
                    id SERIAL PRIMARY KEY,
                    namespace TEXT,
                    metric_name TEXT,
                    timestamp TIMESTAMPTZ,
                    value NUMERIC
                );
            """))

        df.to_sql(db_table, engine, schema=db_schema, if_exists='append', index=False)

        print("CloudWatch metrics ingestion complete.")
    else:
        print("No metrics data to save.")
