from .data_to_df import fetch_data_to_dataframe
import os


def run_ce_operations(start_date, end_date, access_key, secret_key):
    """
    Execute AWS Cost Explorer operations.
    """
    METRICS = "AmortizedCost, BlendedCost, NetAmortizedCost, NetUnblendedCost, NormalizedUsageAmount, UnblendedCost, UsageQuantity"
    DIMENSIONS = "OPERATION, SERVICE"
    # Load dimensions and metrics from environment variables
    dimensions = DIMENSIONS.split(',')
    metrics = METRICS.split(',')

    # Fetch data and store in a DataFrame
    df = fetch_data_to_dataframe(dimensions, metrics, start_date, end_date, access_key, secret_key)
    return df  # Return DataFrame for further processing if needed
