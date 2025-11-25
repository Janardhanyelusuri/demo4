import os
import psycopg2
from urllib.parse import urlparse
import datetime
import asyncpg
import databases
import aiohttp
from typing import List
from tortoise.exceptions import DoesNotExist
from app.models.database import Database
from app.models.project import Project
from app.models.resources_tags import ResourceTag
from tortoise import Tortoise
from app.core.config import settings

DB_HOST_NAME = os.getenv("DB_HOST_NAME")
DB_NAME = os.getenv("DB_NAME")
DB_USER_NAME = os.getenv("DB_USER_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


async def init_tortoise_connection():
    return await Tortoise.init(
        db_url=settings.DATABASE_URL,
        modules={"models": [
            "app.models.user",
            "app.models.project",
            "app.models.aws",
            "app.models.azure",
            "app.models.gcp",
            "app.models.project_access",
            "app.models.database",
            "app.models.snowflake",
            "app.models.sync_status",
            "app.models.alert",
            "app.models.alert_integration",
            "app.models.service",
            "app.models.dashboard_request",
            "app.models.dashboard",
            "app.models.tags",
            "app.models.resources",
            "app.models.resources_tags",
]},)


async def close_tortoise_connection():
    return await Tortoise.close_connections()


async def send_message(webhook_url, message, integration_type='slack'):
    async with aiohttp.ClientSession() as session:
        headers = {}
        
        # Customize message formatting for Teams if needed
        if integration_type == 'microsoft_teams':
            message = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.hero",
                        "content": {
                            "title": message.get("title", "Alert Notification"),
                            "text": message.get("text")
                        }
                    }
                ]
            }
        
        async with session.post(webhook_url, json=message, headers=headers) as response:
            if response.status == 200:
                print(f"Message sent to {integration_type.capitalize()} successfully.")
            else:
                print(f"Failed to send message to {integration_type.capitalize()}: {response.status}")


async def fetch_data_from_database(connection_string: str, query: str):
    conn = await asyncpg.connect(dsn=connection_string)
    try:
        result = await conn.fetch(query)
        data = [dict(record) for record in result]
    finally:
        await conn.close()
    return data


async def fetch_data(connection_string: str, query: str) -> List[str]:
    database = databases.Database(connection_string)
    await database.connect()
    results = await database.fetch_all(query=query)
    await database.disconnect()
    return [row[0] for row in results]


# Function to create project and database
async def create_project_and_database(project_name: str, cloud_platform: str, aws_access_key: str, aws_secret_key: str) -> Database:
    try:
        # Create the project
        project = await Project.create(
            name=project_name,
            status=True,
            date=datetime.date.today(),
            cloud_platform=cloud_platform
        )

        # Create the database for the project
        connection_string = f"postgresql://{aws_access_key}:{aws_secret_key}@your-db-host/{project_name}"
        
        # Connect to the PostgreSQL server and create the database
        conn = await asyncpg.connect(dsn=os.environ.get("DATABASE_URL"))
        await conn.execute(f"CREATE DATABASE {project_name}")
        await conn.close()
        
        # Store the database details in the Database table
        database = await Database.create(
            name=project_name,
            connection_string=connection_string,
            project_id=project.id  # Use the project ID for the foreign key
        )

        return database

    except Exception as e:
        print(f"Error creating project and database: {e}")
        raise


def connection(func):
    def wrapper(*args, **kwargs):
        connection = None
        try:
            # Parse the database URL
            url = urlparse(os.environ.get("DATABASE_URL"))
            print(url)
            connection = psycopg2.connect(
                host=url.hostname,
                database=url.path[1:],
                user=url.username,
                password=url.password,
                port=url.port,
                # sslmode='require'
            )
            print("Connection established successfully")

            result = func(connection, *args, **kwargs)

            return result

        except Exception as error:
            print(f'Error connecting to the database: {error}')

        finally:
            if connection:
                connection.close()
                print("Connection closed.")

    return wrapper


@connection
def execute_query(connection,
                  query,
                  fetch=True):
    result = []
    try:
        cursor = connection.cursor()

        # Execute a query
        cursor.execute(query)

        # fetch result for select queries, for delete, insert, update send fetch = False
        if fetch:
            result = cursor.fetchall()

        # Commit the transaction
        connection.commit()

        # Close the cursor
        cursor.close()
    except Exception as ex:
        print(ex)
    return result


async def fetch_resources_by_tag(tag_id):
    """Fetch resources associated with a tag ID."""
    await init_tortoise_connection()

    # Example PostgreSQL connection using asyncpg
    conn = await asyncpg.connect(user=DB_USER_NAME, password=DB_PASSWORD, database=DB_NAME, host=DB_HOST_NAME)

    try:
        # Fetch resources associated with the tag
        resource_tags = await ResourceTag.filter(tag_id=tag_id).prefetch_related('resource')

        # If no resource_tags found, return an empty list
        if not resource_tags:
            return []

        # Extract the resource names from the related resources
        resource_list = [resource_tag.resource.resource_name for resource_tag in resource_tags]
        
        await conn.close()
        await close_tortoise_connection()

        # Return the resource names as a list of strings
        return resource_list
        
    except DoesNotExist:
        return []  # If the tag does not exist, return an empty list
    except Exception as e:
        # Log the error if needed (optional)
        print(f"Error fetching resources by tag: {e}")
        return []  # Return empty list on any other exception


async def fetch_resource_id_by_tag(tag_id):
    """Fetch resources associated with a tag ID."""
    await init_tortoise_connection()

    # Example PostgreSQL connection using asyncpg
    conn = await asyncpg.connect(user=DB_USER_NAME, password=DB_PASSWORD, database=DB_NAME, host=DB_HOST_NAME)

    try:
        # Fetch resources associated with the tag
        resource_tags = await ResourceTag.filter(tag_id=tag_id).prefetch_related('resource')

        # If no resource_tags found, return an empty list
        if not resource_tags:
            return []

        # Extract the resource names from the related resources
        resource_list = [resource_tag.resource.resource_name for resource_tag in resource_tags]
        
        await conn.close()
        await close_tortoise_connection()

        # Return the resource names as a list of strings
        return resource_list
        
    except DoesNotExist:
        return []  # If the tag does not exist, return an empty list
    except Exception as e:
        # Log the error if needed (optional)
        print(f"Error fetching resources by tag: {e}")
        return []  # Return empty list on any other exception
    

async def build_query(alert_data, schema_name, cloud_platform):

    condition_map = {
        "Less than": "<",
        "Greater than": ">",
        "Equal to": "=",
        "Not equal to": "!=",
        "Greater than equal to": ">=",
        "Less than equal to": "<="
    }

    operation_map = {
        "SUM": "SUM",
        "AVERAGE": "AVG",
        "COUNT": "COUNT"
    }

    alert_type = alert_data['alert_type']
    condition_operator = condition_map[alert_data['condition']]
    operator = operation_map[alert_data['operation']]
    value_threshold = alert_data.get('value_threshold')
    percentage_threshold = alert_data.get('percentage_threshold')
    schedule = alert_data.get('schedule')

    tag_id = alert_data.get('tag_id')
    
    # Determine resource names based on tag_id or provided resource_list
    if tag_id:
        if cloud_platform == 'aws':
            resource_list = await fetch_resource_id_by_tag(tag_id)
        else:
            resource_list = await fetch_resources_by_tag(tag_id)
    else:
        resource_list = alert_data.get('resource_list', [])
    
    # Create resource names string for SQL query
    resource_names = '{' + ','.join(f'"{r}"' for r in resource_list) + '}'
    query = ""

    if cloud_platform == 'azure':
        if alert_type == "Cost":
            if resource_list:
                if value_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{condition_operator}' AS operator_value
                            ),
                            threshold AS (
                                SELECT {value_threshold} AS threshold_value
                            ),
                            resource_list AS (
                                SELECT unnest('{resource_names}'::text[]) AS resource_name
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    {operator}(billed_cost) AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost,
                                    granularity,
                                    resource_list
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN granularity.granularity_value = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN granularity.granularity_value = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_azure_fact_cost WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN granularity.granularity_value = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                                    AND resource_name = ANY(SELECT resource_name FROM resource_list)
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT threshold_value FROM threshold) AS threshold_value,
                                total_billed_cost - (SELECT threshold_value FROM threshold) AS difference,
                                CASE 
                                    WHEN (SELECT operator_value FROM operator) = '>' AND total_billed_cost > (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '=' AND total_billed_cost = (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '<' AND total_billed_cost < (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '>=' AND total_billed_cost >= (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '<=' AND total_billed_cost <= (SELECT threshold_value FROM threshold) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;
                            """
                elif percentage_threshold is not None:
                    query = f"""-- Cost alert with percentage threshold and resource list"""
            else:
                if value_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value 
                            ),
                            operator AS (
                                SELECT '{condition_operator}' AS operator_value
                            ),
                            threshold AS (
                                SELECT {value_threshold} AS threshold_value
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    {operator}(billed_cost) AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost,
                                    granularity
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN granularity.granularity_value = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN granularity.granularity_value = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_azure_fact_cost WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN granularity.granularity_value = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT threshold_value FROM threshold) AS threshold_value,
                                total_billed_cost - (SELECT threshold_value FROM threshold) AS difference,
                                CASE 
                                    WHEN (SELECT operator_value FROM operator) = '>' AND total_billed_cost > (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '=' AND total_billed_cost = (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '<' AND total_billed_cost < (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '>=' AND total_billed_cost >= (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '<=' AND total_billed_cost <= (SELECT threshold_value FROM threshold) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;
                            """
                    
                elif percentage_threshold is not None:
                    query = f"""-- Cost alert with percentage threshold and no resource list"""

        elif alert_type == "Spike":
            if resource_list:
                if value_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{operator}' AS aggregation_operator
                            ),
                            comparison AS (
                                SELECT '{condition_operator}' AS condition_operator
                            ),
                            threshold AS (
                                SELECT {value_threshold} AS percentage_threshold_value
                            ),
                            resource_list AS (
                                SELECT unnest('{resource_names}'::text[]) AS resource_name
                            ),
                            HistoricalCost AS (
                                SELECT 
                                    {operator}(billed_cost) AS avg_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost AS cost
                                JOIN 
                                    {schema_name}.gold_azure_resource_dim AS dim 
                                ON 
                                    cost.resource_id = dim.resource_id
                                JOIN 
                                    resource_list
                                ON 
                                    dim.resource_name = resource_list.resource_name
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE - INTERVAL '1 day'
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                    END
                                    AND charge_period_start < CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    CASE 
                                        WHEN (SELECT aggregation_operator FROM operator) = 'SUM' THEN SUM(billed_cost)
                                        WHEN (SELECT aggregation_operator FROM operator) = 'AVG' THEN AVG(billed_cost)
                                        ELSE NULL
                                    END AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost AS cost
                                JOIN 
                                    {schema_name}.gold_azure_resource_dim AS dim
                                ON 
                                    cost.resource_id = dim.resource_id
                                JOIN 
                                    resource_list
                                ON 
                                    dim.resource_name = resource_list.resource_name
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_azure_fact_cost WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT avg_billed_cost FROM HistoricalCost) AS avg_billed_cost,
                                (SELECT percentage_threshold_value FROM threshold) AS percentage_threshold,
                                ((SELECT avg_billed_cost FROM HistoricalCost) * ((SELECT percentage_threshold_value FROM threshold) / 100.0)) AS spike_threshold_amount,
                                total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost) AS spike_difference,
                                CASE 
                                    WHEN (SELECT avg_billed_cost FROM HistoricalCost) = 0 THEN NULL
                                    ELSE (total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost)) / (SELECT avg_billed_cost FROM HistoricalCost) * 100
                                END AS spike_percentage,
                                CASE 
                                    WHEN (SELECT condition_operator FROM comparison) = '>' AND total_billed_cost > ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '=' AND total_billed_cost = ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '<' AND total_billed_cost < ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '>=' AND total_billed_cost >= ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '<=' AND total_billed_cost <= ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;"""
                elif percentage_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{operator}' AS aggregation_operator
                            ),
                            comparison AS (
                                SELECT '{condition_operator}' AS condition_operator
                            ),
                            value_threshold AS (
                                SELECT {percentage_threshold} AS allowed_spike_value 
                            ),
                            resource_list AS (
                                SELECT unnest('{resource_names}'::text[]) AS resource_name 
                            ),
                            HistoricalCost AS (
                                SELECT 
                                    {operator}(billed_cost) AS avg_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost AS cost
                                JOIN 
                                    {schema_name}.gold_azure_resource_dim AS dim
                                ON 
                                    cost.resource_id = dim.resource_id
                                JOIN 
                                    resource_list
                                ON 
                                    dim.resource_name = resource_list.resource_name
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE - INTERVAL '1 day'
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                    END
                                    AND charge_period_start < CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    CASE 
                                        WHEN (SELECT aggregation_operator FROM operator) = 'SUM' THEN SUM(billed_cost)
                                        WHEN (SELECT aggregation_operator FROM operator) = 'AVG' THEN AVG(billed_cost)
                                        ELSE NULL
                                    END AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost AS cost
                                JOIN 
                                    {schema_name}.gold_azure_resource_dim AS dim 
                                ON 
                                    cost.resource_id = dim.resource_id
                                JOIN 
                                    resource_list
                                ON 
                                    dim.resource_name = resource_list.resource_name
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_azure_fact_cost WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT avg_billed_cost FROM HistoricalCost) AS avg_billed_cost,
                                (SELECT allowed_spike_value FROM value_threshold) AS allowed_spike_value,
                                CASE 
                                    WHEN total_billed_cost > (SELECT avg_billed_cost FROM HistoricalCost) THEN total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost)
                                    ELSE 0 
                                END AS actual_spike,
                                CASE 
                                    WHEN total_billed_cost > ((SELECT avg_billed_cost FROM HistoricalCost) + (SELECT allowed_spike_value FROM value_threshold)) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;"""
            else:
                if value_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{operator}' AS aggregation_operator
                            ),
                            comparison AS (
                                SELECT '{condition_operator}' AS condition_operator
                            ),
                            threshold AS (
                                SELECT {value_threshold} AS percentage_threshold_value
                            ),
                            HistoricalCost AS (
                                SELECT 
                                    {operator}(billed_cost) AS avg_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE - INTERVAL '1 day'
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                    END
                                    AND charge_period_start < CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    CASE 
                                        WHEN (SELECT aggregation_operator FROM operator) = 'SUM' THEN SUM(billed_cost)
                                        WHEN (SELECT aggregation_operator FROM operator) = 'AVG' THEN AVG(billed_cost)
                                        ELSE NULL
                                    END AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_azure_fact_cost WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT avg_billed_cost FROM HistoricalCost) AS avg_billed_cost,
                                (SELECT percentage_threshold_value FROM threshold) AS percentage_threshold,
                                ((SELECT avg_billed_cost FROM HistoricalCost) * ((SELECT percentage_threshold_value FROM threshold) / 100.0)) AS spike_threshold_amount,
                                total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost) AS spike_difference,
                                CASE 
                                    WHEN (SELECT avg_billed_cost FROM HistoricalCost) = 0 THEN NULL
                                    ELSE (total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost)) / (SELECT avg_billed_cost FROM HistoricalCost) * 100
                                END AS spike_percentage,
                                CASE 
                                    WHEN (SELECT condition_operator FROM comparison) = '>' AND total_billed_cost > ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '=' AND total_billed_cost = ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '<' AND total_billed_cost < ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '>=' AND total_billed_cost >= ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '<=' AND total_billed_cost <= ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;"""
                elif percentage_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{operator}' AS aggregation_operator
                            ),
                            comparison AS (
                                SELECT '{condition_operator}' AS condition_operator
                            ),
                            value_threshold AS (
                                SELECT {percentage_threshold} AS allowed_spike_value
                            ),
                            HistoricalCost AS (
                                SELECT 
                                    {operator}(billed_cost) AS avg_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE - INTERVAL '1 day'
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                    END
                                    AND charge_period_start < CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    CASE 
                                        WHEN (SELECT aggregation_operator FROM operator) = 'SUM' THEN SUM(billed_cost)
                                        WHEN (SELECT aggregation_operator FROM operator) = 'AVG' THEN AVG(billed_cost)
                                        ELSE NULL
                                    END AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_azure_fact_cost
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_azure_fact_cost WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT avg_billed_cost FROM HistoricalCost) AS avg_billed_cost,
                                (SELECT allowed_spike_value FROM value_threshold) AS allowed_spike_value,
                                CASE 
                                    WHEN total_billed_cost > (SELECT avg_billed_cost FROM HistoricalCost) THEN total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost)
                                    ELSE 0 
                                END AS actual_spike,
                                CASE 
                                    WHEN total_billed_cost > ((SELECT avg_billed_cost FROM HistoricalCost) + (SELECT allowed_spike_value FROM value_threshold)) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;"""

    elif cloud_platform == 'aws':
        if alert_type == "Cost":
            if resource_list:
                if value_threshold is not None:
                    query = f"""
                            """
                elif percentage_threshold is not None:
                    query = f"""-- Cost alert with percentage threshold and resource list"""
            else:
                if value_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value 
                            ),
                            operator AS (
                                SELECT '{condition_operator}' AS operator_value
                            ),
                            threshold AS (
                                SELECT {value_threshold} AS threshold_value
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    {operator}(list_cost) AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_aws_fact_focus,
                                    granularity
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN granularity.granularity_value = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN granularity.granularity_value = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_aws_fact_focus WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN granularity.granularity_value = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT threshold_value FROM threshold) AS threshold_value,
                                total_billed_cost - (SELECT threshold_value FROM threshold) AS difference,
                                CASE 
                                    WHEN (SELECT operator_value FROM operator) = '>' AND total_billed_cost > (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '=' AND total_billed_cost = (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '<' AND total_billed_cost < (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '>=' AND total_billed_cost >= (SELECT threshold_value FROM threshold) THEN TRUE
                                    WHEN (SELECT operator_value FROM operator) = '<=' AND total_billed_cost <= (SELECT threshold_value FROM threshold) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;
                            """
                    
                elif percentage_threshold is not None:
                    query = f"""-- Cost alert with percentage threshold and no resource list"""

        elif alert_type == "Spike":
            if resource_list:
                if value_threshold is not None:
                    query = f""" 
                            """
                elif percentage_threshold is not None:
                    query = f"""
                            """
            else:
                if value_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{operator}' AS aggregation_operator
                            ),
                            comparison AS (
                                SELECT '{condition_operator}' AS condition_operator
                            ),
                            threshold AS (
                                SELECT {value_threshold} AS percentage_threshold_value
                            ),
                            HistoricalCost AS (
                                SELECT 
                                    {operator}(list_cost) AS avg_billed_cost
                                FROM 
                                    {schema_name}.gold_aws_fact_focus
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE - INTERVAL '1 day'
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                    END
                                    AND charge_period_start < CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    CASE 
                                        WHEN (SELECT aggregation_operator FROM operator) = 'SUM' THEN SUM(billed_cost)
                                        WHEN (SELECT aggregation_operator FROM operator) = 'AVG' THEN AVG(billed_cost)
                                        ELSE NULL
                                    END AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_aws_fact_focus
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_aws_fact_focus WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT avg_billed_cost FROM HistoricalCost) AS avg_billed_cost,
                                (SELECT percentage_threshold_value FROM threshold) AS percentage_threshold,
                                ((SELECT avg_billed_cost FROM HistoricalCost) * ((SELECT percentage_threshold_value FROM threshold) / 100.0)) AS spike_threshold_amount,
                                total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost) AS spike_difference,
                                CASE 
                                    WHEN (SELECT avg_billed_cost FROM HistoricalCost) = 0 THEN NULL
                                    ELSE (total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost)) / (SELECT avg_billed_cost FROM HistoricalCost) * 100
                                END AS spike_percentage,
                                CASE 
                                    WHEN (SELECT condition_operator FROM comparison) = '>' AND total_billed_cost > ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '=' AND total_billed_cost = ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '<' AND total_billed_cost < ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '>=' AND total_billed_cost >= ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    WHEN (SELECT condition_operator FROM comparison) = '<=' AND total_billed_cost <= ((SELECT avg_billed_cost FROM HistoricalCost) + ((SELECT avg_billed_cost FROM HistoricalCost) * (SELECT percentage_threshold_value FROM threshold) / 100.0)) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;"""
                elif percentage_threshold is not None:
                    query = f"""
                            WITH granularity AS (
                                SELECT '{schedule}' AS granularity_value
                            ),
                            operator AS (
                                SELECT '{operator}' AS aggregation_operator
                            ),
                            comparison AS (
                                SELECT '{condition_operator}' AS condition_operator
                            ),
                            value_threshold AS (
                                SELECT {percentage_threshold} AS allowed_spike_value
                            ),
                            HistoricalCost AS (
                                SELECT 
                                    {operator}(list_cost) AS avg_billed_cost
                                FROM 
                                    {schema_name}.gold_aws_fact_focus
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE - INTERVAL '1 week')
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE - INTERVAL '1 day'
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                    END
                                    AND charge_period_start < CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN CURRENT_DATE
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            ),
                            AggregatedCost AS (
                                SELECT 
                                    CASE 
                                        WHEN (SELECT aggregation_operator FROM operator) = 'SUM' THEN SUM(billed_cost)
                                        WHEN (SELECT aggregation_operator FROM operator) = 'AVG' THEN AVG(billed_cost)
                                        ELSE NULL
                                    END AS total_billed_cost
                                FROM 
                                    {schema_name}.gold_aws_fact_focus
                                WHERE 
                                    charge_period_start >= CASE 
                                        WHEN (SELECT granularity_value FROM granularity) = 'Weekly' THEN DATE_TRUNC('week', CURRENT_DATE)
                                        WHEN (SELECT granularity_value FROM granularity) = 'Daily' THEN 
                                            CASE 
                                                WHEN EXISTS (SELECT 1 FROM {schema_name}.gold_aws_fact_focus WHERE charge_period_start = CURRENT_DATE) 
                                                THEN CURRENT_DATE 
                                                ELSE CURRENT_DATE - INTERVAL '1 day'
                                            END
                                        WHEN (SELECT granularity_value FROM granularity) = 'Monthly' THEN DATE_TRUNC('month', CURRENT_DATE)
                                    END
                            )
                            SELECT 
                                total_billed_cost, 
                                (SELECT avg_billed_cost FROM HistoricalCost) AS avg_billed_cost,
                                (SELECT allowed_spike_value FROM value_threshold) AS allowed_spike_value,
                                CASE 
                                    WHEN total_billed_cost > (SELECT avg_billed_cost FROM HistoricalCost) THEN total_billed_cost - (SELECT avg_billed_cost FROM HistoricalCost)
                                    ELSE 0 
                                END AS actual_spike,
                                CASE 
                                    WHEN total_billed_cost > ((SELECT avg_billed_cost FROM HistoricalCost) + (SELECT allowed_spike_value FROM value_threshold)) THEN TRUE
                                    ELSE FALSE 
                                END AS trigger
                            FROM AggregatedCost;"""
    elif cloud_platform == 'gcp':
        pass
    # Add other alert types as necessary

    return query