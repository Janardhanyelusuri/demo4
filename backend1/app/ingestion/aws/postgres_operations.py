import psycopg2
from psycopg2 import sql
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_HOST_NAME = os.getenv("DB_HOST_NAME")
DB_NAME = os.getenv("DB_NAME")
DB_USER_NAME = os.getenv("DB_USER_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


# Decorator to create a connection with PostgreSQL server
def connection(func):
    def wrapper(*args, **kwargs):
        hostname = DB_HOST_NAME
        database = DB_NAME
        username = DB_USER_NAME
        password = DB_PASSWORD
        port = DB_PORT

        connection = None

        try:
            # Establish connection
            connection = psycopg2.connect(
                host=hostname,
                database=database,
                user=username,
                password=password,
                port=port,
                sslmode='require'
            )
            print("Connection established successfully")

            try:
                # Pass the connection to the wrapped function
                result = func(connection, *args, **kwargs)

                # Commit the transaction after function execution
                connection.commit()
                return result

            except Exception as inner_error:
                # Roll back in case of an error
                connection.rollback()
                print(f"Transaction failed and rolled back: {inner_error}")
                raise

        except Exception as outer_error:
            print(f"Error connecting to the database: {outer_error}")
            raise

        finally:
            # Ensure the connection is closed
            if connection:
                connection.close()
                print("Connection closed.")

    return wrapper

@connection
def create_schemas(connection, schema):
    try:
        cursor = connection.cursor()
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        connection.commit()
        print("Schemas created successfully")
        cursor.close()
    except Exception as error:
        print(f'Error creating schemas: {error}')


# @connection
# def dump_to_postgresql(connection, new_data, schema_name, table_name):
#     from psycopg2.extras import execute_values  # Import this utility for bulk inserts
#     try:
#         # Replace empty strings with None for all columns
#         new_data = new_data.replace("", None)

#         # Prepare the data as a list of tuples
#         records = [tuple(row) for row in new_data.to_numpy()]

#         # Debugging: Print some sample records to inspect data
#         # print(f"First 5 records to insert: {records[:5]}")

#         # Get the column names and ensure they are properly quoted
#         columns = ', '.join([f'"{col}"' for col in new_data.columns])

#         # Generate the SQL query for bulk insert
#         insert_query = f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES %s"

#         cursor = connection.cursor()
#         # Use execute_values for efficient bulk insert
#         execute_values(cursor, insert_query, records)
#         connection.commit()  # Commit transaction
#         print(f"Data dumped into {schema_name}.{table_name} table successfully.")

#     except Exception as e:
#         print(f"Error dumping data into {schema_name}.{table_name} table: {e}")
#         connection.rollback()  # Rollback on error
#         raise

from psycopg2.extras import execute_values

@connection
def dump_to_postgresql(connection, new_data, schema_name, table_name):
    try:
        new_data = new_data.replace("", None)
        records = [tuple(row) for row in new_data.to_numpy()]
        columns = ', '.join([f'"{col}"' for col in new_data.columns])
        insert_query = f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES %s"
        
        cursor = connection.cursor()
        # Use batching with page_size to boost speed
        execute_values(cursor, insert_query, records, page_size=10000)
        connection.commit()
        print(f"Data dumped into {schema_name}.{table_name} table successfully.")
    except Exception as e:
        print(f"Error dumping data into {schema_name}.{table_name} table: {e}")
        connection.rollback()
        raise


@connection
def get_tables_in_schema(connection, schema_name):
    try:
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL("SELECT table_name FROM information_schema.tables WHERE table_schema = %s;"),
            [schema_name]
        )
        tables = cursor.fetchall()
        cursor.close()
        return [table[0] for table in tables]
    except Exception as error:
        print(f'Error fetching tables in schema {schema_name}: {error}')
        return []


# @connection  # Ensure this decorator works as expected
# def delete_table(connection, schema_name, table_name):
#     try:
#         cursor = connection.cursor()
#         # Dropping the table with CASCADE
#         drop_command = sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
#             sql.Identifier(schema_name),
#             sql.Identifier(table_name)
#         )
#         cursor.execute(drop_command)
#         connection.commit()  # Commit the transaction
#         print(f"Table {schema_name}.{table_name} and its dependent objects deleted successfully.")
#     except Exception as error:
#         # Detailed error logging
#         print(f'Error deleting table {schema_name}.{table_name}: {error}')
#     finally:
#         cursor.close()  # Always close the cursor


@connection
def fetch_data_from_pg(connection, schema_name, table_or_view_name):
    try:
        engine = create_engine('postgresql+psycopg2://', creator=lambda: connection)
        print("Engine created..")
        query = f'SELECT * FROM {schema_name}.{table_or_view_name}'
        df = pd.read_sql(query, engine)
        print(f"Data fetched from {schema_name}.{table_or_view_name} successfully.")
        return df
    except Exception as e:
        print(f"Error fetching data from {schema_name}.{table_or_view_name}: {e}")
        return None


@connection
def drop_schema(connection, schema_name):
    try:
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(schema_name),
            )
        )
        connection.commit()
        print(f"Schema: {schema_name} dropped successfully.")
        cursor.close()
    except Exception as error:
        print(f'Error in dropping schema: {error}')


@connection
def execute_sql_files(connection, sql_file_path, schema_name, budget):
    try:
        # Read the SQL file
        with open(sql_file_path, 'r') as file:
            sql_script = file.read()
        sql_script = sql_script.replace('__schema__', schema_name).replace('__budget__', str(budget)).replace('__databasename__', DB_NAME).replace('__password__', DB_PASSWORD)

        # Create a cursor object
        cursor = connection.cursor()

        # Execute the SQL script
        cursor.execute(sql_script)
        print(cursor.statusmessage)

        # Commit the transaction
        connection.commit()

        print(f"Executed {sql_file_path} successfully")

        # Close the cursor
        cursor.close()

    except Exception as error:
        print(f"Error executing {sql_file_path}: {error}")

@connection
def fetch_existing_hash_keys(connection, schema_name, table_name):
    try:
        cursor = connection.cursor()
        query = f"SELECT hash_key FROM {schema_name}.{table_name};"
        cursor.execute(query)
        rows = cursor.fetchall()
        existing_keys = set(row[0] for row in rows)
        print(f"Fetched {len(existing_keys)} existing hash keys.")
        return existing_keys

    except Exception as ex:
        print(f"Error fetching hash keys: {ex}")
        return set()
