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
            connection = psycopg2.connect(
                host=hostname,
                database=database,
                user=username,
                password=password,
                port=port,
                sslmode='require'
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
def create_schemas(connection, schema):
    try:
        cursor = connection.cursor()
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        connection.commit()
        print("Schemas created successfully")
        cursor.close()
    except Exception as error:
        print(f'Error creating schemas: {error}')


@connection
def dump_to_postgresql(connection, df, schema_name, table_name):
    try:
        engine = create_engine('postgresql+psycopg2://', creator=lambda: connection)
        print("Engine created..")
        df.to_sql(table_name, engine, schema=schema_name, if_exists='replace', index=False)
        print(f"Data dumped into {schema_name}.{table_name} table successfully.")
    except Exception as e:
        print(f"Error dumping data into {schema_name}.{table_name} table: {e}")


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


@connection
def delete_table(connection, schema_name, table_name):
    try:
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name)
            )
        )
        connection.commit()
        print(f"Table {schema_name}.{table_name} deleted successfully.")
        cursor.close()
    except Exception as error:
        print(f'Error deleting table {schema_name}.{table_name}: {error}')


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
def run_sql_file(connection, sql_file_path, schema_name,dashboardname, budget=0):
    try:
        # Read the SQL file
        with open(sql_file_path, 'r') as file:
            sql_script = file.read()
        sql_script = sql_script.replace('__schema__', schema_name).replace('__budget__', str(budget)).replace('__dashboardname__', dashboardname)

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
