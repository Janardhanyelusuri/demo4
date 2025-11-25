import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
from sqlalchemy.types import String, Integer, Float, DateTime, Boolean
import pandas as pd
from dotenv import load_dotenv
import os

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
        connection = None
        try:
            connection = psycopg2.connect(
                host=DB_HOST_NAME,
                database=DB_NAME,
                user=DB_USER_NAME,
                password=DB_PASSWORD,
                port=DB_PORT,
                sslmode='require'
            )
            print("Connection established successfully")

            func(connection, *args, **kwargs)

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
        # Create a cursor object
        cursor = connection.cursor()

        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))

        # Commit the transaction
        connection.commit()

        print("Schema created successfully")

    except Exception as error:
        print(f'Error creating schema: {error}')

    finally:
        if cursor:
            cursor.close()


@connection
def display_schemas(connection):
    try:
        # Create a cursor object
        cursor = connection.cursor()

        # Execute a query to fetch all schemas
        cursor.execute("SELECT schema_name FROM information_schema.schemata;")

        # Fetch all schema names
        schemas = cursor.fetchall()

        # Print the schemas
        print("Schemas in the database:")
        for schema in schemas:
            print(schema[0])

    except Exception as error:
        print(f'Error displaying schemas: {error}')

    finally:
        if cursor:
            cursor.close()


def get_sqlalchemy_type(dtype):
    """
    Convert a Pandas dtype to a SQLAlchemy type.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return Integer()
    elif pd.api.types.is_float_dtype(dtype):
        return Float()
    elif pd.api.types.is_bool_dtype(dtype):
        return Boolean()
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return DateTime()
    else:
        return String()


@connection
def dump_to_postgresql(connection, df, schema_name, table_name):
    """
    Dump the DataFrame into a PostgreSQL database table within a specific schema.
    """
    try:
        # Create SQLAlchemy engine using the provided connection details
        engine = create_engine(
            f'postgresql+psycopg2://{DB_USER_NAME}:{DB_PASSWORD}@{DB_HOST_NAME}:{DB_PORT}/{DB_NAME}',
            pool_pre_ping=True  # Optional: Enable SQLAlchemy's pool pre-ping feature
        )
        print("Engine created successfully..")

        # Convert DataFrame column types to SQLAlchemy types
        dtype_map = {col: get_sqlalchemy_type(df[col].dtype) for col in df.columns}

        # Dump data to PostgreSQL table
        df.to_sql(table_name, con=engine, schema=schema_name, if_exists='replace', index=False, dtype=dtype_map)
        print(f"Data dumped into {schema_name}.{table_name} table successfully.")

    except Exception as e:
        print(f"Error dumping data into {schema_name}.{table_name} table: {e}")

    finally:
        # Dispose of the engine after use (optional)
        engine.dispose()
