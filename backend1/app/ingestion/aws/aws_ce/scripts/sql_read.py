from ..scripts.postgres_operations import connection


@connection
def run_sql_file(connection, sql_file_path, schema_name):
    try:
        # Read the SQL file
        with open(sql_file_path, 'r') as file:
            sql_script = file.read()
        sql_script = sql_script.replace('__schema__', schema_name)

        # Create a cursor object
        cursor = connection.cursor()

        # Execute the SQL script
        cursor.execute(sql_script)

        # Commit the transaction
        connection.commit()

        print(f"Executed {sql_file_path} successfully")

        # Close the cursor
        cursor.close()

    except Exception as error:
        print(f"Error executing {sql_file_path}: {error}")
