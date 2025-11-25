from app.ingestion.aws.aws_cur.postgres.scripts.postgres_operations import connection

@connection
def run_sql_file(connection, sql_file_path, schema_name, budget):
    try:
        # Read the SQL file
        with open(sql_file_path, 'r') as file:
            sql_script = file.read()
        sql_script = sql_script.replace('__schema__', schema_name).replace('__budget__', str(budget))

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
