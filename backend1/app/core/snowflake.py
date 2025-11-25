import snowflake.connector
from snowflake.connector.errors import DatabaseError, ProgrammingError, OperationalError, InterfaceError

def snowflake_validate_creds(account_name: str, user_name: str, password: str) -> bool:
    try:
        # Attempt to establish a connection to Snowflake
        conn = snowflake.connector.connect(
            account=account_name,
            user=user_name,
            password=password,
        )
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Execute a simple query to test the connection
        cursor.execute("SELECT CURRENT_USER()")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
    
    except DatabaseError as db_error:
        print(f"Snowflake DatabaseError: {db_error}")
        if '250001' in str(db_error):
            print("Connection attempts failed. Please check your network settings and Snowflake account configuration.")
        return False
    
    except ProgrammingError as prog_error:
        print(f"Snowflake ProgrammingError: {prog_error}")
        return False
    
    except OperationalError as op_error:
        print(f"Snowflake OperationalError: {op_error}")
        return False
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    

def list_warehouses(account_name: str, user_name: str, password: str):
    try:
        # Establish a connection to Snowflake
        conn = snowflake.connector.connect(
            account=account_name,
            user=user_name,
            password=password
        )
        
        # Create a cursor object to execute queries
        cursor = conn.cursor()
        
        # Execute the query to list all warehouses
        cursor.execute("SHOW WAREHOUSES")
        
        # Fetch all the results
        warehouses = cursor.fetchall()
        
        if not warehouses:
            return []
        else:
            # Prepare the list of warehouses
            warehouses_info = [
                {"name": warehouse[0], "state": warehouse[1]}
                for warehouse in warehouses
            ]
            return warehouses_info
        
    except DatabaseError as db_error:
        raise Exception(f"Snowflake DatabaseError: {db_error}")
    
    except InterfaceError as intf_error:
        raise Exception(f"Snowflake InterfaceError: {intf_error}")
    
    except OperationalError as op_error:
        raise Exception(f"Snowflake OperationalError: {op_error}")
    
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass