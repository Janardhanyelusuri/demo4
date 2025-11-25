from app.ingestion.dashboard.postgres_operations import run_sql_file
import json
from fastapi import HTTPException
from tortoise.exceptions import DoesNotExist
from app.models.project import Project, Project_Pydantic
from tortoise import Tortoise
from app.core.config import settings
from app.core.misc import init_tortoise_connection, close_tortoise_connection

async def create_dashboard_view(project_ids, project_names, cloud_platforms, dashboard_name):
    await init_tortoise_connection()
    try:
        base_path = "app/ingestion/dashboard"
        schema_name = dashboard_name
        schemas_and_tables = []

        for project_id in project_ids:
            try:
                print(f"Processing project_id: {project_id}")
                
                # Attempt to fetch the project
                try:
                    project = await Project.get(id=project_id)
                except DoesNotExist:
                    print(f"Project with ID {project_id} does not exist. Skipping.")
                    continue
                except Exception as e:
                    print(f"Error fetching project {project_id}: {e}")
                    continue
                
                if not project:
                    print(f"Project.get returned None for ID {project_id}. Skipping.")
                    continue
                print(f"Fetched project: {project}")

                # Convert to Pydantic model
                try:
                    project_data = await Project_Pydantic.from_tortoise_orm(project)
                    if not project_data:
                        print(f"Pydantic conversion returned None for project {project_id}. Skipping.")
                        continue
                    print(f"Converted project_data: {project_data}")
                except Exception as conversion_error:
                    print(f"Error converting project {project_id} to Pydantic: {conversion_error}")
                    continue
                
                # table name based on cloud platform
                table_name = f"{project_data.name}_data"  # Default table name
                if project_data.cloud_platform == 'aws':
                    table_name = "silver_focus_aws"
                elif project_data.cloud_platform == 'gcp':
                    table_name = "bronze_focus_gcp_data"
                elif project_data.cloud_platform == 'azure':
                    table_name = "bronze_azure_focus"

                schemas_and_tables.append({
                    "schema": project_data.name,
                    "table": table_name,
                    "cloud": project_data.cloud_platform
                })

                await close_tortoise_connection()

            except Exception as e:
                print(f"Unexpected error processing project ID {project_id}: {e}")

        # Serialize schemas_and_tables to JSON
        schemas_and_tables_json = json.dumps(schemas_and_tables, indent=4)
        print(f"Constructed schemas_and_tables JSON: {schemas_and_tables_json}")

        # Execute SQL file using the consolidated JSON
        run_sql_file(f'{base_path}/sql/consolidated_data.sql', schemas_and_tables_json, dashboard_name)
        print(f"Schema {schema_name} created successfully.")

        # Execute SQL file using the consolidated JSON
        run_sql_file(f'{base_path}/sql/consolidated_views.sql', schemas_and_tables_json, dashboard_name)
        print(f"views created successfully.")

        return True

    except Exception as ex:
        print(f"Error in create_dashboard_view: {ex}")
        raise HTTPException(status_code=500, detail="Dashboard creation failed.")
