from fastapi import APIRouter, HTTPException, Request
import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

CUBEJS_API_URL = os.getenv("CUBEJS_API_URL")
CUBEJS_API_SECRET = os.getenv("CUBEJS_API_SECRET")


async def fetch_cube_schema(cube_names):
    """
    Fetch cube schema from Cube.js API for multiple cubes.
    
    Args:
    - cube_names (list): List of cube names to fetch schema for.
    
    Returns:
    - dict: Schema information for the specified cubes.
    
    Raises:
    - HTTPException: If cube schema fetch fails.
    """
    url = f"{CUBEJS_API_URL}/meta"
    headers = {"Authorization": f"Bearer {CUBEJS_API_SECRET}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            meta = response.json()
            # print(f"Meta response: {meta}")  # Debug print

            cube_schemas = []
            for cube_name in cube_names:
                found_cube = False
                for cube in meta["cubes"]:
                    if cube["name"] == cube_name:
                        cube_schemas.append(cube)
                        found_cube = True
                        break
                if not found_cube:
                    raise HTTPException(status_code=404, detail=f"Cube '{cube_name}' not found")
            
            return cube_schemas
        except httpx.RequestError as e:
            print(f"Request error occurred while requesting {e.request.url!r}: {e}") 
            raise HTTPException(status_code=500, detail="Error fetching cube schema")
        except httpx.HTTPStatusError as e:
            print(f"HTTP status error occurred: {e}")  # Debug print
            print(f"Response status code: {e.response.status_code}")  # Debug print
            print(f"Response content: {e.response.content}")  # Debug print
            raise HTTPException(status_code=500, detail="Error fetching cube schema")
        except Exception as e:
            print(f"Unexpected error occurred while fetching cube schema: {e}")
            raise HTTPException(status_code=500, detail="Unexpected error fetching cube schema")


@router.get("/data")
async def get_data_get(cloud_provider):
    # Set cube_names based on the cloud_provider parameter
    if cloud_provider == "aws":
        cube_names = ["aws_fact_cost", "aws_dim_account", "aws_tags"]  
    elif cloud_provider == "snowflake":
        cube_names = ["snowflake_fact", "snowflake_dim_account", "snowflake_dim_date"]
    elif cloud_provider == "azure":
        cube_names = ["azure_fact_costs", "azure_dim_resource_groups", "azure_dim_meters", "azure_dim_resources", "azure_tags", "azure_dim_subscriptions"]
    elif cloud_provider == "gcp":
        cube_names = ["gcp_fact_costs", "gcp_dim_location", "gcp_dim_project", "gcp_dim_services", "gcp_labels"]
    else:
        raise HTTPException(status_code=400, detail="Invalid cloud provider")

    print(f"Fetching data for cloud provider: {cloud_provider}")

# async def get_data_get():
#     #cube_names = ["azure_fact_costs", "azure_dim_meters", "azure_dim_resource_groups", "azure_dim_resources", "azure_dim_subscriptions", "aws_fact_cost", "aws_dim_account", "aws_tags", "gcp_fact_costs", "gcp_dim_location", "gcp_dim_project", "gcp_dim_services", "gcp_labels"]
#     cube_names=["azure_fact_costs", "azure_dim_resource_groups", "azure_dim_meters", "azure_dim_resources", "azure_tags", "azure_dim_subscriptions"]
#     #cube_names=["aws_fact_cost", "aws_dim_account", "aws_tags"]
#     #cube_names=["gcp_fact_costs", "gcp_dim_location", "gcp_dim_project", "gcp_dim_services", "gcp_labels"]


#     print("hello")
    try:
        schemas = await fetch_cube_schema(cube_names)
        
        # Combine measures, time dimensions, and regular dimensions from all schemas
        measures = []
        time_dimensions = []
        dimensions = []
        
        # for schema in schemas:
        #     measures.extend([measure["name"] for measure in schema["measures"]])
        #     time_dimensions.extend([{"dimension": dim["name"], "granularity": "month"} for dim in schema["dimensions"] if dim["type"] == "time"])  # Added
        #     # time_dimensions.extend([{"dimension": dim["name"], "granularity": "month"} for dim in schema["dimensions"] if dim["type"] == "time"])
        #     dimensions.extend([dimension["name"] for dimension in schema["dimensions"]])

        for schema in schemas:
            measures.extend([measure["name"] for measure in schema["measures"]])
            # Add time dimension if it exists in the schema
            for dimension in schema["dimensions"]:
                if dimension["type"] == "time":
                    time_dimensions.append({"dimension": dimension["name"], "granularity": "month"})  # Add time dimension
                else:
                    dimensions.append(dimension["name"])

        # print(f"Measures: {measures}")
        # print(f"Time Dimensions: {time_dimensions}")
        # print(f"Dimensions: {dimensions}")
        
        query = {
            "query": {
                "measures": measures,
                # "timeDimensions": [{"dimension": dim["name"], "granularity": "month"} for dim in schema["dimensions"] if dim["type"] == "time"],
                "timeDimensions": time_dimensions,
                "dimensions": dimensions
            }
        }

        url = f"{CUBEJS_API_URL}/load"
        headers = {"Authorization": f"Bearer {CUBEJS_API_SECRET}"}

        failure = True
        while failure:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=query, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    failure = False
            except Exception as ex:
                print("api failed, retrying")
                print(ex)
        return {"message": "Success", "data": data}
    except HTTPException as e:
        print(e)
        raise e
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Unexpected error fetching data")
