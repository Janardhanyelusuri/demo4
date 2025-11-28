import sys
import os
import json
from typing import Optional, Union
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from tortoise.exceptions import DoesNotExist
from app.models.project import Project
from datetime import datetime, date
from typing import List
from app.core.llm_cache_utils import generate_cache_hash_key, get_cached_result, save_to_cache
from app.core.task_manager import task_manager
try:
    from app.ingestion.aws.llm_s3_integration import run_llm_analysis_s3
    from app.ingestion.aws.llm_ec2_integration import run_llm_analysis as run_llm_analysis_ec2
    from app.ingestion.azure.llm_data_fetch import run_llm_analysis
    # from app.ingestion.gcp.llm import run_llm_analysis_gcp # <-- Keeping this commented to resolve the 404
except ImportError as e:
    # If this prints, it means one of the ingestion modules has an internal error.
    print("FATAL IMPORT ERROR DURING LLM ROUTER LOAD:")
    print(f"Error: {e}")

# Router definition
router = APIRouter(tags=["llm"])

# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------
class LLMRequest(BaseModel):
    resource_type: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    resource_id: Optional[str] = None
    schema_name: Optional[str] = None


# ---------------------------------------------------------
# RESPONSE MODEL
# ---------------------------------------------------------
class LLMResponse(BaseModel):
    status: str
    cloud: str
    schema_name: str
    resource_type: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    resource_id: Optional[str]
    recommendations: Optional[str] = None
    details: Optional[dict] = None
    timestamp: datetime
    task_id: Optional[str] = None  # Task ID for cancellation


async def _resolve_schema_name(project_id: Optional[Union[int, str]], schema_name: Optional[str]) -> str:
    if schema_name:
        return schema_name.lower()

    if project_id is None:
        raise HTTPException(status_code=400, detail="Either project_id or schema_name must be provided.")

    project_id_str = str(project_id)
    project = None

    # --- ROBUST LOOKUP LOGIC: Try ID first, then fall back to Name ---

    # 1. Check if the input is numeric (could be an int or a numeric string like '5')
    if project_id_str.isdigit():
        try:
            # Try to look up by Primary Key (id)
            pk = int(project_id_str)
            project = await Project.get(id=pk)
        except DoesNotExist:
            # If the ID is numeric but doesn't exist, we fall through to try by name, 
            # in case a project was named '5', for example.
            pass
    
    # 2. If no project was found yet (either input was non-numeric, or numeric ID was missing)
    if not project:
        try:
            # Try to look up by Name (case-insensitive)
            project = await Project.get(name__iexact=project_id_str)
        except DoesNotExist:
            # If both ID and Name lookups failed, raise the final 404.
            raise HTTPException(status_code=404, detail=f"Project ID/Name '{project_id_str}' not found")

    # If we reached here, 'project' is guaranteed to be a valid Project instance.
    return project.name.lower() 


# ---------------------------------------------------------
# AWS Endpoint (WITH CACHING)
# ---------------------------------------------------------
@router.post("/aws/{project_id}", response_model=LLMResponse, status_code=200)
async def llm_aws(
    project_id: str,
    payload: LLMRequest,
    response: Response,
):
    schema = await _resolve_schema_name(project_id, payload.schema_name)

    # Convert datetime to date for hashing
    start_dt = payload.start_date.date() if payload.start_date else None
    end_dt = payload.end_date.date() if payload.end_date else None

    # Generate hash key for caching
    hash_key = generate_cache_hash_key(
        cloud_platform="aws",
        schema_name=schema,
        resource_type=payload.resource_type,
        start_date=start_dt,
        end_date=end_dt,
        resource_id=payload.resource_id
    )

    print(f"🔑 Generated cache hash_key: {hash_key[:16]}...")

    # Check cache first
    cached_result = await get_cached_result(hash_key)
    task_id = None

    if cached_result:
        # Return cached result
        print(f"📦 Returning cached result for AWS {payload.resource_type}")
        result_list = cached_result
    else:
        # Cache miss - create task and call LLM
        task_id = task_manager.create_task(
            task_type="llm_analysis",
            metadata={
                "cloud": "aws",
                "project_id": project_id,
                "resource_type": payload.resource_type,
                "schema": schema,
                "resource_id": payload.resource_id
            }
        )

        # Set task_id in response header immediately so frontend can cancel even if request is aborted
        response.headers["X-Task-ID"] = task_id

        print(f"🔄 Cache miss - calling LLM for AWS {payload.resource_type} (task: {task_id})")

        try:
            # Route based on resource type
            resource_type_lower = payload.resource_type.lower().strip()

            if resource_type_lower == 's3':
                result = run_llm_analysis_s3(
                    schema_name=schema,
                    start_date=payload.start_date,
                    end_date=payload.end_date,
                    bucket_name=payload.resource_id
                )
            elif resource_type_lower == 'ec2':
                result = run_llm_analysis_ec2(
                    resource_type=payload.resource_type,
                    schema_name=schema,
                    start_date=payload.start_date,
                    end_date=payload.end_date,
                    resource_id=payload.resource_id
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported AWS resource type: {payload.resource_type}. Supported types: s3, ec2"
                )

            # Convert single dict result to list for consistent frontend handling
            if result is not None:
                result_list = [result] if isinstance(result, dict) else result
                print(f'Final response_list: {len(result_list)} resources processed')
            else:
                result_list = []

            # Save to cache (async) - but only if task was NOT cancelled
            if result_list and not task_manager.is_cancelled(task_id):
                await save_to_cache(
                    hash_key=hash_key,
                    cloud_platform="aws",
                    schema_name=schema,
                    resource_type=payload.resource_type,
                    start_date=start_dt,
                    end_date=end_dt,
                    resource_id=payload.resource_id,
                    output_json=result_list
                )
            elif task_manager.is_cancelled(task_id):
                print(f"🚫 NOT saving to cache - task {task_id[:8]}... was cancelled")

            # Mark task as complete
            task_manager.complete_task(task_id)

        except Exception as e:
            # Mark task as complete even on error
            if task_id:
                task_manager.complete_task(task_id)
            raise

    return LLMResponse(
        status="success",
        cloud="aws",
        schema_name=schema,
        resource_type=payload.resource_type,
        start_date=payload.start_date,
        end_date=payload.end_date,
        resource_id=payload.resource_id,
        recommendations=json.dumps(result_list) if result_list else None,
        details=None,
        timestamp=datetime.utcnow(),
        task_id=task_id
    )


# ---------------------------------------------------------
# AZURE Endpoint (WITH CACHING)
# ---------------------------------------------------------
@router.post("/azure/{project_id}", response_model=LLMResponse, status_code=200)
async def llm_azure(
    project_id: str,
    payload: LLMRequest,
    response: Response,
):
    schema = await _resolve_schema_name(project_id, payload.schema_name)

    # Convert datetime to date for hashing
    start_dt = payload.start_date.date() if payload.start_date else None
    end_dt = payload.end_date.date() if payload.end_date else None

    # Generate hash key for caching
    hash_key = generate_cache_hash_key(
        cloud_platform="azure",
        schema_name=schema,
        resource_type=payload.resource_type,
        start_date=start_dt,
        end_date=end_dt,
        resource_id=payload.resource_id
    )

    print(f"🔑 Generated cache hash_key: {hash_key[:16]}...")

    # Check cache first
    cached_result = await get_cached_result(hash_key)
    task_id = None

    if cached_result:
        # Return cached result
        print(f"📦 Returning cached result for Azure {payload.resource_type}")
        result_list = cached_result
    else:
        # Cache miss - create task and call LLM
        task_id = task_manager.create_task(
            task_type="llm_analysis",
            metadata={
                "cloud": "azure",
                "project_id": project_id,
                "resource_type": payload.resource_type,
                "schema": schema,
                "resource_id": payload.resource_id
            }
        )

        # Set task_id in response header immediately so frontend can cancel even if request is aborted
        response.headers["X-Task-ID"] = task_id

        print(f"🔄 Cache miss - calling LLM for Azure {payload.resource_type} (task: {task_id})")

        try:
            result = run_llm_analysis(
                payload.resource_type,
                schema,
                payload.start_date,
                payload.end_date,
                payload.resource_id,
                task_id=task_id
            )

            # Convert single dict result to list for consistent frontend handling
            if result is not None:
                result_list = [result] if isinstance(result, dict) else result
                print(f'Final response__list: {len(result_list)} resources processed')
            else:
                result_list = []

            # Save to cache (async) - but only if task was NOT cancelled
            if result_list and not task_manager.is_cancelled(task_id):
                await save_to_cache(
                    hash_key=hash_key,
                    cloud_platform="azure",
                    schema_name=schema,
                    resource_type=payload.resource_type,
                    start_date=start_dt,
                    end_date=end_dt,
                    resource_id=payload.resource_id,
                    output_json=result_list
                )
            elif task_manager.is_cancelled(task_id):
                print(f"🚫 NOT saving to cache - task {task_id[:8]}... was cancelled")

            # Mark task as complete
            task_manager.complete_task(task_id)

        except Exception as e:
            # Mark task as complete even on error
            if task_id:
                task_manager.complete_task(task_id)
            raise

    return LLMResponse(
        status="success",
        cloud="azure",
        schema_name=schema,
        resource_type=payload.resource_type,
        start_date=payload.start_date,
        end_date=payload.end_date,
        resource_id=payload.resource_id,
        recommendations=json.dumps(result_list) if result_list else None,
        details=None,
        timestamp=datetime.utcnow(),
        task_id=task_id
    )


# ---------------------------------------------------------
# GCP Endpoint (when ready)
# ---------------------------------------------------------
@router.post("/gcp/{project_id}", response_model=LLMResponse, status_code=200)
async def llm_gcp(
    project_id: str, 
    payload: LLMRequest,
):
    schema = await _resolve_schema_name(project_id, payload.schema_name)

    # result = run_llm_analysis_gcp(...) 
    result = {"message": "GCP LLM not implemented yet"}

    return LLMResponse(
        status="success",
        cloud="gcp",
        schema_name=schema,
        resource_type=payload.resource_type,
        start_date=payload.start_date,
        end_date=payload.end_date,
        resource_id=payload.resource_id,
        details=result,
        recommendations=None, 
        timestamp=datetime.utcnow()
    )



@router.get("/{cloud_platform}/{project_id}/resources/{resource_type}")
async def get_resource_ids(
    cloud_platform: str,
    project_id: str,
    resource_type: str,
):
    """
    Fetch available resource IDs for a given resource type and schema.
    Supports Azure (VM, Storage), AWS (EC2, S3), and GCP (future).
    """
    from app.ingestion.azure.postgres_operation import connection
    import pandas as pd

    # Resolve schema name from project_id
    schema = await _resolve_schema_name(project_id, None)

    # Normalize inputs
    cloud_platform = cloud_platform.lower()
    resource_type = resource_type.lower()

    @connection
    def fetch_resource_ids(conn, schema_name: str, res_type: str, cloud: str):
        """Fetch resource IDs from the database based on cloud and resource type."""
        query = None

        if cloud == "azure":
            if res_type in ["vm", "virtualmachine", "virtual_machine"]:
                # Fetch VM resource IDs from Azure (excluding Databricks)
                query = f"""
                    SELECT DISTINCT LOWER(resource_id) as resource_id, resource_name
                    FROM {schema_name}.gold_azure_resource_dim
                    WHERE service_category = 'Compute'
                      AND (LOWER(resource_id) LIKE '%/virtualmachines/%'
                           OR LOWER(resource_id) LIKE '%/compute/virtualmachines%')
                      AND LOWER(resource_id) NOT LIKE '%databricks%'
                    ORDER BY resource_name
                    LIMIT 100;
                """
            elif res_type in ["storage", "storageaccount", "storage_account"]:
                # Fetch Storage Account resource IDs from Azure (excluding Databricks)
                query = f"""
                    SELECT DISTINCT LOWER(resource_id) as resource_id, storage_account_name as resource_name
                    FROM {schema_name}.dim_storage_account
                    WHERE resource_id IS NOT NULL
                      AND LOWER(resource_id) NOT LIKE '%databricks%'
                    ORDER BY storage_account_name
                    LIMIT 100;
                """
            elif res_type in ["publicip", "public_ip", "pip"]:
                # Fetch Public IP resource IDs from Azure
                query = f"""
                    SELECT DISTINCT LOWER(resource_id) as resource_id, public_ip_name as resource_name
                    FROM {schema_name}.dim_public_ip
                    WHERE resource_id IS NOT NULL
                    ORDER BY public_ip_name
                    LIMIT 100;
                """
        elif cloud == "aws":
            if res_type in ["ec2", "instance"]:
                # Fetch EC2 instance IDs from AWS dimension table
                query = f"""
                    SELECT DISTINCT instance_id as resource_id,
                           COALESCE(instance_name, instance_id) as resource_name
                    FROM {schema_name}.dim_ec2_instance
                    WHERE instance_id IS NOT NULL
                    ORDER BY instance_name, instance_id
                    LIMIT 100;
                """
            elif res_type in ["s3", "bucket"]:
                # Fetch S3 bucket names from AWS dimension table
                query = f"""
                    SELECT DISTINCT bucket_name as resource_id,
                           bucket_name as resource_name
                    FROM {schema_name}.dim_s3_bucket
                    WHERE bucket_name IS NOT NULL
                    ORDER BY bucket_name
                    LIMIT 100;
                """

        if not query:
            return []

        try:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                return []
            return df.to_dict('records')
        except Exception as e:
            print(f"Error fetching resource IDs: {e}")
            return []

    # Fetch the resource IDs
    resource_ids = fetch_resource_ids(schema, resource_type, cloud_platform)

    return {
        "status": "success",
        "cloud_platform": cloud_platform,
        "resource_type": resource_type,
        "schema_name": schema,
        "resource_ids": resource_ids,
        "count": len(resource_ids)
    }


# ---------------------------------------------------------
# TASK MANAGEMENT ENDPOINTS
# ---------------------------------------------------------
@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    Cancel a running LLM analysis task.
    """
    print(f"🔔 CANCEL REQUEST RECEIVED for task: {task_id}")

    # List all active tasks for debugging
    active_tasks = task_manager.list_active_tasks()
    print(f"📋 Active tasks: {len(active_tasks)}")
    for task in active_tasks:
        print(f"  - {task['id'][:8]}... status={task['status']}")

    success = task_manager.cancel_task(task_id)

    print(f"{'✅' if success else '❌'} Cancellation result: {success}")

    if success:
        return {
            "status": "success",
            "message": f"Task {task_id} has been cancelled",
            "task_id": task_id
        }
    else:
        print(f"⚠️  Task {task_id} not found in active tasks")
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )


@router.post("/projects/{project_id}/cancel-tasks", status_code=200)
async def cancel_project_tasks(project_id: str):
    """
    Cancel all running LLM analysis tasks for a specific project.
    This is useful when user aborts a request before receiving the task_id.
    """
    try:
        print(f"🔔 CANCEL ALL TASKS REQUEST for project: {project_id}")
        print(f"🔔 Endpoint is being hit!")

        # List all active tasks for debugging
        active_tasks = task_manager.list_active_tasks()
        print(f"📋 Total active tasks: {len(active_tasks)}")

        cancelled_count = task_manager.cancel_tasks_by_project(project_id)

        print(f"✅ Returning response with cancelled_count: {cancelled_count}")

        return {
            "status": "success",
            "message": f"Cancelled {cancelled_count} task(s) for project {project_id}",
            "project_id": project_id,
            "cancelled_count": cancelled_count
        }
    except Exception as e:
        print(f"❌ ERROR in cancel_project_tasks: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Simple health check to verify code is loaded."""
    print("🏥 Health check endpoint hit!")
    return {"status": "healthy", "message": "Backend code loaded successfully", "version": "v2.1"}


@router.get("/tasks")
async def list_tasks():
    """
    List all active tasks.
    """
    tasks = task_manager.list_active_tasks()
    return {
        "status": "success",
        "tasks": tasks,
        "count": len(tasks)
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status of a specific task.
    """
    task_status = task_manager.get_task_status(task_id)

    if task_status.get('status') == 'not_found':
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    return {
        "status": "success",
        "task": task_status
    }