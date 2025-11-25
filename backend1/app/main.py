# app/main.py
import uvicorn
from fastapi import FastAPI, Depends, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise
from app.core.logging import setup_logging
from app.core.config import settings
from app.api.v1.endpoints.user import router as user_router
from app.api.v1.endpoints.queries_metrics import router as queries_metrics_router
from app.api.v1.endpoints.aws import router as aws_router
from app.api.v1.endpoints.azure import router as azure_router
from app.api.v1.endpoints.gcp import router as gcp_router
from app.api.v1.endpoints.project import router as project_router
from app.api.v1.endpoints.project_access import router as project_access_router
from app.api.v1.endpoints.snowflake import router as snowflake_router
from app.api.v1.endpoints.sync_status import router as sync_status_router
from app.api.v1.endpoints.alert import router as alert_router
from app.api.v1.endpoints.data import router as data_router
from app.api.v1.endpoints.database import router as database_router
from app.api.v1.endpoints.slack import router as slack_router
from app.api.v1.endpoints.microsoft_teams import router as teams_router
from app.api.v1.endpoints.alert_integration import router as integration_router
from app.api.v1.endpoints.service import router as service_router
from app.api.v1.endpoints.llm import router as llm_router
from app.api.v1.endpoints.service import create_services
from app.api.v1.endpoints.dashboard_request import router as dashboard_request_router
from app.api.v1.endpoints.dashboard import router as dashboard_router
from app.api.v1.endpoints.queries import queriesrouter as queriesrouter
from app.api.v1.endpoints.resources import router as resource_router
from app.api.v1.endpoints.tags import router as tags_router
from app.core.config import settings
from app.api.v1.dependencies.auth import azure_scheme
# from app.worker.celery_app import celery_app

app = FastAPI(
    openapi_url=f'{settings.API_V1_STR}/openapi.json',
    swagger_ui_oauth2_redirect_url='/oauth2-redirect',
    swagger_ui_init_oauth={
        'usePkceWithAuthorizationCodeGrant': True,
        'clientId': settings.OPENAPI_CLIENT_ID,
    },
    version='1.0.0',
    description='## Welcome to CloudMeter!',
    title=settings.PROJECT_NAME,
)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

# app.celery_app = celery_app

# Register Tortoise ORM with FastAPI
register_tortoise(
    app,
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
        "app.models.llm_cache",

    ]},
    generate_schemas=False,
    add_exception_handlers=True,
)


@app.get("/health")
def health_check():
    """
    The root route which returns a JSON response.
    The JSON response is delivered as:
    {
      'message': 'Hello, World!'
    }
    """
    return {"message": "App okay!"}


@app.post("/cancel-tasks/{project_id}")
async def cancel_tasks_no_auth(project_id: str, response: Response):
    """
    FAST cancel endpoint WITHOUT authentication for immediate task cancellation.
    This endpoint is intentionally unauthenticated to ensure instant response
    when the reset button is clicked, preventing token waste.

    The operation is safe to expose because:
    - Cancelling is idempotent (no side effects from multiple calls)
    - ProjectId is user-known information
    - Cannot corrupt data or cause security issues
    """
    from app.core.task_manager import task_manager

    # Set explicit CORS headers for this endpoint
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    try:
        print(f"🔔 [NO-AUTH] FAST CANCEL REQUEST for project: {project_id}")
        print(f"🔔 [NO-AUTH] Endpoint is being hit!")

        # List all active tasks for debugging
        active_tasks = task_manager.list_active_tasks()
        print(f"📋 [NO-AUTH] Total active tasks: {len(active_tasks)}")

        # DEBUG: Show what project_ids are in active tasks
        for task in active_tasks:
            task_pid = task.get('metadata', {}).get('project_id')
            print(f"   🔍 Task {task['id'][:8]}... has project_id: {repr(task_pid)} (type: {type(task_pid).__name__})")
        print(f"   🎯 Looking for project_id: {repr(project_id)} (type: {type(project_id).__name__})")

        cancelled_count = task_manager.cancel_tasks_by_project(project_id)

        print(f"✅ [NO-AUTH] Returning response with cancelled_count: {cancelled_count}")

        return {
            "status": "success",
            "message": f"Cancelled {cancelled_count} task(s) for project {project_id}",
            "project_id": project_id,
            "cancelled_count": cancelled_count
        }
    except Exception as e:
        print(f"❌ [NO-AUTH] ERROR in cancel_tasks_no_auth: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "project_id": project_id,
            "cancelled_count": 0
        }


@app.options("/cancel-tasks/{project_id}")
async def cancel_tasks_options(project_id: str):
    """Handle OPTIONS preflight for cancel endpoint"""
    print(f"🔄 [NO-AUTH] OPTIONS preflight received for project: {project_id}")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


@app.on_event('startup')
async def load_config() -> None:
    """
    Load OpenID config on startup.
    """
    await azure_scheme.openid_config.load_config()
    # await create_services()  # create services in service table for dashboards and requests



#app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include the user router
app.include_router(user_router, prefix="/users", dependencies=[Depends(azure_scheme)])
app.include_router(project_router, prefix="/project", dependencies=[Depends(azure_scheme)])
app.include_router(project_access_router, prefix="/project_access", dependencies=[Depends(azure_scheme)])
app.include_router(aws_router, prefix="/aws", dependencies=[Depends(azure_scheme)])
app.include_router(azure_router, prefix="/azure", dependencies=[Depends(azure_scheme)])
app.include_router(gcp_router, prefix="/gcp", dependencies=[Depends(azure_scheme)])
app.include_router(snowflake_router, prefix="/snowflake", dependencies=[Depends(azure_scheme)])
app.include_router(database_router, prefix="/database", dependencies=[Depends(azure_scheme)])
app.include_router(sync_status_router, prefix="/sync_status", dependencies=[Depends(azure_scheme)])
app.include_router(alert_router, prefix="/alert", dependencies=[Depends(azure_scheme)])
app.include_router(slack_router, prefix="/slack", dependencies=[Depends(azure_scheme)])
app.include_router(teams_router, prefix="/microsoft_teams", dependencies=[Depends(azure_scheme)])
app.include_router(integration_router, prefix="/integration", dependencies=[Depends(azure_scheme)])
app.include_router(service_router, prefix="/service", dependencies=[Depends(azure_scheme)])
app.include_router(dashboard_request_router, prefix="/dashboard_request", dependencies=[Depends(azure_scheme)])
app.include_router(dashboard_router, prefix="/dashboard", dependencies=[Depends(azure_scheme)])
app.include_router(data_router, prefix="/api/v1", tags=["data"], dependencies=[Depends(azure_scheme)])
app.include_router(queriesrouter, prefix="/queries", dependencies=[Depends(azure_scheme)])
app.include_router(resource_router, prefix="/resources", dependencies=[Depends(azure_scheme)])
app.include_router(tags_router, prefix="/tags", dependencies=[Depends(azure_scheme)])
app.include_router(queries_metrics_router, prefix="/queries_metrics", dependencies=[Depends(azure_scheme)])
app.include_router(llm_router, prefix="/llm", dependencies=[Depends(azure_scheme)])
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
