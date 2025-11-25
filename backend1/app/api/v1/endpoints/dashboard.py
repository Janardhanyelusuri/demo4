from fastapi import FastAPI, APIRouter, HTTPException
from tortoise.exceptions import DoesNotExist
from typing import List
from app.models.dashboard import Dashboard, DashboardIn_Pydantic, Dashboard_Pydantic, DashboardResponse
from app.models.project import Project
from app.worker.celery_worker import task_create_dashboard_view, task_delete_dashboard
from app.schemas.connection import CheckDashboardNameRequest, CheckDashboardNameResponse

router = APIRouter()

@router.post('/', response_model=Dashboard_Pydantic, tags=["dashboard"])
async def add_dashboard(dashboard: DashboardIn_Pydantic):
    try:
        project_ids = []
        project_names = []

        # Validate if the project exists
        for project_id in dashboard.project_ids:
            obj = await Project.get(id=project_id)
            if not obj:
                raise HTTPException(status_code=400, detail="Project not found")
            project_ids.append(obj.id)
            project_names.append(obj.name)

        # Ensure that the cloud_platforms field contains at least one cloud platform
        if not dashboard.cloud_platforms:
            raise HTTPException(status_code=400, detail="At least one cloud platform must be provided")

        # Create Dashboard record
        dashboard_data = dashboard.dict(exclude_unset=True)
        # Convert dashboard name to lowercase
        if 'name' in dashboard_data:
            dashboard_data['name'] = dashboard_data['name'].lower()

        # Loop through personas and create the dashboard for each persona
        dashboard_objs = []  # Collect dashboard objects if you need to create multiple records per persona
        for persona in dashboard.persona:
            dashboard_data["persona"] = [persona]  # Assign persona for each dashboard record creation
            dashboard_obj = await Dashboard.create(**dashboard_data)
            dashboard_objs.append(dashboard_obj)

        # Assuming the last created dashboard is the one to be used for task
        dashboard_obj = dashboard_objs[-1]

        # Run task to create new view
        payload = {
            "cloud_platforms": dashboard_obj.cloud_platforms,  # Changed to cloud_platforms
            "project_ids": project_ids,
            "project_names": project_names,
            "dashboard_id": dashboard_obj.id,
            "dashboard_name": dashboard_obj.name,
        }
        task = task_create_dashboard_view.delay(payload)
        print({"task_id": task.id})

        return await Dashboard_Pydantic.from_tortoise_orm(dashboard_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@router.post('/add_persona', response_model=List[Dashboard_Pydantic], tags=["dashboard"])
async def add_persona_to_dashboard(name: str, personas: List[str]):
    try:
        # Get all existing dashboards with the same name
        existing_dashboards = await Dashboard.filter(name=name).all()
        if not existing_dashboards:
            raise HTTPException(status_code=404, detail="Dashboard with this name not found")

        # Collect all existing personas for this dashboard name
        existing_personas = set()
        for db in existing_dashboards:
            if db.persona:
                existing_personas.update(db.persona)

        # Filter out personas that already exist
        new_personas = [p for p in personas if p not in existing_personas]
        if not new_personas:
            raise HTTPException(status_code=400, detail="All provided personas already exist for this dashboard")

        # Use the first dashboard as a template
        base_dashboard = existing_dashboards[0]
        dashboard_data = base_dashboard.__dict__
        excluded_fields = {"id", "persona", "_prefetched_objects_cache"}
        dashboard_data = {k: v for k, v in dashboard_data.items() if k not in excluded_fields}

        # Create new dashboards for non-duplicate personas
        new_dashboards = []
        for persona in new_personas:
            new_data = {**dashboard_data, "persona": [persona]}
            new_dashboard = await Dashboard.create(**new_data)
            new_dashboards.append(await Dashboard_Pydantic.from_tortoise_orm(new_dashboard))

        return new_dashboards

    except Exception as e:
        print(f"Error in add_persona: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{id}', response_model=Dashboard_Pydantic, tags=["dashboard"])
async def get_dashboard(dashboard_id: int):
    try:
        # No need for select_related if there are no related fields
        dashboard = await Dashboard.get(id=dashboard_id)
        return await Dashboard_Pydantic.from_tortoise_orm(dashboard)
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Dashboard not found")


@router.get('/', response_model=List[DashboardResponse], tags=["dashboard"])
async def get_all_dashboard():
    try:
        items = await Dashboard.all()
        dashboards_to_return = []

        for i in items:
            i.connectors = []
            valid_platforms = set(i.cloud_platforms)  #cloud_platforms list

            for project_id in i.project_ids:
                obj = await Project.filter(id=project_id).first()
                if obj:
                    i.connectors.append({
                        "id": obj.id,
                        "name": obj.name,
                        "cloud_platform": obj.cloud_platform  #Project cloud_platform field
                    })
                    # Ensure the cloud platform is valid
                    if obj.cloud_platform in valid_platforms:
                        valid_platforms.discard(obj.cloud_platform)

            # Update the cloud_platforms to only include valid ones
            i.cloud_platforms = [platform for platform in i.cloud_platforms if platform not in valid_platforms]

            # If connectors exist, add the dashboard to the list to return
            if i.connectors:
                dashboards_to_return.append(i)
            else:
                # If no connectors, delete the dashboard
                await Dashboard.filter(id=i.id).delete()

        return dashboards_to_return
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{dashboard_id}", tags=["dashboard"])
async def delete_dashboard(dashboard_id: int):
    obj = await Dashboard.get(id=dashboard_id)
    payload = {"dashboard_name": obj.name}
    task = task_delete_dashboard.delay(payload)
    print({"task_id": task.id})
    await Dashboard.filter(id=dashboard_id).delete()
    return {"status": True, "message": "Successfully deleted dashboard"}


@router.put("/{dashboard_id}", response_model=Dashboard_Pydantic, tags=["dashboard"])
async def update_dashboard(dashboard_id: int, dashboard: DashboardIn_Pydantic):
    dashboard_data = dashboard.model_dump(exclude_unset=True)
    # Convert dashboard name to lowercase if it's being updated
    if 'name' in dashboard_data:
        dashboard_data['name'] = dashboard_data['name'].lower()
    
    await Dashboard.filter(id=dashboard_id).update(**dashboard_data)
    return await Dashboard_Pydantic.from_queryset_single(Dashboard.get(id=dashboard_id))


@router.get('/{dashboard_id}/run_ingestion', tags=["dashboard"])
async def rerun_ingestion(dashboard_id: int):
    try:

        dashboard = await Dashboard.get(id=dashboard_id)
        dashboard_obj = await Dashboard_Pydantic.from_tortoise_orm(dashboard)

        # Ensure project_ids is not None
        project_ids = dashboard_obj.project_ids or []
        project_names = []

        # Validate if the project exists
        for project_id in project_ids:
            obj = await Project.get(id=project_id)
            if not obj:
                raise HTTPException(status_code=400, detail="Project not found")
            project_names.append(obj.name)

        # run task to create new view
        payload = {
            "cloud_platforms": dashboard_obj.cloud_platforms,
            "project_ids":project_ids,
            "project_names": project_names,
            "dashboard_id": dashboard_obj.id,
            "dashboard_name": dashboard_obj.name,
        }
        task = task_create_dashboard_view.delay(payload)
        print({"task_id": task.id})
        return {"status": True, "message": "Successfully run"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/check_name', response_model=CheckDashboardNameResponse, tags=["dashboard"])
async def check_dashboard_name(dashboard: CheckDashboardNameRequest):
    try:
        # check if project with same name already exists
        obj = await Dashboard.filter(name=dashboard.name).first()
        if obj:
            return CheckDashboardNameResponse(status=False,
                                            message="Project with same name already exists. Please try another name.")
    except Exception as e:
        print(f"Error: {e}")
        return CheckDashboardNameResponse(status=False,
                                        message="Something went wrong")
    return CheckDashboardNameResponse(status=True,
                                    message="Success")