from fastapi import FastAPI, APIRouter, HTTPException
from tortoise.exceptions import DoesNotExist
from typing import List
from app.models.service import Service, ServiceIn_Pydantic, Service_Pydantic
from app.models.project import Project  # Ensure Project model is imported for validation if needed

router = APIRouter()


@router.get('/create_service', tags=["service"])
async def create_service():
    try:
        await create_services()
    except Exception as ex:
        print(ex)
    return True


async def create_services():  # to create services on app startup
    try:
        items = [
            # aws
            {"cloud_platform": "aws", "label": "DynamoDB", "name": "amazon_dynamodb", "status": True},
            {"cloud_platform": "aws", "label": "SageMaker", "name": "amazon_sagemaker", "status": True},
            {"cloud_platform": "aws", "label": "Step Function", "name": "amazon_step_function", "status": True},
            {"cloud_platform": "aws", "label": "Blockchain", "name": "amazon_blockchain", "status": True},
            {"cloud_platform": "aws", "label": "Simple Queue Service", "name": "amazon_sqs", "status": True},
            {"cloud_platform": "aws", "label": "Simple Notification Service", "name": "amazon_sns", "status": True},
            # azure
            {"cloud_platform": "azure", "label": "Blob Storage", "name": "azure_blob_storage", "status": True},
            {"cloud_platform": "azure", "label": "Kubernetes", "name": "azure_kubernetes", "status": True},
            {"cloud_platform": "azure", "label": "Data Factory", "name": "azure_data_factory", "status": True},
            {"cloud_platform": "azure", "label": "Azure Functions", "name": "azure_functions", "status": True},
            # gcp
            {"cloud_platform": "azure", "label": "App Engine", "name": "app_engine", "status": True},
            {"cloud_platform": "azure", "label": "Cloud Filestore", "name": "cloud_filestore", "status": True},
            {"cloud_platform": "azure", "label": "Cloud Bigtable", "name": "cloud_bigtable", "status": True},
            {"cloud_platform": "azure", "label": "Cloud Load Balancing", "name": "cloud_load_balancing", "status": True},
            {"cloud_platform": "azure", "label": "Cloud Logging", "name": "cloud_logging", "status": True},
        ]
        for i in items:
            if not await Service.filter(cloud_platform=i["cloud_platform"],
                                        name=i["name"]).exists():
                service_obj = await Service.create(**i)
            else:
                await Service.filter(cloud_platform=i["cloud_platform"],
                                     name=i["name"]
                                     ).update(status=i["status"],
                                              label=i["label"])
        # return True
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return True


@router.post('/', response_model=Service_Pydantic, tags=["service"])
async def add_service(service: ServiceIn_Pydantic):
    try:
        # Validate that the project exists
        if not await Project.filter(id=service.project_id).exists():
            raise HTTPException(status_code=400, detail="Project not found")

        # Create Service record
        service_obj = await Service.create(**service.dict())
        return await Service_Pydantic.from_tortoise_orm(service_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{id}', response_model=Service_Pydantic, tags=["service"])
async def get_service(service_id: int):
    try:
        service_obj = await Service.get(id=service_id)
        return await Service_Pydantic.from_tortoise_orm(service_obj)
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Service not found")


@router.get('/', response_model=List[Service_Pydantic], tags=["service"])
async def get_all_service():
    try:
        return await Service_Pydantic.from_queryset(Service.all())
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{service_id}", tags=["service"])
async def delete_service(service_id: int):
    await Service.filter(id=service_id).delete()
    return {"status": True, "message": "Successfully deleted service"}


@router.put("/{service_id}", response_model=Service_Pydantic, tags=["service"])
async def update_service(service_id: int, service: ServiceIn_Pydantic):
    await Service.filter(id=service_id).update(**service.model_dump(exclude_unset=True))
    return await Service_Pydantic.from_queryset_single(Service.get(id=service_id))
