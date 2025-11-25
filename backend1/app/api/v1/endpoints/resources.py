from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
import asyncpg
import os
from tortoise import Tortoise
from app.models.resources import Resource
from app.models.project import Project
from tortoise.expressions import Q
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from tortoise.exceptions import DoesNotExist, IntegrityError
from app.models.resources import Resource
from app.models.resources_tags import ResourceTag
from app.models.tags import Tag, Tag_Pydantic
from tortoise.exceptions import DoesNotExist  # Correct Exception
from tortoise import fields
from app.core.logging import setup_logging, logger

router = APIRouter()

setup_logging()

class ApplyTagSchema(BaseModel):
    tag_id: int
    resource_ids: List[int]

async def get_db_connection():
    conn = await asyncpg.connect(
        user=os.getenv("DB_USER_NAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        host=os.getenv("DB_HOST_NAME")
    )
    return conn


@router.post('/sync-resources', tags=["resources"])
async def sync_resources(schema: str, cloudPlatform: str):
    try:

        db_conn = await get_db_connection()

        if cloudPlatform.lower() == 'azure':
            query = f'''
                SELECT rd.*, fc.resource_group_name 
                FROM {schema}.gold_azure_resource_dim rd
                LEFT JOIN {schema}.gold_azure_fact_cost fc
                ON rd.resource_id = fc.resource_id;
            '''
        elif cloudPlatform.lower() == 'aws':
            query = f'''
                SELECT resource_id, resource_name, region_id, region_name,
                    service_category, service_name
                FROM {schema}.gold_aws_fact_focus gaff;
            '''
        elif cloudPlatform.lower() == 'gcp':
            query = f'''
                SELECT resource_id, resource_name, region_id, region_name,
                    service_category, service_name
                FROM {schema}.gold_gcp_fact_dim ggfd;
            '''                            
        resources = await db_conn.fetch(query)
        await db_conn.close()

        # Fetch corresponding project using schema name
        project = await Project.get_or_none(name=schema)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        # Synchronize resources
        for resource in resources:
            # Add resource_group_name only if cloudPlatform is Azure
            resource_data = {
                "resource_id": resource['resource_id'],
                "resource_name": resource['resource_name'],
                "region_id": resource['region_id'],
                "region_name": resource['region_name'],
                "service_category": resource['service_category'],
                "service_name": resource['service_name'],
                "cloud_platform": cloudPlatform,
                "project": project
            }

            if cloudPlatform.lower() == 'azure':
                resource_data["resource_group_name"] = resource.get('resource_group_name')

            try:
                if resource['resource_id']:
                    # update or create based on resource_id
                    existing_resource = await Resource.get_or_none(
                        resource_id=resource['resource_id'], project=project
                    )
                else:
                    # Check for unique field combination when resource_id is null
                    existing_resource = await Resource.get_or_none(
                        project=project,
                        resource_name=resource['resource_name'],
                        region_id=resource['region_id'],
                        region_name=resource['region_name'],
                        service_category=resource['service_category'],
                        service_name=resource['service_name'],
                        resource_group_name=resource.get('resource_group_name'),
                        cloud_platform=resource['cloud_platform']
                    )

                if existing_resource:
                    # Update existing resource
                    for key, value in resource_data.items():
                        setattr(existing_resource, key, value)
                    await existing_resource.save()
                    # logger.info(f"Updated resource: {resource.get('resource_id') or 'unique fields combination'}")
                else:
                    # Create a new resource if it doesn't exist
                    await Resource.create(**resource_data)
                    # logger.info(f"Created new resource: {resource.get('resource_id') or 'unique fields combination'}")

            except Exception as e:
                logger.info(f"Failed to process resource: {resource.get('resource_id') or 'unique fields combination'} - {str(e)}")

        return {"status": True, "message": "Resources synchronized successfully"}

    except Exception as e:
        logger.info(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synchronization failed: {str(e)}")


@router.get('/resources', tags=["resources"])
async def get_resources(
    name: str,  # Project name as input
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1),
    service_name: Optional[str] = None,
    resource_group_name: Optional[str] = None,
    resource_name: Optional[str] = None,
    service_category: Optional[str] = None,
    region_name: Optional[str] = None,
    tag: Optional[str] = None, 
    sort_by: Optional[str] = Query(None, regex="^(resource_name|service_name|service_category|region_name|resource_group_name)$"),
    sort_order: Optional[str] = Query("asc", regex="^(asc|desc)$")
):
    offset = (page - 1) * page_size

    try:
        # Step 1: Fetch project based on name
        project = await Project.get_or_none(name=name)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with name '{name}' not found")

        # Step 2: Build dynamic filter dictionary for ORM filtering
        filters = Q(project=project)  # Filter by project foreign key (project_id)

        # Apply additional filters dynamically
        if service_name:
            filters &= Q(service_name__icontains=service_name)
        if resource_name:
            filters &= Q(resource_name__icontains=resource_name)
        if service_category:
            filters &= Q(service_category__icontains=service_category)
        if region_name:
            filters &= Q(region_name__icontains=region_name)
        if resource_group_name:
            filters &= Q(resource_group_name__icontains=resource_group_name)

        # Step 3: Add tag filter if provided
        if tag:
            tag_key, tag_value = tag.split(":", 1)  # Split the tag string into key and value
            filters &= Q(resource_tags__tag__key=tag_key, resource_tags__tag__value=tag_value)

        # Step 4: Fetch distinct filter values (without pagination)
        distinct_filters = {
            "service_names": await Resource.filter(Q(project=project)).distinct().values_list('service_name', flat=True),
            "resource_names": await Resource.filter(Q(project=project)).distinct().values_list('resource_name', flat=True),
            "service_categories": await Resource.filter(Q(project=project)).distinct().values_list('service_category', flat=True),
            "region_names": await Resource.filter(Q(project=project)).distinct().values_list('region_name', flat=True),
            "resource_group_name": await Resource.filter(Q(project=project)).distinct().values_list('resource_group_name', flat=True),
        }

        # Determine the sorting field and order
        if sort_by:
            sort_field = sort_by
            sort_direction = "-" if sort_order == "desc" else ""
            sort_expression = f"{sort_direction}{sort_field}"
        else:
            sort_expression = "resource_name"  # Default sort field if not provided

        # Step 5: Query the Resource model with filters, pagination, sorting, and prefetch related tags from ResourceTag
        resources = await Resource.filter(filters).offset(offset).limit(page_size).order_by(sort_expression).prefetch_related(
            "resource_tags__tag"  # Prefetch the related tags via the ResourceTag table
        )

        # Step 6: Transform the flat data into the desired nested structure
        formatted_resources = []
        for resource in resources:
            resource_data = {
                'id': resource.id,
                'resource_id': resource.resource_id, 
                'resource_name': resource.resource_name,
                'region_id': resource.region_id,
                'region_name': resource.region_name,
                'service_category': resource.service_category,
                'service_name': resource.service_name,
                'resource_group_name': resource.resource_group_name,
                'tags': []
            }

            # Loop through the related tags if any exist
            for resource_tag in resource.resource_tags:
                if resource_tag.tag:  # Make sure tag exists
                    resource_data['tags'].append({
                        'key': resource_tag.tag.key,
                        'value': resource_tag.tag.value,
                        'tag_id': resource_tag.tag.tag_id
                    })

            formatted_resources.append(resource_data)

        # Step 7: Return the resources and filter options in JSON format
        total_items = await Resource.filter(filters).count()  # Get total items count before pagination
        return {
            "resources": formatted_resources,
            "total_items": total_items,
            "filter_options": distinct_filters  # Include available filters based on the full dataset
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

@router.get('/resources_dim', tags=["resources"])
async def get_resources_from_dim(
    schema: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1),
    service_name: Optional[str] = None,
    resource_name: Optional[str] = None,
    service_category: Optional[str] = None,
    region_name: Optional[str] = None
):
    offset = (page - 1) * page_size
    db_conn = await get_db_connection()

    # Base SQL query with schema name
    query = f"SELECT * FROM {schema}.gold_azure_resource_dim WHERE 1=1"
    
    # Apply filters dynamically
    params = []
    param_count = 1
    
    if service_name:
        query += f" AND service_name ILIKE ${param_count}"
        params.append(f"%{service_name}%")
        param_count += 1
    if resource_name:
        query += f" AND resource_name ILIKE ${param_count}"
        params.append(f"%{resource_name}%")
        param_count += 1
    if service_category:
        query += f" AND service_category ILIKE ${param_count}"
        params.append(f"%{service_category}%")
        param_count += 1
    if region_name:
        query += f" AND region_name ILIKE ${param_count}"
        params.append(f"%{region_name}%")
        param_count += 1

    # Append pagination
    query += f" OFFSET ${param_count} LIMIT ${param_count + 1}"
    params.extend([offset, page_size])

    # Execute the query
    try:
        resources = await db_conn.fetch(query, *params)
        await db_conn.close()

        # Return the resources in JSON format
        return {"resources": [dict(record) for record in resources]}
    except Exception as e:
        await db_conn.close()
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

@router.post('/apply-tag', tags=["resources"])
async def apply_tag_to_resources(apply_tag_data: ApplyTagSchema):
    try:
        # Fetch the tag by its ID
        tag = await Tag.get(tag_id=apply_tag_data.tag_id)
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    resource_status = []

    for resource_id in apply_tag_data.resource_ids:
        try:
            # Fetch the resource by its ID
            resource = await Resource.get(id=resource_id)
            
            # Check if the tag is already applied to the resource
            if resource.tag_id == tag.tag_id:
                # Skip and record message if the tag is already applied
                resource_status.append({
                    "id": resource.id,
                    "status": "already_applied",
                    "message": f"Tag {tag.tag_id} is already applied to resource {resource.id}."
                })
                continue
            
            # Create or update the ResourceTag relationship
            await ResourceTag.update_or_create(resource=resource, tag=tag)
            
            # Directly update the tag_id field in the resource table (resource_dim)
            resource.tag_id = tag.tag_id
            await resource.save()

            resource_status.append({
                "id": resource.id,
                "status": "success",
                "message": f"Tag {tag.tag_id} successfully applied to resource {resource.id}."
            })
        except Resource.DoesNotExist:
            resource_status.append({
                "id": resource_id,  # Still refer to resource_id in case resource doesn't exist
                "status": "failed",
                "message": f"Resource {resource_id} does not exist."
            })
        except IntegrityError as e:
            resource_status.append({
                "id": resource_id,
                "status": "failed",
                "message": f"IntegrityError while applying tag to resource {resource_id}: {e}"
            })
        except Exception as e:
            resource_status.append({
                "id": resource_id,
                "status": "failed",
                "message": f"Error while applying tag to resource {resource_id}: {e}"
            })

    return {
        "resource_status": resource_status
    }

@router.get("/api/resources/{id}/tags", tags=["resources"])
async def get_tags_for_resource(id: int):
    try:
        # Fetch the resource by its ID
        resource = await Resource.get(id=id)
        
        # Retrieve all ResourceTag objects related to the resource
        resource_tags = await ResourceTag.filter(resource=resource).select_related('tag')

        # Extract tag information
        tags = [f"{rt.tag.key}: {rt.tag.value}" for rt in resource_tags if rt.tag]

        return {"tags": tags}
    except Resource.DoesNotExist:
        raise HTTPException(status_code=404, detail="Resource not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/resources", tags=["resources"])
async def filter_resources_by_tag(tag_key: str, tag_value: str):
    try:
        # Find all tags with the given key and value
        tag = await Tag.get(key=tag_key, value=tag_value)
        
        # Retrieve all ResourceTag objects associated with the found tag
        resource_tags = await ResourceTag.filter(tag=tag).select_related('resource')

        # Return the list of resources that are associated with the found tag
        return {
            "resources": [
                {
                    "resource_id": resource_tag.resource.id,
                    "resource_name": resource_tag.resource.resource_name,
                    "resource_region": resource_tag.resource.region_name,
                    "service_category": resource_tag.resource.service_category,
                    "service_name": resource_tag.resource.service_name,
                    "cloud_platform": resource_tag.resource.cloud_platform
                } for resource_tag in resource_tags
            ]
        }
    except Tag.DoesNotExist:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete('/remove-tag', tags=["resources"])
async def remove_tag_from_resource(id: int, tag_id: int):
    try:
        # Fetch the resource and tag by their IDs
        resource = await Resource.get(id=id)
        tag = await Tag.get(tag_id=tag_id)
        
        # Attempt to delete the ResourceTag relationship
        resource_tag = await ResourceTag.get_or_none(resource=resource, tag=tag)
        
        if resource_tag:
            await resource_tag.delete()
            return {"status": True, "message": f"Tag {tag_id} removed from resource {id}"}
        else:
            raise HTTPException(status_code=404, detail="Tag not applied to this resource")

    except Resource.DoesNotExist:
        raise HTTPException(status_code=404, detail="Resource not found")
    except Tag.DoesNotExist:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove tag: {e}")