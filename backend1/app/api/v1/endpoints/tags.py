from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.tags import Tag, Tag_Pydantic, TagIn_Pydantic
from typing import List
from tortoise.exceptions import DoesNotExist 
from tortoise.exceptions import IntegrityError
from app.models.resources_tags import ResourceTag
from app.schemas.connection import TagRequest

router = APIRouter()

@router.post('/tags', response_model=Tag_Pydantic, tags=["tags"])
async def add_tag(tag: TagIn_Pydantic):
    try:
        tag_obj = await Tag.create(**tag.dict())
        return await Tag_Pydantic.from_tortoise_orm(tag_obj)
    except IntegrityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@router.get('/tags/{tag_id}', response_model=Tag_Pydantic, tags=["tags"])
async def get_tag(tag_id: int):
    try:
        # Fetch a single tag by its ID
        tag = await Tag.get(tag_id=tag_id)
        # Convert the ORM model to a Pydantic model
        return await Tag_Pydantic.from_tortoise_orm(tag)
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/tags', response_model=List[Tag_Pydantic], tags=["tags"])
async def get_all_tags():
    try:
        # Fetch all tags using a queryset
        tags = Tag.all()  # Should return a queryset, not a list
        # Convert the queryset to a list of Pydantic models
        return await Tag_Pydantic.from_queryset(tags)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.put('/tags/{tag_id}', response_model=Tag_Pydantic, tags=["tags"])
async def update_tag(tag_id: int, tag_in: TagIn_Pydantic):
    try:
        # Update the tag based on the primary key (id)
        updated_count = await Tag.filter(tag_id=tag_id).update(**tag_in.dict(exclude_unset=True))
        
        if updated_count == 0:
            raise HTTPException(status_code=404, detail="Tag not found")

        # Fetch and return the updated tag
        updated_tag = await Tag.get(tag_id=tag_id)
        return await Tag_Pydantic.from_tortoise_orm(updated_tag)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete('/tags/{tag_id}', tags=["tags"])
async def delete_tag(tag_id: int):
    try:
        # Attempt to delete the tag by its ID
        deleted_count = await Tag.filter(tag_id=tag_id).delete()

        # Check if the tag was actually deleted (i.e., if it existed)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Tag not found")

        return {"detail": f"Tag with id {tag_id} successfully deleted"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/tags/resources', response_model=str, tags=["tags"])
async def post_resources_by_tag(request: TagRequest):
    try:
        # Extract tag_id from the request body
        tag_id = request.tag_id
        
        # Check if the tag exists
        tag = await Tag.get(tag_id=tag_id)
        
        # Fetch resources associated with the tag
        resource_tags = await ResourceTag.filter(tag_id=tag_id).prefetch_related('resource')

        if not resource_tags:
            raise HTTPException(status_code=404, detail="No resources found for this tag")
        
        # Extract the resource names from the related resources
        resource_names = [resource_tag.resource.resource_name for resource_tag in resource_tags]
        
        # Return the resource names as a comma-separated string
        return ",".join(resource_names)
    
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    

@router.post('/tags/resource_ids', response_model=str, tags=["tags"])
async def post_resources_by_tag(request: TagRequest):
    try:
        # Extract tag_id from the request body
        tag_id = request.tag_id
        
        # Check if the tag exists
        tag = await Tag.get(tag_id=tag_id)
        
        # Fetch resources associated with the tag
        resource_tags = await ResourceTag.filter(tag_id=tag_id).prefetch_related('resource')

        if not resource_tags:
            raise HTTPException(status_code=404, detail="No resources found for this tag")
        
        # Extract the resource names from the related resources
        resource_names = [resource_tag.resource.resource_id for resource_tag in resource_tags]
        
        # Return the resource names as a comma-separated string
        return ",".join(resource_names)
    
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
        
        
# @router.get('/tags/{tag_id}/resources', response_model=str, tags=["tags"])
# async def get_resources_by_tag(tag_id: int):
#     try:
#         # Check if the tag exists
#         tag = await Tag.get(tag_id=tag_id)
        
#         # Fetch resources associated with the tag
#         resource_tags = await ResourceTag.filter(tag_id=tag_id).prefetch_related('resource')

#         if not resource_tags:
#             raise HTTPException(status_code=404, detail="No resources found for this tag")
        
#         # Extract the resource names from the related resources
#         resource_names = [resource_tag.resource.resource_name for resource_tag in resource_tags]
        
#         # Return the resource names as a comma-separated string
#         return ",".join(resource_names)
    
#     except DoesNotExist:
#         raise HTTPException(status_code=404, detail="Tag not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="An unexpected error occurred")
    