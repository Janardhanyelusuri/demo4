from tortoise import fields
from tortoise.models import Model
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise.exceptions import IntegrityError
from pydantic import BaseModel
from tortoise.exceptions import IntegrityError
from typing import Optional

class Tag(Model):
    """
    ORM Model representing the Tag entity.
    """
    tag_id = fields.IntField(pk=True)
    key = fields.CharField(max_length=255)
    value = fields.CharField(max_length=255)
    budget = fields.IntField(null=True)

    class Meta:
        table = 'tag'
        unique_together = (('key', 'value'),)  # Ensure key-value pairs are unique

    def __str__(self):
        return f"{self.key}: {self.value}"

class TagIn_Pydantic(BaseModel):
    key: str
    value: str
    budget: Optional[int] = None

Tag_Pydantic = pydantic_model_creator(Tag, name="Tag")
TagIn_Pydantic = pydantic_model_creator(Tag, name="TagIn", exclude_readonly=True)

async def create_tag(tag_data: TagIn_Pydantic):
    try:
        # Attempt to create a new tag in the database
        tag = await Tag.create(**tag_data.dict())
        return tag
    except IntegrityError as e:
        # Handle unique constraint violation (duplicate key-value pair)
        raise IntegrityError("A tag with the same key and value already exists.")
    except Exception as e:
        # Catch any other exceptions
        raise e