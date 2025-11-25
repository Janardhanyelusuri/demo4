from tortoise import fields, models
from enum import Enum
from pydantic import BaseModel, validator
from typing import Optional
from tortoise.contrib.pydantic import pydantic_model_creator


class IntegrationType(str, Enum):
    slack = "slack"
    microsoft_teams="microsoft_teams"

class IntegrationRequest(BaseModel):
    name: str
    integration_type: IntegrationType
    url: Optional[str] = None
    notification_template: dict  # JSON data as a dictionary

class IntegrationResponse(BaseModel):
    status: bool
    message: str

class Integration(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=100, null=False)
    integration_type = fields.CharField(max_length=50, default='slack', null=False)
    url = fields.CharField(max_length=255, null=True)
    notification_template = fields.JSONField()  # Use JSONField to store the JSON data

    project = fields.ForeignKeyField('models.Project', related_name='integrations', on_delete=fields.CASCADE)

    class PydanticMeta:
        model_config = {'extra': 'allow'}

# Create Pydantic models from Tortoise ORM models
Integration_Pydantic = pydantic_model_creator(Integration, name="Integration")
IntegrationIn_Pydantic = pydantic_model_creator(Integration, name="IntegrationIn", exclude_readonly=True)
