from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator


class Service(models.Model):
    id = fields.IntField(pk=True)
    cloud_platform = fields.CharField(max_length=100, null=False)
    label = fields.CharField(max_length=100, null=True)
    name = fields.CharField(max_length=100, null=False)
    status = fields.BooleanField(default=False, null=False)

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
Service_Pydantic = pydantic_model_creator(Service, name="Service")
ServiceIn_Pydantic = pydantic_model_creator(Service, name="ServiceIn", exclude_readonly=True)
