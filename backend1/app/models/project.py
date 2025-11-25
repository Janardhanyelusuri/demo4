from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
from pydantic import BaseModel, HttpUrl


class Project(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=50)
    status = fields.BooleanField(default=True, null=False)
    date = fields.DateField(null=False)
    cloud_platform = fields.CharField(max_length=50, null=True)


# Create Pydantic models from Tortoise ORM models
Project_Pydantic = pydantic_model_creator(Project, name="Project")
ProjectIn_Pydantic = pydantic_model_creator(Project, name="ProjectIn", exclude_readonly=True)
