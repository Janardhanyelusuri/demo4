from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
from pydantic import BaseModel
import datetime


class Dashboard(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=50)
    status = fields.BooleanField(default=True, null=False)
    date = fields.DateField(null=False)
    cloud_platforms = fields.JSONField(null=True, default=[])
    persona = fields.JSONField(null=True, default=[])
    project_ids = fields.JSONField(null=True, default=[])


# Create Pydantic models from Tortoise ORM models
Dashboard_Pydantic = pydantic_model_creator(Dashboard, name="Dashboard")
DashboardIn_Pydantic = pydantic_model_creator(Dashboard, name="DashboardIn", exclude_readonly=True)


class DashboardResponse(BaseModel):
    id: int
    name: str
    status: bool
    date: datetime.date
    cloud_platforms: list = []
    persona: list = []
    project_ids: list = []
    connectors: list = []
