from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator


class DashboardRequest(models.Model):
    id = fields.IntField(pk=True)
    requested_on = fields.DatetimeField(null=False, auto_now_add=True)
    requested_by = fields.CharField(max_length=100, null=False)
    status = fields.BooleanField(default=False, null=False)
    message = fields.CharField(max_length=100, null=True)

    project = fields.ForeignKeyField('models.Project', related_name='dashboard_request')
    service = fields.ForeignKeyField('models.Service', related_name='dashboard_request')
    # user = fields.ForeignKeyField('models.Users', related_name='user')

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
DashboardRequest_Pydantic = pydantic_model_creator(DashboardRequest, name="DashboardRequest")
DashboardRequestIn_Pydantic = pydantic_model_creator(DashboardRequest, name="DashboardRequestIn", exclude_readonly=True)
