from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator


class SyncStatus(models.Model):
    id = fields.IntField(pk=True)
    module = fields.CharField(max_length=100, null=False)
    status = fields.CharField(max_length=100, null=False)
    start_date = fields.DatetimeField(null=False, auto_now_add=True)
    end_date = fields.DatetimeField(null=True)

    project = fields.ForeignKeyField('models.Project', related_name='sync_status')

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
SyncStatus_Pydantic = pydantic_model_creator(SyncStatus, name="SyncStatus")
SyncStatusIn_Pydantic = pydantic_model_creator(SyncStatus, name="SyncStatusIn", exclude_readonly=True)
