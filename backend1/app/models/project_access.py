from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator


class ProjectAccess(models.Model):
    id = fields.IntField(pk=True)
    access_granted_by = fields.CharField(max_length=100, null=False)
    access_granted_on = fields.DateField(null=False)
    status = fields.BooleanField(default=True, null=False)

    project = fields.ForeignKeyField('models.Project', related_name='project_access')
    user = fields.ForeignKeyField('models.Users', related_name='project_access')

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
ProjectAccess_Pydantic = pydantic_model_creator(ProjectAccess, name="ProjectAccess")
ProjectAccessIn_Pydantic = pydantic_model_creator(ProjectAccess, name="ProjectAccessIn", exclude_readonly=True)
