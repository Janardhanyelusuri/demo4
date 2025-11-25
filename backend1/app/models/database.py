from tortoise import fields, models
from pydantic import BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

class Database(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    connection_string = fields.CharField(max_length=255)

    project = fields.ForeignKeyField('models.Project', related_name='database')
    
    class PydanticMeta:
        model_config = {'extra': 'allow'}

class DataResponse(BaseModel):
    data: list

Database_Pydantic = pydantic_model_creator(Database, name="Database")
DatabaseIn_Pydantic = pydantic_model_creator(Database, name="DatabaseIn", exclude_readonly=True)
