from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
from pydantic import BaseModel, HttpUrl


class Users(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=50)
    email = fields.CharField(max_length=50, unique=True)
    username = fields.CharField(max_length=20, unique=True)
    family_name = fields.CharField(max_length=50, null=True)
    
    def full_name(self) -> str:
        """
        Returns the best name
        """
        if self.name or self.family_name:
            return f"{self.name or ''} {self.family_name or ''}".strip()
        return self.username
    
    class PydanticMeta:
        computed = ["full_name"]
        exclude = ["password_hash"]


# Create Pydantic models from Tortoise ORM models
User_Pydantic = pydantic_model_creator(Users, name="User")
UserIn_Pydantic = pydantic_model_creator(Users, name="UserIn", exclude_readonly=True)
