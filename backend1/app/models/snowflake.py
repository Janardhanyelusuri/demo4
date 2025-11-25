from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
import os
from app.core.encryption import encrypt_data


class SnowflakeConnection(models.Model):
    id = fields.IntField(pk=True)
    account_name = fields.CharField(max_length=100, null=False)
    user_name = fields.CharField(max_length=100, null=False)
    password = fields.CharField(max_length=100, null=False)
    warehouse_name = fields.CharField(max_length=100, null=False)

    project = fields.ForeignKeyField('models.Project', related_name='snowflake_connection')

    async def save(self, *args, **kwargs):
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        # Encrypt azure_tenant_id
        encrypted_account_name = encrypt_data(self.account_name, encryption_key)
        self.account_name = encrypted_account_name

        # Encrypt azure_client_id
        encrypted_user_name = encrypt_data(self.user_name, encryption_key)
        self.user_name = encrypted_user_name

        # Encrypt azure_client_secret
        encrypted_password = encrypt_data(self.password, encryption_key)
        self.password = encrypted_password

        await super().save(*args, **kwargs)

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
SnowflakeConnection_Pydantic = pydantic_model_creator(SnowflakeConnection, name="SnowflakeConnection")
SnowflakeConnectionIn_Pydantic = pydantic_model_creator(SnowflakeConnection, name="SnowflakeConnectionIn", exclude_readonly=True)
