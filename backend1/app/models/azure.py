from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
import os
from app.core.encryption import encrypt_data


class AzureConnection(models.Model):
    id = fields.IntField(pk=True)
    azure_tenant_id = fields.CharField(max_length=100, null=False)
    azure_client_id = fields.CharField(max_length=100, null=False)
    azure_client_secret = fields.CharField(max_length=100, null=False)
    subscription_info = fields.JSONField(null=False)
    date = fields.DateField(null=False)

    # to maintain state on project
    status = fields.BooleanField(default=False, null=False)

    # to maintain budget of project
    monthly_budget = fields.IntField(null=True)
    yearly_budget = fields.IntField(null=True)
    quarterly_budget = fields.IntField(null=True)

    export = fields.BooleanField(default=False, null=False)
    storage_account_name = fields.CharField(max_length=100, null=False)
    resource_group_name = fields.CharField(max_length=100, null=True)
    container_name = fields.CharField(max_length=100, null=True)

    project = fields.ForeignKeyField('models.Project', related_name='azure_connection')

    async def save(self, *args, **kwargs):
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        # Encrypt azure_tenant_id
        encrypted_tenant_id = encrypt_data(self.azure_tenant_id, encryption_key)
        self.azure_tenant_id = encrypted_tenant_id

        # Encrypt azure_client_id
        encrypted_client_id = encrypt_data(self.azure_client_id, encryption_key)
        self.azure_client_id = encrypted_client_id

        # Encrypt azure_client_secret
        encrypted_client_secret = encrypt_data(self.azure_client_secret, encryption_key)
        self.azure_client_secret = encrypted_client_secret

        await super().save(*args, **kwargs)

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
AzureConnection_Pydantic = pydantic_model_creator(AzureConnection, name="AzureConnection")
AzureConnectionIn_Pydantic = pydantic_model_creator(AzureConnection, name="AzureConnectionIn", exclude_readonly=True)
