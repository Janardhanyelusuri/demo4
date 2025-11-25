from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
import json
import os
from app.core.encryption import encrypt_data, decrypt_data


class GCPConnection(models.Model):
    id = fields.IntField(pk=True)
    credentials = fields.JSONField(null=False)
    project_info = fields.JSONField(null=False)
    date = fields.DateField(null=False)

    # to maintain state on project
    status = fields.BooleanField(default=False, null=False)

    # to maintain budget of project
    monthly_budget = fields.IntField(null=True)
    yearly_budget = fields.IntField(null=True)
    quarterly_budget = fields.IntField(null=True)

    export = fields.BooleanField(default=False, null=False)
    dataset_id = fields.CharField(max_length=100, null=True)
    billing_account_id = fields.CharField(max_length=100, null=True)

    project = fields.ForeignKeyField('models.Project', related_name='gcp_connection')

    async def save(self, *args, **kwargs):
        # Hardcoded encryption key (replace with your actual key)
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        # Encrypt credentials JSON payload
        credentials_json = json.dumps(self.credentials)
        encrypted_credentials = encrypt_data(credentials_json, encryption_key)

        # Store the encrypted credentials as a JSON object
        self.credentials = {"encrypted_credentials": encrypted_credentials}

        await super().save(*args, **kwargs)

    async def decrypt_credentials(self):
        # Hardcoded encryption key (replace with your actual key)
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        # Retrieve the encrypted credentials
        encrypted_credentials = self.credentials.get("encrypted_credentials")

        # Decrypt the credentials
        decrypted_credentials_json = decrypt_data(encrypted_credentials, encryption_key)

        # Convert the JSON string back to a dictionary
        decrypted_credentials = json.loads(decrypted_credentials_json)

        return decrypted_credentials

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
GCPConnection_Pydantic = pydantic_model_creator(GCPConnection, name="GCPConnection")
GCPConnectionIn_Pydantic = pydantic_model_creator(GCPConnection, name="GCPConnectionIn", exclude_readonly=True)
