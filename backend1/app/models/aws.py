from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
import os
from app.core.encryption import encrypt_data, decrypt_data


class AwsConnection(models.Model):
    id = fields.IntField(pk=True)
    aws_access_key = fields.CharField(max_length=100, null=False)
    aws_secret_key = fields.CharField(max_length=100, null=False)
    date = fields.DateField(null=False)
    
    # to maintain state on project
    status = fields.BooleanField(default=True, null=False)
    
    # to maintain budget of project
    monthly_budget = fields.IntField(null=True)
    yearly_budget = fields.IntField(null=True)
    quarterly_budget = fields.IntField(null=True)
    
    # if enable export: false > generate export and update export_location
    export = fields.BooleanField(default=False, null=False)
    
    # enable export: false > take s3 path
    export_location = fields.CharField(max_length=100, null=True)
    
    project = fields.ForeignKeyField('models.Project', related_name='aws_connection')

    async def save(self, *args, **kwargs):
        # Retrieve encryption key from environment variable
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        # Convert encryption key from string to bytes
        encryption_key = bytes.fromhex(encryption_key)

        # Encrypt aws_access_key
        encrypted_access_key = encrypt_data(self.aws_access_key, encryption_key)
        self.aws_access_key = encrypted_access_key

        # Encrypt aws_secret_key
        encrypted_secret_key = encrypt_data(self.aws_secret_key, encryption_key)
        self.aws_secret_key = encrypted_secret_key

        await super().save(*args, **kwargs)

    async def decrypt_credentials(self):
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("Encryption key not found in environment variables")

        encryption_key = bytes.fromhex(encryption_key)

        decrypted_aws_access_key = decrypt_data(self.aws_access_key, encryption_key)
        decrypted_aws_secret_key = decrypt_data(self.aws_secret_key, encryption_key)

        return {
            "aws_access_key": decrypted_aws_access_key,
            "aws_secret_key": decrypted_aws_secret_key
        }

    class PydanticMeta:
        model_config = {'extra': 'allow'}


# Create Pydantic models from Tortoise ORM models
AwsConnection_Pydantic = pydantic_model_creator(AwsConnection, name="AwsConnection")
AwsConnectionIn_Pydantic = pydantic_model_creator(AwsConnection, name="AwsConnectionIn", exclude_readonly=True)
