# app/models/llm_cache.py

from tortoise import fields
from tortoise.models import Model


class LLMCache(Model):
    """
    Model for caching LLM recommendation outputs to avoid redundant API calls.
    Hash key is generated from input parameters (start_date, end_date, resource_type, resource_id).
    """
    id = fields.IntField(pk=True)
    hash_key = fields.CharField(max_length=64, unique=True, index=True, description="MD5 hash of input parameters")
    schema_name = fields.CharField(max_length=255, description="Schema/project name")
    cloud_platform = fields.CharField(max_length=50, description="Cloud platform (aws, azure, gcp)")
    resource_type = fields.CharField(max_length=100, description="Resource type (vm, storage, ec2, s3, etc.)")
    resource_id = fields.TextField(null=True, description="Specific resource ID (null if fetching all)")
    start_date = fields.DateField(null=True, description="Analysis start date")
    end_date = fields.DateField(null=True, description="Analysis end date")
    output_json = fields.JSONField(description="Cached LLM response as JSON")
    created_at = fields.DatetimeField(auto_now_add=True, description="Cache creation timestamp")
    updated_at = fields.DatetimeField(auto_now=True, description="Cache last update timestamp")

    class Meta:
        table = "llm_cache"

    def __str__(self):
        return f"LLMCache({self.hash_key[:8]}... - {self.cloud_platform}/{self.resource_type})"
