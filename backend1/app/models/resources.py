from tortoise import fields
from tortoise.models import Model

class Resource(Model):
    """
    ORM Model representing the Resource entity.
    """
    id = fields.IntField(pk=True)  # Auto-generated integer primary key
    resource_id = fields.CharField(max_length=1024, null=True)  # Set unique=True to prevent duplicates
    resource_name = fields.CharField(max_length=255, null=True)
    region_id = fields.CharField(max_length=100, null=True)
    region_name = fields.CharField(max_length=255, null=True)
    service_category = fields.CharField(max_length=255, null=True)
    service_name = fields.CharField(max_length=255, null=True)
    resource_group_name = fields.CharField(max_length=255, null=True)
    cloud_platform = fields.CharField(max_length=255, null=True)
    
    project = fields.ForeignKeyField('models.Project', related_name='resources', on_delete=fields.CASCADE)
    tag = fields.ForeignKeyField('models.Tag', related_name='resource_tags', on_delete=fields.CASCADE, to_field='tag_id', null=True)

    class Meta:
        table = 'resource_dim'

    def __str__(self):
        return f"{self.resource_name} ({self.resource_id})"