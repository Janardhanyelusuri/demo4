from tortoise import fields
from tortoise.models import Model

class ResourceTag(Model):
    id = fields.IntField(pk=True)  # Add the primary key to ResourceTag

    resource = fields.ForeignKeyField('models.Resource', related_name='resource_tags', on_delete=fields.CASCADE)
    tag = fields.ForeignKeyField('models.Tag', related_name='tagged_resources', on_delete=fields.CASCADE, null=True)

    class Meta:
        table = 'resource_tag'
        unique_together = (('resource', 'tag'),)

    def __str__(self):
        return f"Resource: {self.resource.resource_id} - Tag: {self.tag.key if self.tag else 'NULL'}: {self.tag.value if self.tag else 'NULL'}"