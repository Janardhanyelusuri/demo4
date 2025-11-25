from app.core.config import settings

TORTOISE_ORM = {
    "connections": {"default": settings.DATABASE_URL},
    "apps": {
        "models": {
            # add here to add in migrations
            "models": [
                "app.models.user",
                "app.models.project",
                "app.models.aws",
                "app.models.azure",
                "app.models.gcp",
                "app.models.project_access",
                "app.models.database",
                "app.models.snowflake",
                "app.models.sync_status",
                "app.models.alert",
                "app.models.alert_integration",
                "app.models.service",
                "app.models.dashboard_request",
                "app.models.dashboard",
                "app.models.tags",
                "app.models.resources",
                "app.models.resources_tags",
                "aerich.models"
            ],
            "default_connection": "default",
        },
    },
}
