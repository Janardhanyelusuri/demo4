from pydantic import BaseModel
from datetime import date
from typing import Dict, List, Optional


class TableColumnsResponse(BaseModel):
    table: str
    columns: List[str]


class AWSConnectionRequest(BaseModel):
    aws_access_key: str
    aws_secret_key: str


class AWSConnectionResponse(BaseModel):
    status: bool
    message: str


class SlackConnectionResponse(BaseModel):
    status: bool
    message: str

class TeamsConnectionResponse(BaseModel):
    status: bool
    message: str


class WebhookURL(BaseModel):
    webhook_url: str


class Message(BaseModel):
    text: str
    username: str = "WebhookBot"  # Default value
    icon_emoji: str = ":robot_face:"  # Default value


class SlackMessage(BaseModel):
    text: str = None  # Optional, will be set based on message_type if not provided
    username: str = "WebhookBot"  # Default value
    icon_emoji: str = ":robot_face:"  # Default value
    message_type: str = None  # Optional, used internally to determine text


class SnowflakeConnectionRequest(BaseModel):
    account_name: str
    user_name: str
    password: str


class SnowflakeConnectionResponse(BaseModel):
    status: bool
    message: str


class AzureConnectionRequest(BaseModel):
    azure_client_id: str
    azure_client_secret: str
    azure_tenant_id: str


class SubscriptionInfo(BaseModel):
    subscription_id: str
    display_name: str


class WarehouseInfo(BaseModel):
    name: str
    state: str


class SubscriptionsResponse(BaseModel):
    subscriptions: List[SubscriptionInfo]


class WarehousesResponse(BaseModel):
    warehouses: List[WarehouseInfo]


class AzureConnectionResponse(BaseModel):
    status: bool
    message: str


class GCPConnectionRequest(BaseModel):
    credentials_file: str
    project_name: str


class GCPConfirmRequest(BaseModel):
    service_account_number: str
    service_account_id: str
    project_name: str
    date: date


class GCPConnectionResponse(BaseModel):
    status: bool
    message: str


class AzureConfirmRequest(BaseModel):
    azure_client_id: str
    azure_client_secret: str
    azure_tenant_id: str
    subscription: str
    date: date
    # param1: bool
    # param2: bool


class AWSConfirmRequest(BaseModel):
    aws_access_key: str
    aws_secret_key: str
    date: date


class GCPConfirmationResponse(BaseModel):
    status: bool
    message: str


class AWSConfirmationResponse(BaseModel):
    status: bool


class ProjectInfo(BaseModel):
    project_id: str
    project_name: str


class ProjectsResponse(BaseModel):
    projects: List[ProjectInfo]


class CheckProjectNameRequest(BaseModel):
    name: str


class CheckProjectNameResponse(BaseModel):
    status: bool
    message: str


class CheckDashboardNameRequest(BaseModel):
    name: str


class CheckDashboardNameResponse(BaseModel):
    status: bool
    message: str


class DeleteAwsProjectConfirmation(BaseModel):
    delete_s3: bool = False
    delete_export: bool = False
    delete_container: bool = False


class DeleteAwsS3Bucket(BaseModel):
    aws_access_key: str
    aws_secret_key: str
    aws_region: str
    s3_bucket: str


class DeleteAwsExport(BaseModel):
    aws_access_key: str
    aws_secret_key: str
    aws_region: str
    export_name: str


class BillingAccountResponse(BaseModel):
    name: str
    display_name: str
    parent: str

class QueriesRequest(BaseModel):
    cloud_provider: Optional[str] = None
    query_type: str
    project_id: str = ""
    dashboard_id: str = ""
    granularity: str = ""
    resource_names: str = ""
    tag_id: int = 0
    service_names: str = ""
    duration: str = ""

class TagRequest(BaseModel):
    tag_id: int

class GenerateRecommendationRequest(BaseModel):
    data: List = []  # Assuming data is a list

class GetUtilizationTable(BaseModel):
    provider: str
    project_id: int
