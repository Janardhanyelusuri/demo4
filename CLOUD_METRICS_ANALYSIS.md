# Cloud Metrics & Recommendations Application - Technical Analysis

## Executive Summary

This is a **Cloud FinOps Platform** that ingests cost and performance metrics from AWS and Azure, stores them in PostgreSQL, and uses AI (Azure OpenAI) to generate cost optimization recommendations for cloud resources.

**Key Capabilities:**
- Multi-cloud metrics ingestion (AWS CloudWatch, Azure Monitor)
- Cost data aggregation from billing exports (AWS FOCUS, Azure FOCUS)
- AI-powered cost optimization recommendations
- Metric-driven analysis with LLM integration
- Background task processing with Celery
- RESTful API with FastAPI

---

## Architecture Overview

### Technology Stack

**Backend:**
- **Framework:** FastAPI (Python)
- **ORM:** Tortoise ORM with Pydantic models
- **Database:** PostgreSQL (with schemas per project)
- **Task Queue:** Celery with Redis/RabbitMQ
- **AI/ML:** Azure OpenAI (GPT-based models)
- **Cloud SDKs:** boto3 (AWS), azure-sdk (Azure)

**Data Pipeline:**
- **Bronze Layer:** Raw ingestion from S3/Blob Storage
- **Silver Layer:** Cleaned and deduplicated data
- **Gold Layer:** Business logic views and aggregations

**Frontend:** (Exists in `/frontend` directory)

---

## 1. Metrics Ingestion Architecture

### AWS Metrics Ingestion

#### 1.1 Cost Data Ingestion (`backend1/app/ingestion/aws/main.py`)

**Data Sources:**
- **AWS Cost Explorer (CE):** Historical cost data
- **AWS FOCUS Exports:** Detailed billing data in FOCUS format (parquet/CSV)
- **S3 Storage:** Export destination

**Workflow:**
1. **Export Creation:**
   - Creates S3 bucket with proper IAM policies
   - Enables FOCUS export for billing data
   - Files: `aws_cur/exports/scripts/create_export.py`

2. **Data Ingestion:**
   ```python
   aws_run_ingestion(project_name, monthly_budget, aws_access_key,
                     aws_secret_key, aws_region, s3_bucket, s3_prefix,
                     export_name, billing_period)
   ```

3. **Processing Steps:**
   - Downloads latest parquet/CSV files from S3
   - Generates MD5 hash keys for deduplication
   - Filters new data only (hash-based comparison)
   - Loads data in chunks (100K rows) to PostgreSQL
   - Executes SQL transformations (bronze → silver → gold)

**Key Files:**
- `backend1/app/ingestion/aws/main.py` - Main orchestration
- `backend1/app/ingestion/aws/s3.py` - S3 operations
- `backend1/app/ingestion/aws/postgres_operations.py` - DB operations
- `backend1/app/ingestion/aws/export_ops.py` - FOCUS export management

#### 1.2 CloudWatch Metrics (`backend1/app/ingestion/aws/resource_metrics.py`)

**Metrics Collected:**
```python
services_metrics = {
    'AWS/EC2': ['CPUUtilization', 'DiskReadOps', 'DiskWriteOps',
                'NetworkIn', 'NetworkOut', 'StatusCheckFailed'],
    'AWS/RDS': ['CPUUtilization', 'DatabaseConnections', 'FreeableMemory',
                'ReadIOPS', 'WriteIOPS', 'FreeStorageSpace'],
    'AWS/S3': ['BucketSizeBytes', 'NumberOfObjects', 'AllRequests',
               'GetRequests', 'PutRequests', '4xxErrors', '5xxErrors'],
    'AWS/Lambda': ['Invocations', 'Errors', 'Duration', 'Throttles'],
    'AWS/ELB': ['RequestCount', 'HealthyHostCount', 'Latency'],
    'AWS/DynamoDB': ['ConsumedReadCapacityUnits', 'ThrottledRequests'],
    'AWS/EBS': ['VolumeReadOps', 'VolumeWriteOps', 'BurstBalance'],
    'AWS/ECS': ['CPUUtilization', 'MemoryUtilization'],
    'AWS/ApiGateway': ['Count', '4xxError', '5xxError', 'Latency']
}
```

**Collection Process:**
- Fetches last 14 days of metrics
- 1-hour granularity (period = 3600)
- Stores in `<schema>.metrics_details` table
- Runs via `fetch_and_store_cloudwatch_metrics()`

#### 1.3 S3 Metrics (`backend1/app/ingestion/aws/metrics_s3.py`)

**Purpose:** Collects S3 bucket-level metrics for storage optimization

**Implementation:**
```python
metrics_dump(aws_access_key, aws_secret_key, aws_region, schema_name)
```

**SQL Transformations:**
- `sql/bronze_s3_metrics.sql` - Raw S3 metrics
- `sql/silver_s3_metrics.sql` - Cleaned metrics
- `sql/gold_s3_metrics.sql` - Business aggregations

---

### Azure Metrics Ingestion

#### 2.1 Cost Data Ingestion (`backend1/app/ingestion/azure/main.py`)

**Data Sources:**
- **Azure FOCUS Export:** Billing data in FOCUS format
- **Azure Blob Storage:** Export container

**Workflow:**
1. **Export Creation:**
   - Creates blob container via Azure Functions
   - Configures Cost Management Export API
   - Files: `backend1/app/ingestion/azure/azure_ops.py`

2. **Data Ingestion:**
   ```python
   azure_main(project_name, budget, tenant_id, client_id,
              client_secret, storage_account_name, container_name,
              subscription_id)
   ```

3. **Processing Steps:**
   - Downloads parquet files from Blob Storage
   - Creates hash keys for deduplication
   - Filters new records only
   - Loads to PostgreSQL bronze table
   - Executes SQL transformations

**Key Files:**
- `backend1/app/ingestion/azure/main.py` - Main orchestration
- `backend1/app/ingestion/azure/blob.py` - Blob storage operations
- `backend1/app/ingestion/azure/postgres_operation.py` - DB operations

#### 2.2 Azure Monitor Metrics

##### Virtual Machine Metrics (`backend1/app/ingestion/azure/metrics_vm.py`)

**Metrics Collected:**
```python
RELEVANT_METRICS = [
    "Percentage CPU",
    "Available Memory Bytes",
    "Disk Read Operations/Sec",
    "Disk Write Operations/Sec",
    "Network In Total",
    "Network Out Total",
    "OS Disk IOPS Consumed Percentage",
    "Data Disk IOPS Consumed Percentage"
]
```

**Features:**
- Dynamic metric discovery per VM
- 90-day historical data
- Daily granularity (P1D interval)
- Handles Azure API rate limits
- Stores instance type and vCPU mapping

**Implementation:**
```python
metrics_dump(tenant_id, client_id, client_secret, subscription_id,
             schema_name, "bronze_azure_vm_metrics")
```

##### Storage Account Metrics (`backend1/app/ingestion/azure/metrics_storage_account.py`)

**Metrics:**
- `UsedCapacity`, `Transactions`, `Ingress`, `Egress`
- `Availability`, `SuccessServerLatency`
- `BlobCapacity`, `FileCapacity`

##### Public IP Metrics (`backend1/app/ingestion/azure/metrics_public_ip.py`)

**Purpose:** Tracks public IP utilization and DDoS metrics

---

## 2. Data Transformation Pipeline

### Bronze Layer (Raw Data)
- Direct ingestion from cloud exports
- Hash-based deduplication
- Minimal transformations
- Tables: `bronze_azure_focus`, `bronze_focus_aws_data`

### Silver Layer (Cleaned Data)
- Data type conversions
- Null handling
- Join operations with dimension tables
- Tables: `silver_focus_aws`, `silver_azure_resource_dim`

**Example SQL:** `backend1/app/ingestion/azure/sql/silver.sql`
```sql
CREATE OR REPLACE VIEW <schema>.silver_azure_resource_dim AS
SELECT DISTINCT
    resource_id,
    resource_name,
    resource_type,
    service_category,
    service_name,
    region,
    availability_zone,
    publisher_name,
    publisher_type
FROM <schema>.bronze_azure_focus
WHERE resource_id IS NOT NULL;
```

### Gold Layer (Business Logic)
- Cost aggregations
- Budget tracking
- Forecast calculations
- Resource utilization summaries
- Tables: `gold_aws_fact_focus`, `gold_azure_cost_by_service`

**Example SQL:** `backend1/app/ingestion/aws/sql/parquet_gold_views.sql`
```sql
CREATE OR REPLACE VIEW <schema>.gold_aws_resource_dim AS
SELECT DISTINCT
    resource_id,
    service_code,
    service_name,
    region,
    pricing_category,
    charge_category,
    commitment_discount_type
FROM <schema>.silver_focus_aws
WHERE resource_id IS NOT NULL;
```

---

## 3. AI-Powered Recommendations System

### Architecture Overview

**Components:**
1. **LLM Engine:** Azure OpenAI GPT-4 models
2. **Prompt Engineering:** Resource-specific prompts
3. **Data Aggregation:** Metrics + Cost data joins
4. **Caching:** Hash-based result caching
5. **Task Management:** Cancellable background tasks

### 3.1 AWS Recommendations

#### EC2 Instance Optimization (`backend1/app/ingestion/aws/llm_ec2_vpc_integration.py`)

**Data Fetched:**
```sql
WITH metric_agg AS (
    SELECT
        m.instance_id,
        m.instance_type,
        m.region,
        m.metric_name,
        AVG(metric_value) AS avg_value,
        MAX(metric_value) AS max_value,
        FIRST_VALUE(timestamp) AS max_date
    FROM fact_ec2_metrics m
    JOIN dim_ec2_instance i ON m.instance_id = i.instance_id
    WHERE m.timestamp BETWEEN :start_date AND :end_date
    GROUP BY instance_id, metric_name
)
SELECT
    instance_id,
    instance_type,
    metrics_json,
    SUM(billed_cost) AS billed_cost
FROM metric_agg
JOIN gold_aws_fact_focus ff ON ff.resource_id LIKE '%' || instance_id || '%'
GROUP BY instance_id;
```

**Prompt Structure:**
```python
prompt = f"""
You are a cloud cost optimization expert for AWS. Analyze the following EC2 instance:

**EC2 Instance Details:**
- Instance ID: {instance_id}
- Instance Type: {instance_type}
- Region: {region}
- Total Billed Cost: ${billed_cost:.2f}

**Performance Metrics:**
- CPU Utilization: Avg {cpu_avg:.2f}%, Max {cpu_max:.2f}%
- Network In: Avg {network_in_avg:.2f} bytes
- Disk Read Ops: Avg {disk_read_avg:.2f} ops/sec

**Your Task:**
1. Is the instance right-sized?
2. Spot/Reserved Instance opportunities?
3. Scheduling opportunities?

**Response Format (JSON only):**
{{
  "recommendations": {{
    "effective_recommendation": {{
      "text": "Primary recommendation",
      "saving_pct": <percentage>
    }},
    "additional_recommendation": [...]
  }},
  "cost_forecasting": {{
    "monthly": <cost>,
    "annually": <cost>
  }},
  "anomalies": [...],
  "contract_deal": {{...}}
}}
"""
```

**LLM Call:** `backend1/app/core/genai.py`
```python
def llm_call(prompt: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION
    )
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a FinOps assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message.content
```

**Features:**
- Exponential backoff for rate limits
- 3-second delay between requests
- 5 retry attempts
- JSON extraction from LLM response

#### VPC Resource Optimization

**Supported Resources:**
- NAT Gateways
- VPN Connections
- VPC Endpoints
- VPCs

**Data Query:** Similar to EC2, fetches from `fact_vpc_metrics`

**Recommendations:**
- NAT Gateway → NAT instance alternatives
- Unused VPN connections
- VPC endpoint optimization

#### S3 Bucket Optimization (`backend1/app/ingestion/aws/llm_s3_integration.py`)

**Analysis Focus:**
- Storage class optimization
- Lifecycle policies
- Access patterns
- Data transfer costs

### 3.2 Azure Recommendations

#### Virtual Machine Optimization (`backend1/app/ingestion/azure/llm_analysis.py`)

**Data Fetched:**
```python
def _format_metrics_for_llm(resource_data, resource_type="vm"):
    RELEVANT_METRICS = {
        "vm": [
            "Percentage CPU",
            "Available Memory Bytes",
            "Disk Read Operations/Sec",
            "Network In Total"
        ]
    }
    formatted_metrics = {}
    for metric_name in unique_metric_names:
        entry = {
            "Avg": resource_data.get(f"metric_{metric_name}_Avg"),
            "Max": resource_data.get(f"metric_{metric_name}_Max"),
            "MaxDate": resource_data.get(f"metric_{metric_name}_MaxDate")
        }
        formatted_metrics[metric_name] = entry
    return formatted_metrics
```

**Prompt Rules:**
```python
prompt = f"""
Azure VM FinOps. Analyze metrics, output JSON only.

CONTEXT: {resource_id} | SKU: {current_sku} | Cost: ${billed_cost:.2f}

METRICS:
{json.dumps(formatted_metrics, indent=2)}

RULES:
1. Use exact values+units from METRICS (e.g., "CPU: 12.3% avg")
2. State exact SKU names+specs
3. Show calculations
4. BANNED: "consider", "review", "optimize"
5. Use action verbs: Downsize, Upsize, Purchase, Enable
6. Always include units: %, GB, vCPU, $

DECIDE: Primary optimization? VM size change? 2-3 additional optimizations?

JSON (MUST: 2-3 additional_recommendation, 2-3 anomalies):
{{...}}
"""
```

**Cost Forecasting:**
```python
def _extrapolate_costs(billed_cost: float, duration_days: int):
    avg_daily_cost = billed_cost / duration_days
    monthly = avg_daily_cost * 30.4375
    annually = avg_daily_cost * 365
    return {"monthly": monthly, "annually": annually}
```

#### Storage Account Optimization

**Analysis Focus:**
- Access tier optimization (Hot/Cool/Archive)
- Redundancy optimization (LRS/GRS/ZRS)
- Transaction patterns
- Capacity planning

**Prompt Engineering:**
```python
prompt = f"""
Azure Storage FinOps. Current: {current_sku} {current_tier}

METRICS: {formatted_metrics}

RULES:
1. Use exact values (e.g., "478.3 GB", "1,247 tx/day")
2. State exact pricing (e.g., "Hot $0.0184/GB → Cool $0.01/GB")
3. Show calculations
4. Every recommendation needs explanation with metrics

JSON: {{...}}
"""
```

### 3.3 Caching & Task Management

#### LLM Cache (`backend1/app/core/llm_cache_utils.py`)

**Purpose:** Avoid duplicate LLM calls for same parameters

**Hash Key Generation:**
```python
def generate_cache_hash_key(cloud_platform, schema_name, resource_type,
                           start_date, end_date, resource_id):
    key_str = f"{cloud_platform}|{schema_name}|{resource_type}|{start_date}|{end_date}|{resource_id}"
    return hashlib.sha256(key_str.encode()).hexdigest()
```

**Cache Storage:**
```python
async def save_to_cache(hash_key, cloud_platform, schema_name,
                       resource_type, output_json):
    await LLMCache.create(
        hash_key=hash_key,
        cloud_platform=cloud_platform,
        schema_name=schema_name,
        resource_type=resource_type,
        output_json=output_json,
        created_at=datetime.utcnow()
    )
```

**Cache Retrieval:**
```python
async def get_cached_result(hash_key):
    cached = await LLMCache.filter(hash_key=hash_key).first()
    if cached:
        return cached.output_json
    return None
```

#### Task Manager (`backend1/app/core/task_manager.py`)

**Features:**
- Cancellable LLM tasks
- Project-level task tracking
- Task status monitoring

**Usage:**
```python
# Create task
task_id = task_manager.create_task(
    task_type="llm_analysis",
    metadata={"cloud": "azure", "resource_type": "vm"}
)

# Check cancellation
if task_manager.is_cancelled(task_id):
    return None

# Complete task
task_manager.complete_task(task_id)
```

---

## 4. API Endpoints

### 4.1 Connection Management

**AWS Connection:** `POST /api/v1/aws/`
```python
@router.post('/', response_model=AwsConnection_Pydantic)
async def add_aws_connection(aws_connection: AwsConnectionIn_Pydantic):
    # Validate credentials
    # Calculate budgets
    # Create FOCUS export
    # Trigger ingestion task
    task_run_ingestion_aws.delay({...})
```

**Azure Connection:** `POST /api/v1/azure/`
```python
@router.post('/', response_model=AzureConnection_Pydantic)
async def add_azure_connection(azure_connection: AzureConnectionIn_Pydantic):
    # Validate credentials
    # Create blob container
    # Create FOCUS export
    # Trigger ingestion task
    task_run_ingestion_azure.delay({...})
```

### 4.2 LLM Recommendations

**AWS Recommendations:** `POST /api/v1/llm/aws/{project_id}`
```python
@router.post("/aws/{project_id}", response_model=LLMResponse)
async def llm_aws(project_id: str, payload: LLMRequest):
    # Route based on resource_type: s3, ec2, vpc
    result = run_llm_analysis_ec2_vpc(
        resource_type=payload.resource_type,
        schema_name=schema,
        start_date=payload.start_date,
        end_date=payload.end_date,
        resource_id=payload.resource_id
    )
    return LLMResponse(...)
```

**Azure Recommendations:** `POST /api/v1/llm/azure/{project_id}`
```python
@router.post("/azure/{project_id}", response_model=LLMResponse)
async def llm_azure(project_id: str, payload: LLMRequest):
    # Check cache
    hash_key = generate_cache_hash_key(...)
    cached_result = await get_cached_result(hash_key)

    if cached_result:
        return cached_result

    # Create cancellable task
    task_id = task_manager.create_task(...)

    # Run LLM analysis
    result = run_llm_analysis(
        payload.resource_type,
        schema,
        payload.start_date,
        payload.end_date,
        payload.resource_id,
        task_id=task_id
    )

    # Save to cache
    await save_to_cache(hash_key, ...)

    return LLMResponse(...)
```

**Task Cancellation:** `POST /api/v1/llm/tasks/{task_id}/cancel`
```python
@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    success = task_manager.cancel_task(task_id)
    return {"status": "success", "task_id": task_id}
```

### 4.3 Resource Discovery

**Get Resource IDs:** `GET /api/v1/llm/{cloud}/{project_id}/resources/{resource_type}`
```python
@router.get("/{cloud_platform}/{project_id}/resources/{resource_type}")
async def get_resource_ids(cloud_platform: str, project_id: str, resource_type: str):
    # For Azure VMs
    query = f"""
        SELECT DISTINCT resource_id, resource_name
        FROM {schema_name}.gold_azure_resource_dim
        WHERE service_category = 'Compute'
        AND resource_id LIKE '%/virtualmachines/%'
        LIMIT 100;
    """
    return {"resource_ids": [...], "count": len(resource_ids)}
```

---

## 5. Background Task Processing

### Celery Worker (`backend1/app/worker/celery_worker.py`)

**Tasks:**

1. **Daily Ingestion:** `task_run_daily_ingestion()`
   - Runs for all active projects
   - Fetches latest cost exports
   - Updates metrics tables

2. **AWS Export Creation:** `task_create_aws_export()`
   - Creates S3 bucket
   - Enables FOCUS export

3. **AWS Ingestion:** `task_run_ingestion_aws()`
   - Downloads from S3
   - Processes billing data
   - Collects CloudWatch metrics

4. **Azure Export Creation:** `task_create_azure_export()`
   - Creates blob container
   - Configures Cost Management Export

5. **Azure Ingestion:** `task_run_ingestion_azure()`
   - Downloads from Blob Storage
   - Processes billing data
   - Collects Azure Monitor metrics

6. **Alert Processing:**
   - `run_daily_alerts()`
   - `run_weekly_alerts()`
   - `run_monthly_alerts()`

**Configuration:** `backend1/app/worker/celery_app.py`
```python
celery_app = Celery('worker',
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    'run-daily-alerts': {
        'task': 'run_daily_alerts',
        'schedule': crontab(hour=0, minute=0)
    },
    'run-daily-ingestion': {
        'task': 'task_run_daily_ingestion',
        'schedule': crontab(hour=2, minute=0)
    }
}
```

---

## 6. Data Models

### Project Model
```python
class Project(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=100)
    cloud_platform = fields.CharField(max_length=20)  # aws/azure/gcp
    status = fields.BooleanField(default=False)
```

### AWS Connection Model (`backend1/app/models/aws.py`)
```python
class AwsConnection(models.Model):
    id = fields.IntField(pk=True)
    aws_access_key = fields.CharField(max_length=100)  # Encrypted
    aws_secret_key = fields.CharField(max_length=100)  # Encrypted
    date = fields.DateField()
    status = fields.BooleanField(default=True)
    monthly_budget = fields.IntField()
    yearly_budget = fields.IntField()
    export = fields.BooleanField(default=False)
    export_location = fields.CharField(max_length=100)
    project = fields.ForeignKeyField('models.Project')
```

### Azure Connection Model (`backend1/app/models/azure.py`)
```python
class AzureConnection(models.Model):
    id = fields.IntField(pk=True)
    azure_tenant_id = fields.CharField(max_length=100)  # Encrypted
    azure_client_id = fields.CharField(max_length=100)  # Encrypted
    azure_client_secret = fields.CharField(max_length=100)  # Encrypted
    subscription_info = fields.JSONField()
    storage_account_name = fields.CharField(max_length=100)
    container_name = fields.CharField(max_length=100)
    project = fields.ForeignKeyField('models.Project')
```

### LLM Cache Model (`backend1/app/models/llm_cache.py`)
```python
class LLMCache(models.Model):
    id = fields.IntField(pk=True)
    hash_key = fields.CharField(max_length=64, unique=True)
    cloud_platform = fields.CharField(max_length=20)
    schema_name = fields.CharField(max_length=100)
    resource_type = fields.CharField(max_length=50)
    start_date = fields.DateField()
    end_date = fields.DateField()
    resource_id = fields.CharField(max_length=255)
    output_json = fields.JSONField()
    created_at = fields.DatetimeField(auto_now_add=True)
```

---

## 7. Security & Encryption

### Credential Encryption (`backend1/app/core/encryption.py`)

**Encryption:**
```python
def encrypt_data(data: str, key: bytes) -> str:
    cipher = Fernet(base64.urlsafe_b64encode(key))
    encrypted = cipher.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()
```

**Decryption:**
```python
def decrypt_data(encrypted_data: str, key: bytes) -> str:
    cipher = Fernet(base64.urlsafe_b64encode(key))
    decrypted = cipher.decrypt(base64.urlsafe_b64decode(encrypted_data))
    return decrypted.decode()
```

**Usage in Models:**
```python
async def save(self, *args, **kwargs):
    encryption_key = os.getenv("ENCRYPTION_KEY")
    encryption_key = bytes.fromhex(encryption_key)

    encrypted_access_key = encrypt_data(self.aws_access_key, encryption_key)
    self.aws_access_key = encrypted_access_key

    await super().save(*args, **kwargs)
```

---

## 8. Database Schema Design

### Per-Project Schema Pattern

Each project gets its own PostgreSQL schema:
- **Schema Name:** Lowercase project name
- **Isolation:** Prevents data mixing between projects
- **Scaling:** Easy to drop/recreate per project

**Example Schemas:**
- `myproject_aws` - AWS project
- `myproject_azure` - Azure project

### Table Naming Convention

**Bronze Layer:**
- `bronze_azure_focus` - Raw Azure billing
- `bronze_focus_aws_data` - Raw AWS billing
- `bronze_azure_vm_metrics` - Raw VM metrics

**Silver Layer:**
- `silver_azure_resource_dim` - Cleaned resources
- `silver_focus_aws` - Cleaned AWS billing

**Gold Layer:**
- `gold_azure_cost_by_service` - Cost aggregations
- `gold_aws_fact_focus` - AWS cost facts
- `fact_ec2_metrics` - EC2 metric facts
- `dim_ec2_instance` - EC2 dimension table

---

## 9. Key Optimization Techniques

### 9.1 Deduplication Strategy

**Hash-Based Deduplication:**
```python
def generate_hash_key(df):
    df = df.fillna("")
    def hash_row(row):
        concatenated_values = "".join(map(str, row))
        return hashlib.md5(concatenated_values.encode('utf-8')).hexdigest()
    df['hash_key'] = df.apply(hash_row, axis=1)
    return df

def filter_new_data(df, schema_name, table_name):
    existing_hash_keys = fetch_existing_hash_keys(schema_name, table_name)
    new_data = df[~df['hash_key'].isin(existing_hash_keys)]
    return new_data
```

**Benefits:**
- Avoids duplicate billing records
- Idempotent ingestion
- Reduces storage costs

### 9.2 Chunked Loading

```python
chunk_size = 100000
for start in range(0, len(new_data), chunk_size):
    chunk = new_data.iloc[start:start + chunk_size]
    dump_to_postgresql(chunk, schema_name, table_name)
```

**Benefits:**
- Handles large datasets
- Prevents memory overflow
- Better error recovery

### 9.3 LLM Rate Limiting

**Exponential Backoff:**
```python
for attempt in range(MAX_RETRIES):
    try:
        response = client.chat.completions.create(...)
        time.sleep(REQUEST_DELAY)  # 3 seconds
        return response
    except RateLimitError:
        backoff_time = INITIAL_BACKOFF * (2 ** attempt)
        time.sleep(backoff_time)
```

**Token Optimization:**
```python
max_tokens=800  # Limits response size
temperature=0.3  # More deterministic
```

---

## 10. Recommendation Output Format

### Standard JSON Response

```json
{
  "resource_id": "i-1234567890abcdef0",
  "recommendations": {
    "effective_recommendation": {
      "text": "Downsize from t3.large to t3.medium (4 vCPU → 2 vCPU)",
      "explanation": "CPU utilization averages 12.3% with max 35.7%, indicating over-provisioning",
      "saving_pct": 50
    },
    "additional_recommendation": [
      {
        "text": "Purchase 1-year Reserved Instance",
        "explanation": "Stable workload running 24/7 for past 90 days",
        "saving_pct": 40
      },
      {
        "text": "Enable auto-shutdown during off-hours (8PM-6AM)",
        "explanation": "Network activity drops to <1% during these hours",
        "saving_pct": 25
      }
    ],
    "base_of_recommendations": [
      "CPU Utilization: 12.3% avg, 35.7% max",
      "Network In: 1.2 MB/hr avg",
      "Current cost: $70.08/month"
    ]
  },
  "cost_forecasting": {
    "monthly": 35.04,
    "annually": 420.48
  },
  "anomalies": [
    {
      "metric_name": "CPUUtilization",
      "timestamp": "2025-01-15 14:23:00",
      "value": 87.3,
      "reason_short": "Deployment spike, returned to baseline within 1 hour"
    }
  ],
  "contract_deal": {
    "assessment": "bad",
    "for_sku": "t3.large",
    "reason": "On-Demand pricing 62% more expensive than 1yr RI",
    "monthly_saving_pct": 40,
    "annual_saving_pct": 40
  }
}
```

---

## 11. Monitoring & Observability

### Celery Flower Dashboard
- **URL:** `http://localhost:5556/workers`
- **Features:** Task monitoring, worker status, task history

### Database Connection
```bash
docker compose exec db psql -U postgres
\c <database_name>
\dt  # List tables
```

### Application Logs
```bash
docker compose logs -f server
```

---

## 12. Deployment & Configuration

### Environment Variables (`.env`)

```bash
# Database
DB_HOST_NAME=postgres
DB_NAME=cloudmeter
DB_USER_NAME=postgres
DB_PASSWORD=<password>
DB_PORT=5432

# Celery
CELERY_BROKER=redis://redis:6379/0
CELERY_BACKEND=redis://redis:6379/0

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<instance>.openai.azure.com/
AZURE_OPENAI_KEY=<key>
AZURE_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_VERSION=2024-02-01

# Encryption
ENCRYPTION_KEY=<hex_key>
```

### Docker Deployment

**Start Services:**
```bash
docker compose up --build
```

**Database Migrations:**
```bash
docker compose exec server aerich init -t app.db.base.TORTOISE_ORM
docker compose exec server aerich upgrade
```

**Create New Migration:**
```bash
docker compose exec server aerich migrate
docker compose exec server aerich upgrade
```

---

## 13. Workflow Example: End-to-End

### Scenario: Add AWS Project & Get EC2 Recommendations

1. **User creates AWS connection via API:**
   ```bash
   POST /api/v1/aws/
   {
     "project_id": 1,
     "aws_access_key": "AKIA...",
     "aws_secret_key": "...",
     "yearly_budget": 120000,
     "export": false,
     "date": "2025-01-01"
   }
   ```

2. **Backend validates credentials:**
   ```python
   aws_validate_creds(aws_access_key, aws_secret_key)
   ```

3. **Creates FOCUS export:**
   ```python
   task_create_aws_export.delay({
     "s3_bucket": "cloud-meter-1-1737820800",
     "s3_prefix": "cm",
     "export_name": "cloud-meter-1-1737820800"
   })
   ```

4. **Celery task creates S3 bucket & export:**
   ```python
   aws_create_focus_export(aws_region, aws_access_key, aws_secret_key,
                          export_name, s3_bucket, s3_prefix)
   ```

5. **After export is ready, triggers ingestion:**
   ```python
   task_run_ingestion_aws.delay({
     "project_name": "MyProject",
     "aws_access_key": "...",
     "s3_bucket": "cloud-meter-1-1737820800",
     "billing_period": "2025-01"
   })
   ```

6. **Ingestion process:**
   - Downloads parquet files from S3
   - Generates hash keys
   - Filters new data
   - Loads to PostgreSQL
   - Executes SQL transformations
   - Collects CloudWatch metrics

7. **Database tables created:**
   - `myproject.bronze_focus_aws_data`
   - `myproject.silver_focus_aws`
   - `myproject.gold_aws_fact_focus`
   - `myproject.fact_ec2_metrics`
   - `myproject.dim_ec2_instance`

8. **User requests EC2 recommendations:**
   ```bash
   POST /api/v1/llm/aws/1
   {
     "resource_type": "ec2",
     "start_date": "2025-01-01",
     "end_date": "2025-01-25",
     "resource_id": null  # All EC2 instances
   }
   ```

9. **Backend queries metrics + cost:**
   ```python
   df = fetch_ec2_utilization_data(schema_name, start_date, end_date)
   # Returns: instance_id, instance_type, cpu_avg, cpu_max, billed_cost
   ```

10. **For each EC2 instance, calls LLM:**
    ```python
    prompt = generate_ec2_prompt(instance_data)
    llm_response = llm_call(prompt)
    recommendation = extract_json(llm_response)
    ```

11. **Returns aggregated recommendations:**
    ```json
    {
      "status": "success",
      "cloud": "aws",
      "schema_name": "myproject",
      "resource_type": "ec2",
      "recommendations": "[{...}, {...}]",
      "timestamp": "2025-01-25T10:30:00Z"
    }
    ```

---

## 14. Key Strengths

1. **Multi-Cloud Support:** Unified API for AWS & Azure
2. **Scalable Architecture:** Schema-per-project isolation
3. **AI-Driven Insights:** Contextual recommendations based on real metrics
4. **Deduplication:** Hash-based idempotent ingestion
5. **Caching:** Reduces LLM costs via intelligent caching
6. **Task Management:** Cancellable, trackable background jobs
7. **Security:** Encrypted credentials at rest
8. **Observability:** Celery Flower monitoring

---

## 15. Potential Improvements

1. **Real-Time Metrics:** Add streaming ingestion (Kafka/Kinesis)
2. **More Cloud Providers:** GCP support (partially implemented)
3. **Custom Rules Engine:** User-defined optimization rules
4. **Anomaly Detection:** ML-based anomaly detection
5. **Cost Attribution:** Tag-based cost allocation
6. **Forecasting:** Time-series cost forecasting
7. **Alerting:** Proactive cost alerts (partially implemented)
8. **Multi-Tenancy:** Support multiple organizations
9. **API Rate Limiting:** Protect against abuse
10. **CI/CD Pipeline:** Automated testing & deployment

---

## 16. File Structure Summary

```
backend1/
├── app/
│   ├── api/v1/endpoints/
│   │   ├── aws.py              # AWS connection API
│   │   ├── azure.py            # Azure connection API
│   │   └── llm.py              # LLM recommendations API
│   ├── ingestion/
│   │   ├── aws/
│   │   │   ├── main.py         # AWS ingestion orchestration
│   │   │   ├── resource_metrics.py  # CloudWatch metrics
│   │   │   ├── metrics_s3.py   # S3 metrics
│   │   │   ├── llm_ec2_vpc_integration.py  # EC2/VPC recommendations
│   │   │   └── llm_s3_integration.py  # S3 recommendations
│   │   └── azure/
│   │       ├── main.py         # Azure ingestion orchestration
│   │       ├── metrics_vm.py   # Azure Monitor VM metrics
│   │       ├── metrics_storage_account.py  # Storage metrics
│   │       ├── llm_analysis.py # VM/Storage recommendations
│   │       └── llm_data_fetch.py  # Data aggregation for LLM
│   ├── core/
│   │   ├── genai.py            # Azure OpenAI integration
│   │   ├── encryption.py       # Credential encryption
│   │   ├── llm_cache_utils.py  # LLM caching
│   │   └── task_manager.py     # Task cancellation
│   ├── models/
│   │   ├── aws.py              # AWS connection model
│   │   ├── azure.py            # Azure connection model
│   │   ├── llm_cache.py        # LLM cache model
│   │   └── project.py          # Project model
│   └── worker/
│       ├── celery_app.py       # Celery configuration
│       └── celery_worker.py    # Background tasks
└── Dockerfile
```

---

## Conclusion

This application is a sophisticated **Cloud FinOps Platform** that:
1. **Ingests** cost and performance metrics from AWS and Azure
2. **Transforms** raw data into actionable insights via SQL pipelines
3. **Analyzes** resources using AI to generate cost optimization recommendations
4. **Presents** findings via a RESTful API with caching and task management

The architecture is well-designed for scalability, security, and extensibility, making it a robust solution for multi-cloud cost optimization.

---

**Generated:** 2025-01-25
**Author:** Claude Code Analysis
**Version:** 1.0
