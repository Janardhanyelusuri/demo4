import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
import asyncpg
from app.models.project import Project
from app.schemas.connection import GetUtilizationTable
from fastapi.responses import JSONResponse
import traceback
router = APIRouter()

# -------------------------------
# Unified logic for all providers
# -------------------------------
@router.post('/fetch_metrics', tags=["queries_metrics"])
async def get_provider_metrics(payload: GetUtilizationTable) -> List[Dict[str, Any]]:
    provider = payload.provider.lower()
    project_id = payload.project_id
    project_obj = await Project.filter(id=project_id).first()
    name = project_obj.name if project_obj else None
    if not name:
        raise HTTPException(status_code=404, detail="Project not found or name is missing")

    if provider == "azure":
        query = f"""
            SELECT
                vm_name,
                instance_type,
                total_cost,
                average_percentage_cpu,
                average_network_in,
                average_network_out,
                average_disk_read_bytes,
                average_disk_write_bytes,
                average_burst_iops,
                suggested_instance_type,
                suggested_reason,
                suggested_cost,
                cost_saving
            FROM {name}.genai_response;
        """

        conn = None
        try:
            conn = await asyncpg.connect(
                user=os.getenv("DB_USER_NAME"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                host=os.getenv("DB_HOST_NAME")
            )

            rows = await conn.fetch(query)

            return [
                {
                    "VM Name": row["vm_name"],
                    "Instance Type": row["instance_type"],
                    "Current Cost ($)": f"{row['total_cost']:.2f}",
                    "Suggested Instance Type": row["suggested_instance_type"],
                    "Recommendation Reason": row["suggested_reason"],
                    "Suggested Cost ($)": f"{row['suggested_cost']:.2f}",
                    "Estimated Cost Saving ($)": f"{row['cost_saving']:.2f}",
                    "Other": {
                        "Average CPU (%)": f"{row['average_percentage_cpu']}",
                        "Avg Network In (MB)": f"{row['average_network_in']}",
                        "Avg Network Out (MB)": f"{row['average_network_out']}",
                        "Avg Disk Read (Bytes)": f"{row['average_disk_read_bytes']}",
                        "Avg Disk Write (Bytes)": f"{row['average_disk_write_bytes']}",
                        "Avg Burst IOPS": str(row["average_burst_iops"]),
                    }
                }
                for row in rows
            ]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            if conn:
                await conn.close()

    elif provider in ["aws", "gcp"]:
        return []

    else:
        raise HTTPException(status_code=400, detail="Unsupported provider or project ID")


def serialize_record(record: dict) -> dict:
    """Convert datetime fields to ISO format strings for JSON serialization"""
    return {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in record.items()
    }

@router.post('/fetch_raw_metrics', tags=["queries_metrics"])
async def get_raw_vm_metrics(payload: GetUtilizationTable):
    provider = payload.provider.lower()
    project_id = payload.project_id

    project_obj = await Project.filter(id=project_id).first()
    schema_name = project_obj.name if project_obj else None
    if not schema_name:
        raise HTTPException(status_code=404, detail="Project not found or name is missing")

    if provider == "azure":
        # Calculate date 3 months before current date
        three_months_ago = datetime.now() - relativedelta(months=3)
        three_months_ago_str = three_months_ago.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT *
            FROM {schema_name}.silver_azure_vm_metrics 
            where value is not null and timestamp > '{three_months_ago_str}';
        """

        conn = None
        try:
            conn = await asyncpg.connect(
                user=os.getenv("DB_USER_NAME"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                host=os.getenv("DB_HOST_NAME")
            )

            rows = await conn.fetch(query)

            # Convert each record and serialize datetime
            result = [serialize_record(dict(row)) for row in rows]

            return JSONResponse(content=result)

        except Exception as e:
            print("🔥 ERROR IN fetch_raw_metrics:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            if conn:
                await conn.close()

    elif provider in ["aws", "gcp"]:
        return JSONResponse(content=[])

    else:
        raise HTTPException(status_code=400, detail="Unsupported provider or project ID")
