from typing import List
from fastapi import APIRouter, HTTPException
from app.models.alert import Alert, Alert_Pydantic, AlertIn_Pydantic
from app.models.alert_integration import Integration
from datetime import datetime
from tortoise.exceptions import DoesNotExist

router = APIRouter()


@router.post('/', response_model=Alert_Pydantic, tags=["alert"])
async def add_alert(alert: AlertIn_Pydantic):
    try:
        alert_obj = await Alert.create(**alert.dict())
        return await Alert_Pydantic.from_tortoise_orm(alert_obj)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get('/fetch-integrations', tags=['alert'])
async def fetch_integrations(recipient: str):
    try:
        integrations= await Integration.filter(integration_type=recipient.lower()).all()

        integration_list= [{"id": integration.id, "name": integration.name} for integration in integrations]

        return integration_list
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/{alert_id}', response_model=Alert_Pydantic, tags=["alert"])
async def get_alert(alert_id: int):
    try:
        alert = await Alert.get(id=alert_id)
        return await Alert_Pydantic.from_tortoise_orm(alert)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/', response_model=List[Alert_Pydantic], tags=["alert"])
async def get_all_alert():
    try:
        present = datetime.now().date() 
        alerts = await Alert_Pydantic.from_queryset(Alert.all())

        for alert in alerts:
            alert_instance = await Alert.get(id=alert.id)

            ends_on = alert_instance.ends_on
            #if the end date has passed, update the status to false
            if ends_on < present:
                alert_instance.status = False
                await alert_instance.save()

        return await Alert_Pydantic.from_queryset(Alert.all())  #updated list of alerts
    except DoesNotExist:
        raise HTTPException(status_code=404, detail="Alert not found.")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{alert_id}", tags=["alert"])
async def delete_alert(alert_id: int):
    await Alert.filter(id=alert_id).delete()
    return {"status": True, "message": "Successfully deleted alert"}


@router.put("/{alert_id}", response_model=Alert_Pydantic, tags=["alert"])
async def update_alert(alert_id: int, alert: AlertIn_Pydantic):
    await Alert.filter(id=alert_id).update(**alert.model_dump(exclude_unset=True))
    return await Alert_Pydantic.from_queryset_single(Alert.get(id=alert_id))
