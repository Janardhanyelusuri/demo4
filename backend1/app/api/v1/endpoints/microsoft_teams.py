from fastapi import APIRouter, HTTPException
import os
import json
import requests
from app.schemas.connection import TeamsConnectionResponse, WebhookURL, Message, SlackMessage
from app.models.alert_integration import Integration

router = APIRouter()

@router.post("/test-teams-connection", response_model=TeamsConnectionResponse, tags=['microsoft_teams'])
async def test_microsoft_connection(data: WebhookURL):
    payload = {
        "text": "Test message: Checking Microsoft connection.",
        "username": "TestBot",
        "icon_emoji": ":white_check_mark:",
    }
    response = requests.post(data.webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        
    if response.status_code == 200:
        return TeamsConnectionResponse(status=True, message='Connection successful')
    else:
        return TeamsConnectionResponse(status=False, message='Connection failed')