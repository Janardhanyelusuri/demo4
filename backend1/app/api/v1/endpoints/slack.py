from fastapi import APIRouter, HTTPException
import os
import json
import requests
from app.schemas.connection import SlackConnectionResponse, WebhookURL, Message, SlackMessage
from app.models.alert_integration import Integration

router = APIRouter()


@router.post("/test-slack-connection", response_model=SlackConnectionResponse, tags=['slack'])
async def test_slack_connection(data: WebhookURL):
    # Default message details
    payload = {
        'text': 'Test message: Checking Slack connection.',
        'username': 'TestBot',
        'icon_emoji': ':white_check_mark:'
    }
    
    # Send the test message to the provided webhook URL
    response = requests.post(data.webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        return SlackConnectionResponse(status=True, message='Connection successful')
    else:
        return SlackConnectionResponse(status=False, message='Connection failed')


@router.post("/send-message", tags=['slack'])
async def send_message(project_id: int, integration_id: int, message: Message):
    try:
        integration = await Integration.get(id=integration_id, project_id=project_id)
        webhook_url = integration.url

        payload = {
            'text': message.text,
            'username': message.username,
            'icon_emoji': message.icon_emoji
        }
        
        response = requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            return {"message": "Message posted successfully."}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Integration.DoesNotExist:
        raise HTTPException(status_code=404, detail="Integration not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-message_type", tags=['slack'])
async def send_message_type(project_id: int, integration_id: int, message: SlackMessage):
    try:
        integration = await Integration.get(id=integration_id, project_id=project_id)
        webhook_url = integration.url

        if not message.text:
            if message.message_type == 'warning':
                message.text = 'Warning: Threshold exceeded!'
            elif message.message_type == 'info':
                message.text = 'Info: System is running smoothly.'
            elif message.message_type == 'alert':
                message.text = 'Alert: Immediate action required!'
            else:
                message.text = 'Default message text.'

        payload = {
            'text': message.text,
            'username': message.username,
            'icon_emoji': message.icon_emoji
        }

        response = requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            return {"message": "Message posted successfully."}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Integration.DoesNotExist:
        raise HTTPException(status_code=404, detail="Integration not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))