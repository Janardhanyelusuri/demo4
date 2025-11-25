import os
from celery import Celery, Task
from celery.schedules import crontab
from celery.signals import worker_ready

celery_app = None

celery_app = Celery(
    "worker",
    backend=os.environ.get("CELERY_RESULT_BACKEND"),
    broker=os.environ.get("CELERY_BROKER_URL")
)

celery_app.conf.update(
    task_track_started=True,
    beat_scheduler='celery.beat.Scheduler',
    enable_utc=True,
    timezone="UTC",
    beat_schedule={
        'run-ingestion-every-day': {
            'task': 'task_run_daily_ingestion',
            'schedule': crontab(hour=7, minute=00),  # Run every day at 07:00 UTC
        },
        'run-dashboard-ingestion-every-day': {
            'task': 'task_run_daily_dashboard_ingestion',
            'schedule': crontab(hour=7, minute=30),  # Run every day at 07:30 UTC
        },
        'run-daily-alerts': {
            'task': 'run_daily_alerts', 
            'schedule': crontab(minute=0, hour=8),  # Run every day at 08:00 UTC
        },
        'run-weekly-alerts': {
            'task': 'run_weekly_alerts',  
            'schedule': crontab(minute=0, hour=8, day_of_week=0),  # Run every Monday at 08:00 UTC
        },
        'run-monthly-alerts': {
            'task': 'run_monthly_alerts', 
            'schedule': crontab(minute=0, hour=8, day_of_month=1),  # Run on the 1st of each month at 08:00 UTC
        },
    }
)
