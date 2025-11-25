# app/core/task_manager.py

import threading
import uuid
import json
import os
from typing import Dict, Set, Optional
import redis

class TaskManager:
    """
    Redis-backed task manager to track and cancel running LLM analysis tasks.
    Shared across all Uvicorn worker processes.
    """
    def __init__(self):
        # Connect to Redis (shared across all workers!)
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self._lock = threading.Lock()

        print(f"🔴 TaskManager initialized with Redis: {redis_url}")

        # Test Redis connection
        try:
            self.redis_client.ping()
            print(f"✅ Redis connection successful")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            raise

    def _task_key(self, task_id: str) -> str:
        """Generate Redis key for task data."""
        return f"task:{task_id}"

    def _project_tasks_key(self, project_id: str) -> str:
        """Generate Redis key for project's task list."""
        return f"project:{project_id}:tasks"

    def create_task(self, task_type: str, metadata: dict = None) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        project_id = (metadata or {}).get('project_id')

        task_data = {
            'id': task_id,
            'type': task_type,
            'metadata': json.dumps(metadata or {}),
            'status': 'running'
        }

        with self._lock:
            # Check if this project has a pending cancellation
            if project_id and self.redis_client.sismember('pending_cancel_projects', project_id):
                print(f"⚠️  Task {task_id} created but project {project_id} has pending cancellation - cancelling immediately")

                # Store task as cancelled
                task_data['status'] = 'cancelled'
                self.redis_client.hset(self._task_key(task_id), mapping=task_data)
                self.redis_client.sadd('cancelled_tasks', task_id)

                # Add to project's task list
                if project_id:
                    self.redis_client.sadd(self._project_tasks_key(project_id), task_id)

                # Remove pending cancellation
                self.redis_client.srem('pending_cancel_projects', project_id)

                print(f"✅ Created task {task_id} ({task_type}) - IMMEDIATELY CANCELLED")
                return task_id

            # Store task in Redis
            self.redis_client.hset(self._task_key(task_id), mapping=task_data)

            # Add to project's task list
            if project_id:
                self.redis_client.sadd(self._project_tasks_key(project_id), task_id)

        print(f"✅ Created task {task_id} ({task_type}) for project {project_id}")
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """Mark a task as cancelled."""
        with self._lock:
            task_key = self._task_key(task_id)
            if self.redis_client.exists(task_key):
                # Mark as cancelled
                self.redis_client.hset(task_key, 'status', 'cancelled')
                self.redis_client.sadd('cancelled_tasks', task_id)
                print(f"🛑 Cancelled task {task_id}")
                return True
            return False

    def cancel_tasks_by_project(self, project_id: str) -> int:
        """Cancel all tasks for a given project. Returns count of cancelled tasks."""
        cancelled_count = 0
        already_cancelled_count = 0

        with self._lock:
            # Get all task IDs for this project
            task_ids = self.redis_client.smembers(self._project_tasks_key(project_id))

            print(f"🔍 Found {len(task_ids)} task(s) for project {project_id}")

            for task_id in task_ids:
                task_key = self._task_key(task_id)
                task_data = self.redis_client.hgetall(task_key)

                if task_data:
                    status = task_data.get('status')
                    if status == 'running':
                        # Cancel the task
                        self.redis_client.hset(task_key, 'status', 'cancelled')
                        self.redis_client.sadd('cancelled_tasks', task_id)
                        print(f"🛑 Cancelled task {task_id} for project {project_id}")
                        cancelled_count += 1
                    elif status == 'cancelled':
                        # Task was already cancelled (e.g., by a previous cancel request)
                        already_cancelled_count += 1
                        print(f"ℹ️  Task {task_id} already cancelled")

            # Only set pending cancellation if no tasks exist at all
            # If we found cancelled or completed tasks, cancellation already happened
            if cancelled_count == 0 and already_cancelled_count == 0 and len(task_ids) == 0:
                self.redis_client.sadd('pending_cancel_projects', project_id)
                print(f"⚠️  No running tasks yet - set pending cancellation for project {project_id}")
                print(f"   Any tasks created for this project will be immediately cancelled")
            else:
                # Clear any pending cancellation flag since tasks exist (running, cancelled, or completed)
                # This prevents duplicate cancel requests from setting the flag
                self.redis_client.srem('pending_cancel_projects', project_id)
                if cancelled_count > 0 or already_cancelled_count > 0:
                    print(f"🧹 Cleared pending cancellation flag for project {project_id}")

        if cancelled_count > 0:
            print(f"✅ Cancelled {cancelled_count} task(s) for project {project_id}")
        elif already_cancelled_count > 0:
            print(f"ℹ️  {already_cancelled_count} task(s) already cancelled for project {project_id}")
        else:
            print(f"ℹ️  No active tasks found for project {project_id}")

        return cancelled_count

    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task has been cancelled."""
        # Check Redis set (shared across all workers!)
        return self.redis_client.sismember('cancelled_tasks', task_id)

    def complete_task(self, task_id: str):
        """Mark a task as completed and clean up."""
        with self._lock:
            task_key = self._task_key(task_id)
            if self.redis_client.exists(task_key):
                self.redis_client.hset(task_key, 'status', 'completed')
                print(f"✅ Completed task {task_id}")
            # Clean up cancelled flag
            self.redis_client.srem('cancelled_tasks', task_id)

    def get_task_status(self, task_id: str) -> dict:
        """Get task status."""
        task_data = self.redis_client.hgetall(self._task_key(task_id))
        if task_data:
            # Deserialize metadata
            if 'metadata' in task_data:
                task_data['metadata'] = json.loads(task_data['metadata'])
            return task_data
        return {'status': 'not_found'}

    def list_active_tasks(self) -> list:
        """List all active tasks across ALL workers."""
        tasks = []

        # Scan for all task keys
        for key in self.redis_client.scan_iter("task:*"):
            task_data = self.redis_client.hgetall(key)
            if task_data:
                # Deserialize metadata
                if 'metadata' in task_data:
                    task_data['metadata'] = json.loads(task_data['metadata'])
                tasks.append(task_data)

        return tasks

    def cleanup_completed_tasks(self):
        """Remove completed tasks from registry."""
        with self._lock:
            # Scan for all task keys
            for key in self.redis_client.scan_iter("task:*"):
                task_data = self.redis_client.hgetall(key)
                if task_data and task_data.get('status') in ['completed', 'cancelled']:
                    task_id = task_data.get('id')

                    # Get project_id from metadata
                    metadata = json.loads(task_data.get('metadata', '{}'))
                    project_id = metadata.get('project_id')

                    # Remove from Redis
                    self.redis_client.delete(key)
                    self.redis_client.srem('cancelled_tasks', task_id)

                    # Remove from project's task list
                    if project_id:
                        self.redis_client.srem(self._project_tasks_key(project_id), task_id)

# Global singleton instance (but now backed by Redis, shared across workers!)
task_manager = TaskManager()
