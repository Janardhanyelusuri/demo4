from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
from datetime import datetime


class Alert(models.Model):
    id = fields.IntField(pk=True)
    created_at = fields.DatetimeField(null=False, auto_now_add=True)
    name = fields.CharField(max_length=100, null=False)
    recipient = fields.CharField(max_length=100, null=False)
    integration = fields.ForeignKeyField('models.Integration', related_name='integration')
    ends_on = fields.DateField(null=False) 
    schedule = fields.CharField(max_length=100, null=False)
    state = fields.JSONField(null=True)
    status = fields.BooleanField(null=False, default=False)
    type = fields.CharField(max_length=100, null=False)
    alert_type = fields.CharField(max_length=100, null=False)
    resource_list = fields.JSONField(null=True, default=list) 
    percentage_threshold = fields.FloatField(null=True) 
    value_threshold = fields.FloatField(null=True)
    condition = fields.CharField(max_length=100, null=False)
    operation = fields.CharField(max_length=100, null=False)
    tag_ids = fields.JSONField(null=True, default=[])

    project_ids = fields.JSONField(null=True, default=[])

    class PydanticMeta:
        model_config = {'extra': 'allow'}

    async def save(self, *args, **kwargs):
        """
        Override save method to ensure proper state initialization and management.
        """
        current_time = datetime.utcnow().isoformat()

        # Initialize the state if it's not already set
        if not self.state:
            self.state = {
                "status": "disabled" if not self.status else "pending",  # Set initial status based on the alert status
                "updated_at": current_time,
                "history": [{
                    "previous_state": None,  # No previous state for a new alert
                    "new_state": "disabled" if not self.status else "pending",  # Set the initial state to 'disabled' or 'pending'
                    "timestamp": current_time,
                    "initial_state": True  # Mark it as the initial state
                }]
            }
        else:
            # Ensure the state is a dictionary if it wasn't already
            if not isinstance(self.state, dict):
                self.state = {}

            # Initialize 'history' if not present
            if "history" not in self.state:
                self.state["history"] = [{
                    "previous_state": None,
                    "new_state": self.state.get("status", "disabled" if not self.status else "pending"),
                    "timestamp": current_time,
                    "initial_state": True
                }]
            
            # If the alert status changes, update the state
            current_state = self.state.get("status")
            if not self.status and current_state != "disabled":
                await self.update_state("disabled")  # Transition to 'disabled' if the status is False
            elif self.status and current_state == "disabled":
                await self.update_state("pending")  # Transition to 'pending' if the status is True and current state is 'disabled'

            # Update the 'updated_at' field in the state
            if "updated_at" not in self.state:
                self.state["updated_at"] = current_time

        # Call the parent class's save method
        await super().save(*args, **kwargs)

    async def update_state(self, new_status: str, additional_info: dict = None):
            """
            Updates the state JSON field with a new status and optional additional info.
            """
            valid_states = {"pending", "firing", "resolved", "snoozed", "error", "disabled"}
            if new_status not in valid_states:
                # Ensure the new status is one of the valid states
                raise ValueError(f"Invalid state '{new_status}'. Must be one of: {', '.join(valid_states)}")

            current_time = datetime.utcnow().isoformat()

            # Ensure state is a dictionary
            if not isinstance(self.state, dict):
                self.state = {}

            # Initialize 'history' if it's not already present
            if "history" not in self.state:
                self.state["history"] = []

            # Record the state change in the history
            history_entry = {
                "previous_state": self.state.get("status"),  # Store the previous state
                "new_state": new_status,  # Store the new state
                "timestamp": current_time  # Store the timestamp of the state change
            }
            if additional_info:
                history_entry["additional_info"] = additional_info  # Add any additional info if provided

            # Append the state change history entry
            self.state["history"].append(history_entry)

            # Limit the history to the last 100 entries
            self.state["history"] = self.state["history"][-100:]

            # Special handling for the 'resolved' state
            if new_status == "resolved":
                self.state["last_resolved_at"] = current_time  # Track the timestamp when the alert was resolved
                if self.status:
                    # If the alert was previously active, set it to 'pending' upon resolution
                    self.state["status"] = "pending"
                    self.state["updated_at"] = current_time
                    self.state["history"].append({
                        "previous_state": "resolved",
                        "new_state": "pending",  # Auto-transition to 'pending'
                        "timestamp": current_time,
                        "auto_transition": True  # Mark this as an auto-transition
                    })
            else:
                # If the new status is not 'resolved', just update the status
                self.state["status"] = new_status
                self.state["updated_at"] = current_time

            # Update the state with any additional info
            if additional_info:
                self.state.update(additional_info)

            # Save the updated state in the database
            await self.save(update_fields=["state"])


# Create Pydantic models from Tortoise ORM models
Alert_Pydantic = pydantic_model_creator(Alert, name="Alert")
AlertIn_Pydantic = pydantic_model_creator(Alert, name="AlertIn", exclude_readonly=True)