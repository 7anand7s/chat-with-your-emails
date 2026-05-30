"""Pipeline state manager — tracks every email through every stage.

Persists to disk after every update. Survives crashes, kills, and reboots.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime
from enum import Enum

STATE_FILE = "data/pipeline_state.json"


class PipelineStage(str, Enum):
    FETCHED = "fetched"
    CLEANED = "cleaned"
    LLM_EXTRACTED = "llm_extracted"
    VLM_PROCESSED = "vlm_processed"
    EMBEDDED = "embedded"
    STORED = "stored"


class PipelineStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


# Stage order for determining progress
STAGE_ORDER = [
    PipelineStage.FETCHED,
    PipelineStage.CLEANED,
    PipelineStage.LLM_EXTRACTED,
    PipelineStage.VLM_PROCESSED,
    PipelineStage.EMBEDDED,
    PipelineStage.STORED,
]


def stage_index(stage: PipelineStage) -> int:
    return STAGE_ORDER.index(stage)


class EmailState:
    """State of a single email in the pipeline."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def message_id(self) -> str:
        return self._data.get("message_id", "")

    @property
    def stage(self) -> str:
        return self._data.get("stage", PipelineStage.FETCHED.value)

    @property
    def error(self) -> str | None:
        return self._data.get("error")

    @property
    def retries(self) -> int:
        return self._data.get("retries", 0)

    @property
    def chunks_created(self) -> int:
        return self._data.get("chunks_created", 0)

    def to_dict(self) -> dict:
        return self._data.copy()


class PipelineStateManager:
    """Manages pipeline state with atomic persistence.

    Every mutation writes to disk immediately. Survives crashes.
    """

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._state: dict = {}
        self.load()

    def load(self):
        """Load state from disk or create fresh state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._new_state()
        else:
            self._new_state()

    def _new_state(self):
        """Initialize a fresh pipeline state."""
        self._state = {
            "run_id": str(uuid.uuid4()),
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": PipelineStatus.RUNNING.value,
            "total_emails": 0,
            "stages": {
                stage.value: {"completed": 0, "failed": 0}
                for stage in STAGE_ORDER
            },
            "emails": {},
        }

    def save(self):
        """Atomically persist state to disk."""
        self._state["updated_at"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        # Atomic write: write to temp file, then rename
        dir_name = os.path.dirname(self.state_file)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @property
    def status(self) -> str:
        return self._state.get("status", PipelineStatus.RUNNING.value)

    @property
    def total_emails(self) -> int:
        return self._state.get("total_emails", 0)

    @property
    def emails(self) -> dict:
        return self._state.get("emails", {})

    @property
    def stages(self) -> dict:
        return self._state.get("stages", {})

    # ── Email registration ──

    def register_emails(self, message_ids: list[str], subjects: dict = None, senders: dict = None):
        """Register a batch of emails to track. Only adds new ones."""
        subjects = subjects or {}
        senders = senders or {}
        added = 0

        for mid in message_ids:
            if mid not in self._state["emails"]:
                self._state["emails"][mid] = {
                    "message_id": mid,
                    "subject": subjects.get(mid, ""),
                    "sender": senders.get(mid, ""),
                    "stage": None,
                    "error": None,
                    "retries": 0,
                    "started_at": None,
                    "completed_at": None,
                    "chunks_created": 0,
                }
                added += 1

        self._state["total_emails"] = len(self._state["emails"])
        if added > 0:
            self.save()

    # ── Stage transitions ──

    def set_stage(self, message_id: str, stage: PipelineStage):
        """Advance an email to a new stage."""
        if message_id not in self._state["emails"]:
            return

        email = self._state["emails"][message_id]
        old_stage = email.get("stage")

        # Update stage counts
        if old_stage and old_stage != stage.value:
            old_counts = self._state["stages"].get(old_stage, {"completed": 0, "failed": 0})
            old_counts["completed"] = max(0, old_counts["completed"] - 1)
            self._state["stages"][old_stage] = old_counts

        email["stage"] = stage.value
        email["error"] = None
        email["started_at"] = email.get("started_at") or datetime.now().isoformat()

        if stage == PipelineStage.STORED:
            email["completed_at"] = datetime.now().isoformat()

        # Increment new stage count
        new_counts = self._state["stages"].get(stage.value, {"completed": 0, "failed": 0})
        new_counts["completed"] += 1
        self._state["stages"][stage.value] = new_counts

        self.save()

    def set_error(self, message_id: str, error: str, stage: PipelineStage):
        """Record an error for an email at a specific stage."""
        if message_id not in self._state["emails"]:
            return

        email = self._state["emails"][message_id]
        email["error"] = error
        email["retries"] = email.get("retries", 0) + 1

        # Increment failed count for the stage
        counts = self._state["stages"].get(stage.value, {"completed": 0, "failed": 0})
        counts["failed"] += 1
        self._state["stages"][stage.value] = counts

        self.save()

    def set_chunks_created(self, message_id: str, count: int):
        """Record how many chunks were created for an email."""
        if message_id in self._state["emails"]:
            self._state["emails"][message_id]["chunks_created"] = count
            self.save()

    # ── Queries ──

    def get_email_state(self, message_id: str) -> EmailState | None:
        """Get the state of a single email."""
        data = self._state["emails"].get(message_id)
        return EmailState(data) if data else None

    def get_emails_at_stage(self, stage: PipelineStage) -> list[str]:
        """Get all message_ids currently at a given stage."""
        return [
            mid for mid, data in self._state["emails"].items()
            if data.get("stage") == stage.value
        ]

    def get_emails_needing_processing(self, target_stage: PipelineStage) -> list[str]:
        """Get message_ids that haven't reached the target stage yet.

        Returns emails that are either:
        - At an earlier stage than target_stage
        - Have an error at any stage before target_stage
        - Have no stage assigned yet
        """
        target_idx = stage_index(target_stage)
        result = []

        for mid, data in self._state["emails"].items():
            current_stage = data.get("stage")
            if current_stage is None:
                # Not yet started
                result.append(mid)
                continue

            try:
                current_idx = STAGE_ORDER.index(PipelineStage(current_stage))
            except ValueError:
                result.append(mid)
                continue

            if current_idx < target_idx:
                result.append(mid)
            elif current_idx == target_idx and data.get("error"):
                # Failed at this stage — needs retry
                result.append(mid)

        return result

    def get_progress(self) -> dict:
        """Get a progress summary."""
        total = self._state["total_emails"]
        stages = self._state["stages"]

        # Calculate overall percentage based on weighted stage completion
        weights = {
            PipelineStage.FETCHED.value: 0.05,
            PipelineStage.CLEANED.value: 0.10,
            PipelineStage.LLM_EXTRACTED.value: 0.40,
            PipelineStage.VLM_PROCESSED.value: 0.15,
            PipelineStage.EMBEDDED.value: 0.20,
            PipelineStage.STORED.value: 0.10,
        }

        overall_pct = 0.0
        for stage_name, weight in weights.items():
            completed = stages.get(stage_name, {}).get("completed", 0)
            if total > 0:
                overall_pct += (completed / total) * weight

        # Count emails with errors
        total_errors = sum(
            1 for data in self._state["emails"].values()
            if data.get("error")
        )

        return {
            "status": self._state["status"],
            "run_id": self._state["run_id"],
            "started_at": self._state["started_at"],
            "updated_at": self._state["updated_at"],
            "total_emails": total,
            "overall_pct": round(overall_pct * 100, 1),
            "stages": stages,
            "total_errors": total_errors,
        }

    def is_complete(self) -> bool:
        """Check if all emails are at the stored stage."""
        if not self._state["emails"]:
            return False
        return all(
            data.get("stage") == PipelineStage.STORED.value
            for data in self._state["emails"].values()
        )

    def get_failed_emails(self) -> list[dict]:
        """Get all emails with errors."""
        return [
            {
                "message_id": data["message_id"],
                "subject": data.get("subject", ""),
                "sender": data.get("sender", ""),
                "stage": data.get("stage", ""),
                "error": data.get("error", ""),
                "retries": data.get("retries", 0),
            }
            for data in self._state["emails"].values()
            if data.get("error")
        ]

    # ── Status management ──

    def set_status(self, status: PipelineStatus):
        """Set the overall pipeline status."""
        self._state["status"] = status.value
        self.save()

    def mark_complete(self):
        """Mark the pipeline run as completed."""
        self.set_status(PipelineStatus.COMPLETED)

    def mark_failed(self, reason: str = ""):
        """Mark the pipeline run as failed."""
        self._state["failure_reason"] = reason
        self.set_status(PipelineStatus.FAILED)

    def should_pause(self) -> bool:
        """Check if the pipeline should pause."""
        return self._state["status"] == PipelineStatus.PAUSED.value

    def reset(self):
        """Clear all state and start fresh."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        self._new_state()
