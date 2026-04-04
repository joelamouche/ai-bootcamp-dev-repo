"""
In-memory global registry for ETHGlobal meetup matching (~100 users for MVP).
Replace with Qdrant + persistence later.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()

# user_id -> profile record (latest wins on append)
_registry: dict[str, dict[str, Any]] = {}

_proposals: list[dict[str, Any]] = []

# Outbound notifications for the Telegram (or other) layer to deliver
_pending_notifications: list[dict[str, Any]] = []


def append_or_update_profile(
    user_id: str,
    telegram_handle: str,
    want_to_learn: str,
    can_teach: str,
) -> dict[str, Any]:
    with _lock:
        record = {
            "user_id": user_id,
            "telegram_handle": telegram_handle,
            "want_to_learn": want_to_learn.strip(),
            "can_teach": can_teach.strip(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _registry[user_id] = record
        return dict(record)


def get_registry_copy() -> dict[str, dict[str, Any]]:
    with _lock:
        return {k: dict(v) for k, v in _registry.items()}


def create_proposal(
    topic: str,
    teacher_user_id: str,
    student_user_ids: list[str],
    session_summary: str,
) -> dict[str, Any]:
    with _lock:
        prop_id = str(uuid.uuid4())
        entry = {
            "proposal_id": prop_id,
            "topic": topic.strip(),
            "teacher_user_id": teacher_user_id,
            "student_user_ids": list(student_user_ids),
            "session_summary": session_summary.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _proposals.append(entry)
        return dict(entry)


def enqueue_notifications_for_proposal(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-participant notification payloads (includes handles for the transport layer)."""
    with _lock:
        teacher = _registry.get(proposal["teacher_user_id"])
        out: list[dict[str, Any]] = []
        topic = proposal["topic"]
        pid = proposal["proposal_id"]

        if teacher:
            msg = (
                f"[Meetup proposal {pid}] You're suggested as a session lead for «{topic}». "
                f"Summary: {proposal['session_summary']}"
            )
            n = {"proposal_id": pid, "user_id": teacher["user_id"], "telegram_handle": teacher["telegram_handle"], "message": msg}
            _pending_notifications.append(n)
            out.append(dict(n))

        for uid in proposal["student_user_ids"]:
            prof = _registry.get(uid)
            if not prof:
                continue
            msg = (
                f"[Meetup proposal {pid}] A learning session on «{topic}» may run; you've been matched as a participant. "
                f"Summary: {proposal['session_summary']}"
            )
            n = {"proposal_id": pid, "user_id": prof["user_id"], "telegram_handle": prof["telegram_handle"], "message": msg}
            _pending_notifications.append(n)
            out.append(dict(n))

        return out


def pending_notification_count() -> int:
    with _lock:
        return len(_pending_notifications)


def drain_notifications() -> list[dict[str, Any]]:
    """Optional helper for tests or a worker; not required for the graph."""
    with _lock:
        batch = list(_pending_notifications)
        _pending_notifications.clear()
        return batch
