"""
Global registry for ETHGlobal meetup matching (~100 users for MVP).
In-memory plus optional JSON file so admin /api survives uvicorn --reload and restarts.
Replace with Qdrant + persistence later.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# user_id -> profile record (latest wins on append)
_registry: dict[str, dict[str, Any]] = {}

_proposals: list[dict[str, Any]] = []

# Outbound notifications for the Telegram (or other) layer to deliver
_pending_notifications: list[dict[str, Any]] = []

def _state_file_path() -> str:
    """JSON persistence path from app settings (reads .env via pydantic). Empty string disables disk I/O."""
    from api.core.config import config

    raw = config.MEETUP_STATE_FILE
    if raw is None:
        return "/tmp/meetup_agent_state.json"
    s = str(raw).strip()
    if s == "":
        return ""
    return s


def _persist() -> None:
    path = _state_file_path()
    if not path:
        return
    with _lock:
        payload = {
            "registry": {k: dict(v) for k, v in _registry.items()},
            "proposals": [dict(p) for p in _proposals],
            "pending_notifications": [dict(n) for n in _pending_notifications],
        }
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("meetup_context: could not persist state to %s: %s", path, e)


def _load_state(*, silent: bool = False) -> None:
    global _registry, _proposals, _pending_notifications
    path = _state_file_path()
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        with _lock:
            reg = data.get("registry") or {}
            _registry = {str(k): dict(v) for k, v in reg.items()}
            _proposals = [dict(p) for p in data.get("proposals", [])]
            _pending_notifications = [dict(n) for n in data.get("pending_notifications", [])]
        n_users = len(_registry)
        if silent:
            logger.debug(
                "meetup_context: refreshed from %s (users=%d)",
                path,
                n_users,
            )
        else:
            logger.info(
                "meetup_context: loaded state from %s (users=%d)",
                path,
                n_users,
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("meetup_context: could not load state from %s: %s", path, e)


_load_state()


def append_or_update_profile(
    user_id: str,
    telegram_handle: str,
    want_to_learn: str,
    can_teach: str,
) -> dict[str, Any]:
    wl = want_to_learn.strip()
    ct = can_teach.strip()
    with _lock:
        prev = _registry.get(user_id)
        if prev:
            if not wl:
                wl = (prev.get("want_to_learn") or "").strip()
            if not ct:
                ct = (prev.get("can_teach") or "").strip()
        tg = (telegram_handle or "").strip() or (
            (prev or {}).get("telegram_handle") or ""
        )
        record = {
            "user_id": user_id,
            "telegram_handle": tg,
            "want_to_learn": wl,
            "can_teach": ct,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _registry[user_id] = record
        out = dict(record)
    _persist()
    logger.info("meetup_context: saved profile user_id=%s (registry_size=%d)", user_id, len(_registry))
    return out


def get_registry_copy() -> dict[str, dict[str, Any]]:
    with _lock:
        return {k: dict(v) for k, v in _registry.items()}


def get_proposals_copy() -> list[dict[str, Any]]:
    with _lock:
        return [dict(p) for p in _proposals]


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
        out = dict(entry)
    _persist()
    return out


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

        result = list(out)
    _persist()
    return result


def pending_notification_count() -> int:
    with _lock:
        return len(_pending_notifications)


def drain_notifications() -> list[dict[str, Any]]:
    """Optional helper for tests or a worker; not required for the graph."""
    with _lock:
        batch = list(_pending_notifications)
        _pending_notifications.clear()
    _persist()
    return batch


def _read_state_file_snapshot() -> dict[str, Any] | None:
    """Read persisted JSON without mutating globals (used for admin view merge)."""
    path = _state_file_path()
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("meetup_context: could not read state file for admin snapshot: %s", e)
        return None


def pop_notifications_for_user(user_id: str) -> list[dict[str, Any]]:
    """Remove and return queued outreach messages for this user (e.g. Streamlit thread_id == user_id)."""
    uid = str(user_id or "").strip()
    if not uid:
        return []
    with _lock:
        kept: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []
        for n in _pending_notifications:
            if n.get("user_id") == uid:
                out.append(dict(n))
            else:
                kept.append(n)
        _pending_notifications[:] = kept
    if out:
        _persist()
    return out


def get_admin_snapshot() -> dict[str, Any]:
    """Read-only view of registry, proposals, and queued notifications.

    Refreshes in-memory state from disk first (same file as append_or_update_profile) so admin
    matches persisted data after reload or across multiple API workers.

    Merges on-disk state with in-process state for the response only. In-process entries win on
    duplicate user_ids so we never overwrite fresh RAM with a stale or empty file (a previous bug
    when `_load_state` ran on every admin GET).
    """
    _load_state(silent=True)
    file_data = _read_state_file_snapshot()
    with _lock:
        mem_reg = {k: dict(v) for k, v in _registry.items()}
        mem_prop = [dict(p) for p in _proposals]
        mem_pend = [dict(n) for n in _pending_notifications]

    fr: dict[str, dict[str, Any]] = {}
    if file_data:
        reg = file_data.get("registry") or {}
        fr = {str(k): dict(v) for k, v in reg.items()}
    merged_reg = {**fr, **mem_reg}

    fp_list = [dict(p) for p in (file_data.get("proposals", []) if file_data else [])]
    by_pid: dict[str, dict[str, Any]] = {}
    for p in fp_list + mem_prop:
        pid = p.get("proposal_id")
        if pid:
            by_pid[str(pid)] = p
    merged_prop = list(by_pid.values())

    fpend = [dict(n) for n in (file_data.get("pending_notifications", []) if file_data else [])]
    pend_seen: set[tuple[Any, ...]] = set()
    merged_pend: list[dict[str, Any]] = []
    for n in fpend + mem_pend:
        key = (n.get("proposal_id"), n.get("user_id"))
        if key in pend_seen:
            continue
        pend_seen.add(key)
        merged_pend.append(dict(n))

    stats = {
        "user_count": len(merged_reg),
        "proposal_count": len(merged_prop),
        "pending_notification_count": len(merged_pend),
    }
    if stats["user_count"] == 0:
        p = _state_file_path()
        logger.info(
            "meetup admin snapshot: empty registry; state_file=%r file_exists=%s",
            p,
            os.path.isfile(p) if p else False,
        )
    return {
        "registry": merged_reg,
        "proposals": merged_prop,
        "pending_notifications": merged_pend,
        "stats": stats,
    }
