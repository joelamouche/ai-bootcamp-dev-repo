"""
Tools for the meetup organizer agent (global in-memory registry; Qdrant later).
"""

import json
import logging

from api.agent import meetup_context as ctx


logger = logging.getLogger(__name__)


def append_my_learning_profile(
    user_id: str,
    telegram_handle: str,
    want_to_learn: str,
    can_teach: str,
) -> str:
    """Save or update this user's learning goals and what they can teach in the global meetup registry.

    Call this when the user states what they want to learn and/or what they could teach (e.g. hackathon interests).

    Args:
        user_id: Stable user id (e.g. Telegram chat/thread id).
        telegram_handle: Public handle or username for coordination (e.g. @nickname).
        want_to_learn: Topics or skills they want to learn.
        can_teach: Topics or skills they offer to teach or facilitate.

    Returns:
        Confirmation text for the agent to relay to the user (no other users' data).
    """
    logger.info(
        "append_my_learning_profile: user_id=%s want_to_learn_len=%d can_teach_len=%d",
        user_id,
        len(want_to_learn or ""),
        len(can_teach or ""),
    )

    rec = ctx.append_or_update_profile(
        user_id=user_id,
        telegram_handle=telegram_handle,
        want_to_learn=want_to_learn,
        can_teach=can_teach,
    )
    return (
        "Saved your profile in the shared registry for matching. "
        f"Entry id: user_id={rec['user_id']}, updated_at={rec['updated_at']}."
    )


def get_meetup_community_registry() -> str:
    """Return the full internal registry JSON for planning matches and meetups.

    Contains all participants' user_ids, telegram handles, want_to_learn and can_teach fields.
    Use only for reasoning about matches and proposals — do not copy other users' handles or profiles into user-facing replies.

    Returns:
        JSON string of the registry (may be empty object).
    """
    data = ctx.get_registry_copy()
    logger.info("get_meetup_community_registry: entries=%d", len(data))
    return json.dumps(data, ensure_ascii=False, indent=2)


def create_meetup_proposal_and_notify(
    topic: str,
    teacher_user_id: str,
    student_user_ids: str,
    session_summary: str,
) -> str:
    """Create a proposed learning session and enqueue outreach to the teacher and matched students.

    Args:
        topic: Short title for the session (e.g. "Intro to Rust for smart contract devs").
        teacher_user_id: Registry user_id of the suggested teacher/facilitator.
        student_user_ids: Comma-separated registry user_ids of participants to invite (no spaces or JSON array as string).
        session_summary: Neutral description of the session for notifications (avoid leaking unrelated private details).

    Returns:
        Summary including proposal id and notification dispatch status (for the agent; user-facing text must stay privacy-safe).
    """
    raw = student_user_ids.replace(" ", "")
    if not raw:
        ids: list[str] = []
    else:
        ids = [x for x in raw.split(",") if x]

    logger.info(
        "create_meetup_proposal_and_notify: topic=%r teacher=%s students=%d",
        topic,
        teacher_user_id,
        len(ids),
    )

    prop = ctx.create_proposal(
        topic=topic,
        teacher_user_id=teacher_user_id,
        student_user_ids=ids,
        session_summary=session_summary,
    )
    sent = ctx.enqueue_notifications_for_proposal(prop)
    return json.dumps(
        {
            "proposal_id": prop["proposal_id"],
            "topic": prop["topic"],
            "notifications_enqueued": len(sent),
            "notification_handles": [x["telegram_handle"] for x in sent],
        },
        ensure_ascii=False,
    )
