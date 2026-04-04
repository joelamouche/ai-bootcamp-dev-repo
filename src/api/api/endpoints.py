import logging
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.api.models import (
    AgentRequest,
    FeedbackRequest,
    FeedbackResponse,
    MeetupAdminContextResponse,
    MeetupPendingNotificationsResponse,
)
from api.agent.graph import run_agent_stream_wrapper
from api.agent.meetup_context import get_admin_snapshot, pop_notifications_for_user
from api.api.processors.submit_feedback import submit_feedback
from api.core.config import config


logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def _admin_password_ok(got: str, expected: str) -> bool:
    """Length-safe compare to avoid leaking password length and compare_digest length errors."""
    a, b = got.encode("utf-8"), expected.encode("utf-8")
    if len(a) != len(b):
        return False
    return secrets.compare_digest(a, b)


def require_meetup_admin(x_admin_password: str | None = Header(None, alias="X-Admin-Password")) -> None:
    if x_admin_password is None:
        raise HTTPException(status_code=401, detail="Missing X-Admin-Password header")
    if not _admin_password_ok(x_admin_password, config.MEETUP_ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Unauthorized")


@rag_router.get("/meetup-pending-notifications", response_model=MeetupPendingNotificationsResponse)
def get_meetup_pending_notifications(
    request: Request,
    thread_id: str,
) -> MeetupPendingNotificationsResponse:
    """Pop queued meetup outreach for this chat thread (user_id == thread_id). No admin password."""
    notes = pop_notifications_for_user(thread_id)
    return MeetupPendingNotificationsResponse(
        request_id=request.state.request_id,
        notifications=notes,
    )


@admin_router.get(
    "/meetup-context",
    response_model=MeetupAdminContextResponse,
    dependencies=[Depends(require_meetup_admin)],
)
def get_meetup_context(request: Request) -> MeetupAdminContextResponse:
    """Return meetup registry and queue state (refreshes from MEETUP_STATE_FILE disk snapshot first). Admin only."""
    snap = get_admin_snapshot()
    return MeetupAdminContextResponse(
        request_id=request.state.request_id,
        registry=snap["registry"],
        proposals=snap["proposals"],
        pending_notifications=snap["pending_notifications"],
        stats=snap["stats"],
    )

@rag_router.post("/")
def rag(
    request: Request,
    payload: AgentRequest
) -> StreamingResponse:

    q = (payload.query or "").replace("\n", " ").strip()
    q_preview = q[:200] + ("…" if len(q) > 200 else "")
    logger.info(
        "POST /rag thread_id=%s query_len=%d preview=%r",
        payload.thread_id,
        len(payload.query or ""),
        q_preview,
    )

    return StreamingResponse(
        run_agent_stream_wrapper(
            payload.query,
            payload.thread_id,
            telegram_handle=payload.telegram_handle,
        ),
        media_type="text/event-stream"
    )


@feedback_router.post("/")
def send_feedback(
    request: Request,
    payload: FeedbackRequest
) -> FeedbackResponse:

    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)

    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success"
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])
api_router.include_router(admin_router)