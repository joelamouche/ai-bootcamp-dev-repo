from pydantic import BaseModel, Field
from typing import Any, Optional, List, Union


class AgentRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: str = Field(..., description="The thread ID")
    telegram_handle: Optional[str] = Field(
        default=None,
        description="Optional Telegram @handle or username; defaults to thread_id if omitted",
    )


class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The image URL of the item")
    price: Optional[float] = Field(..., description="The price of the item")
    description: str = Field(..., description="The description of the item")

class AgentResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: List[RAGUsedContext] = Field(..., description="Information about items used to answer the query")


class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(..., description="1 if the feedback is positive, 0 if the feedback is negative")
    feedback_text: str = Field(..., description="The feedback text")
    trace_id: str = Field(..., description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")
    feedback_source_type: str = Field(..., description="The type of feedback. Human or API")

class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")


class MeetupPendingNotificationsResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    notifications: List[dict[str, Any]] = Field(
        default_factory=list,
        description="Queued meetup outreach for this thread/user (popped from the server queue)",
    )


class MeetupAdminContextResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    registry: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="user_id → saved profile (want_to_learn, can_teach, telegram_handle, …)",
    )
    proposals: List[dict[str, Any]] = Field(default_factory=list)
    pending_notifications: List[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)