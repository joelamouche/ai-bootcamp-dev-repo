from fastapi import APIRouter, Request
import logging

from api.api.models import AgentRequest, AgentResponse, RAGUsedContext

from api.agent.graph import run_agent_wrapper


logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def rag(
    request: Request,
    payload: AgentRequest
) -> AgentResponse:

    answer = run_agent_wrapper(payload.query, payload.session_id)

    return AgentResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
        used_context=[RAGUsedContext(**used_context) for used_context in answer["used_context"]]
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])