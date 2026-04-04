import logging
from pydantic import BaseModel
from typing import List, Dict, Any, Annotated, cast
from operator import add

from langchain_core.runnables import RunnableConfig
from api.agent.agents import (
    coordinator_agent,
    profile_intake_agent,
    meetup_coordination_agent,
    ToolCall,
    RAGUsedContext,
    Delegation,
)
from api.agent.utils.utils import get_tool_descriptions
from api.agent.meetup_tools import (
    append_my_learning_profile,
    get_meetup_community_registry,
    create_meetup_proposal_and_notify,
)
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
# from qdrant_client import QdrantClient
# from qdrant_client.models import Filter, FieldCondition, MatchValue
# import numpy as np
import json
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import Field

from api.agent import meetup_context as meetup_context_store


logger = logging.getLogger(__name__)


def _state_field_as_dict(value: Any) -> dict[str, Any]:
    """LangGraph value updates may be plain dicts or Pydantic models depending on the merge path."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []


class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""


class State(BaseModel):
    """
    LangGraph state for the ETHGlobal meetup organizer.

    - coordinator_agent: routes each turn to profile intake and/or meetup coordination.
    - profile_intake_agent: captures learn/teach text and appends to global in-memory registry.
    - meetup_coordination_agent: reads the registry, proposes sessions, enqueues participant outreach.
    """

    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    profile_intake_agent: AgentProperties = Field(default_factory=AgentProperties)
    meetup_coordination_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(default_factory=CoordinatorAgentProperties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    telegram_handle: str = ""
    trace_id: str = ""


#### Routers


def profile_intake_agent_tool_router(state) -> str:
    # If the model emitted tool_calls, run tools first — even when final_answer is also true.
    # Otherwise the graph skips tools, and the next LLM call hits OpenAI's
    # "assistant with tool_calls must be followed by tool messages" error.
    if len(state.profile_intake_agent.tool_calls) > 0:
        return "tools"
    if state.profile_intake_agent.final_answer:
        return "end"
    if state.profile_intake_agent.iteration > 4:
        return "end"
    return "end"


def meetup_coordination_agent_tool_router(state) -> str:
    if len(state.meetup_coordination_agent.tool_calls) > 0:
        return "tools"
    if state.meetup_coordination_agent.final_answer:
        return "end"
    if state.meetup_coordination_agent.iteration > 4:
        return "end"
    return "end"


def coordinator_router(state) -> str:
    if state.coordinator_agent.iteration > 8:
        return "end"
    if state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0:
        return "end"
    if state.coordinator_agent.next_agent == "profile_intake_agent":
        return "profile_intake_agent"
    if state.coordinator_agent.next_agent == "meetup_coordination_agent":
        return "meetup_coordination_agent"
    return "end"


#### Workflow

workflow = StateGraph(State)

profile_intake_tools = [append_my_learning_profile]
profile_intake_tool_node = ToolNode(profile_intake_tools)
profile_intake_tool_descriptions = cast(
    List[Dict[str, Any]], get_tool_descriptions(profile_intake_tools)
)

meetup_coordination_tools = [
    get_meetup_community_registry,
    create_meetup_proposal_and_notify,
]
meetup_coordination_tool_node = ToolNode(meetup_coordination_tools)
meetup_coordination_tool_descriptions = cast(
    List[Dict[str, Any]], get_tool_descriptions(meetup_coordination_tools)
)

workflow.add_node("profile_intake_agent", profile_intake_agent)
workflow.add_node("meetup_coordination_agent", meetup_coordination_agent)
workflow.add_node("coordinator_agent", coordinator_agent)

workflow.add_node("profile_intake_agent_tool_node", profile_intake_tool_node)
workflow.add_node("meetup_coordination_agent_tool_node", meetup_coordination_tool_node)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_router,
    {
        "profile_intake_agent": "profile_intake_agent",
        "meetup_coordination_agent": "meetup_coordination_agent",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "profile_intake_agent",
    profile_intake_agent_tool_router,
    {
        "tools": "profile_intake_agent_tool_node",
        "end": "coordinator_agent",
    },
)

workflow.add_conditional_edges(
    "meetup_coordination_agent",
    meetup_coordination_agent_tool_router,
    {
        "tools": "meetup_coordination_agent_tool_node",
        "end": "coordinator_agent",
    },
)

workflow.add_edge("profile_intake_agent_tool_node", "profile_intake_agent")
workflow.add_edge("meetup_coordination_agent_tool_node", "meetup_coordination_agent")


#### Agent execution (SSE)


def run_agent_stream_wrapper(question: str, thread_id: str, telegram_handle: str | None = None):

    def _string_for_sse(message: str):
        return f"data: {message}\n\n"

    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                return "Running tools..."
        else:
            return False

    handle = telegram_handle if telegram_handle else thread_id

    logger.info(
        "run_agent_stream: start thread_id=%s telegram_handle=%s query_len=%d",
        thread_id,
        handle,
        len(question or ""),
    )

    result: dict[str, Any] = {}

    initial_state = State(
        messages=[{"role": "user", "content": question}],
        user_intent="",
        profile_intake_agent=AgentProperties(
            iteration=0,
            final_answer=False,
            available_tools=profile_intake_tool_descriptions,
            tool_calls=[],
        ),
        meetup_coordination_agent=AgentProperties(
            iteration=0,
            final_answer=False,
            available_tools=meetup_coordination_tool_descriptions,
            tool_calls=[],
        ),
        coordinator_agent=CoordinatorAgentProperties(
            iteration=0,
            final_answer=False,
            plan=[],
            next_agent="",
        ),
        user_id=thread_id,
        telegram_handle=handle,
    )
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": thread_id}},
    )

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        # Required once: creates checkpoint_migrations, checkpoints, etc. (see PostgresSaver.setup docstring)
        checkpointer.setup()
        logger.info("PostgresSaver checkpoint tables ready")

        graph = workflow.compile(checkpointer=checkpointer)

        for raw_chunk in graph.stream(
            initial_state,
            config=config,
            stream_mode=["debug", "values"],
        ):
            chunk = cast(tuple[str, Any], raw_chunk)
            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                payload = chunk[1]
                if isinstance(payload, dict):
                    result = payload
                elif isinstance(payload, State):
                    result = payload.model_dump()
                else:
                    result = {}
                ans = result.get("answer")
                ca = _state_field_as_dict(result.get("coordinator_agent"))
                logger.info(
                    "graph values update: answer_len=%s coord_iter=%s final=%s next=%s",
                    len(ans) if isinstance(ans, str) else "n/a",
                    ca.get("iteration"),
                    ca.get("final_answer"),
                    ca.get("next_agent"),
                )

    used_context: list = []

    # Legacy Amazon pipeline: resolve RAG references via Qdrant (disabled for meetup MVP).
    # qdrant_client = QdrantClient(url="http://qdrant:6333")
    # dummy_vector = np.zeros(1536).tolist()
    # for item in result.get("references", []):
    #     payload = qdrant_client.query_points(
    #         collection_name="Amazon-items-collection-01-hybrid-search",
    #         query=dummy_vector,
    #         using="text-embedding-3-small",
    #         limit=1,
    #         with_payload=True,
    #         query_filter=Filter(
    #             must=[
    #                 FieldCondition(
    #                     key="parent_asin",
    #                     match=MatchValue(value=item.id),
    #                 )
    #             ]
    #         ),
    #     ).points[0].payload
    #     image_url = payload.get("image")
    #     price = payload.get("price")
    #     if image_url:
    #         used_context.append(
    #             {"image_url": image_url, "price": price, "description": item.description}
    #         )

    registry = meetup_context_store.get_registry_copy()

    final_answer = result.get("answer")
    if not (isinstance(final_answer, str) and final_answer.strip()):
        final_answer = (
            "Thanks for your message! I'm here to help connect people in the community. "
            "If you share what you'd like to learn and what you can teach, I can register "
            "you and coordinate with others."
        )
        logger.warning(
            "run_agent_stream: empty final answer after graph; using fallback (thread_id=%s)",
            thread_id,
        )

    logger.info(
        "run_agent_stream: done thread_id=%s answer_len=%d registry_users=%d",
        thread_id,
        len(final_answer),
        len(registry),
    )

    yield _string_for_sse(
        json.dumps(
            {
                "type": "final_result",
                "data": {
                    "answer": final_answer,
                    "used_context": used_context,
                    "trace_id": result.get("trace_id"),
                    "meetup_registry_user_count": len(registry),
                    "pending_notifications_count": meetup_context_store.pending_notification_count(),
                    # Legacy shopping cart field (Amazon template); kept for API compatibility.
                    "shopping_cart": [],
                },
            },
            default=float,
        )
    )
