import logging
from typing import Any, List

from pydantic import BaseModel, Field, model_validator
from langsmith import traceable
from langchain_core.messages import convert_to_openai_messages, AIMessage
from openai import OpenAI
import instructor
from api.agent.utils.utils import format_ai_message
from api.agent.utils.prompt_management import prompt_template_config
from langsmith import get_current_run_tree
from litellm import completion


logger = logging.getLogger(__name__)


def _log_answer_preview(node: str, answer: str, limit: int = 160) -> None:
    text = (answer or "").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    logger.info("%s: answer (%d chars): %s", node, len(answer or ""), text or "(empty)")


### QnA Agent Structured output schemas

class ToolCall(BaseModel):
    """LLMs often omit arguments for no-param tools or use \"parameters\" instead — normalize."""

    name: str
    arguments: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_arguments(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        merged = dict(data)
        args = merged.get("arguments")
        if args is None and "parameters" in merged:
            args = merged.get("parameters")
        merged["arguments"] = args if isinstance(args, dict) else {}
        return merged

class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question.")
    description: str = Field(description="Short description of the item used to answer the question.")

class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Shopping Cart Agent Structured output schemas

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Warehouse Manager Agent Structured output schemas

class WarehouseManagerAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Coordinator Agent Structured output schemas

class Delegation(BaseModel):
    agent: str
    task: str

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: List[Delegation]
    final_answer: bool
    answer: str


### Meetup organizer — Profile intake

class ProfileIntakeAgentResponse(BaseModel):
    answer: str = Field(description="Reply to the user (privacy-safe).")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Meetup organizer — Coordination / matching

class MeetupCoordinationAgentResponse(BaseModel):
    answer: str = Field(description="Reply to the user (privacy-safe).")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


@traceable(
    name="profile_intake_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def profile_intake_agent(state, models=None):
    if models is None:
        models = ["gpt-4.1"]

    prompts = {}
    for model in models:
        template = prompt_template_config("src/api/agent/prompts/profile_intake_agent.yml", model)
        prompt = template.render(
            available_tools=state.profile_intake_agent.available_tools,
            user_id=state.user_id,
            telegram_handle=state.telegram_handle,
        )
        prompts[model] = prompt

    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ProfileIntakeAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            logger.warning("profile_intake_agent: model %s failed: %s", model, e)
            continue
    else:
        raise RuntimeError(f"All models failed for profile_intake_agent: {models!r}")

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    logger.info(
        "profile_intake_agent: user_id=%s iteration=%s final_answer=%s tool_calls=%d",
        state.user_id,
        state.profile_intake_agent.iteration + 1,
        response.final_answer,
        len(response.tool_calls),
    )
    _log_answer_preview("profile_intake_agent", response.answer)

    return {
        "messages": [ai_message],
        "profile_intake_agent": {
            "iteration": state.profile_intake_agent.iteration + 1,
            "final_answer": response.final_answer,
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "available_tools": state.profile_intake_agent.available_tools,
        },
        "answer": response.answer,
    }


@traceable(
    name="meetup_coordination_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def meetup_coordination_agent(state, models=None):
    if models is None:
        models = ["gpt-4.1"]

    prompts = {}
    for model in models:
        template = prompt_template_config("src/api/agent/prompts/meetup_coordination_agent.yml", model)
        prompt = template.render(
            available_tools=state.meetup_coordination_agent.available_tools,
            user_id=state.user_id,
            telegram_handle=state.telegram_handle,
        )
        prompts[model] = prompt

    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=MeetupCoordinationAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            logger.warning("meetup_coordination_agent: model %s failed: %s", model, e)
            continue
    else:
        raise RuntimeError(f"All models failed for meetup_coordination_agent: {models!r}")

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    logger.info(
        "meetup_coordination_agent: user_id=%s iteration=%s final_answer=%s tool_calls=%d",
        state.user_id,
        state.meetup_coordination_agent.iteration + 1,
        response.final_answer,
        len(response.tool_calls),
    )
    _log_answer_preview("meetup_coordination_agent", response.answer)

    return {
        "messages": [ai_message],
        "meetup_coordination_agent": {
            "iteration": state.meetup_coordination_agent.iteration + 1,
            "final_answer": response.final_answer,
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "available_tools": state.meetup_coordination_agent.available_tools,
        },
        "answer": response.answer,
    }


### Legacy QnA Agent Node (Amazon shopping template; unused by meetup graph)

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def product_qa_agent(state, models=None) -> dict:
    if models is None:
        models = ["gpt-4.1"]

    prompts = {}

    for model in models:
        template = prompt_template_config("src/api/agent/prompts/qa_agent.yaml", model)
        prompt = template.render(
            available_tools=state.product_qa_agent.available_tools
        )
        prompts[model] = prompt

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ProductQAAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue
    else:
        raise RuntimeError(f"All models failed for product_qa_agent: {models!r}")

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "product_qa_agent": {
        "iteration": state.product_qa_agent.iteration + 1,
        "final_answer": response.final_answer,
        "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
        "available_tools": state.product_qa_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references
    }


## Shopping Cart Agent Node

@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent(state, models=None) -> dict:
    if models is None:
        models = ["gpt-4.1"]

    prompts = {}

    for model in models:
        template = prompt_template_config("src/api/agent/prompts/shopping_cart_agent.yaml", model)
        prompt = template.render(
            available_tools=state.shopping_cart_agent.available_tools,
            user_id=state.user_id,
            cart_id=state.cart_id
        )
        prompts[model] = prompt
   
    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ShoppingCartAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue
    else:
        raise RuntimeError(f"All models failed for shopping_cart_agent: {models!r}")

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "shopping_cart_agent": {
        "iteration": state.shopping_cart_agent.iteration + 1,
        "final_answer": response.final_answer,
        "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
        "available_tools": state.shopping_cart_agent.available_tools
        },
        "answer": response.answer,
    }


### Warehouse Manager Agent Node

@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def warehouse_manager_agent(state, models=None) -> dict:
    if models is None:
        models = ["gpt-4.1"]

    prompts = {}

    for model in models:
        template = prompt_template_config("src/api/agent/prompts/warehouse_manager_agent.yaml", model)
        prompt = template.render(
            available_tools=state.warehouse_manager_agent.available_tools
        )
        prompts[model] = prompt

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=WarehouseManagerAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue
    else:
        raise RuntimeError(f"All models failed for warehouse_manager_agent: {models!r}")

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "warehouse_manager_agent": {
        "iteration": state.warehouse_manager_agent.iteration + 1,
        "final_answer": response.final_answer,
        "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
        "available_tools": state.warehouse_manager_agent.available_tools
        },
        "answer": response.answer,
    }

### Coordinator Agnet Node

@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def coordinator_agent(state, models=None):
    if models is None:
        models = ["gpt-4.1"]

    prompts = {}

    for model in models:
        template = prompt_template_config("src/api/agent/prompts/coordinator_agent.yml", model)
        prompt = template.render()
        prompts[model] = prompt

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=CoordinatorAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            logger.warning("coordinator_agent: model %s failed: %s", model, e)
            continue
    else:
        raise RuntimeError(f"All models failed for coordinator_agent: {models!r}")

    current_run = get_current_run_tree()

    trace_id = ""
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    fallback_answer = (
        "Thanks for reaching out! I'm glad you're here. I've noted what you shared "
        "and will help coordinate with other participants in the community where it makes sense."
    )

    if response.final_answer:
        text = (response.answer or "").strip()
        if not text:
            logger.warning("coordinator_agent: final_answer=True but empty answer; using fallback")
            text = fallback_answer
        ai_message = [AIMessage(content=text)]
        answer_for_state = text
    else:
        ai_message = []
        answer_for_state = None

    logger.info(
        "coordinator_agent: iteration=%s final_answer=%s next_agent=%s plan_steps=%d",
        state.coordinator_agent.iteration + 1,
        response.final_answer,
        response.next_agent or "(none)",
        len(response.plan),
    )
    if response.final_answer:
        _log_answer_preview("coordinator_agent", answer_for_state or "")

    out: dict[str, Any] = {
        "messages": ai_message,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": [data.model_dump() for data in response.plan]
        },
        "trace_id": trace_id,
    }
    # Only publish answer when this node is meant to speak to the user; delegating must
    # not overwrite specialist answers with an empty string.
    if response.final_answer:
        out["answer"] = answer_for_state
    return out