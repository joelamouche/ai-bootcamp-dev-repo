from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable
from langchain_core.messages import convert_to_openai_messages, AIMessage
from openai import OpenAI
import instructor
from api.agent.utils.utils import format_ai_message
from api.agent.utils.prompt_management import prompt_template_config
from langsmith import get_current_run_tree



### QnA Agent Structured output schemas

class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question.")
    description: str = Field(description="Short description of the item used to answer the question.")

class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Shopping Cart Agent Structured output schemas

class ToolCall(BaseModel):
    name: str
    arguments: dict

class ShoppingCartAgentResponse(BaseModel):
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

### QnA Agent Node

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def product_qa_agent(state) -> dict:

   template = prompt_template_config("src/api/agent/prompts/qa_agent.yaml", "qa_agent")
   
   prompt = template.render(
      available_tools=state.product_qa_agent.available_tools
   )

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
   )

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
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def shopping_cart_agent(state) -> dict:

   template = prompt_template_config("src/api/agent/prompts/shopping_cart_agent.yaml", "shopping_cart_agent")
   
   prompt = template.render(
      available_tools=state.shopping_cart_agent.available_tools,
      user_id=state.user_id,
      cart_id=state.cart_id
   )

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
   )

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


### Intent Router Node

@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def coordinator_agent(state):

   template = prompt_template_config("src/api/agent/prompts/coordinator_agent.yaml", "coordinator_agent")
   
   prompt = template.render()

   messages = state.messages

   conversation = []

   for message in messages:
        conversation.append(convert_to_openai_messages(message))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=CoordinatorAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
   )

   current_run = get_current_run_tree()

   if response.final_answer:
      ai_message = [AIMessage(
         content=response.answer,
      )]
   else:
      ai_message = []

   if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

   return {
      "messages": ai_message,
      "answer": response.answer,
      "coordinator_agent": {
         "iteration": state.coordinator_agent.iteration + 1,
         "final_answer": response.final_answer,
         "next_agent": response.next_agent,
         "plan": [data.model_dump() for data in response.plan]
      },
      "trace_id": trace_id
   }