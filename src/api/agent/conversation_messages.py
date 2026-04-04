"""Shared helpers for LangGraph message lists (avoid duplicate logic / import cycles)."""

from __future__ import annotations

from typing import Any


def sanitize_messages_for_llm(messages: Any) -> list[Any]:
    """Drop or fix assistant turns so OpenAI never sees tool_calls without matching tool messages.

    Checkpointed threads can retain invalid tails (e.g. after partial runs or router skips); the
    coordinator would otherwise fail with tool_call_id errors on the next user message.
    """
    if not messages:
        return []
    from langchain_core.messages import AIMessage

    def ai_has_tool_calls(msg: Any) -> bool:
        if isinstance(msg, dict):
            return bool(msg.get("tool_calls"))
        return bool(getattr(msg, "tool_calls", None))

    def tool_call_ids_from_ai(msg: Any) -> list[str]:
        out: list[str] = []
        raw = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        for tc in raw or []:
            if isinstance(tc, dict) and tc.get("id"):
                out.append(str(tc["id"]))
            else:
                tid = getattr(tc, "id", None)
                if tid:
                    out.append(str(tid))
        return out

    def is_tool_msg(msg: Any) -> bool:
        return message_role(msg) in ("tool",)

    def tool_msg_call_id(msg: Any) -> str | None:
        if isinstance(msg, dict):
            tid = msg.get("tool_call_id") or (msg.get("additional_kwargs") or {}).get("tool_call_id")
            return str(tid) if tid else None
        tid = getattr(msg, "tool_call_id", None)
        return str(tid) if tid else None

    def strip_ai_tool_calls(msg: Any) -> Any:
        if isinstance(msg, AIMessage):
            return AIMessage(content=msg.content or "")
        if isinstance(msg, dict):
            d = {k: v for k, v in msg.items() if k != "tool_calls"}
            return d
        return msg

    msgs = list(messages)
    out: list[Any] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if not ai_has_tool_calls(m):
            out.append(m)
            i += 1
            continue
        ids = tool_call_ids_from_ai(m)
        j = i + 1
        matched: dict[str, bool] = {tid: False for tid in ids}
        while j < len(msgs) and is_tool_msg(msgs[j]):
            tid = tool_msg_call_id(msgs[j])
            if tid and tid in matched:
                matched[tid] = True
            j += 1
        if ids and all(matched.get(tid) for tid in ids):
            out.extend(msgs[i:j])
            i = j
            continue
        out.append(strip_ai_tool_calls(m))
        k = i + 1
        id_set = set(ids)
        while k < len(msgs) and is_tool_msg(msgs[k]):
            tid = tool_msg_call_id(msgs[k])
            if tid and tid in id_set:
                k += 1
            else:
                break
        i = k
    return out


def message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "")
    return str(getattr(msg, "type", None) or getattr(msg, "role", None) or "")


def tool_message_name(msg: Any) -> str | None:
    if isinstance(msg, dict):
        if msg.get("role") not in ("tool",):
            return None
        return msg.get("name") or (msg.get("additional_kwargs") or {}).get("name")
    name = getattr(msg, "name", None)
    if name:
        return str(name)
    return None


def append_profile_tool_ran_after_last_user(messages: Any) -> bool:
    """True if append_my_learning_profile already returned after the latest user/human message."""
    if not messages:
        return False
    last_user_i: int | None = None
    for i, m in enumerate(messages):
        if message_role(m) in ("user", "human"):
            last_user_i = i
    if last_user_i is None:
        return False
    for m in messages[last_user_i + 1 :]:
        if tool_message_name(m) == "append_my_learning_profile":
            return True
    return False
