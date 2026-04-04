import json
import logging
import os
import secrets

import requests
import streamlit as st

from core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def api_base_url() -> str:
    """Resolve API base URL for server-side HTTP calls (Streamlit → FastAPI).

    In Docker, the API is always reachable at the compose service name and *container* port
    (http://api:8000). Host port mappings (API_HOST_PORT) do not change that.

    If .env sets API_URL to http://localhost:... for host-only use, that breaks inside a
    container — we rewrite when running in Docker.
    """
    raw = (
        os.environ.get("API_URL") or getattr(config, "API_URL", None) or "http://api:8000"
    )
    raw = str(raw).strip().rstrip("/")

    in_docker = os.path.exists("/.dockerenv")
    if in_docker and raw and ("localhost" in raw or "127.0.0.1" in raw):
        logger.warning(
            "API_URL=%s is invalid inside Docker (localhost is this container). Using http://api:8000",
            raw,
        )
        return "http://api:8000"

    return raw or "http://api:8000"


st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_thread_id_in_url() -> str:
    """Persist a random thread id in ?thread= (bookmarkable; matches LangGraph / TG thread semantics)."""
    qp = st.query_params
    raw = qp.get("thread")
    if raw:
        tid = raw[0] if isinstance(raw, list) else raw
        tid = str(tid).strip()
        if tid:
            return tid
    new_id = secrets.token_urlsafe(12)
    qp["thread"] = new_id
    return new_id


def fake_telegram_handle(thread_id: str) -> str:
    """Map thread id to a plausible @username (Telegram allows 5–32 chars: [a-zA-Z0-9_])."""
    body = "".join(c if (c.isalnum() or c == "_") else "_" for c in thread_id)
    body = ("tg_" + body.strip("_"))[:32]
    if len(body) < 5:
        body = (body + "abcde")[:5]
    return "@" + body


thread_id = ensure_thread_id_in_url()

def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


def api_call_stream(method, url, **kwargs):
    """Stream SSE lines from the API. Yields decoded text lines only on success."""

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    kwargs.setdefault("timeout", 600)

    try:
        response = getattr(requests, method)(url, **kwargs)
    except requests.exceptions.ConnectionError as e:
        logger.exception("API stream connection error url=%s", url)
        _show_error_popup(
            "Connection error — cannot reach the API. If you use Docker Compose, API_URL must be http://api:8000 "
            "(set in docker-compose for streamlit-app)."
        )
        return iter(())
    except requests.exceptions.Timeout:
        logger.error("API stream timeout url=%s", url)
        _show_error_popup("The request timed out. Please try again later.")
        return iter(())
    except Exception as e:
        logger.exception("API stream error url=%s", url)
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return iter(())

    if not response.ok:
        body = (response.text or "")[:800]
        logger.error(
            "API stream HTTP %s url=%s body_preview=%r",
            response.status_code,
            url,
            body,
        )
        _show_error_popup(f"API error {response.status_code}. Is API_URL correct? (Docker: http://api:8000)")
        return iter(())

    return response.iter_lines(decode_unicode=True)


def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None 
    
    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": thread_id,
        "feedback_source_type": "api"
    }

    logger.info(f"Feedback data: {feedback_data}")
    
    status, response = api_call("post", f"{api_base_url()}/submit_feedback", json=feedback_data)
    return status, response


if "by_thread" not in st.session_state:
    st.session_state.by_thread = {}


def _default_thread_bucket():
    return {
        "messages": [{"role": "assistant", "content": "Hello! How can I assist you today?"}],
        "used_context": [],
        "trace_id": None,
        "latest_feedback": None,
        "show_feedback_box": False,
        "feedback_submission_status": None,
        "delivered_meetup_notification_ids": [],
    }


if thread_id not in st.session_state.by_thread:
    st.session_state.by_thread[thread_id] = _default_thread_bucket()

_tb = st.session_state.by_thread[thread_id]
st.session_state.messages = _tb["messages"]
st.session_state.used_context = _tb["used_context"]
st.session_state.trace_id = _tb["trace_id"]
st.session_state.latest_feedback = _tb["latest_feedback"]
st.session_state.show_feedback_box = _tb["show_feedback_box"]
st.session_state.feedback_submission_status = _tb["feedback_submission_status"]


def poll_meetup_outreach_banners() -> None:
    """Pop queued meetup notifications from the API and append them as persistent chat messages.

    Using st.info() alone made banners disappear on rerun; Telegram-like threads need durable messages.
    """
    if "delivered_meetup_notification_ids" not in _tb:
        _tb["delivered_meetup_notification_ids"] = []
    delivered: list = _tb["delivered_meetup_notification_ids"]

    try:
        r = requests.get(
            f"{api_base_url()}/rag/meetup-pending-notifications",
            params={"thread_id": thread_id},
            timeout=5,
        )
        if not r.ok:
            return
        for n in r.json().get("notifications") or []:
            pid = str(n.get("proposal_id") or "").strip()
            msg = (n.get("message") or "").strip()
            if not msg:
                continue
            dedup_key = pid or msg[:200]
            if dedup_key in delivered:
                continue
            delivered.append(dedup_key)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Meetup notification\n\n{msg}"}
            )
        _tb["messages"] = st.session_state.messages
    except Exception:
        logger.debug("meetup pending notifications poll failed", exc_info=True)


poll_meetup_outreach_banners()

with st.sidebar:
    with st.expander("Test session (Telegram simulation)", expanded=True):
        st.caption(
            "`thread_id` is stored in the URL as `?thread=` for stable checkpoints per chat. "
            "`telegram_handle` is derived for the API like a TG username."
        )
        st.text_input("thread_id", value=thread_id, disabled=True, help="Sent as thread_id to POST /rag/")
        st.text_input("telegram_handle", value=fake_telegram_handle(thread_id), disabled=True, help="Sent to POST /rag/")
        if st.button("New random thread", help="New ?thread= and empty chat (like a new TG user)"):
            st.query_params["thread"] = secrets.token_urlsafe(12)
            st.rerun()

    with st.expander("API connection", expanded=False):
        base = api_base_url()
        st.code(base, language="text")
        try:
            r = requests.get(f"{base}/health", timeout=5)
            if r.ok:
                st.success(f"OK: {r.json()}")
            else:
                st.error(f"HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
        st.caption("Docker Compose should set API_URL=http://api:8000 for this container.")

    # Create tabs in the sidebar
    suggestions_tab, = st.tabs(["🔍 Suggestions"])
    
    # Suggestions Tab
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get('description', 'No description'))
                if 'image_url' in item:
                    st.image(item["image_url"], width=250)
                st.caption(f"Price: {item['price']} USD")
                st.divider()
        else:
            st.info("No suggestions yet")


for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = (
            message["role"] == "assistant" and 
            idx == len(st.session_state.messages) - 1 and 
            idx > 0
        )
        
        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_key = f"feedback_{thread_id}_{len(st.session_state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)
            
            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"
                
                # Only submit if this is a new/different feedback
                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(feedback_type=feedback_type)
                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = (feedback_type == "negative")
                            _tb["latest_feedback"] = feedback_type
                            _tb["feedback_submission_status"] = "success"
                            _tb["show_feedback_box"] = (feedback_type == "negative")
                        else:
                            st.session_state.feedback_submission_status = "error"
                            _tb["feedback_submission_status"] = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()
            
            # Show feedback status message
            if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")
            
            # Show feedback text box if thumbs down was pressed
            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption("Your negative feedback has already been recorded. You can optionally provide additional details below.")
                
                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key=f"feedback_text_{thread_id}_{len(st.session_state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100
                )
                
                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button("Send Additional Details", key=f"send_additional_{thread_id}_{len(st.session_state.messages)}"):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(feedback_text=feedback_text)
                                if status:
                                    st.success("✅ Thank you! Your additional feedback has been recorded.")
                                    st.session_state.show_feedback_box = False
                                    _tb["show_feedback_box"] = False
                                else:
                                    st.error("❌ Failed to submit additional feedback. Please try again.")
                        else:
                            st.warning("Please enter some feedback text before submitting.")
                        st.rerun()
                
                with col_close:
                    if st.button("Close", key=f"close_feedback_{thread_id}_{len(st.session_state.messages)}"):
                        st.session_state.show_feedback_box = False
                        _tb["show_feedback_box"] = False
                        st.rerun()


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        message_placeholder = st.empty()

        base = api_base_url()
        tg = fake_telegram_handle(thread_id)
        logger.info(
            "POST %s/rag/ (stream) thread_id=%s telegram_handle=%s",
            base,
            thread_id,
            tg,
        )

        got_final = False
        for line in api_call_stream(
            "post",
            f"{base}/rag/",
            json={
                "query": prompt,
                "thread_id": thread_id,
                "telegram_handle": tg,
            },
            stream=True,
            headers={"Accept": "text/event-stream"},
        ):
            if not line:
                continue
            line_text = line if isinstance(line, str) else line.decode("utf-8")

            if line_text.startswith("data: "):
                data = line_text[6:]

                try:
                    output = json.loads(data)

                    if output["type"] == "final_result":
                        answer = output["data"]["answer"]
                        used_context = output["data"]["used_context"]
                        trace_id = output["data"]["trace_id"]

                        st.session_state.used_context = used_context
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.trace_id = trace_id

                        st.session_state.latest_feedback = None
                        st.session_state.show_feedback_box = False
                        st.session_state.feedback_submission_status = None
                        _tb["used_context"] = used_context
                        _tb["trace_id"] = trace_id
                        _tb["latest_feedback"] = None
                        _tb["show_feedback_box"] = False
                        _tb["feedback_submission_status"] = None

                        status_placeholder.empty()
                        message_placeholder.markdown(answer)
                        got_final = True
                        break

                except json.JSONDecodeError:
                    status_placeholder.markdown(f"*{data}*")

        if not got_final:
            logger.warning("SSE stream ended without final_result (API_URL=%s)", api_base_url())
            message_placeholder.error(
                "No assistant reply received. Check the API container logs and that API_URL reaches the FastAPI "
                "service (Docker: http://api:8000)."
            )

    st.rerun()