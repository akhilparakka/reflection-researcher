from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from typing import cast, Any
from uuid import uuid4
import json
from pydantic import BaseModel, Field
from src.builder import workflow
from langgraph.types import Command

app = FastAPI(
    title="Report Generator",
    description="API for streaming report generation",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(description="User input message to generate the report")
    thread_id: str = Field(default="__default__", description="Unique thread identifier")
    already_clarified_topic: bool = Field(default=False, description="Whether the topic has been clarified")
    max_search_iterations: int = Field(default=3, description="Maximum number of search iterations")
    auto_accept_plan: bool = Field(default=False, description="Whether to auto-accept the report plan")
    interrupt_feedback: str = Field(default="", description="Feedback for resuming after an interrupt")

async def _astream_report_generator(
    message: str,
    already_clarified_topic: bool = False,
    thread_id: str = "default-thread",
    max_search_iterations: int = 3,
    auto_accept_plan: bool = False,
    interrupt_feedback: str = "",
):
    """
    Async generator to stream events from the report generation workflow with interrupt handling.

    Args:
        message: Single user input message to initialize or resume the workflow.
        already_clarified_topic: Whether the topic has been clarified (defaults to False).
        thread_id: Unique identifier for the thread (defaults to 'default-thread').
        max_search_iterations: Maximum number of search iterations (defaults to 3).
        auto_accept_plan: Whether to auto-accept the report plan (defaults to False).
        interrupt_feedback: Feedback for resuming after an interrupt (defaults to '').

    Yields:
        Formatted event strings containing message chunks, tool calls, section completions, or interrupts.
    """
    # Convert the input message to a HumanMessage
    messages = [HumanMessage(content=message)]

    input_ = {
        "messages": messages,
        "already_clarified_topic": already_clarified_topic,
        "feedback_on_report_plan": [],
        "sections": [],
        "completed_sections": [],
        "report_sections_from_research": "",
        "final_report": "",
        "source_str": "",
    }

    # Handle interrupt feedback with Command(resume=...)
    if not auto_accept_plan and interrupt_feedback:
        resume_msg = f"[{interrupt_feedback}] {message}"
        print(interrupt_feedback, "Checkkk")
        input_ = Command(resume=resume_msg)

    async for agent, _, event_data in workflow.astream(
        input_,
        config={
            "thread_id": thread_id,
            "max_search_iterations": max_search_iterations,
            "config_settings": {"include_source_str": True},
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        if isinstance(event_data, dict):
            # Handle interrupts (e.g., from clarify_with_user or human_feedback)
            if "__interrupt__" in event_data:
                interrupt_data = event_data["__interrupt__"][0]
                content = interrupt_data.value if hasattr(interrupt_data, "value") else ""
                agent_name = agent[0].split(":")[0] if agent else "unknown"
                options = []
                if agent_name == "clarify_with_user":
                    # Try to parse content as JSON for ClarifyWithUser
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "question" in parsed:
                            content = parsed["question"]
                    except json.JSONDecodeError:
                        pass
                    options = [
                        {"text": "Provide clarification", "value": "clarify"},
                        {"text": "Proceed with report", "value": "proceed"},
                    ]
                elif agent_name == "human_feedback":
                    options = [
                        {"text": "Approve plan", "value": "true"},
                        {"text": "Provide feedback", "value": "feedback"},
                    ]
                yield _make_event(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "id": interrupt_data.ns[0] if hasattr(interrupt_data, "ns") else f"interrupt-{thread_id}",
                        "role": "assistant",
                        "content": content,
                        "finish_reason": "interrupt",
                        "options": options,
                    },
                )
            # Handle section completions
            elif "completed_sections" in event_data:
                section_data = event_data.get("completed_sections", [])
                for section in section_data:
                    yield _make_event(
                        "section_completed",
                        {
                            "thread_id": thread_id,
                            "agent": "build_section_with_web_research",
                            "section_name": section.name,
                            "section_content": section.content,
                            "source_str": event_data.get("source_str", ""),
                        },
                    )
            continue

        message_chunk, message_metadata = cast(
            tuple[AIMessageChunk, dict[str, Any]], event_data
        )
        event_stream_message: dict[str, Any] = {
            "thread_id": thread_id,
            "agent": agent[0].split(":")[0] if agent else "unknown",
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
        }

        # Handle finish reason
        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get("finish_reason")

        # Handle tool messages
        if isinstance(message_chunk, ToolMessage):
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
            event_stream_message["tool_calls"] = message_chunk.tool_calls
            yield _make_event("tool_calls", event_stream_message)
        else:
            yield _make_event("message_chunk", event_stream_message)

def _make_event(event_type: str, data: dict[str, Any]) -> str:
    if data.get("content") == "":
        data.pop("content", None)
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
    print(thread_id, "Check thread Id")
    if thread_id == "__default__":
        thread_id = str(uuid4())

    return StreamingResponse(
        _astream_report_generator(
            message=request.message,
            already_clarified_topic=request.already_clarified_topic,
            thread_id=thread_id,
            max_search_iterations=request.max_search_iterations,
            auto_accept_plan=request.auto_accept_plan,
            interrupt_feedback=request.interrupt_feedback,
        ),
        media_type="text/event-stream",
    )
