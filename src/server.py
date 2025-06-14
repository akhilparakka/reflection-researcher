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
from src.streaming_events import (
    WorkflowPhase, InterruptType, SectionInfo,
    create_workflow_status_event, create_section_completed_event,
    create_search_results_event, create_interrupt_event,
    create_final_report_event, create_report_plan_event,
    parse_sources_from_string, get_phase_progress_percentage,
    get_phase_message
)

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
    Async generator to stream events from the report generation workflow with enhanced UI-friendly events.

    Args:
        message: Single user input message to initialize or resume the workflow.
        already_clarified_topic: Whether the topic has been clarified (defaults to False).
        thread_id: Unique identifier for the thread (defaults to 'default-thread').
        max_search_iterations: Maximum number of search iterations (defaults to 3).
        auto_accept_plan: Whether to auto-accept the report plan (defaults to False).
        interrupt_feedback: Feedback for resuming after an interrupt (defaults to '').

    Yields:
        Enhanced event strings optimized for different UI components:
        - workflow_status: Overall progress updates
        - clarification_request: User clarification needed
        - report_plan_generated: Structured report plan
        - section_progress: Individual section updates
        - search_results: Search sources for UI display
        - section_completed: Finished section content
        - final_report: Complete report ready
        - interrupt: User input required
    """
    messages = [HumanMessage(content=message)]

    input_ = {
        "messages": messages,
        "already_clarified_topic": already_clarified_topic,
    }

    if interrupt_feedback:
        input_ = Command(resume=interrupt_feedback)

    # Track workflow state for enhanced UI updates
    current_phase = "initializing"
    sections_status = {}

    # Send initial workflow status
    initial_status = create_workflow_status_event(
        thread_id=thread_id,
        phase=WorkflowPhase.STARTING,
        message=get_phase_message(WorkflowPhase.STARTING),
        progress_percentage=5
    )
    yield _make_event("workflow_status", initial_status)

    async for agent, _, event_data in workflow.astream(
        input_,
        config={
            "configurable": {
                "thread_id": thread_id,
                "max_search_iterations": max_search_iterations,
                "include_source_str": True,
                "sections_user_approval": not auto_accept_plan,
                "clarify_with_user": not already_clarified_topic,
            },
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        agent_name = agent[0].split(":")[0] if agent else "unknown"

        if isinstance(event_data, dict):
            # Handle interrupts with enhanced UI data
            if "__interrupt__" in event_data:
                interrupt_data = event_data["__interrupt__"][0]
                content = interrupt_data.value if hasattr(interrupt_data, "value") else ""

                # Determine interrupt type for better UI handling
                interrupt_type = InterruptType.USER_INPUT_REQUIRED
                if "clarify" in agent_name.lower():
                    interrupt_type = InterruptType.CLARIFICATION_REQUEST
                elif "feedback" in agent_name.lower():
                    interrupt_type = InterruptType.PLAN_FEEDBACK_REQUEST

                event_data = create_interrupt_event(
                    thread_id=thread_id,
                    interrupt_type=interrupt_type,
                    agent=agent_name,
                    question=content,
                    event_id=interrupt_data.ns[0] if hasattr(interrupt_data, "ns") else f"interrupt-{thread_id}"
                )
                yield _make_event("interrupt", event_data)
                continue

            # Handle section completions with rich data
            elif "completed_sections" in event_data:
                section_data = event_data.get("completed_sections", [])
                sources = event_data.get("source_str", "")

                for section in section_data:
                    sections_status[section.name] = "completed"

                    # Create section info
                    section_info = SectionInfo(
                        name=section.name,
                        description=section.description,
                        content=section.content,
                        word_count=len(section.content.split()) if section.content else 0,
                        has_research=section.research,
                        status="completed"
                    )

                    # Send section completion event
                    completion_event = create_section_completed_event(
                        thread_id=thread_id,
                        section=section_info
                    )
                    yield _make_event("section_completed", completion_event)

                    # Send search sources if available
                    if sources and section.research:
                        parsed_sources = parse_sources_from_string(sources)
                        if parsed_sources:
                            search_event = create_search_results_event(
                                thread_id=thread_id,
                                section_name=section.name,
                                sources=parsed_sources
                            )
                            yield _make_event("search_results", search_event)
                continue

            # Handle report sections generation
            elif "sections" in event_data:
                sections = event_data.get("sections", [])
                section_infos = [
                    SectionInfo(
                        name=s.name,
                        description=s.description,
                        has_research=s.research,
                        status="planned"
                    ) for s in sections
                ]

                plan_event = create_report_plan_event(
                    thread_id=thread_id,
                    sections=section_infos
                )
                yield _make_event("report_plan_generated", plan_event)
                continue

            # Handle final report
            elif "final_report" in event_data:
                final_event = create_final_report_event(
                    thread_id=thread_id,
                    content=event_data["final_report"],
                    sections_completed=len(sections_status)
                )
                yield _make_event("final_report", final_event)
                continue

        # Handle message chunks with agent context - check if event_data is a tuple
        if not isinstance(event_data, tuple) or len(event_data) != 2:
            # Skip non-message events that don't have the tuple structure
            continue

        message_chunk, message_metadata = cast(
            tuple[AIMessageChunk, dict[str, Any]], event_data
        )

        # Update current phase based on agent
        new_phase = _determine_phase(agent_name)
        if new_phase != current_phase:
            current_phase = new_phase
            progress = get_phase_progress_percentage(
                WorkflowPhase(current_phase),
                total_sections=len(sections_status),
                completed_sections=sum(1 for status in sections_status.values() if status == "completed")
            )

            status_event = create_workflow_status_event(
                thread_id=thread_id,
                phase=WorkflowPhase(current_phase),
                message=get_phase_message(WorkflowPhase(current_phase)),
                agent=agent_name,
                progress_percentage=progress
            )
            yield _make_event("workflow_status", status_event)

        event_stream_message: dict[str, Any] = {
            "thread_id": thread_id,
            "agent": agent_name,
            "phase": current_phase,
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
            "timestamp": _get_timestamp(),
            "ui_component": "workflow_status"
        }

        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get("finish_reason")

        # Enhanced tool handling
        if isinstance(message_chunk, ToolMessage):
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
            event_stream_message["tool_calls"] = message_chunk.tool_calls
            # Send tool calls with context
            yield _make_event("tool_calls", event_stream_message)
        else:
            # Send regular message chunks with phase context
            yield _make_event("message_chunk", event_stream_message)

def _make_event(event_type: str, data: dict[str, Any]) -> str:
    """Create a Server-Sent Event with enhanced structure for UI components."""
    if data.get("content") == "":
        data.pop("content", None)
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()

def _determine_phase(agent_name: str) -> str:
    """Determine workflow phase based on agent name."""
    phase_mapping = {
        "clarify_with_user": WorkflowPhase.CLARIFICATION.value,
        "generate_report_plan": WorkflowPhase.PLANNING.value,
        "human_feedback": WorkflowPhase.FEEDBACK.value,
        "build_section_with_web_research": WorkflowPhase.RESEARCH_WRITING.value,
        "gather_completed_sections": WorkflowPhase.COMPILATION.value,
        "write_final_sections": WorkflowPhase.FINAL_WRITING.value,
        "compile_final_report": WorkflowPhase.FINALIZATION.value
    }
    return phase_mapping.get(agent_name, WorkflowPhase.RESEARCH_WRITING.value)

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
