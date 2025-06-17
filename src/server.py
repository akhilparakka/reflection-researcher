from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessageChunk
from typing import cast, Any
from uuid import uuid4
import json
from pydantic import BaseModel, Field
from src.builder import workflow
from langgraph.types import Command

app = FastAPI(
    title="Report Generator",
    description="API for streaming report generation with artifacts",
    version="0.1.0"
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
    messages = [HumanMessage(content=message)]

    input_ = {
        "messages": messages,
        "already_clarified_topic": already_clarified_topic,
    }

    if interrupt_feedback:
        input_ = Command(resume=interrupt_feedback)

    current_phase = "starting"

    yield _create_event("status", {
        "thread_id": thread_id,
        "phase": current_phase,
        "message": "🚀 Starting report generation...",
        "timestamp": _get_timestamp()
    })

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
            # Handle interrupts - need user input
            if "__interrupt__" in event_data:
                interrupt_data = event_data["__interrupt__"][0]
                content = interrupt_data.value if hasattr(interrupt_data, "value") else ""

                yield _create_event("interrupt", {
                    "thread_id": thread_id,
                    "agent": agent_name,
                    "question": content,
                    "interrupt_id": interrupt_data.ns[0] if hasattr(interrupt_data, "ns") else f"interrupt-{thread_id}",
                    "timestamp": _get_timestamp()
                })
                continue

            # Handle report plan - send as artifact
            elif "sections" in event_data:
                sections = event_data.get("sections", [])
                plan_content = _format_plan_for_artifact(sections)

                yield _create_event("artifact", {
                    "thread_id": thread_id,
                    "type": "report_plan",
                    "title": "Report Plan",
                    "content": plan_content,
                    "sections": [
                        {
                            "name": s.name,
                            "description": s.description,
                            "has_research": s.research
                        } for s in sections
                    ],
                    "timestamp": _get_timestamp()
                })

                # Update phase
                yield _create_event("status", {
                    "thread_id": thread_id,
                    "phase": "planning_complete",
                    "message": "📋 Report plan generated - please review",
                    "timestamp": _get_timestamp()
                })
                continue

            # Handle final report - send as artifact
            elif "final_report" in event_data:
                final_report = event_data["final_report"]

                yield _create_event("artifact", {
                    "thread_id": thread_id,
                    "type": "final_report",
                    "title": "Final Report",
                    "content": final_report,
                    "word_count": len(final_report.split()) if final_report else 0,
                    "timestamp": _get_timestamp()
                })

                # Update phase to complete
                yield _create_event("status", {
                    "thread_id": thread_id,
                    "phase": "completed",
                    "message": "✅ Report generation completed!",
                    "timestamp": _get_timestamp()
                })
                continue

            # Handle section completion progress
            elif "completed_sections" in event_data:
                completed_sections = event_data.get("completed_sections", [])

                yield _create_event("progress", {
                    "thread_id": thread_id,
                    "completed_sections": len(completed_sections),
                    "message": f"📝 Completed {len(completed_sections)} sections",
                    "timestamp": _get_timestamp()
                })
                continue

        # Handle streaming messages
        if isinstance(event_data, tuple) and len(event_data) == 2:
            message_chunk, message_metadata = cast(
                tuple[AIMessageChunk, dict[str, Any]], event_data
            )

            # Update phase based on agent
            new_phase = _determine_phase(agent_name)
            if new_phase != current_phase:
                current_phase = new_phase
                yield _create_event("status", {
                    "thread_id": thread_id,
                    "phase": current_phase,
                    "agent": agent_name,
                    "message": _get_phase_message(current_phase),
                    "timestamp": _get_timestamp()
                })

            # Send message chunks for real-time updates
            if message_chunk.content:
                yield _create_event("message", {
                    "thread_id": thread_id,
                    "agent": agent_name,
                    "phase": current_phase,
                    "content": message_chunk.content,
                    "timestamp": _get_timestamp()
                })

def _create_event(event_type: str, data: dict[str, Any]) -> str:
    """Create a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()

def _determine_phase(agent_name: str) -> str:
    """Determine workflow phase based on agent name."""
    phase_mapping = {
        "clarify_with_user": "clarification",
        "generate_report_plan": "planning",
        "human_feedback": "feedback",
        "build_section_with_web_research": "research_writing",
        "gather_completed_sections": "compilation",
        "write_final_sections": "final_writing",
        "compile_final_report": "finalization"
    }
    return phase_mapping.get(agent_name, "processing")

def _get_phase_message(phase: str) -> str:
    """Get user-friendly message for each phase."""
    messages = {
        "starting": "🚀 Initializing report generation",
        "clarification": "❓ Understanding your requirements",
        "planning": "📋 Creating report structure",
        "feedback": "⏱️ Waiting for your feedback",
        "research_writing": "📚 Researching and writing sections",
        "compilation": "📝 Organizing sections",
        "final_writing": "✍️ Writing final sections",
        "finalization": "🔧 Compiling final report",
        "completed": "✅ Report completed!"
    }
    return messages.get(phase, "🔄 Processing...")

def _format_plan_for_artifact(sections) -> str:
    """Format the report plan as markdown for artifact display."""
    if not sections:
        return "# Report Plan\n\nNo sections planned yet."

    content = "# Report Plan\n\n"
    content += f"This report will contain **{len(sections)} sections**.\n\n"

    for i, section in enumerate(sections, 1):
        research_indicator = "🔍 Research Required" if section.research else "📝 Writing Only"
        content += f"## {i}. {section.name}\n\n"
        content += f"**{research_indicator}**\n\n"
        content += f"{section.description}\n\n"
        content += "---\n\n"

    content += "\n*Please review this plan and provide your feedback.*"
    return content

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
