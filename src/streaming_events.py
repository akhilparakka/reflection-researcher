"""
Enhanced streaming event types and utilities for beautiful UI components.
This module provides structured event types that map directly to UI components.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class UIComponent(str, Enum):
    """UI components that will handle different event types."""
    WORKFLOW_STATUS = "workflow_status"
    CLARIFICATION_DIALOG = "clarification_dialog"
    PLAN_DISPLAY = "plan_display"
    SECTION_PROGRESS = "section_progress"
    SECTION_DISPLAY = "section_display"
    SOURCES_PANEL = "sources_panel"
    FINAL_REPORT_DISPLAY = "final_report_display"
    INTERRUPT_DIALOG = "interrupt_dialog"
    SEARCH_INDICATOR = "search_indicator"
    ERROR_DISPLAY = "error_display"

class WorkflowPhase(str, Enum):
    """Different phases of the report generation workflow."""
    STARTING = "starting"
    CLARIFICATION = "clarification"
    PLANNING = "planning"
    FEEDBACK = "feedback"
    RESEARCH_WRITING = "research_writing"
    COMPILATION = "compilation"
    FINAL_WRITING = "final_writing"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    ERROR = "error"

class InterruptType(str, Enum):
    """Types of interrupts that require user interaction."""
    CLARIFICATION_REQUEST = "clarification_request"
    PLAN_FEEDBACK_REQUEST = "plan_feedback_request"
    SECTION_FEEDBACK_REQUEST = "section_feedback_request"
    USER_INPUT_REQUIRED = "user_input_required"

class SourceInfo(BaseModel):
    """Information about a research source."""
    title: str
    url: str
    summary: str
    source_number: int
    relevance_score: Optional[float] = None
    domain: Optional[str] = None

class SectionInfo(BaseModel):
    """Information about a report section."""
    name: str
    description: str
    content: Optional[str] = None
    word_count: Optional[int] = None
    has_research: bool = False
    status: Literal["planned", "in_progress", "completed", "error"] = "planned"
    sources_count: Optional[int] = None

class BaseStreamEvent(BaseModel):
    """Base class for all streaming events."""
    event_type: str
    thread_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    ui_component: UIComponent

class WorkflowStatusEvent(BaseStreamEvent):
    """Event for overall workflow status updates."""
    event_type: str = "workflow_status"
    ui_component: UIComponent = UIComponent.WORKFLOW_STATUS
    phase: WorkflowPhase
    agent: Optional[str] = None
    message: str
    progress_percentage: Optional[int] = None
    estimated_time_remaining: Optional[str] = None

class ClarificationRequestEvent(BaseStreamEvent):
    """Event when user clarification is needed."""
    event_type: str = "clarification_request"
    ui_component: UIComponent = UIComponent.CLARIFICATION_DIALOG
    question: str
    context: Optional[str] = None
    suggestions: Optional[List[str]] = None

class ReportPlanEvent(BaseStreamEvent):
    """Event when report plan is generated."""
    event_type: str = "report_plan_generated"
    ui_component: UIComponent = UIComponent.PLAN_DISPLAY
    sections: List[SectionInfo]
    total_sections: int
    estimated_research_sections: int
    estimated_word_count: Optional[int] = None

class SectionProgressEvent(BaseStreamEvent):
    """Event for section writing progress."""
    event_type: str = "section_progress"
    ui_component: UIComponent = UIComponent.SECTION_PROGRESS
    section_name: str
    status: Literal["started", "researching", "writing", "completed"]
    progress_percentage: Optional[int] = None
    current_step: Optional[str] = None

class SectionCompletedEvent(BaseStreamEvent):
    """Event when a section is completed."""
    event_type: str = "section_completed"
    ui_component: UIComponent = UIComponent.SECTION_DISPLAY
    section: SectionInfo
    research_quality_score: Optional[float] = None

class SearchResultsEvent(BaseStreamEvent):
    """Event for search results/sources."""
    event_type: str = "search_results"
    ui_component: UIComponent = UIComponent.SOURCES_PANEL
    section_name: str
    sources: List[SourceInfo]
    search_query: Optional[str] = None
    total_sources_found: int

class FinalReportEvent(BaseStreamEvent):
    """Event when final report is ready."""
    event_type: str = "final_report"
    ui_component: UIComponent = UIComponent.FINAL_REPORT_DISPLAY
    content: str
    word_count: int
    sections_completed: int
    total_sources_used: Optional[int] = None
    quality_score: Optional[float] = None

class InterruptEvent(BaseStreamEvent):
    """Event for user interrupts."""
    event_type: str = "interrupt"
    ui_component: UIComponent = UIComponent.INTERRUPT_DIALOG
    interrupt_type: InterruptType
    agent: str
    question: str
    id: str
    options: Optional[List[str]] = None
    default_response: Optional[str] = None

class SearchIndicatorEvent(BaseStreamEvent):
    """Event for search/research indicators."""
    event_type: str = "search_indicator"
    ui_component: UIComponent = UIComponent.SEARCH_INDICATOR
    is_searching: bool
    search_query: Optional[str] = None
    search_engine: Optional[str] = None
    section_name: Optional[str] = None

class ErrorEvent(BaseStreamEvent):
    """Event for error handling."""
    event_type: str = "error"
    ui_component: UIComponent = UIComponent.ERROR_DISPLAY
    error_type: str
    error_message: str
    is_recoverable: bool = True
    suggested_action: Optional[str] = None

class MessageChunkEvent(BaseStreamEvent):
    """Event for streaming message chunks."""
    event_type: str = "message_chunk"
    ui_component: UIComponent = UIComponent.WORKFLOW_STATUS
    agent: str
    phase: WorkflowPhase
    content: str
    role: str = "assistant"
    id: str
    finish_reason: Optional[str] = None

# Event factory functions for easy creation

def create_workflow_status_event(
    thread_id: str,
    phase: WorkflowPhase,
    message: str,
    agent: Optional[str] = None,
    progress_percentage: Optional[int] = None
) -> Dict[str, Any]:
    """Create a workflow status event."""
    event = WorkflowStatusEvent(
        thread_id=thread_id,
        phase=phase,
        message=message,
        agent=agent,
        progress_percentage=progress_percentage
    )
    return event.model_dump()

def create_section_completed_event(
    thread_id: str,
    section: SectionInfo,
    research_quality_score: Optional[float] = None
) -> Dict[str, Any]:
    """Create a section completed event."""
    event = SectionCompletedEvent(
        thread_id=thread_id,
        section=section,
        research_quality_score=research_quality_score
    )
    return event.model_dump()

def create_search_results_event(
    thread_id: str,
    section_name: str,
    sources: List[SourceInfo],
    search_query: Optional[str] = None
) -> Dict[str, Any]:
    """Create a search results event."""
    event = SearchResultsEvent(
        thread_id=thread_id,
        section_name=section_name,
        sources=sources,
        search_query=search_query,
        total_sources_found=len(sources)
    )
    return event.model_dump()

def create_interrupt_event(
    thread_id: str,
    interrupt_type: InterruptType,
    agent: str,
    question: str,
    event_id: str,
    options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create an interrupt event."""
    event = InterruptEvent(
        thread_id=thread_id,
        interrupt_type=interrupt_type,
        agent=agent,
        question=question,
        id=event_id,
        options=options
    )
    return event.model_dump()

def create_final_report_event(
    thread_id: str,
    content: str,
    sections_completed: int,
    total_sources_used: Optional[int] = None
) -> Dict[str, Any]:
    """Create a final report event."""
    word_count = len(content.split()) if content else 0
    event = FinalReportEvent(
        thread_id=thread_id,
        content=content,
        word_count=word_count,
        sections_completed=sections_completed,
        total_sources_used=total_sources_used
    )
    return event.model_dump()

def create_report_plan_event(
    thread_id: str,
    sections: List[SectionInfo]
) -> Dict[str, Any]:
    """Create a report plan event."""
    research_sections = sum(1 for s in sections if s.has_research)
    event = ReportPlanEvent(
        thread_id=thread_id,
        sections=sections,
        total_sections=len(sections),
        estimated_research_sections=research_sections
    )
    return event.model_dump()

# Utility functions for parsing and formatting

def parse_sources_from_string(source_str: str) -> List[SourceInfo]:
    """Parse source string into structured SourceInfo objects."""
    sources = []
    if not source_str:
        return sources

    sections = source_str.split("--- SOURCE")
    for i, section in enumerate(sections[1:], 1):
        lines = section.strip().split('\n')
        if len(lines) >= 2:
            title = lines[0].replace(f"{i}:", "").strip()
            url_line = next((line for line in lines if line.startswith("URL:")), "")
            url = url_line.replace("URL:", "").strip() if url_line else ""

            # Extract domain from URL
            domain = ""
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                except:
                    pass

            # Extract summary/content
            summary = ""
            content_start = False
            for line in lines:
                if line.startswith("SUMMARY:") or line.startswith("Most relevant content"):
                    content_start = True
                    if ":" in line:
                        summary += line.split(":", 1)[1].strip() + " "
                    continue
                elif line.startswith("FULL CONTENT:") or line.startswith("---") or line.startswith("==="):
                    break
                elif content_start and line.strip():
                    summary += line.strip() + " "

            # Truncate summary if too long
            if len(summary) > 300:
                summary = summary[:297] + "..."

            sources.append(SourceInfo(
                title=title,
                url=url,
                summary=summary.strip(),
                source_number=i,
                domain=domain
            ))

    return sources

def get_phase_progress_percentage(phase: WorkflowPhase, total_sections: int = 0, completed_sections: int = 0) -> int:
    """Calculate progress percentage based on workflow phase."""
    phase_weights = {
        WorkflowPhase.STARTING: 5,
        WorkflowPhase.CLARIFICATION: 10,
        WorkflowPhase.PLANNING: 20,
        WorkflowPhase.FEEDBACK: 25,
        WorkflowPhase.RESEARCH_WRITING: 30 + (completed_sections / max(total_sections, 1)) * 40,
        WorkflowPhase.COMPILATION: 80,
        WorkflowPhase.FINAL_WRITING: 90,
        WorkflowPhase.FINALIZATION: 95,
        WorkflowPhase.COMPLETED: 100,
        WorkflowPhase.ERROR: 0
    }
    return min(100, max(0, int(phase_weights.get(phase, 0))))

def get_phase_message(phase: WorkflowPhase) -> str:
    """Get user-friendly message for each workflow phase."""
    messages = {
        WorkflowPhase.STARTING: "🚀 Initializing report generation",
        WorkflowPhase.CLARIFICATION: "❓ Understanding your requirements",
        WorkflowPhase.PLANNING: "📋 Creating detailed report structure",
        WorkflowPhase.FEEDBACK: "⏱️ Waiting for your feedback",
        WorkflowPhase.RESEARCH_WRITING: "📚 Researching and writing sections",
        WorkflowPhase.COMPILATION: "📝 Organizing completed sections",
        WorkflowPhase.FINAL_WRITING: "✍️ Writing conclusions and final sections",
        WorkflowPhase.FINALIZATION: "🔧 Compiling your final report",
        WorkflowPhase.COMPLETED: "✅ Report generation completed!",
        WorkflowPhase.ERROR: "❌ An error occurred"
    }
    return messages.get(phase, "🔄 Processing your request")
