"""
Simplified streaming events focused on artifacts and core functionality.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ArtifactEvent(BaseModel):
    """Event for artifact content (plans and reports)."""
    thread_id: str
    type: str  # "report_plan" or "final_report"
    title: str
    content: str  # Markdown content for the artifact
    sections: Optional[List[Dict[str, Any]]] = None  # For plan metadata
    word_count: Optional[int] = None  # For final report
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class InterruptEvent(BaseModel):
    """Event for user interrupts/questions."""
    thread_id: str
    agent: str
    question: str
    interrupt_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class StatusEvent(BaseModel):
    """Event for workflow status updates."""
    thread_id: str
    phase: str
    message: str
    agent: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MessageEvent(BaseModel):
    """Event for streaming message chunks."""
    thread_id: str
    agent: str
    phase: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ProgressEvent(BaseModel):
    """Event for progress updates."""
    thread_id: str
    completed_sections: int
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Event types that your frontend should handle:
"""
1. "status" - StatusEvent: Overall workflow status
2. "interrupt" - InterruptEvent: When user input is needed
3. "artifact" - ArtifactEvent: Plan or final report for artifact display
4. "message" - MessageEvent: Real-time message chunks
5. "progress" - ProgressEvent: Section completion progress
"""
