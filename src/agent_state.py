from typing import Annotated, List, Literal, Optional
from langgraph.graph import MessagesState
import operator
from pydantic import BaseModel, Field

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ClarifyWithUser(BaseModel):
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )

class SectionOutput(BaseModel):
    section_content: str = Field(
        description="The content of the section.",
    )

class ReportStateInput(MessagesState):
    """InputState is only 'messages'"""
    already_clarified_topic: Optional[bool] = None # If the user has clarified the topic with the agent

class ReportStateOutput(MessagesState):
    final_report: str
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

class ReportState(MessagesState):
    already_clarified_topic: Optional[bool] = None # If the user has clarified the topic with the agent
    feedback_on_report_plan: Annotated[list[str], operator.add] # List of feedback on the report plan
    sections: list[Section] # List of report sections
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: Annotated[str, operator.add] # String of formatted source content from web search



class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class SectionState(MessagesState):
    section: Section
    search_iterations: int
    search_queries: list[SearchQuery]
    source_str: str
    report_sections_from_research: str
    completed_sections: list[Section]

class SectionOutputState(MessagesState):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search
