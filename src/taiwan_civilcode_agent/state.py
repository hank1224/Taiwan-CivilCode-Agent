"""This module defines the state and actions for the Taiwan Civil Code Agent.

It includes:
- AgentState: A TypedDict for managing agent messages.
- Plan: A BaseModel for defining future plans.
- ParallelPlanExecute: A TypedDict for parallel plan execution.
- Response: A BaseModel for user responses.
- Act: A BaseModel for actions to perform.
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Plan to follow in future."""

    steps: list[str] = Field(
        description="該執行的計劃的子任務"
    )


# class PlanExecute(TypedDict):
#     input: str
#     plan: List[str]
#     past_steps: Annotated[List[Tuple], operator.add]
#     response: str
#     current_step_index: int

class ParallelPlanExecute(TypedDict):
    """A dictionary for parallel plan execution.

    Attributes:
    ----------
    input : str
        The input string for the plan.
    plan : List[str]
        The list of steps in the plan.
    past_steps : Annotated[List[Tuple[str, str]], operator.add]
        The list of past steps executed, annotated with their results.
    response : str
        The response generated after executing the plan.
    """
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Response | Plan = Field(
        description="如果你想要回應使用者，請使用「Response」。如果你需要進一步使用工具來獲得答案，請使用「Plan」。"
    )
