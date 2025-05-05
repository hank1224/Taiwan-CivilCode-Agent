import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Plan(BaseModel):
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
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str


class Response(BaseModel):
    response: str


class Act(BaseModel):
    action: Response | Plan = Field(
        description="如果你想要回應使用者，請使用「Response」。如果你需要進一步使用工具來獲得答案，請使用「Plan」。"
    )

class Question(BaseModel):
    question: str = Field(description="問題")
    option_a: str = Field(description="選項A內容")
    option_b: str = Field(description="選項B內容")
    option_c: str = Field(description="選項C內容")
    option_d: str = Field(description="選項D內容")
    question_type: str = Field(description="問題類型")
