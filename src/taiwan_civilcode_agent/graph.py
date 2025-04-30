from langgraph.graph import END, START, StateGraph

from taiwan_civilcode_agent.nodes import (
    execute_parallel,
    plan_step,
    replan_step,
    should_end,
)
from taiwan_civilcode_agent.state import ParallelPlanExecute

workflow = StateGraph(ParallelPlanExecute)

workflow.add_node("planner", plan_step)

workflow.add_node("agent", execute_parallel)

workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

workflow.add_edge("planner", "agent")

workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

graph = workflow.compile(name="ReAct Agent")
