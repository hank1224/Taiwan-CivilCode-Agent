import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import END
from langgraph.prebuilt import create_react_agent

from taiwan_civilcode_agent.configuration import GPT4oMini
from taiwan_civilcode_agent.prompts import (
    LAW_TASK_SLOVER_SYSTEM_PROMPT,
    REPLANNER_PROMPT,
    TASK_PLANNER_PROMPT,
)
from taiwan_civilcode_agent.state import Act, ParallelPlanExecute, Plan, Response
from taiwan_civilcode_agent.tools import TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def execute_parallel(state: ParallelPlanExecute):
    """Executes all tasks in the current plan in parallel.

    Uses a ThreadPoolExecutor to run each planned step concurrently by invoking
    a React agent. Collects results for all tasks and updates the state's
    'past_steps' with the outcomes.

    Args:
        state: The current state of the graph, containing the 'plan' (list of tasks)
               and the original 'input'.

    Returns:
        The updated state with execution results added to the 'past_steps' key.
        Returns the state unchanged if the plan is empty.
    """
    plan = state.get("plan", [])
    original_input = state.get("input", "")

    if not plan:
        logger.info("Plan is empty, no tasks to execute.")
        state["past_steps"] = []  # Ensure past_steps is initialized even if empty
        return state

    logger.info(f"Submitting {len(plan)} tasks for parallel execution...")

    # --- Parallel Execution ---
    # Create a dictionary to map futures to task info (index, task) for result tracking
    future_to_task_info = {}
    # Use ThreadPoolExecutor for I/O-bound tasks (like LLM API calls).
    # Consider ProcessPoolExecutor if the agent_executor is heavily CPU-bound and GIL-limited.
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Initialize the agent executor instance once for all tasks within the pool context
        # Assuming Gemini20Flash and create_react_agent are thread-safe or inexpensive to create
        llm_instance = GPT4oMini()
        agent_executor = create_react_agent(llm_instance, TOOLS, state_modifier=LAW_TASK_SLOVER_SYSTEM_PROMPT)

        # Submit all tasks to the executor
        for i, task in enumerate(plan):
            # Format the task instruction for the agent, including context and task number
            task_formatted = (
                f"這是被已拆分的主要問題：{original_input}，在拆分的任務中，"
                f"你被分配到了第 {i + 1} 個任務，內容為： {task}."
            )
            # Submit the task and store the future, mapping it back to the original task/index
            future = executor.submit(
                agent_executor.invoke, {"messages": [("user", task_formatted)]}
            )
            future_to_task_info[future] = {"index": i, "task": task}

        # Use as_completed to retrieve results as they finish
        results_dict = {}  # Dictionary to temporarily store results, keyed by index for ordering
        for future in as_completed(future_to_task_info):
            task_info = future_to_task_info[future]
            task_index = task_info["index"]
            original_task = task_info["task"]
            try:
                # Get result from the future
                agent_response = future.result()
                # Extract the final message content from the agent's response
                response_content = agent_response["messages"][-1].content
                results_dict[task_index] = (original_task, response_content)
                logger.info(f"Task {task_index + 1} ('{original_task[:50]}...') completed successfully.")
            except Exception as e:
                logger.error(f"Task {task_index + 1} ('{original_task[:50]}...') failed: {e}", exc_info=True)
                # Store error message if execution fails
                results_dict[task_index] = (original_task, f"ERROR: {e}")
                # TODO: Implement specific error handling/retry logic if needed

    # --- Compile and Organize Results ---
    # Sort results by the original task order based on the index
    ordered_results = [results_dict[i] for i in range(len(plan))]
    logger.info(f"All {len(plan)} parallel tasks finished.")

    # --- Update State ---
    # Update 'past_steps' with the results from this parallel execution phase.
    # Note: LangGraph state updates using assignment (=) overwrite the key.
    # If using operator.add (+), results would be appended. For parallel execution,
    # overwriting with the results of the current batch is typical.
    state["past_steps"] = ordered_results

    # Return the updated state for the next node in the graph
    # Node functions typically return the state dictionary or a modification of it.
    return state


def plan_step(state: ParallelPlanExecute):
    """Generates the initial plan based on the input.

    Uses an LLM with a specific prompt and structured output
    to break down the initial input into a sequence of tasks.

    Args:
        state: The current state of the graph, containing the initial 'input'.

    Returns:
        A dictionary containing the generated 'plan' (list of steps) and
        initializes 'current_step_index' to 0.
    """
    logger.info("Generating initial plan...")
    llm_instance = GPT4oMini()
    # Use the planner prompt and structured output model to create the plan
    planner = TASK_PLANNER_PROMPT | llm_instance.with_structured_output(Plan)
    plan = planner.invoke({"messages": [("user", state["input"])]})
    logger.info(f"Initial plan generated with {len(plan.steps)} steps.")
    return {
        "plan": plan.steps,
        "current_step_index": 0  # Initialize current_step_index
    }


def replan_step(state: ParallelPlanExecute):
    """Updates the plan based on the current state and execution results.

    Uses a replanner agent (LLM) to analyze the execution results ('past_steps')
    and the original plan/input. It decides whether a final 'response' can be
    given or if the plan needs to be updated. If a new plan is generated, it
    appends the new steps to the existing plan in the state.

    Args:
        state: The current state of the graph, containing 'input', 'plan',
               and 'past_steps' (results from parallel execution).

    Returns:
        A dictionary updating the state. Contains 'response' if the replanner
        decides to respond, or updates 'plan' if new steps are generated.

    Raises:
        ValueError: If the replanner returns an unexpected output type.
    """
    logger.info("Executing replan step...")
    llm_instance = GPT4oMini()
    # Use the replanner prompt and structured output to get the action (Respond or Plan)
    replanner = REPLANNER_PROMPT | llm_instance.with_structured_output(Act)
    output = replanner.invoke(state)
    logger.info(f"Replanner output action type: {type(output.action).__name__}")


    if isinstance(output.action, Response):
        # If the replanner outputs a Response, the task is complete
        return {"response": output.action.response}

    elif isinstance(output.action, Plan):
        # If the replanner outputs a Plan, update the existing plan
        # Get existing steps from the state's plan
        # Note: In the current parallel flow, execute_parallel processes the entire plan
        # and does not advance current_step_index. Thus, existing_steps might be the
        # complete original plan if this is the first replan cycle after execution.
        # The following logic appends new steps to the current plan state value.
        current_plan = state.get("plan", [])
        # Get newly generated steps from the replanner's output
        new_steps = output.action.steps

        # Combine steps: append the new steps to the existing ones
        combined_steps = current_plan + new_steps

        logger.info(f"Replanning added {len(new_steps)} new steps. Total plan length: {len(combined_steps)}")

        return {"plan": combined_steps}

    else:
        # Handle unexpected output types from the replanner
        logger.error(f"Unexpected output type from replanner: {type(output.action).__name__}")
        # Potentially set an error state or a default response
        raise ValueError(f"Unexpected output type from replanner: {type(output.action).__name__}")

def should_end(state: ParallelPlanExecute):
    """Determines the next node in the graph execution.

    The graph ends if a final 'response' has been populated in the state,
    indicating the task is complete. Otherwise, it directs the flow to
    the 'agent' node (intended for replanning or error handling).

    Args:
        state: The current state of the graph.

    Returns:
        END if a response is present, otherwise the string "agent".
    """
    # Check if the 'response' key exists and is not empty in the state
    if state.get("response"):
        logger.info("Response found in state, ending graph execution.")
        return END
    else:
        logger.info("No final response found, proceeding to agent step (replan/handle).")
        return "agent"
