import pytest
from langsmith import unit

from taiwan_civilcode_agent import graph


# TODO: this test is not working
@pytest.mark.asyncio
@unit
async def test_taiwan_civilcode_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},
        {"configurable": {"system_prompt": "You are a helpful AI assistant."}},
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
