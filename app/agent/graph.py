import logging
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    load_context_node,
    agent_node,
    tools_node,
    respond_node,
    should_continue,
)

logger = logging.getLogger(__name__)


def build_agent_graph():
    """
    Builds and compiles the LangGraph agent workflow.

    The Graph Flow:
        load_context → agent → [tools → agent]* → respond → END

    The agent and tools nodes loop until the LLM stops calling tools.
    should_continue is a conditional edge that routes to 'tools' or 'respond'.

    Returns:
        CompiledGraph: Ready-to-invoke LangGraph agent
    """
    graph = StateGraph(AgentState)

    # 1. Register all nodes
    graph.add_node("load_context", load_context_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("respond", respond_node)

    # Set the entry point
    graph.set_entry_point("load_context")

    # Define standard linear edges
    graph.add_edge("load_context", "agent")
    graph.add_edge("tools", "agent")  # After tools run, loop back to the agent
    graph.add_edge("respond", END)

    # Define the conditional routing logic
    # After the agent thinks, the 'should_continue' function decides the next step
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # Route to tools if the LLM requested them
            "respond": "respond",  # Route to final response if the LLM is done
        },
    )

    compiled_graph = graph.compile()
    logger.info("Agent graph compiled successfully for async execution")

    return compiled_graph


# Compile the graph once at module load to save CPU cycles on every request
agent_graph = build_agent_graph()
