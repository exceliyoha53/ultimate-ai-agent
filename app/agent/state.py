from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    """
    The state object that flows through every node in the LangGraph agent.
    LangGraph passes this between nodes — each node reads from it and updates it.
    The Annotated[list, operator.add] on messages means new messages
    are appended to the list rather than replacing it.

    Fields:
        messages: Full conversation history for this turn
        session_id: Identifies the user's session for memory retrieval
        user_message: The current user input
        memories: User's stored preferences from Redis
        tool_results: Results from tool calls this turn
        final_response: The agent's final text response
        audio_path: Path to generated audio file if voice was requested
        should_speak: Whether to generate voice response
    """

    messages: Annotated[list, operator.add]
    session_id: str
    user_message: str
    memories: dict
    tool_results: list
    final_response: str
    audio_path: str | None
    should_speak: bool
