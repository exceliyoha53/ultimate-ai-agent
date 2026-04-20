import logging
import sys
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

from app.agent.state import AgentState
from app.memory.redis_memory import save_message, get_history, get_all_memories
from app.tools.search_tools import search_web, get_weather, get_news
from app.tools.email_tools import send_email
from app.tools.memory_tools import remember_fact, recall_fact
from app.voice.tts import text_to_speech


load_dotenv()
logger = logging.getLogger(__name__)

# --- JOB TOOLS CHECK ---


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from app.tools.job_tools import (
        search_jobs as _search_jobs,
        get_latest_jobs as _get_latest_jobs,
        get_jobs_by_keyword as _get_jobs_by_keyword,
    )

    JOBS_AVAILABLE = True

except Exception:
    JOBS_AVAILABLE = False
    logger.warning("Job tools not available - DATABASE_URL may not be set")


# -- ASYNC LANGCHAIN TOOLS -------------------------
@tool
async def search_web_tool(query: str, num_results: int = 5) -> list:
    """
    Search the web for current information, news, or general knowledge.
    Use for questions about recent events, facts not in local data,
    or any topic requiring up-to-date information.
    """
    return await search_web(query, num_results)


@tool
async def get_weather_tool(city: str) -> dict:
    """
    Get current weather conditions for any city.
    Use when user asks about weather or temperature in a location.
    """
    return await get_weather(city)


@tool
async def get_news_tool(topic: str, num_articles: int = 5) -> list:
    """
    Get latest news articles on any topic.
    Use when user asks about current events, Nigerian news, tech news, etc.
    """
    return await get_news(topic, num_articles)


@tool
async def send_email_tool(to: str, subject: str, body: str) -> dict:
    """
    Send an email on behalf of the user.
    Use when user explicitly asks to send an email or follow up with someone.
    Always confirm the recipient, subject, and body before sending.
    """
    return await send_email(to, subject, body)


@tool
async def remember_fact_tool(session_id: str, key: str, value: str) -> dict:
    """
    Remember a fact about the user for future conversations.
    Use proactively when user reveals their name, location, job preferences,
    salary expectations, or any personal detail worth remembering.
    """
    return await remember_fact(session_id, key, value)


@tool
async def recall_fact_tool(session_id: str, key: str) -> dict:
    """
    Recall a previously stored fact about the user.
    Use when you need to retrieve stored preferences or personal details.
    """
    return await recall_fact(session_id, key)


@tool
def search_jobs_tool(location: str, limit: int = 5) -> list:
    """
    Search Nigerian job listings by city or region.
    Use when user asks about jobs in a specific location like Lagos or Abuja.
    """
    if JOBS_AVAILABLE:
        return _search_jobs(location, limit)
    return [{"error": "Job database not available"}]


@tool
def get_latest_jobs_tool(limit: int = 5) -> list:
    """
    Get the most recently scraped Nigerian job listings.
    Use when user asks for latest jobs without specifying a location.
    """
    if JOBS_AVAILABLE:
        return _get_latest_jobs(limit)
    return [{"error": "Job database not available"}]


@tool
def get_jobs_by_keyword_tool(keyword: str, limit: int = 10) -> list:
    """
    Search Nigerian jobs by job title keyword.
    Use when user asks for a specific type of job like engineer, manager, or analyst.
    """
    if JOBS_AVAILABLE:
        return _get_jobs_by_keyword(keyword, limit)
    return [{"error": "Job database not available"}]


ALL_TOOLS = [
    search_web_tool,
    get_weather_tool,
    get_news_tool,
    send_email_tool,
    remember_fact_tool,
    recall_fact_tool,
    search_jobs_tool,
    get_latest_jobs_tool,
    get_jobs_by_keyword_tool,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,  # controls randomness or creativity of the models response
).bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """
You are an intelligent personal assistant with access to 
real-time tools. You can search the web, find Nigerian jobs, check weather, 
get news, send emails, and remember things about users across conversations.

When the user tells you personal information (their name, location, job 
preferences, etc.), use remember_fact_tool to store it proactively.

At the start of conversations, check memories to personalise your responses.

Be conversational, helpful, and context-aware. Format job listings clearly.
Always use tools to get real data rather than making things up.
For Nigerian context: salary ranges in Naira are relevant, acknowledge 
Nigerian cities, be aware of the local job market.
"""


# --- LANGRAPH NODES -----------------------------
async def load_context_node(state: AgentState) -> AgentState:
    """
    LangGraph node — runs first every turn.
    Loads conversation history and user memories from Redis.
    Builds the full message list for the LLM including system prompt
    and all past conversation turns.

    Parameters:
        state (AgentState): Current agent state

    Returns:
        AgentState: Updated with messages and memories populated
    """
    session_id = state["session_id"]

    memories = await get_all_memories(session_id)
    history = await get_history(session_id)

    # build memory context string
    memory_context = ""
    if memories:
        memory_lines = "\n".join(f"- {k}: {v}" for k, v in memories.items())
        memory_context = f"\n\nWhat I know about this user:\n{memory_lines}"

    messages = [SystemMessage(content=SYSTEM_PROMPT + memory_context)]
    for msg in history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(
        HumanMessage(content=state["user_message"])
    )  # users newest current input
    logger.info(f"Context loaded for session {session_id}")
    return {**state, "messages": messages, "memories": memories}


async def agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node — the LLM thinks and decides what to do.
    Sends the full message history to Gemini.
    Gemini either calls tools or gives a final response.

    Parameters:
        state (AgentState): Current state with messages populated

    Returns:
        AgentState: Updated with LLM response appended to messages
    """
    logger.info("Agent node: invoking LLM")
    response = await llm.ainvoke(state["messages"])
    return {**state, "messages": state["messages"] + [response]}


async def tools_node(state: AgentState) -> AgentState:
    """
    LangGraph node — executes any tool calls the LLM requested.
    Finds tool_call blocks in the last LLM message, executes each tool,
    and appends the results as ToolMessages for the LLM to read.

    Parameters:
        state (AgentState): Current state after agent_node ran

    Returns:
        AgentState: Updated with tool results appended to messages
    """
    last_message = state["messages"][-1]
    tool_messages = []
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if "session_id" in tool_args:
            tool_args["session_id"] = state["session_id"]

        logger.info(f"Executing tool: {tool_name}")

        if tool_name in TOOL_MAP:
            result = await TOOL_MAP[tool_name].ainvoke(tool_args)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        tool_results.append({"tool": tool_name, "result": result})
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {
        **state,
        "messages": state["messages"] + tool_messages,
        "tool_results": state.get("tool_results", []) + tool_results,
    }


async def response_node(state: AgentState) -> AgentState:
    """
    LangGraph node — extracts the final text response and saves to Redis.
    After all tool calls complete, Gemini gives a final text response.
    This node extracts it, saves the exchange to history, and optionally
    generates audio via text-to-speech.

    Parameters:
        state (AgentState): Final state after all tool calls

    Returns:
        AgentState: Updated with final_response and optional audio_path
    """
    # find last AI msg with actual context
    final_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content
            break

    # save to redis history
    await save_message(state["session_id"], "user", state["user_message"])
    await save_message(state["session_id"], "assistant", final_text)

    audio_path = None
    if state.get("should_speak", False):
        audio_path = await text_to_speech(final_text)

    logger.info(f"Response generated for session {state['session_id']}")
    return {**state, "final_response": final_text, "audio_path": audio_path}


def should_continue(state: AgentState) -> str:
    """
    LangGraph conditional edge — decides whether to call tools or finish.
    If the last LLM message contains tool calls → go to tools_node.
    If no tool calls → go to respond_node (LLM has a final answer).

    Parameters:
        state (AgentState): Current state

    Returns:
        str: 'tools' or 'respond' — names of next nodes
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "respond"
