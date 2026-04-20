import logging
import os
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.memory.redis_memory import clear_session, get_history, get_all_memories

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


class ChatRequest(BaseModel):
    """
    Parameters:
        message (str): User's message
        session_id (str): Conversation session identifier
        voice (bool): Whether to generate audio response
    """

    message: str
    session_id: str = "default"
    voice: bool = False


class ChatResponse(BaseModel):
    """
    Parameters:
        response (str): Agent's text response
        session_id (str): Echo of session ID
        audio_url (str | None): URL to download audio if voice=True
        tools_used (list): Names of tools called this turn
    """

    response: str
    session_id: str
    audio_url: str | None = None
    tools_used: list = []


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main chat interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """
    Main agent endpoint. Runs the full LangGraph agent pipeline.
    Supports voice responses via voice=True in the request.

    Parameters:
        request (ChatRequest): Message, session ID, and voice flag
        background_tasks: FastAPI background tasks for audio cleanup

    Returns:
        ChatResponse: Agent response with optional audio URL
    """
    try:
        # build initial state
        initial_state: AgentState = {
            "messages": [],
            "session_id": request.session_id,
            "user_message": request.message,
            "memories": {},
            "tool_results": [],
            "final_response": "",
            "audio_path": None,
            "should_speak": request.voice,
        }

        # run the agent graph
        final_state = await agent_graph.ainvoke(initial_state)

        # build response
        audio_url = None
        if final_state.get("audio_path"):
            audio_url = f"/audio/{os.path.basename(final_state['audio_path'])}"
            # schedule audio file deletion after 60 seconds
            background_tasks.add_task(
                _delete_file_after_delay, final_state["audio_path"]
            )

        tools_used = [t["tool"] for t in final_state.get("tool_results", [])]

        return ChatResponse(
            response=final_state["final_response"],
            session_id=request.session_id,
            audio_url=audio_url,
            tools_used=tools_used,
        )

    except Exception as e:
        logger.error(f"Chat error for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serves a generated audio file for download.
    Deletes the file after serving to keep disk clean.

    Parameters:
        filename (str): Audio filename from the chat response audio_url
    """
    filepath = os.path.join("generated_audio", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(path=filepath, media_type="audio/mpeg", filename=filename)


@router.delete("/chat/{session_id}")
async def reset_chat(session_id: str) -> dict:
    """
    Clears all history and memory for a session.

    Parameters:
        session_id (str): Session to clear
    """
    await clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/memory/{session_id}")
async def get_memories(session_id: str) -> dict:
    """
    Returns all stored memories and recent history for a session.
    Useful for debugging and showing users what the agent remembers.

    Parameters:
        session_id (str): Session to inspect
    """
    memories = await get_all_memories(session_id)
    history = await get_history(session_id)

    return {
        "session_id": session_id,
        "memories": memories,
        "recent_history": history[-5:],
    }


async def _delete_file_after_delay(filepath: str) -> None:
    """Background task: deletes after 60s."""
    await asyncio.sleep(60)
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"Cleaned up audio file: {filepath}")
