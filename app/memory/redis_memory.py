import json
import os
import logging
from datetime import datetime, UTC
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# connect to Redis — stores all conversation memory
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,  # returns strings not bytes
)

MEMORY_TTL = 60 * 60 * 24 * 30  # 30 days — memories persist for a month
MAX_HISTORY = 20  # keep last 20 messages per session


async def save_message(session_id: str, role: str, content: str) -> None:
    """
    Saves a single message to conversation history in Redis.
    History is stored as a JSON list and capped at MAX_HISTORY messages.

    Parameters:
        session_id (str): Unique conversation identifier
        role (str): 'user' or 'assistant'
        content (str): The message text
    """
    key = f"history:{session_id}"
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(key, json.dumps(message))
        pipe.ltrim(key, -MAX_HISTORY, -1)
        pipe.expire(key, MEMORY_TTL)
        await pipe.execute()

    logger.debug(f"Saved {role} message to session {session_id}")


async def get_history(session_id: str) -> list[dict]:
    """
    Retrieves full conversation history for a session.

    Parameters:
        session_id (str): Unique conversation identifier

    Returns:
        list[dict]: List of messages with role, content, timestamp
    """
    key = f"history:{session_id}"
    raw = await redis_client.lrange(key, 0, -1)
    return [json.loads(m) for m in raw]


async def save_memory(session_id: str, key: str, value: str) -> None:
    """
    Saves a specific memory fact about the user.
    Examples: preferred location, job type, name, salary expectations.
    These persist longer than conversation history.

    Parameters:
        session_id (str): Unique conversation identifier
        key (str): Memory key e.g. 'preferred_location', 'name'
        value (str): Memory value e.g. 'Lagos', 'Excel'
    """
    hash_key = f"memory:{session_id}"
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.hset(hash_key, key, value)
        pipe.expire(hash_key, MEMORY_TTL)
        await pipe.execute()

    logger.info(f"Saved memory [{key}={value}] for session {session_id}")


async def get_memory(session_id: str, key: str) -> str | None:
    """
    Retrieves a specific memory fact.

    Parameters:
        session_id (str): Unique conversation identifier
        key (str): Memory key to retrieve

    Returns:
        str | None: The stored value or None if not found
    """
    hash_key = f"memory:{session_id}"
    return await redis_client.hget(hash_key, key)


async def get_all_memories(session_id: str) -> dict:
    """
    Retrieves all stored memory facts for a session.
    Used to build context for the agent at the start of each conversation.

    Parameters:
        session_id (str): Unique conversation identifier

    Returns:
        dict: All memory key-value pairs for this session
    """
    hash_key = f"memory:{session_id}"
    return await redis_client.hgetall(hash_key)


async def clear_session(session_id: str) -> None:
    """
    Clears all history and memory for a session.
    Called when user wants to start fresh.

     Parameters:
        session_id (str): Session to clear
    """
    history_key = f"history:{session_id}"
    memory_key = f"memory:{session_id}"

    await redis_client.delete(history_key, memory_key)

    logger.info(f"Cleared session {session_id}")
