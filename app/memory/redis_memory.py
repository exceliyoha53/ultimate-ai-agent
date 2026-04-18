import redis
import json
import os
import logging
from datetime import datetime, UTC
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


def save_message(session_id: str, role: str, content: str) -> None:
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
    # append to list
    redis_client.rpush(key, json.dumps(message))  # right push

    # trim to last MAX_HISTORY messages
    redis_client.ltrim(key, -MAX_HISTORY, -1)  # list trim

    # reset TTL on every transaction
    redis_client.expire(key, MEMORY_TTL)
    logger.debug(f"Saved {role} message to session {session_id}")


def get_history(session_id: str) -> list[dict]:
    """
    Retrieves full conversation history for a session.

    Parameters:
        session_id (str): Unique conversation identifier

    Returns:
        list[dict]: List of messages with role, content, timestamp
    """
    key = f"history:{session_id}"
    raw = redis_client.lrange(key, 0, -1)
    return [json.loads(m) for m in raw]


def save_memory(session_id: str, key: str, value: str) -> None:
    """
    Saves a specific memory fact about the user.
    Examples: preferred location, job type, name, salary expectations.
    These persist longer than conversation history.

    Parameters:
        session_id (str): Unique conversation identifier
        key (str): Memory key e.g. 'preferred_location', 'name'
        value (str): Memory value e.g. 'Lagos', 'Excel'
    """
    memory_key = f"memory:{session_id}:{key}"
    redis_client.set(memory_key, value, ex=MEMORY_TTL)
    logger.info(f"Saved memory [{key}={value}] for session {session_id}")


def get_memory(session_id: str, key: str) -> str | None:
    """
    Retrieves a specific memory fact.

    Parameters:
        session_id (str): Unique conversation identifier
        key (str): Memory key to retrieve

    Returns:
        str | None: The stored value or None if not found
    """
    memory_key = f"memory:{session_id}:{key}"
    return redis_client.get(memory_key)


def get_all_memories(session_id: str) -> dict:
    """
    Retrieves all stored memory facts for a session.
    Used to build context for the agent at the start of each conversation.

    Parameters:
        session_id (str): Unique conversation identifier

    Returns:
        dict: All memory key-value pairs for this session
    """
    pattern = f"memory:{session_id}:*"
    keys = redis_client.keys(pattern)
    memories = {}
    for k in keys:
        # extract just the memory key name from the full Redis key
        memory_key = k.split(":")[-1]
        memories[memory_key] = redis_client.get(k)
    return memories


def clear_session(session_id: str) -> None:
    """
    Clears all history and memory for a session.
    Called when user wants to start fresh.

     Parameters:
        session_id (str): Session to clear
    """
    # delete history
    redis_client.delete(f"history:{session_id}")
    # delete all memory keys for this session
    pattern = f"memory:{session_id}:*"
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)

    logger.info(f"Cleared session {session_id}")
