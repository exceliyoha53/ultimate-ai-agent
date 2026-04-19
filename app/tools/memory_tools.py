import logging
from app.memory.redis_memory import save_memory, get_memory

logger = logging.getLogger(__name__)


async def remember_fact(session_id: str, key: str, value: str) -> dict:
    """
    Stores a fact about the user for future conversations.
    Use this proactively when the user reveals preferences, their name,
    location, job interests, or any personal detail worth remembering.
    Examples: remember_fact(session_id, 'name', 'Excel')
              remember_fact(session_id, 'preferred_location', 'Lagos')
              remember_fact(session_id, 'job_interest', 'software engineering')

    Parameters:
        session_id (str): The current session ID
        key (str): What to remember e.g. name, location, job_interest
        value (str): The value to remember

    Returns:
        dict: Confirmation of what was stored
    """
    await save_memory(session_id, key, value)

    logger.info(f"Remembered [{key}={value}] for session {session_id}")

    return {
        "stored": True,
        "key": key,
        "value": value,
        "message": f"I'll remember that your {key} is {value}",
    }


async def recall_fact(session_id: str, key: str) -> dict:
    """
    Retrieves a specific stored fact about the user.

    Parameters:
        session_id (str): The current session ID
        key (str): The fact to retrieve

    Returns:
        dict: The stored value or a not-found message
    """
    value = await get_memory(session_id, key)
    if value:
        return {"found": True, "key": key, "value": value}
    return {"found": False, "key": key, "message": f"No memory stored for {key}"}
