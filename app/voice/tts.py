import os
import logging
import uuid
import edge_tts
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

AUDIO_DIR = "generated_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


async def text_to_speech(text: str, voice: str = "en-GB-SoniaNeural") -> str | None:
    """
    Converts text to speech using Edge TTS.
    Saves the audio as an MP3 file and returns the file path.
    The calling router serves this file and deletes it after sending.

    Parameters:
        text (str): The text to convert to speech
        voice (str): Neural voice code. 'en-GB-SoniaNeural' is a good default.

    Returns:
        str | None: File path to the generated MP3 file, or None if failed.
    """
    try:
        # truncate very long responses for audio — nobody wants to hear 500 words
        if len(text) > 500:
            text = text[:500] + "... See the full response in text above."

        filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filepath)

        logger.info(f"Audio generated: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None
