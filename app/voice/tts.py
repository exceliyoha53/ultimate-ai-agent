import os
import re
import logging
import uuid
import edge_tts
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

AUDIO_DIR = "generated_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def clean_text_for_tts(text: str) -> str:
    """Removes markdown artifacts so the TTS doesn't read them out."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = text.replace('* ', ' ')
    return text


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
        clean_text = clean_text_for_tts(text)
        # truncate very long responses for audio — nobody wants to hear 500 words
        if len(clean_text) > 500:
            text = text[:500] + "... See the full response in text above."

        filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(filepath)

        logger.info(f"Audio generated: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None
