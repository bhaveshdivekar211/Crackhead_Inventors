"""Audio stream handling for microphone input and speaker output."""

import asyncio
import logging
import pyaudio
from google.genai import types
from config import (
    AUDIO_FORMAT,
    AUDIO_CHANNELS,
    AUDIO_RATE,
    AUDIO_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


def initialize_audio():
    p = pyaudio.PyAudio()
    
    try:
        audio_stream = p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE
        )
        logger.info("Input audio stream opened")
        return p, audio_stream
    except Exception as e:
        logger.error(f"Failed to open audio input stream: {e}")
        p.terminate()
        raise


def cleanup_audio(p, audio_stream):
    try:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
            logger.info("Input stream closed")
    except Exception as e:
        logger.error(f"Error closing input stream: {e}")
    
    try:
        p.terminate()
        logger.info("PyAudio terminated")
    except Exception as e:
        logger.error(f"Error terminating PyAudio: {e}")


async def send_audio(session, audio_stream):
    logger.info("Microphone is live. Start speaking...")
    
    try:
        while True:
            try:
                data = audio_stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                if not data:
                    await asyncio.sleep(0.01)
                    continue
                
                # Send raw audio to Gemini Live
                await session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                await asyncio.sleep(0.1)
                
    except asyncio.CancelledError:
        logger.info("Audio send task cancelled")



