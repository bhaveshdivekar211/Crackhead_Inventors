"""Main orchestration for Gemini Live voice chatbot with document retrieval."""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Modality

from config import (
    GEMINI_MODEL,
    SYSTEM_INSTRUCTION,
    LOG_LEVEL,
    LOG_FORMAT,
    AUDIO_FORMAT,
    AUDIO_CHANNELS,
    AUDIO_RATE,
    AUDIO_CHUNK_SIZE,
)
from vector_store import load_vector_store
from audio_handler import (
    initialize_audio,
    cleanup_audio,
    send_audio,
)
from tool_handler import (
    handle_tool_call,
    get_tool_definitions,
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


def validate_environment():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        sys.exit(1)
    logger.info("GEMINI_API_KEY found")


async def receive_responses(session, p, vector_store):
    output_stream = None
    
    try:
        # Open output stream for model's audio responses
        output_stream = p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            output=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE
        )
        logger.info("Output stream opened")
        
        # THIS IS THE ONLY RECV LOOP - no nested receives!
        async for response in session.receive():
            try:
                # Check if response has text directly
                if response.text:
                    logger.info(f"FOUND TEXT AT TOP LEVEL: {response.text[:200]}")
                    print(response.text, end="", flush=True)
                
                server_content = response.server_content
                
                # 1. Handle audio/text responses from model
                if server_content is not None:
                    logger.info(f"server_content exists, checking model_turn...")
                    model_turn = server_content.model_turn
                    logger.info(f"model_turn is None: {model_turn is None}")
                    if model_turn is not None:
                        logger.info(f"model_turn has {len(model_turn.parts) if model_turn.parts else 0} parts")
                        for i, part in enumerate(model_turn.parts):
                            logger.info(f"  Part {i}: has_inline_data={part.inline_data is not None}, has_text={part.text is not None}")
                            # Play audio
                            if part.inline_data:
                                try:
                                    logger.info(f"    Writing audio data, size: {len(part.inline_data.data)} bytes")
                                    output_stream.write(part.inline_data.data)
                                except Exception as e:
                                    logger.error(f"Error writing audio to output: {e}")
                            
                            # Print text transcripts
                            if part.text:
                                logger.info(f"    Text content: {part.text[:100]}")
                                print(part.text, end="", flush=True)
                                logger.debug(f"Model response: {part.text}")
                            # Play audio
                            if part.inline_data:
                                try:
                                    output_stream.write(part.inline_data.data)
                                except Exception as e:
                                    logger.error(f"Error writing audio to output: {e}")
                            
                            # Print text transcripts
                            if part.text:
                                print(part.text, end="", flush=True)
                                logger.debug(f"Model response: {part.text}")
                    
                    if server_content.turn_complete:
                        print("\n[Turn Complete]")
                        logger.info("Turn completed by model")
                
                # 2. Handle tool/function calls (search_docs, etc.)
                if response.tool_call is not None:
                    for call in response.tool_call.function_calls:
                        await handle_tool_call(session, call, vector_store)
                        
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                continue
                
    except asyncio.CancelledError:
        logger.info("Response handling cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in response handling: {e}")
    finally:
        if output_stream is not None:
            try:
                output_stream.stop_stream()
                output_stream.close()
                logger.info("Output stream closed")
            except Exception as e:
                logger.error(f"Error closing output stream: {e}")

async def run_session(client, vector_store, p, audio_stream):
    # Configure session parameters
    config = types.LiveConnectConfig(
        response_modalities=[Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]
        ),
        tools=get_tool_definitions()
    )

    send_task = None
    receive_task = None
    
    try:
        # Establish WebSocket connection to Gemini Live
        async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
            logger.info("Connected to Gemini Live")
            print("Connected to Gemini Live. System operational.")
            
            # Run input and output streams concurrently
            send_task = asyncio.create_task(send_audio(session, audio_stream))
            receive_task = asyncio.create_task(
                receive_responses(session, p, vector_store)
            )
            
            # Wait for both tasks; cancel on first exception
            done, pending = await asyncio.wait(
                [send_task, receive_task],
                return_when=asyncio.FIRST_EXCEPTION
            )
            
            # Check for exceptions
            for task in done:
                if task.exception():
                    logger.error(f"Task failed: {task.exception()}")
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"Error during session: {e}")
    finally:
        # Cleanup tasks
        if send_task and not send_task.done():
            send_task.cancel()
        if receive_task and not receive_task.done():
            receive_task.cancel()


async def main():
    logger.info("Starting Gemini Live Voice Chatbot...")
    
    # Validate environment
    validate_environment()
    
    # Load vector store
    vector_store = load_vector_store()
    
    # Initialize GenAI client
    try:
        client = genai.Client()
        logger.info("Google GenAI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")
        sys.exit(1)
    
    # Initialize audio
    p, audio_stream = initialize_audio()
    
    try:
        await run_session(client, vector_store, p, audio_stream)
    finally:
        cleanup_audio(p, audio_stream)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSession ended by user.")
        logger.info("Program terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
