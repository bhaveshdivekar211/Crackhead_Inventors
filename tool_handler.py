"""Tool call handling for Gemini Live agent."""

import logging
import json
from google.genai import types

logger = logging.getLogger(__name__)


async def handle_tool_call(session, call, vector_store):
    if call.name == "search_docs":
        query = call.args.get('query', '')
        print(f"\n [Tool Call] Searching documents for: '{query}'...")
        logger.info(f"Tool call: search_docs('{query}')")
        
        try:
            from vector_store import search_docs
            import asyncio
            
            # Execute search
            logger.info("Running embedding in background thread...")
            loop = asyncio.get_event_loop()
            tool_result = await loop.run_in_executor(None, search_docs, vector_store, query)
            logger.info("Embedding complete, processing response...")
            logger.info(f"Tool result type: {type(tool_result)}")
            logger.info(f"Tool result length: {len(tool_result)}")
            logger.info(f"Tool result preview: {tool_result[:200]}")

            # Create the response dict
            response_dict = {"result": tool_result}
            logger.info(f"Response dict type: {type(response_dict)}")
            logger.info(f"Response dict: {response_dict}")
            

            # Send result back to Gemini
            try:
                func_response = types.FunctionResponse(
                    name=call.name,
                    id=call.id,
                    response=response_dict
                )
                logger.info(f"FunctionResponse created successfully: {func_response}")
                
                await session.send_tool_response(
                    function_responses=[func_response]
                )
                logger.info(f"Tool response sent for: {call.name}")
                return True
            except Exception as inner_e:
                logger.error(f"Failed to create/send FunctionResponse: {inner_e}")
                logger.error(f"FunctionResponse type: {type(types.FunctionResponse)}")
                raise
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")


            # Send error response back
            try:
                await session.send_tool_response(
                    function_responses=[
                        types.FunctionResponse(
                            name=call.name,
                            id=call.id,
                            response={"error": str(e)}
                        )
                    ]
                )
            except Exception as send_err:
                logger.error(f"Failed to send error response: {send_err}")
                return False
    else:
        logger.warning(f"Unknown tool called: {call.name}")
        return False


def get_tool_definitions():
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_docs",
                    description="Search the ingested documents and return the most relevant excerpts.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(
                                type=types.Type.STRING,
                                description="The search query terms extracted from user conversation."
                            )
                        },
                        required=["query"]
                    )
                )
            ]
        )
    ]
