# enhanced_server.py
import asyncio
import json
import logging
from typing import Dict, List
import functools

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import the base agent and tools
from app.agent.manus import Manus
from app.tools.collection import ToolCollection
from app.tools.python_execute import PythonExecute
from app.tools.google_search import GoogleSearch
from app.tools.file_saver import FileSaver
from app.tools.bash import Bash
from app.tools.planning_tool import PlanningTool
from app.tools.str_replace_editor import StrReplaceEditor
from app.tools.terminate import Terminate
from app.logger import logger
from app.agent.base import BaseAgent
from app.agent.toolcall import ToolCallAgent

# Configure logging for development
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="OpenManus Enhanced Server")

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom log handler to capture logs for specific clients
class WSLogHandler(logging.Handler):
    def __init__(self, client_id, send_update_func):
        super().__init__()
        self.client_id = client_id
        self.send_update_func = send_update_func
        self.setLevel(logging.INFO)

    def emit(self, record):
        log_entry = self.format(record)
        if hasattr(record, 'client_id') and record.client_id == self.client_id:
            asyncio.create_task(self.send_update_func(
                self.client_id,
                {"status": "step", "message": log_entry}
            ))


# Custom Manus agent without browser tool and with step reporting
class EnhancedManus(Manus):
    def __init__(self, client_id=None, update_callback=None, *args, **kwargs):
        # Create a modified tool collection without the browser tool
        tools = [
            PythonExecute(),
            GoogleSearch(),
            FileSaver(),
            Bash(),
            PlanningTool(),
            StrReplaceEditor(),
            Terminate()
        ]

        self.client_id = client_id
        self.update_callback = update_callback

        # Call the parent constructor with our custom tools
        super().__init__(*args, tools=ToolCollection(tools), **kwargs)

        # Patch the run method to report steps
        original_run = self.run

        @functools.wraps(original_run)
        async def run_with_updates(request):
            if self.update_callback:
                await self.update_callback(
                    self.client_id,
                    {"status": "starting", "message": f"Starting to process: {request}"}
                )
            result = await original_run(request)
            return result

        self.run = run_with_updates

        # Patch the agent's step execution
        original_step = BaseAgent._execute_step

        @functools.wraps(original_step)
        async def step_with_updates(agent_self, step_num, max_steps):
            if self.update_callback:
                await self.update_callback(
                    self.client_id,
                    {"status": "step", "message": f"Executing step {step_num}/{max_steps}"}
                )
            return await original_step(agent_self, step_num, max_steps)

        BaseAgent._execute_step = step_with_updates

        # Patch the think method to report thoughts
        original_think = ToolCallAgent.think

        @functools.wraps(original_think)
        async def think_with_updates(agent_self):
            result = await original_think(agent_self)
            if self.update_callback and hasattr(agent_self, 'thoughts') and agent_self.thoughts:
                thought_summary = agent_self.thoughts[:150] + "..." if len(
                    agent_self.thoughts) > 150 else agent_self.thoughts
                await self.update_callback(
                    self.client_id,
                    {"status": "thinking", "message": f"Thinking: {thought_summary}"}
                )
            return result

        ToolCallAgent.think = think_with_updates

        # Patch the tool execution method to report tool usage
        original_execute = ToolCallAgent.execute_tool

        @functools.wraps(original_execute)
        async def execute_with_updates(agent_self, tool_name, tool_input):
            if self.update_callback:
                await self.update_callback(
                    self.client_id,
                    {"status": "tool", "message": f"Using tool: {tool_name}"}
                )
            result = await original_execute(agent_self, tool_name, tool_input)
            if self.update_callback:
                result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                await self.update_callback(
                    self.client_id,
                    {"status": "tool_result", "message": f"Tool {tool_name} result: {result_summary}"}
                )
            return result

        ToolCallAgent.execute_tool = execute_with_updates


# Connection manager to handle multiple clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_instances: Dict[str, EnhancedManus] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.agent_instances[client_id] = EnhancedManus(
            client_id=client_id,
            update_callback=self.send_message
        )
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agent_instances:
            del self.agent_instances[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message):
        if client_id in self.active_connections:
            if isinstance(message, str):
                await self.active_connections[client_id].send_text(message)
            else:
                await self.active_connections[client_id].send_text(json.dumps(message))

    async def process_request(self, request: str, client_id: str):
        if client_id not in self.agent_instances:
            await self.send_message(client_id, {"status": "error", "error": "No agent instance found"})
            return

        agent = self.agent_instances[client_id]

        # Store original messages to track new ones
        original_messages = agent.memory.messages.copy()

        try:
            logger.info(f"Processing request from client {client_id}: {request}")

            # Send initial processing message
            await self.send_message(
                client_id,
                {"status": "processing", "message": "Starting to process your request..."}
            )

            # Process request with the agent
            result = await agent.run(request)

            # Get new messages generated during processing
            new_messages = agent.memory.messages[len(original_messages):]

            # Send the full result to the client
            await self.send_message(
                client_id,
                {
                    "status": "complete",
                    "result": result,
                    "messages": [msg.to_dict() for msg in new_messages]
                }
            )

            logger.info(f"Request from client {client_id} processed successfully")
        except Exception as e:
            logger.error(f"Error processing request from client {client_id}: {e}")
            await self.send_message(
                client_id,
                {"status": "error", "error": str(e)}
            )


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request_data = json.loads(data)
                if "request" in request_data:
                    # Process request in background task
                    asyncio.create_task(
                        manager.process_request(request_data["request"], client_id)
                    )
            except json.JSONDecodeError:
                await manager.send_message(
                    client_id,
                    {"status": "error", "error": "Invalid JSON"}
                )
    except WebSocketDisconnect:
        manager.disconnect(client_id)


class RequestModel(BaseModel):
    request: str


@app.post("/api/request")
async def handle_request(request_data: RequestModel):
    """HTTP endpoint for non-WebSocket clients"""
    client_id = f"http_{id(request_data)}"

    # Create a temporary agent instance
    agent = EnhancedManus()

    try:
        # Process request
        result = await agent.run(request_data.request)

        # Return result
        return {
            "status": "complete",
            "result": result,
            "messages": [msg.to_dict() for msg in agent.memory.messages]
        }
    except Exception as e:
        logger.error(f"Error processing HTTP request: {e}")
        return {"status": "error", "error": str(e)}


# Serve the client HTML file
@app.get("/")
async def serve_client():
    try:
        return FileResponse("client/index.html")
    except:
        return {"message": "Welcome to Enhanced OpenManus API. Please use the standalone client to connect."}


# Mount static files for the web client
try:
    app.mount("/client", StaticFiles(directory="client"), name="client")
except:
    logger.warning("Client directory not found. Web client will not be available.")


@app.get("/health")
async def health_check():
    return {"status": "OK", "version": "0.1.0", "mode": "Enhanced"}


if __name__ == "__main__":
    port = 3000
    logger.info(f"Starting Enhanced OpenManus server on port {port}")
    uvicorn.run("enhanced_server:app", host="127.0.0.1", port=port, log_level="info", reload=True)