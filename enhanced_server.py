# enhanced_server.py
import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any

# Add aiohttp for HTML fetching in the MockBrowserTool
import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import the base agent and tools
from app.agent.manus import Manus
from app.tool.tool_collection import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.google_search import GoogleSearch
from app.tool.file_saver import FileSaver
from app.tool.bash import Bash
from app.tool.planning import PlanningTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate
from app.tool.base import BaseTool, ToolResult
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


# Create a mock browser tool that doesn't require actual browser installation
class MockBrowserTool(BaseTool):
    name: str = "browser_use"
    description: str = "Interact with a web browser (server environment simulation)"
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate", "click", "input_text", "screenshot", "get_html",
                    "execute_js", "scroll", "switch_tab", "new_tab", "close_tab", "refresh"
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'navigate' or 'new_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click' or 'input_text' actions",
            },
            "text": {"type": "string", "description": "Text for 'input_text' action"},
            "script": {
                "type": "string",
                "description": "JavaScript code for 'execute_js' action",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll for 'scroll' action",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
        },
        "required": ["action"],
    }

    # Track current browser state
    current_url: Optional[str] = None
    current_html: Optional[str] = None
    tabs: Dict[int, Dict[str, str]] = {}
    active_tab: int = 0

    async def fetch_html(self, url: str) -> str:
        """Fetch real HTML content from a URL."""
        import aiohttp
        from urllib.parse import urlparse

        # If URL doesn't have a scheme, add http://
        if not urlparse(url).scheme:
            url = "http://" + url

        try:
            # Try to fetch real content
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        logger.info(f"Successfully fetched HTML from {url} ({len(html_content)} bytes)")
                        return html_content
                    else:
                        logger.warning(f"Failed to fetch HTML from {url}, status: {response.status}")
                        return f"<html><body><p>Error: HTTP {response.status}</p></body></html>"
        except Exception as e:
            logger.error(f"Error fetching HTML from {url}: {str(e)}")
            return f"<html><body><p>Error fetching content: {str(e)}</p></body></html>"

    async def navigate_to(self, url: str) -> str:
        """Navigate to a URL and fetch its HTML content."""
        self.current_url = url
        self.current_html = await self.fetch_html(url)

        # Store in current tab
        self.tabs[self.active_tab] = {
            "url": url,
            "html": self.current_html
        }

        return f"Navigated to {url}"

    async def execute(self, action: str, **kwargs) -> Any:
        """Simulate browser actions, fetching real HTML when possible."""
        try:
            if action == "navigate":
                url = kwargs.get('url')
                if not url:
                    return ToolResult(error="URL parameter is required for navigate action")

                result = await self.navigate_to(url)
                return ToolResult(output=result)

            elif action == "get_html":
                # Return current HTML if available, otherwise return message
                if self.current_html:
                    # Truncate if too long to prevent excessive responses
                    html = self.current_html
                    if len(html) > 15000:
                        html = html[:15000] + "\n... [content truncated due to length] ..."
                    return ToolResult(output=html)
                else:
                    return ToolResult(
                        output="<html><body><p>No page has been loaded yet. Use 'navigate' first.</p></body></html>")

            elif action == "refresh":
                if self.current_url:
                    result = await self.navigate_to(self.current_url)
                    return ToolResult(output=f"Page refreshed: {result}")
                else:
                    return ToolResult(output="No page to refresh. Navigate to a URL first.")

            elif action == "new_tab":
                url = kwargs.get('url')
                if not url:
                    return ToolResult(error="URL parameter is required for new_tab action")

                new_tab_id = max(self.tabs.keys() or [-1]) + 1
                self.active_tab = new_tab_id
                result = await self.navigate_to(url)
                return ToolResult(output=f"Opened new tab (ID: {new_tab_id}) and {result}")

            elif action == "switch_tab":
                tab_id = kwargs.get('tab_id')
                if tab_id is None:
                    return ToolResult(error="tab_id parameter is required for switch_tab action")

                if tab_id not in self.tabs:
                    return ToolResult(error=f"Tab with ID {tab_id} does not exist")

                self.active_tab = tab_id
                self.current_url = self.tabs[tab_id].get("url")
                self.current_html = self.tabs[tab_id].get("html")
                return ToolResult(output=f"Switched to tab {tab_id} ({self.current_url})")

            elif action == "close_tab":
                if not self.tabs:
                    return ToolResult(output="No tabs to close")

                if self.active_tab in self.tabs:
                    del self.tabs[self.active_tab]

                if self.tabs:
                    # Switch to another tab
                    self.active_tab = next(iter(self.tabs.keys()))
                    self.current_url = self.tabs[self.active_tab].get("url")
                    self.current_html = self.tabs[self.active_tab].get("html")
                    return ToolResult(output=f"Closed tab and switched to tab {self.active_tab}")
                else:
                    self.current_url = None
                    self.current_html = None
                    return ToolResult(output="Closed the last tab")

            # Handle other actions with mock responses
            actions = {
                "click": f"[MOCK] Clicked element at index {kwargs.get('index', 'unknown')}",
                "input_text": f"[MOCK] Input text '{kwargs.get('text', '')}' at index {kwargs.get('index', 'unknown')}",
                "screenshot": "[MOCK] Screenshot captured (simulated in server environment)",
                "execute_js": f"[MOCK] Executed JavaScript: {kwargs.get('script', 'unknown script')}",
                "scroll": f"[MOCK] Scrolled {kwargs.get('scroll_amount', 0)} pixels",
            }

            result = actions.get(action, f"[MOCK] Unknown browser action: {action}")
            logger.info(f"MockBrowserTool executing: {action}, result summary: {result[:50]}...")

            return ToolResult(output=result)

        except Exception as e:
            logger.error(f"Error in MockBrowserTool.execute({action}): {str(e)}")
            return ToolResult(error=f"Browser action failed: {str(e)}")


class EnhancedManus(Manus):
    client_id: Optional[str] = Field(default=None)
    update_callback: Optional[Callable] = Field(default=None)

    # Override the available_tools to exclude the real BrowserUseTool and add our MockBrowserTool
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            GoogleSearch(),
            FileSaver(),
            Bash(),
            PlanningTool(),
            StrReplaceEditor(),
            #MockBrowserTool(),  Use the mock instead of the real browser tool
            Terminate()
        )
    )

    async def run(self, request: Optional[str] = None) -> str:
        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "starting", "message": f"Starting to process: {request}"}
            )
        return await super().run(request)

    async def step(self) -> str:
        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "step", "message": f"Executing step {self.current_step}/{self.max_steps}"}
            )
        return await super().step()

    async def think(self) -> bool:
        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "thinking", "message": "Thinking..."}
            )
        return await super().think()

    async def execute_tool(self, command) -> str:
        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "tool", "message": f"Using tool: {command.function.name}"}
            )
        result = await super().execute_tool(command)
        if self.update_callback:
            result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            await self.update_callback(
                self.client_id,
                {"status": "tool_result", "message": f"Tool result: {result_summary}"}
            )
        return result


# Connection manager to handle multiple clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_instances: Dict[str, EnhancedManus] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # Create agent instance with the modified tools
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


# Serve static client HTML file if available
@app.get("/")
async def serve_client():
    try:
        return FileResponse("client/index.html")
    except:
        return {"message": "Welcome to Enhanced OpenManus API. Please use the standalone client to connect."}


# Try to mount static files for the web client
try:
    app.mount("/client", StaticFiles(directory="client"), name="client")
except:
    logger.warning("Client directory not found. Web client will not be available.")


@app.get("/health")
async def health_check():
    return {"status": "OK", "version": "0.1.0", "mode": "Enhanced"}


if __name__ == "__main__":
    port = 2000
    logger.info(f"Starting Enhanced OpenManus server on port {port}")
    uvicorn.run("enhanced_server:app", host="127.0.0.1", port=port, log_level="info", reload=True)