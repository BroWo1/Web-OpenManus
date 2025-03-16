# enhanced_server.py

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Callable, Any, Set
from app.agent.base import BaseAgent, AgentState

# Add aiohttp for HTML fetching in the MockBrowserTool
import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Try to import crawl4ai for markdown conversion
try:
    import crawl4ai

    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logging.warning("crawl4ai library not found. Will fall back to regular HTML fetching.")

# Import the base agent and tools
from app.agent.manus import Manus
from app.tool.tool_collection import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch  # Updated from GoogleSearch to WebSearch
from app.tool.file_saver import FileSaver
from app.tool.bash import Bash
from app.tool.planning import PlanningTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate
from app.tool.base import BaseTool, ToolResult
from app.logger import logger
from app.agent.base import BaseAgent
from app.agent.toolcall import ToolCallAgent
import asyncio
import sys
from typing import Dict, Optional, Callable, Set

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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

# Import workspace directory from app.config
from app.config import WORKSPACE_ROOT

# Directory for storing files that will be available for download
FILES_DIR = WORKSPACE_ROOT
os.makedirs(FILES_DIR, exist_ok=True)


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


# File information model
class FileInfo(BaseModel):
    file_id: str
    file_path: str
    file_name: str
    content_type: str = "application/octet-stream"
    description: str = ""
    size: int = 0

    def to_dict(self):
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "content_type": self.content_type,
            "description": self.description,
            "size": self.size,
            "download_url": f"/api/files/{self.file_id}"
        }


# Web content fetch tool that uses crawl4ai for extracting structured content
# Keeping the original WebContentTool as instructed
class WebContentTool(BaseTool):
    name: str = "browser_use"  # Keep same name for backward compatibility
    description: str = "Fetch and extract structured content from websites using crawl4ai, or read local files"
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "get_html"],
                "description": "The action to perform: 'navigate' to fetch content from a URL, 'get_html' to view current content",
            },
            "url": {
                "type": "string",
                "description": "URL for the 'navigate' action; can be a web URL or a local file URL (file://...)",
            },
        },
        "required": ["action"],
    }

    # Track current content
    current_url: Optional[str] = None
    current_html: Optional[str] = None
    current_markdown: Optional[str] = None

    async def fetch_content(self, url: str) -> tuple:
        """Fetch content from a URL and convert to both HTML and markdown using crawl4ai."""
        from urllib.parse import urlparse

        # If URL doesn't have a scheme, add http://
        if not urlparse(url).scheme:
            url = "http://" + url

        # Handle local file URLs
        if url.startswith("file://"):
            file_path = url[7:]
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    logger.info(f"Successfully read local file: {file_path}")
                    return html_content, f"# Content from local file: {file_path}\n\n{html_content}"
                else:
                    return f"<html><body><p>Error: File not found at {file_path}</p></body></html>", f"# Error\n\nFile not found at {file_path}"
            except Exception as e:
                return f"<html><body><p>Error: {str(e)}</p></body></html>", f"# Error\n\nCould not read file: {str(e)}"

        # Process web URLs
        if CRAWL4AI_AVAILABLE:
            try:
                logger.info(f"Using crawl4ai to fetch and convert content from {url}")

                # Import proper classes from crawl4ai
                from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
                from crawl4ai.content_filter_strategy import PruningContentFilter
                from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

                # Configure browser and crawler
                browser_config = BrowserConfig(
                    headless=True,
                    verbose=False,
                )

                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.ENABLED,
                    markdown_generator=DefaultMarkdownGenerator(
                        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed",
                                                            min_word_threshold=0)
                    ),
                )

                # Create crawler and fetch content
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(
                        url=url,
                        config=run_config
                    )

                    # Extract markdown from result
                    markdown_content = result.markdown.fit_markdown

                    # Create simple HTML representation
                    html_content = f"<html><body><pre>{markdown_content}</pre></body></html>"

                    logger.info(f"Successfully fetched markdown from {url} ({len(markdown_content)} bytes)")
                    return html_content, markdown_content

            except Exception as e:
                logger.warning(f"Error using crawl4ai for {url}: {str(e)}")
                error_msg = f"# Error using crawl4ai\n\nFailed to fetch and convert content from {url}: {str(e)}"
                return f"<html><body><p>{error_msg}</p></body></html>", error_msg
        else:
            error_msg = f"# Error: crawl4ai not available\n\nPlease install the crawl4ai library to enable web content fetching and extraction.\n\nUse: pip install crawl4ai"
            logger.error("crawl4ai library is not available")
            return f"<html><body><p>{error_msg}</p></body></html>", error_msg

    async def execute(self, action: str, **kwargs) -> Any:
        """Execute the requested action."""
        try:
            if action == "navigate":
                url = kwargs.get('url')
                if not url:
                    return ToolResult(error="URL parameter is required for navigate action")

                # Fetch content from the URL
                self.current_url = url
                self.current_html, self.current_markdown = await self.fetch_content(url)
                return ToolResult(output=f"Successfully fetched content from {url}")

            elif action == "get_html":
                # Return markdown content if available
                if not self.current_markdown:
                    return ToolResult(output="No content has been loaded yet. Use 'navigate' with a URL first.")

                content = self.current_markdown
                if len(content) > 15000:
                    content = content[:15000] + "\n... [content truncated due to length] ..."
                return ToolResult(output=content)

            else:
                return ToolResult(
                    error=f"Unsupported action: {action}. This tool only supports 'navigate' and 'get_html' actions.")

        except Exception as e:
            logger.error(f"Error in WebContentTool.execute({action}): {str(e)}")
            return ToolResult(error=f"Web content action failed: {str(e)}")

    # Add this method to WebContentTool class in enhanced_server.py
    async def cleanup(self):
        """Placeholder cleanup method for compatibility."""
        # This is a no-op method to prevent errors when called
        pass


class EnhancedFileSaver(FileSaver):
    # Properly declare file_tracker as a field with proper typing
    file_tracker: Optional[Any] = Field(default=None, description="Tracker for generated files")

    async def execute(self, content: str, file_path: str, mode: str = "w") -> str:
        """Save content to a file and track it for download."""
        # Use the file_path directly to save in the workspace directory
        # The parent FileSaver already saves to WORKSPACE_ROOT

        # Call the parent's execute to save the file in the workspace
        result = await super().execute(content, file_path, mode)

        # Get the actual file path from the result
        import re
        actual_path_match = re.search(r"saved to (.+)", result)
        if actual_path_match:
            absolute_file_path = actual_path_match.group(1)
            safe_filename = os.path.basename(absolute_file_path)
        else:
            # Fallback if regex doesn't match
            safe_filename = os.path.basename(file_path)
            absolute_file_path = os.path.join(WORKSPACE_ROOT, safe_filename)

        # If file was saved successfully and we have a tracker
        if "successfully saved" in result and self.file_tracker:
            # Generate a unique ID for this file
            file_id = str(uuid.uuid4())

            # Determine file size
            file_size = 0
            if os.path.exists(absolute_file_path):
                file_size = os.path.getsize(absolute_file_path)

            # Guess content type
            content_type = self._guess_content_type(absolute_file_path)

            # Create file info and add to tracker
            file_info = FileInfo(
                file_id=file_id,
                file_path=absolute_file_path,
                file_name=safe_filename,
                content_type=content_type,
                description=f"File created during agent interaction",
                size=file_size
            )

            self.file_tracker.add_file(file_info, client_id=getattr(self, 'client_id', None))

            # Update the result message to include download info
            result = f"{result} (File ID: {file_id}, available for download)"

        return result

    def _guess_content_type(self, file_path: str) -> str:
        """Guess the content type based on file extension."""
        import mimetypes
        # Ensure common file types are registered
        mimetypes.add_type('text/x-python', '.py')
        mimetypes.add_type('text/javascript', '.js')
        mimetypes.add_type('text/css', '.css')
        mimetypes.add_type('text/html', '.html')
        mimetypes.add_type('text/markdown', '.md')

        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or "application/octet-stream"


class EnhancedManus(Manus):
    client_id: Optional[str] = Field(default=None)
    update_callback: Optional[Callable] = Field(default=None)
    file_tracker: Any = Field(default=None)
    # Add a flag to track cancellation
    cancel_requested: bool = Field(default=False)

    # Override the available_tools to use enhanced versions
    available_tools: ToolCollection = Field(default_factory=lambda: None)  # Will be set in __init__

    def __init__(self, **data):
        # Call the parent class's __init__ first
        super().__init__(**data)

        # Initialize tools collection with the enhanced FileSaver
        if self.file_tracker is not None:
            # Create the enhanced file saver with the file_tracker
            enhanced_file_saver = EnhancedFileSaver(file_tracker=self.file_tracker)

            # Create a fresh tool collection with our enhanced tools
            self.available_tools = ToolCollection(
                PythonExecute(),
                WebSearch(),  # Updated from GoogleSearch to WebSearch
                enhanced_file_saver,  # Use the enhanced version
                Bash(),
                PlanningTool(),
                StrReplaceEditor(),
                WebContentTool(),  # Use the simplified web content tool
                Terminate()
            )

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cancellation support."""
        # Reset cancel flag at the start of processing
        self.cancel_requested = False

        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "starting", "message": f"Starting to process: {request}"}
            )
        return await super().run(request)

    async def step(self) -> str:
        """Execute a single step with cancellation check."""
        # Check if cancellation was requested
        if self.cancel_requested:
            logger.info(f"Cancellation requested for client {self.client_id}, stopping processing")
            self.state = AgentState.FINISHED  # Set state to finished to stop processing
            if self.update_callback:
                await self.update_callback(
                    self.client_id,
                    {"status": "cancelled", "message": "Processing cancelled by user"}
                )
            return "Processing cancelled by user"

        # Send step update if callback is available
        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "step", "message": f"Executing step {self.current_step}/{self.max_steps}"}
            )
        return await super().step()

    async def think(self) -> bool:
        """Thinking phase with cancellation check and progress updates."""
        # Check for cancellation first
        if self.cancel_requested:
            return False

        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "thinking", "message": "Thinking..."}
            )
        return await super().think()

    async def execute_tool(self, command) -> str:
        """Execute a tool with cancellation check and progress updates."""
        # Check for cancellation first
        if self.cancel_requested:
            return "Tool execution cancelled"

        if self.update_callback:
            await self.update_callback(
                self.client_id,
                {"status": "tool", "message": f"Using tool: {command.function.name}"}
            )
        result = await super().execute_tool(command)

        # Check if the result indicates a file creation (from FileSaver)
        if command.function.name == "file_saver" and self.file_tracker and "File ID:" in result:
            # Extract file ID from result
            import re
            file_id_match = re.search(r"File ID: ([a-f0-9-]+)", result)
            if file_id_match:
                file_id = file_id_match.group(1)
                # Inform the client about the new file
                if self.update_callback:
                    file_info = self.file_tracker.get_file_info(file_id)
                    if file_info:
                        await self.update_callback(
                            self.client_id,
                            {
                                "status": "file_created",
                                "file": file_info.to_dict()
                            }
                        )

        if self.update_callback:
            result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            await self.update_callback(
                self.client_id,
                {"status": "tool_result", "message": f"Tool result: {result_summary}"}
            )
        return result


class FileTracker:
    def __init__(self):
        self.files: Dict[str, FileInfo] = {}
        self.client_files: Dict[str, Set[str]] = {}  # Maps client_id to set of file_ids

    def add_file(self, file_info: FileInfo, client_id: Optional[str] = None):
        """Add a file to the tracker."""
        self.files[file_info.file_id] = file_info

        # Associate file with client if provided
        if client_id:
            if client_id not in self.client_files:
                self.client_files[client_id] = set()
            self.client_files[client_id].add(file_info.file_id)

    def get_file_info(self, file_id: str) -> Optional[FileInfo]:
        """Get information about a file."""
        return self.files.get(file_id)

    def get_client_files(self, client_id: str) -> List[FileInfo]:
        """Get all files associated with a client."""
        if client_id not in self.client_files:
            return []

        return [self.files[file_id] for file_id in self.client_files[client_id]
                if file_id in self.files]

    def remove_file(self, file_id: str):
        """Remove a file from the tracker."""
        if file_id in self.files:
            del self.files[file_id]

            # Remove from client associations
            for client_id in self.client_files:
                self.client_files[client_id].discard(file_id)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_instances: Dict[str, EnhancedManus] = {}
        self.file_tracker = FileTracker()
        # Add a dictionary to track processing tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept the connection and initialize agent instance."""
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # Create agent instance with the modified tools and file tracker
        self.agent_instances[client_id] = EnhancedManus(
            client_id=client_id,
            update_callback=self.send_message,
            file_tracker=self.file_tracker
        )
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Handle client disconnection and cleanup resources."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agent_instances:
            del self.agent_instances[client_id]
        # Also clean up any processing tasks
        if client_id in self.processing_tasks:
            task = self.processing_tasks[client_id]
            if not task.done():
                task.cancel()
            del self.processing_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            if isinstance(message, str):
                await self.active_connections[client_id].send_text(message)
            else:
                await self.active_connections[client_id].send_text(json.dumps(message))

    async def cancel_processing(self, client_id: str):
        """Cancel the current processing task for a client."""
        if client_id in self.processing_tasks:
            task = self.processing_tasks[client_id]
            if not task.done():
                # Set the cancel flag on the agent
                if client_id in self.agent_instances:
                    self.agent_instances[client_id].cancel_requested = True

                # Send cancellation status message
                await self.send_message(
                    client_id,
                    {"status": "cancelling", "message": "Cancellation requested. Stopping processing..."}
                )

                # Wait for task to acknowledge cancellation
                try:
                    # Give it a short timeout to react to cancellation
                    await asyncio.wait_for(asyncio.shield(task), 2.0)
                except asyncio.TimeoutError:
                    # If it doesn't respond in time, forcibly cancel
                    task.cancel()
                    await self.send_message(
                        client_id,
                        {"status": "cancelled", "message": "Processing was forcibly cancelled"}
                    )

                return True

        return False

    async def process_request(self, request: str, client_id: str):
        """Process a client request by creating and tracking a task."""
        if client_id not in self.agent_instances:
            await self.send_message(client_id, {"status": "error", "error": "No agent instance found"})
            return

        # Create and store the processing task
        task = asyncio.create_task(self._process_request_impl(request, client_id))
        self.processing_tasks[client_id] = task

        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Request processing for client {client_id} was cancelled")
            await self.send_message(
                client_id,
                {"status": "cancelled", "message": "Processing cancelled by user"}
            )
        finally:
            # Clean up task reference
            if client_id in self.processing_tasks:
                del self.processing_tasks[client_id]

    async def _process_request_impl(self, request: str, client_id: str):
        """Implementation of request processing."""
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

            # If we get here and cancellation was requested, still treat as cancelled
            if agent.cancel_requested:
                await self.send_message(
                    client_id,
                    {"status": "cancelled", "message": "Processing was cancelled successfully"}
                )
                return

            # Get new messages generated during processing
            new_messages = agent.memory.messages[len(original_messages):]

            # Get files created during processing
            client_files = self.file_tracker.get_client_files(client_id)
            file_data = [file_info.to_dict() for file_info in client_files]

            # Send the full result to the client
            await self.send_message(
                client_id,
                {
                    "status": "complete",
                    "result": result,
                    "messages": [msg.to_dict() for msg in new_messages],
                    "files": file_data
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

                # Handle different message types
                if "request" in request_data:
                    # Process request in background task
                    asyncio.create_task(
                        manager.process_request(request_data["request"], client_id)
                    )
                elif "command" in request_data and request_data["command"] == "cancel":
                    # Handle cancellation command
                    logger.info(f"Received cancellation request from client {client_id}")
                    await manager.cancel_processing(client_id)
                else:
                    logger.warning(f"Unknown message format from client {client_id}: {request_data}")

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
    agent = EnhancedManus(
        client_id=client_id,
        file_tracker=manager.file_tracker
    )

    try:
        # Process request
        result = await agent.run(request_data.request)

        # Get files created during processing
        client_files = manager.file_tracker.get_client_files(client_id)
        file_data = [file_info.to_dict() for file_info in client_files]

        # Return result
        return {
            "status": "complete",
            "result": result,
            "messages": [msg.to_dict() for msg in agent.memory.messages],
            "files": file_data
        }
    except Exception as e:
        logger.error(f"Error processing HTTP request: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/api/files")
async def list_files(client_id: Optional[str] = None):
    """List available files for a client."""
    if client_id:
        files = manager.file_tracker.get_client_files(client_id)
        return {"files": [file_info.to_dict() for file_info in files]}
    else:
        # Return all files if no client_id specified
        return {"files": [file_info.to_dict() for file_info in manager.file_tracker.files.values()]}


@app.get("/api/files/{file_id}")
async def get_file(file_id: str):
    """Serve a file by its ID."""
    file_info = manager.file_tracker.get_file_info(file_id)

    if not file_info:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")

    if not os.path.exists(file_info.file_path):
        raise HTTPException(status_code=404, detail=f"File {file_info.file_name} no longer exists on the server")

    return FileResponse(
        path=file_info.file_path,
        filename=file_info.file_name,
        media_type=file_info.content_type
    )


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