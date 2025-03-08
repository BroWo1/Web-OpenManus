import asyncio
import json
import logging
import time
import os
import sys
from typing import Dict, List, Optional, Callable, Any, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from pydantic import Field

# Import required components from OpenManus
from app.agent.manus import Manus
from app.tool.tool_collection import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.google_search import GoogleSearch
from app.tool.file_saver import FileSaver
from app.tool.bash import Bash
from app.tool.planning import PlanningTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate
# Import the original browser tool instead of creating a mock version
from app.tool.browser_use_tool import BrowserUseTool
from app.logger import logger
from app.agent.base import BaseAgent

# Configure logging for development
logging.basicConfig(level=logging.INFO)
import json
import os

def load_translation(lang_code):
    locales_path = os.path.join(os.path.dirname(__file__), "locales")
    lang_file = os.path.join(locales_path, f"{lang_code}.json")
    with open(lang_file, "r", encoding="utf-8") as f:
        return json.load(f)


class GradioManus(Manus):
    status_callback: Optional[Callable] = Field(default=None)
    html_callback: Optional[Callable] = Field(default=None)

    # Override the available_tools to use the original BrowserUseTool
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            GoogleSearch(),
            FileSaver(),
            Bash(),
            PlanningTool(),
            StrReplaceEditor(),
            BrowserUseTool(),  # Use the real browser tool
            Terminate()
        )
    )

    async def run(self, request: Optional[str] = None) -> str:
        if self.status_callback:
            self.status_callback(f"Starting to process: {request}")
        return await super().run(request)

    async def step(self) -> str:
        if self.status_callback:
            self.status_callback(f"Executing step {self.current_step}/{self.max_steps}")
        return await super().step()

    async def think(self) -> bool:
        if self.status_callback:
            self.status_callback("Thinking about what to do next...")
        return await super().think()

    async def execute_tool(self, command) -> str:
        tool_name = command.function.name
        if self.status_callback:
            self.status_callback(f"Using tool: {tool_name}")

        result = await super().execute_tool(command)

        if self.status_callback:
            result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            self.status_callback(f"Tool result: {result_summary}")

        # For the browser tool, update the HTML preview when get_html is called
        if tool_name == "browser_use" and self.html_callback:
            try:
                args = json.loads(command.function.arguments)
                action = args.get("action")

                if action == "get_html":
                    # The real BrowserUseTool will return the HTML in the result
                    if hasattr(result, 'output') and result.output:
                        url = "Current Page"
                        html_content = str(result.output)
                        self.html_callback(url, html_content)

                elif action == "navigate" and args.get("url"):
                    # After navigation, capture the URL at least
                    url = args.get("url")
                    self.html_callback(url, f"<div>Navigated to {url}</div>")

            except Exception as e:
                logger.error(f"Error updating HTML preview: {e}")

        return result


class OpenManusInterface:
    def __init__(self):
        self.agent = None
        self.conversation_history = []
        self.status_messages = []
        self.is_processing = False

        # HTML preview state
        self.current_url = None
        self.current_html = None

    def initialize_agent(self):
        """Initialize the agent."""
        self.agent = GradioManus(
            status_callback=self.update_status,
            html_callback=self.update_html_preview
        )
        return "Agent initialized successfully."

    def update_status(self, message: str):
        """Update status messages list."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_messages.append(f"[{timestamp}] {message}")

    def get_status_updates(self) -> str:
        """Get status updates as a string."""
        return "\n".join(self.status_messages[-15:]) if self.status_messages else "No status updates yet."

    def update_html_preview(self, url: str, html_content: str):
        """Update the HTML preview."""
        self.current_url = url
        self.current_html = html_content

    def get_html_preview(self) -> Tuple[str, str]:
        """Get the current HTML preview URL and content."""
        return (
            self.current_url or "No page loaded",
            f"<div style='width:100%; height:600px;'><iframe srcdoc='{self.current_html}' style='width:100%; height:100%;' frameborder='0'></iframe></div>"
            if self.current_html else "No HTML content available"
        )

    async def process_message(self, message: str) -> List[List[str]]:
        """Process a user message and return the updated conversation history."""
        if not self.agent:
            self.initialize_agent()

        if not message.strip():
            return self.get_conversation_history()

        self.is_processing = True
        self.status_messages = []
        self.update_status(f"Received request: {message}")

        try:
            # Record start time
            start_time = time.time()

            # Add user message to conversation in the format Gradio expects
            self.conversation_history.append({"role": "user", "content": message})

            # Process the message with the agent
            result = await self.agent.run(message)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update status
            self.update_status(f"Completed in {processing_time:.2f} seconds")

            # Add agent response to conversation
            self.conversation_history.append({"role": "assistant", "content": result})

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.update_status(f"Error: {str(e)}")
            self.conversation_history.append(
                {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})

        finally:
            self.is_processing = False

        return self.get_conversation_history()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history in Gradio chat format."""
        return self.conversation_history

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.status_messages = []
        return []


import gradio as gr

def create_interface():
    interface = OpenManusInterface()

    with gr.Blocks(title="OpenManus WebUI") as demo:
        title_md = gr.Markdown()
        description_md = gr.Markdown()


        # Your existing chatbot and status code here
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    value=interface.get_conversation_history(),
                    type="messages"
                )

                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Enter your message here...",
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear Conversation")

            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="Agent Status",
                    placeholder="Agent status will be displayed here...",
                    lines=20,
                    max_lines=20,
                    every=0.5,
                    value=interface.get_status_updates
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        lang_dropdown = gr.Dropdown(
                            label="🌐 Select Language",
                            choices=["en", "zh"],
                            value="en",
                        )



                def update_lang(lang):
                    translations = load_translation(lang)
                    return translations["title"], translations["description"]

                lang_dropdown.change(
                    update_lang,
                    inputs=lang_dropdown,
                    outputs=[title_md, description_md]
                )

                # Set default language content on load
                demo.load(
                    fn=lambda: update_lang("zh"),
                    outputs=[title_md, description_md]
                )

        def process_message_sync(message):
            return asyncio.run(interface.process_message(message))

        submit_btn.click(
            fn=process_message_sync,
            inputs=msg,
            outputs=chatbot
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        msg.submit(
            fn=process_message_sync,
            inputs=msg,
            outputs=chatbot
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        clear_btn.click(
            fn=interface.clear_conversation,
            outputs=[chatbot]
        )

        demo.load(fn=interface.initialize_agent, outputs=None)

    return demo


# Launch the Gradio interface
if __name__ == "__main__":
    # Install browsers for Playwright if needed
    try:
        import subprocess

        print("Checking if Playwright browsers are installed...")
        subprocess.run(["python", "-m", "playwright", "install", "--with-deps"])
        print("Playwright setup complete.")
    except Exception as e:
        print(f"Warning: Playwright browser installation failed: {e}")
        print("You may need to run 'python -m playwright install' manually.")

    # Launch the Gradio app
    demo = create_interface()
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",  # Make accessible from other computers
        server_port=7860,  # Default Gradio port
        share=True  # Generate a public link
    )