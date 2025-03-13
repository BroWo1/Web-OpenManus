# Create this as a new file named fix_file_functions.py
import os
import aiofiles
from pathlib import Path

# Directory for storing generated files
GENERATED_FILES_DIR = 'generated_files'

# Ensure the directory exists
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)


async def save_file(content, filename):
    """
    Simple function to save content to a file in the generated_files directory.

    Args:
        content (str): The content to save
        filename (str): The name of the file (without path)

    Returns:
        str: Path to the saved file
    """
    # Ensure filename doesn't contain path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(GENERATED_FILES_DIR, safe_filename)

    try:
        # Write the content to the file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


async def read_file(filename):
    """
    Read content from a file in the generated_files directory.

    Args:
        filename (str): The name of the file (without path)

    Returns:
        str: Content of the file or None if file doesn't exist
    """
    # Ensure filename doesn't contain path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(GENERATED_FILES_DIR, safe_filename)

    if not os.path.exists(file_path):
        return None

    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def list_generated_files():
    """
    List all files in the generated_files directory.

    Returns:
        list: List of filenames
    """
    try:
        return os.listdir(GENERATED_FILES_DIR)
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


# Usage example
if __name__ == "__main__":
    import asyncio


    async def test():
        # Save a file
        file_path = await save_file("print('Hello, World!')", "hello_world.py")
        print(f"File saved to: {file_path}")

        # Read the file
        content = await read_file("hello_world.py")
        print(f"File content: {content}")

        # List all files
        files = list_generated_files()
        print(f"Generated files: {files}")


    asyncio.run(test())