"""
Gen Image Tools Executor: tool entrypoints for the image generation task.
"""
from __future__ import annotations

from vision_deepresearch_async_workflow.tools.gen_web_tools import (
    UniversalImageSearchTool,
    WebTextSearchTool,
    JinaBrowseTool,
)


def create_gen_image_tools() -> dict:
    """
    Create the tool set used by the image generation task.
    
    Uses the same tools as DeepResearch:
    - image_search: image search
    - search: text search
    - browse: web browsing
    """
    print("[GenImageTools] Creating tools...")
    
    tools = {
        "image_search": UniversalImageSearchTool(),
        "search": WebTextSearchTool(),
        "browse": JinaBrowseTool(),
    }
    
    print(f"[GenImageTools] Tools created: {list(tools.keys())}")
    
    return tools
