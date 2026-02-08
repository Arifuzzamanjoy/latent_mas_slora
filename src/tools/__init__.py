"""
Tool Use Module for Agent Actions

Provides:
- Tool definition and registration
- Tool execution framework
- Built-in tools (calculator, code execution, web search)
"""

from .tool_registry import ToolRegistry, Tool, ToolResult
from .executor import ToolExecutor, ReActExecutor
from .builtin_tools import (
    CalculatorTool,
    PythonExecutorTool,
    SearchTool,
    FileReaderTool,
    WebFetchTool,
)

__all__ = [
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "ToolExecutor",
    "ReActExecutor",
    "CalculatorTool",
    "PythonExecutorTool",
    "SearchTool",
    "FileReaderTool",
    "WebFetchTool",
]
