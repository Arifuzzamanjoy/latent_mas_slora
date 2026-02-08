"""
Built-in Tools - Ready-to-use tools for common tasks

Includes:
- Calculator: Mathematical expressions
- Python Executor: Run Python code
- Search: Web/document search
- File Reader: Read local files
- Web Fetch: Fetch web content
"""

import os
import re
import json
import math
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .tool_registry import Tool, ToolParameter, ToolParameterType, ToolResult


class CalculatorTool(Tool):
    """
    Mathematical calculator tool.
    
    Safely evaluates mathematical expressions.
    """
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions. Supports basic arithmetic, trigonometry, and common math functions.",
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ToolParameterType.STRING,
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sin(3.14)', 'sqrt(16)')",
                    required=True,
                ),
            ],
            function=self._calculate,
            category="math",
        )
        
        # Safe math functions
        self._safe_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'floor': math.floor,
            'ceil': math.ceil,
            'pi': math.pi,
            'e': math.e,
        }
    
    def _calculate(self, expression: str) -> Union[float, str]:
        """Safely evaluate a mathematical expression"""
        # Clean expression
        expression = expression.strip()
        
        # Remove any dangerous characters
        allowed_chars = set('0123456789+-*/.()^%,_ ')
        for func_name in self._safe_functions:
            allowed_chars.update(func_name)
        
        # Replace common aliases
        expression = expression.replace('^', '**')
        expression = expression.replace('×', '*')
        expression = expression.replace('÷', '/')
        
        try:
            # Evaluate with restricted globals
            result = eval(expression, {"__builtins__": {}}, self._safe_functions)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    return int(result)
                return round(result, 10)
            return result
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"


class PythonExecutorTool(Tool):
    """
    Python code execution tool.
    
    Executes Python code in a restricted environment.
    USE WITH CAUTION - should be sandboxed in production.
    """
    
    def __init__(self, allowed_modules: Optional[List[str]] = None):
        super().__init__(
            name="python_executor",
            description="Execute Python code and return the result. Limited to safe operations.",
            parameters=[
                ToolParameter(
                    name="code",
                    type=ToolParameterType.STRING,
                    description="Python code to execute",
                    required=True,
                ),
            ],
            function=self._execute_python,
            category="code",
            requires_confirmation=True,  # Safety: require confirmation
        )
        
        self.allowed_modules = allowed_modules or ['math', 'json', 're', 'datetime', 'collections']
    
    def _execute_python(self, code: str) -> str:
        """Execute Python code safely"""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Restricted globals
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'abs': abs,
                'round': round,
            }
        }
        
        # Import allowed modules
        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        local_vars = {}
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)
            
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            
            if errors:
                return f"Output:\n{output}\n\nWarnings/Errors:\n{errors}"
            
            # If no print output, try to return last expression value
            if not output and local_vars:
                # Get the last assigned variable
                result_vars = {k: v for k, v in local_vars.items() if not k.startswith('_')}
                if result_vars:
                    last_key = list(result_vars.keys())[-1]
                    return f"Result ({last_key}): {result_vars[last_key]}"
            
            return output or "Code executed successfully (no output)"
            
        except Exception as e:
            return f"Execution error: {type(e).__name__}: {str(e)}"


class SearchTool(Tool):
    """
    Search tool for querying knowledge bases or web.
    
    In production, integrate with actual search APIs.
    """
    
    def __init__(self, search_function: Optional[callable] = None):
        super().__init__(
            name="search",
            description="Search for information on a topic. Returns relevant results.",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    type=ToolParameterType.INTEGER,
                    description="Number of results to return",
                    required=False,
                    default=5,
                ),
            ],
            function=self._search,
            category="search",
        )
        
        self._custom_search = search_function
    
    def _search(self, query: str, num_results: int = 5) -> str:
        """Perform search"""
        if self._custom_search:
            return self._custom_search(query, num_results)
        
        # Default: return placeholder (integrate with actual search in production)
        return f"""Search results for: "{query}"

Note: This is a placeholder. Integrate with a search API (Google, Bing, etc.) for real results.

To enable real search:
1. Install search library: pip install googlesearch-python
2. Or use an API: SerpAPI, Bing Search API, etc.
3. Pass a custom search_function to SearchTool()
"""


class FileReaderTool(Tool):
    """
    Read content from local files.
    """
    
    def __init__(self, allowed_paths: Optional[List[str]] = None):
        super().__init__(
            name="read_file",
            description="Read content from a local file.",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type=ToolParameterType.STRING,
                    description="Path to the file to read",
                    required=True,
                ),
                ToolParameter(
                    name="max_chars",
                    type=ToolParameterType.INTEGER,
                    description="Maximum characters to read",
                    required=False,
                    default=10000,
                ),
            ],
            function=self._read_file,
            category="file",
        )
        
        self.allowed_paths = allowed_paths
    
    def _read_file(self, file_path: str, max_chars: int = 10000) -> str:
        """Read file content"""
        path = Path(file_path)
        
        # Security check
        if self.allowed_paths:
            allowed = any(
                str(path.absolute()).startswith(str(Path(p).absolute()))
                for p in self.allowed_paths
            )
            if not allowed:
                return f"Error: Access denied to {file_path}"
        
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        if not path.is_file():
            return f"Error: Not a file: {file_path}"
        
        try:
            content = path.read_text(encoding='utf-8')
            
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... [Truncated, {len(content)} total chars]"
            
            return content
            
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WebFetchTool(Tool):
    """
    Fetch content from web URLs.
    """
    
    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="Fetch and extract text content from a web URL.",
            parameters=[
                ToolParameter(
                    name="url",
                    type=ToolParameterType.STRING,
                    description="URL to fetch",
                    required=True,
                ),
                ToolParameter(
                    name="max_chars",
                    type=ToolParameterType.INTEGER,
                    description="Maximum characters to return",
                    required=False,
                    default=5000,
                ),
            ],
            function=self._fetch_url,
            category="web",
        )
    
    def _fetch_url(self, url: str, max_chars: int = 5000) -> str:
        """Fetch URL content"""
        try:
            import urllib.request
            from html.parser import HTMLParser
            
            # Simple HTML text extractor
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
                    self._skip = False
                
                def handle_starttag(self, tag, attrs):
                    if tag in self.skip_tags:
                        self._skip = True
                
                def handle_endtag(self, tag):
                    if tag in self.skip_tags:
                        self._skip = False
                
                def handle_data(self, data):
                    if not self._skip:
                        text = data.strip()
                        if text:
                            self.text.append(text)
            
            # Fetch URL
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; LatentMAS/1.0)'}
            request = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(request, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
            
            # Extract text
            extractor = TextExtractor()
            extractor.feed(content)
            text = ' '.join(extractor.text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            return text
            
        except Exception as e:
            return f"Error fetching URL: {str(e)}"


# Factory function to create default tools
def create_default_tools() -> List[Tool]:
    """Create and return all default built-in tools"""
    return [
        CalculatorTool(),
        PythonExecutorTool(),
        SearchTool(),
        FileReaderTool(),
        WebFetchTool(),
    ]


def register_default_tools(registry) -> None:
    """Register all default tools to a registry"""
    for tool in create_default_tools():
        registry.register(tool)
