"""
Tool Registry - Define and manage available tools

Provides a framework for:
- Tool definition with schemas
- Tool validation
- Tool discovery and listing
"""

import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect


class ToolParameterType(Enum):
    """Supported parameter types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format"""
        schema = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert result to string for model consumption"""
        if self.success:
            if isinstance(self.output, (dict, list)):
                return json.dumps(self.output, indent=2)
            return str(self.output)
        return f"Error: {self.error}"


@dataclass
class Tool:
    """
    Definition of an executable tool.
    
    Tools can be used by agents to perform actions like:
    - Calculations
    - Code execution
    - Web searches
    - File operations
    
    Example:
        @Tool.create("calculator", "Perform mathematical calculations")
        def calculate(expression: str) -> float:
            return eval(expression)  # simplified
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    category: str = "general"
    requires_confirmation: bool = False
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible function schema"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def to_prompt_description(self) -> str:
        """Generate description for inclusion in prompts"""
        params_desc = []
        for p in self.parameters:
            req = "(required)" if p.required else "(optional)"
            params_desc.append(f"  - {p.name} ({p.type.value}) {req}: {p.description}")
        
        params_str = "\n".join(params_desc)
        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
{params_str}"""
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments"""
        try:
            # Validate parameters
            for param in self.parameters:
                if param.required and param.name not in kwargs:
                    if param.default is not None:
                        kwargs[param.name] = param.default
                    else:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Missing required parameter: {param.name}",
                        )
            
            # Execute function
            result = self.function(**kwargs)
            
            return ToolResult(
                success=True,
                output=result,
                metadata={"tool": self.name, "args": kwargs},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                metadata={"tool": self.name, "args": kwargs},
            )
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        category: str = "general",
        requires_confirmation: bool = False,
    ) -> Callable:
        """
        Decorator to create a tool from a function.
        
        Example:
            @Tool.create("add", "Add two numbers")
            def add(a: int, b: int) -> int:
                return a + b
        """
        def decorator(func: Callable) -> "Tool":
            # Extract parameters from function signature
            sig = inspect.signature(func)
            parameters = []
            
            for param_name, param in sig.parameters.items():
                # Get type annotation
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        param_type = ToolParameterType.STRING
                    elif param.annotation == int:
                        param_type = ToolParameterType.INTEGER
                    elif param.annotation == float:
                        param_type = ToolParameterType.FLOAT
                    elif param.annotation == bool:
                        param_type = ToolParameterType.BOOLEAN
                    elif param.annotation == list:
                        param_type = ToolParameterType.ARRAY
                    elif param.annotation == dict:
                        param_type = ToolParameterType.OBJECT
                    else:
                        param_type = ToolParameterType.STRING
                else:
                    param_type = ToolParameterType.STRING
                
                # Check if required
                required = param.default == inspect.Parameter.empty
                default = None if required else param.default
                
                # Get description from docstring (simplified)
                param_desc = f"Parameter {param_name}"
                
                parameters.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_desc,
                    required=required,
                    default=default,
                ))
            
            return cls(
                name=name,
                description=description,
                parameters=parameters,
                function=func,
                category=category,
                requires_confirmation=requires_confirmation,
            )
        
        return decorator


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Example:
        registry = ToolRegistry()
        registry.register(calculator_tool)
        registry.register(search_tool)
        
        # Get tool for execution
        tool = registry.get("calculator")
        result = tool.execute(expression="2 + 2")
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)
        
        print(f"[Tools] Registered: {tool.name} ({tool.category})")
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> Tool:
        """Register a function as a tool"""
        name = name or func.__name__
        description = description or func.__doc__ or f"Execute {name}"
        
        tool = Tool.create(name, description, **kwargs)(func)
        self.register(tool)
        return tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools"""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """List tool categories"""
        return list(self._categories.keys())
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())
    
    def get_schemas(self, tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get JSON schemas for tools (OpenAI function format)"""
        if tools is None:
            tools = list(self._tools.keys())
        
        return [self._tools[name].to_schema() for name in tools if name in self._tools]
    
    def get_prompt_description(self, tools: Optional[List[str]] = None) -> str:
        """Get tool descriptions for prompt injection"""
        if tools is None:
            tools = list(self._tools.keys())
        
        descriptions = []
        for name in tools:
            if name in self._tools:
                descriptions.append(self._tools[name].to_prompt_description())
        
        return "\n\n".join(descriptions)
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool"""
        if name not in self._tools:
            return False
        
        tool = self._tools.pop(name)
        if tool.category in self._categories:
            self._categories[tool.category].remove(name)
        
        return True
    
    def clear(self) -> None:
        """Clear all tools"""
        self._tools.clear()
        self._categories.clear()


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get or create global tool registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: Tool) -> None:
    """Register tool to global registry"""
    get_registry().register(tool)


def tool(
    name: str,
    description: str,
    category: str = "general",
    **kwargs,
) -> Callable:
    """
    Decorator to register a function as a tool.
    
    Example:
        @tool("multiply", "Multiply two numbers", category="math")
        def multiply(a: float, b: float) -> float:
            return a * b
    """
    def decorator(func: Callable) -> Tool:
        t = Tool.create(name, description, category=category, **kwargs)(func)
        register_tool(t)
        return t
    return decorator
