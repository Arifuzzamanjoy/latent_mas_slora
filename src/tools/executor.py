"""
Tool Executor - Execute tools based on model output

Handles:
- Parsing tool calls from model output
- Tool execution
- Result formatting
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union

from .tool_registry import ToolRegistry, Tool, ToolResult, get_registry


class ToolExecutor:
    """
    Execute tools based on model output.
    
    Supports multiple formats:
    - JSON function calls
    - Natural language tool invocations
    - Structured XML-style calls
    
    Example:
        executor = ToolExecutor(registry)
        
        # Parse and execute from model output
        result = executor.execute_from_text('''
            I need to calculate this.
            <tool>calculator</tool>
            <args>{"expression": "2 + 2"}</args>
        ''')
    """
    
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        max_iterations: int = 5,
        require_confirmation: bool = False,
    ):
        self.registry = registry or get_registry()
        self.max_iterations = max_iterations
        self.require_confirmation = require_confirmation
        
        self._execution_history: List[Dict[str, Any]] = []
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments for the tool
            
        Returns:
            ToolResult with output or error
        """
        tool = self.registry.get(tool_name)
        
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}. Available: {self.registry.list_tools()}",
            )
        
        # Check confirmation requirement
        if (tool.requires_confirmation or self.require_confirmation) and not kwargs.pop('confirmed', False):
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' requires confirmation. Set confirmed=True to execute.",
                metadata={"requires_confirmation": True},
            )
        
        # Execute
        result = tool.execute(**kwargs)
        
        # Log execution
        self._execution_history.append({
            "tool": tool_name,
            "args": kwargs,
            "success": result.success,
            "output": str(result.output)[:500] if result.output else None,
            "error": result.error,
        })
        
        return result
    
    def parse_tool_call(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse tool calls from model output.
        
        Supports formats:
        1. JSON: {"tool": "name", "args": {...}}
        2. XML-style: <tool>name</tool><args>{...}</args>
        3. Function-style: tool_name(arg1="value", arg2=123)
        """
        calls = []
        
        # Try JSON format
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*"args"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                tool_name = match.group(1)
                args = json.loads(match.group(2))
                calls.append((tool_name, args))
            except json.JSONDecodeError:
                pass
        
        if calls:
            return calls
        
        # Try XML-style format
        xml_pattern = r'<tool>([^<]+)</tool>\s*<args>([^<]+)</args>'
        for match in re.finditer(xml_pattern, text, re.DOTALL):
            try:
                tool_name = match.group(1).strip()
                args = json.loads(match.group(2).strip())
                calls.append((tool_name, args))
            except json.JSONDecodeError:
                pass
        
        if calls:
            return calls
        
        # Try function-style format
        func_pattern = r'(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, text):
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Check if it's a registered tool
            if self.registry.get(tool_name) is None:
                continue
            
            # Parse arguments
            args = {}
            if args_str.strip():
                # Simple key=value parsing
                arg_pattern = r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\d+\.?\d*|true|false|\[.*?\]|\{.*?\})'
                for arg_match in re.finditer(arg_pattern, args_str, re.IGNORECASE):
                    key = arg_match.group(1)
                    value = arg_match.group(2)
                    
                    # Parse value
                    if value.startswith('"') or value.startswith("'"):
                        args[key] = value[1:-1]
                    elif value.lower() == 'true':
                        args[key] = True
                    elif value.lower() == 'false':
                        args[key] = False
                    elif '.' in value:
                        args[key] = float(value)
                    elif value.isdigit():
                        args[key] = int(value)
                    else:
                        try:
                            args[key] = json.loads(value)
                        except:
                            args[key] = value
            
            calls.append((tool_name, args))
        
        return calls
    
    def execute_from_text(self, text: str) -> List[ToolResult]:
        """
        Parse and execute all tool calls from text.
        
        Args:
            text: Model output text containing tool calls
            
        Returns:
            List of ToolResults
        """
        calls = self.parse_tool_call(text)
        results = []
        
        for tool_name, args in calls:
            result = self.execute(tool_name, **args)
            results.append(result)
        
        return results
    
    def format_results_for_model(self, results: List[ToolResult]) -> str:
        """Format tool results for injection back into model context"""
        if not results:
            return "No tools were executed."
        
        formatted = []
        for i, result in enumerate(results):
            if result.success:
                formatted.append(f"Tool {i+1} ({result.metadata.get('tool', 'unknown')}):\n{result.to_string()}")
            else:
                formatted.append(f"Tool {i+1} failed: {result.error}")
        
        return "\n\n".join(formatted)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of tool executions"""
        return self._execution_history.copy()
    
    def clear_history(self) -> None:
        """Clear execution history"""
        self._execution_history.clear()
    
    def generate_tool_prompt(self, tools: Optional[List[str]] = None) -> str:
        """
        Generate prompt text describing available tools.
        
        For injection into system prompt to enable tool use.
        """
        tool_desc = self.registry.get_prompt_description(tools)
        
        return f"""You have access to the following tools:

{tool_desc}

To use a tool, format your request as:
<tool>tool_name</tool>
<args>{{"param1": "value1", "param2": "value2"}}</args>

Only use tools when necessary. Always explain your reasoning before using a tool."""


class ReActExecutor(ToolExecutor):
    """
    ReAct-style executor with reasoning traces.
    
    Implements the Thought → Action → Observation loop.
    """
    
    def execute_react_step(
        self,
        model,
        tokenizer,
        question: str,
        context: str = "",
        max_steps: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute a ReAct reasoning step.
        
        Args:
            model: LLM for reasoning
            tokenizer: Tokenizer
            question: User question
            context: Previous context
            max_steps: Maximum reasoning steps
            
        Returns:
            Dict with final answer and trace
        """
        device = next(model.parameters()).device
        
        # Build prompt
        tool_desc = self.generate_tool_prompt()
        
        system_prompt = f"""You are a helpful assistant that can use tools to answer questions.

{tool_desc}

Use the following format:
Thought: [Your reasoning about what to do]
Action: <tool>tool_name</tool><args>{{"param": "value"}}</args>
Observation: [Tool result will appear here]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Final Answer: [Your final answer]"""

        trace = []
        full_context = context
        
        for step in range(max_steps):
            # Build prompt for this step
            prompt = f"""{system_prompt}

Question: {question}

{full_context}

Thought:"""
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse response
            if "Final Answer:" in response:
                # Extract final answer
                final = response.split("Final Answer:")[-1].strip()
                trace.append({"step": step + 1, "type": "final", "content": final})
                
                return {
                    "answer": final,
                    "trace": trace,
                    "steps": step + 1,
                }
            
            # Extract thought and action
            thought = ""
            if "Thought:" in response or response.strip():
                thought = response.split("Action:")[0].replace("Thought:", "").strip()
            
            trace.append({"step": step + 1, "type": "thought", "content": thought})
            
            # Execute tool calls
            results = self.execute_from_text(response)
            
            if results:
                observation = self.format_results_for_model(results)
                trace.append({"step": step + 1, "type": "action", "results": [r.to_string() for r in results]})
                trace.append({"step": step + 1, "type": "observation", "content": observation})
                
                full_context += f"\nThought: {thought}\nAction: {response.split('Action:')[-1].split('Observation:')[0] if 'Action:' in response else ''}\nObservation: {observation}\n"
            else:
                # No tool call found, might be reasoning only
                full_context += f"\nThought: {thought}\n"
        
        # Max steps reached
        return {
            "answer": "I was unable to complete the task within the allowed steps.",
            "trace": trace,
            "steps": max_steps,
        }
