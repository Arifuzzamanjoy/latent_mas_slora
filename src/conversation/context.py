"""
Context Window Management

Handles:
- Token counting and budgeting
- Context truncation strategies
- Sliding window management
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .manager import Message, MessageRole, Conversation


class ContextStrategy(Enum):
    """Strategy for managing context when it exceeds limits"""
    TRUNCATE_OLDEST = "truncate_oldest"  # Remove oldest messages
    TRUNCATE_MIDDLE = "truncate_middle"  # Keep first and last, remove middle
    SUMMARIZE = "summarize"  # Summarize older messages
    SLIDING_WINDOW = "sliding_window"  # Keep last N tokens


@dataclass
class ContextWindow:
    """
    Manages context window for conversations.
    
    Handles token budgeting and truncation to fit model limits.
    
    Example:
        window = ContextWindow(max_tokens=4096, tokenizer=tokenizer)
        
        # Add messages
        window.add(Message(role=MessageRole.USER, content="Hello"))
        window.add(Message(role=MessageRole.ASSISTANT, content="Hi there!"))
        
        # Get context that fits
        context = window.get_context()
    """
    max_tokens: int = 4096
    reserved_tokens: int = 512  # Reserve for response
    strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW
    tokenizer: Any = None  # Optional tokenizer for accurate counting
    
    # Internal state
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    _token_counts: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self._available_tokens = self.max_tokens - self.reserved_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: approximate ~4 chars per token
        return len(text) // 4 + 1
    
    def add(self, message: Message) -> bool:
        """
        Add a message to the context window.
        
        Returns True if added successfully, False if needed truncation.
        """
        token_count = self.count_tokens(message.content)
        
        self.messages.append(message)
        self._token_counts.append(token_count)
        
        # Check if we need to truncate
        if self.total_tokens > self._available_tokens:
            self._apply_strategy()
            return False
        
        return True
    
    def add_from_conversation(self, conversation: Conversation) -> None:
        """Load messages from a conversation"""
        self.system_prompt = conversation.system_prompt
        
        for msg in conversation.messages:
            self.add(msg)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens in context"""
        system_tokens = self.count_tokens(self.system_prompt) if self.system_prompt else 0
        return system_tokens + sum(self._token_counts)
    
    @property
    def remaining_tokens(self) -> int:
        """Get remaining tokens available"""
        return max(0, self._available_tokens - self.total_tokens)
    
    def _apply_strategy(self) -> None:
        """Apply truncation strategy to fit context"""
        if self.strategy == ContextStrategy.TRUNCATE_OLDEST:
            self._truncate_oldest()
        elif self.strategy == ContextStrategy.TRUNCATE_MIDDLE:
            self._truncate_middle()
        elif self.strategy == ContextStrategy.SLIDING_WINDOW:
            self._sliding_window()
        elif self.strategy == ContextStrategy.SUMMARIZE:
            self._summarize()
    
    def _truncate_oldest(self) -> None:
        """Remove oldest messages until within limit"""
        while self.total_tokens > self._available_tokens and len(self.messages) > 1:
            self.messages.pop(0)
            self._token_counts.pop(0)
    
    def _truncate_middle(self) -> None:
        """Keep first and last messages, remove middle"""
        while self.total_tokens > self._available_tokens and len(self.messages) > 2:
            # Remove from middle
            mid = len(self.messages) // 2
            self.messages.pop(mid)
            self._token_counts.pop(mid)
    
    def _sliding_window(self) -> None:
        """Keep most recent messages that fit"""
        # Start from the end and work backwards
        kept_messages = []
        kept_counts = []
        running_total = self.count_tokens(self.system_prompt) if self.system_prompt else 0
        
        for msg, count in zip(reversed(self.messages), reversed(self._token_counts)):
            if running_total + count <= self._available_tokens:
                kept_messages.insert(0, msg)
                kept_counts.insert(0, count)
                running_total += count
            else:
                break
        
        self.messages = kept_messages
        self._token_counts = kept_counts
    
    def _summarize(self) -> None:
        """Summarize older messages (requires model)"""
        # For now, fall back to truncate_oldest
        # In production, use the model to summarize
        self._truncate_oldest()
    
    def get_context(
        self,
        format: str = "messages",
        include_system: bool = True,
    ) -> Any:
        """
        Get the current context.
        
        Args:
            format: "messages" (list of dicts), "string" (prompt string), "tokens" (token ids)
            include_system: Whether to include system prompt
            
        Returns:
            Context in requested format
        """
        if format == "messages":
            messages = []
            if include_system and self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            for msg in self.messages:
                messages.append(msg.to_chat_format())
            return messages
        
        elif format == "string":
            parts = []
            if include_system and self.system_prompt:
                parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")
            for msg in self.messages:
                role = msg.role.value
                parts.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
            return "\n".join(parts)
        
        elif format == "tokens":
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for 'tokens' format")
            text = self.get_context(format="string", include_system=include_system)
            return self.tokenizer.encode(text)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context window statistics"""
        return {
            "num_messages": len(self.messages),
            "total_tokens": self.total_tokens,
            "available_tokens": self._available_tokens,
            "remaining_tokens": self.remaining_tokens,
            "utilization": self.total_tokens / self._available_tokens if self._available_tokens > 0 else 0,
            "strategy": self.strategy.value,
        }
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()
        self._token_counts.clear()
    
    def optimize_for_response(self, expected_response_tokens: int = 512) -> int:
        """
        Optimize context to leave room for expected response.
        
        Returns number of tokens freed.
        """
        original = self.total_tokens
        target = self._available_tokens - expected_response_tokens
        
        while self.total_tokens > target and len(self.messages) > 1:
            self._apply_strategy()
        
        return original - self.total_tokens


class ContextWindowManager:
    """
    Manages context windows for multiple conversations.
    
    Provides conversation-level context management with
    automatic optimization and caching.
    """
    
    def __init__(
        self,
        max_tokens: int = 4096,
        strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
        tokenizer: Any = None,
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.tokenizer = tokenizer
        
        self._windows: Dict[str, ContextWindow] = {}
    
    def get_window(self, conversation_id: str) -> ContextWindow:
        """Get or create a context window for a conversation"""
        if conversation_id not in self._windows:
            self._windows[conversation_id] = ContextWindow(
                max_tokens=self.max_tokens,
                strategy=self.strategy,
                tokenizer=self.tokenizer,
            )
        return self._windows[conversation_id]
    
    def sync_from_conversation(self, conversation: Conversation) -> ContextWindow:
        """Sync a context window with a conversation"""
        window = self.get_window(conversation.conversation_id)
        window.clear()
        window.add_from_conversation(conversation)
        return window
    
    def delete_window(self, conversation_id: str) -> bool:
        """Delete a context window"""
        if conversation_id in self._windows:
            del self._windows[conversation_id]
            return True
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all windows"""
        return {
            conv_id: window.get_stats()
            for conv_id, window in self._windows.items()
        }
