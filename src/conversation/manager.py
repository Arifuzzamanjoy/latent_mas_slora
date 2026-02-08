"""
Conversation Manager - Multi-turn dialogue handling

Features:
- Message history tracking
- Turn management
- Conversation state
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MessageRole(Enum):
    """Role of a message sender"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"  # For multi-agent responses


@dataclass
class Message:
    """A single message in a conversation"""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Optional agent info for multi-agent scenarios
    agent_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "message_id": self.message_id,
            "agent_name": self.agent_name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id", str(uuid.uuid4())[:8]),
            agent_name=data.get("agent_name"),
        )
    
    def to_chat_format(self) -> Dict[str, str]:
        """Convert to chat completion format"""
        return {
            "role": self.role.value if self.role != MessageRole.AGENT else "assistant",
            "content": self.content,
        }


@dataclass
class Conversation:
    """
    A conversation containing multiple messages.
    
    Tracks the full dialogue history between user and system,
    including multi-agent interactions.
    """
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # System prompt (persists across turns)
    system_prompt: Optional[str] = None
    
    # Conversation state
    is_active: bool = True
    total_tokens: int = 0
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        agent_name: Optional[str] = None,
        **metadata,
    ) -> Message:
        """Add a new message to the conversation"""
        message = Message(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata,
        )
        self.messages.append(message)
        self.updated_at = time.time()
        return message
    
    def add_user_message(self, content: str, **metadata) -> Message:
        """Add a user message"""
        return self.add_message(MessageRole.USER, content, **metadata)
    
    def add_assistant_message(self, content: str, **metadata) -> Message:
        """Add an assistant message"""
        return self.add_message(MessageRole.ASSISTANT, content, **metadata)
    
    def add_agent_message(self, content: str, agent_name: str, **metadata) -> Message:
        """Add a message from a specific agent"""
        return self.add_message(MessageRole.AGENT, content, agent_name=agent_name, **metadata)
    
    def add_tool_result(self, content: str, tool_name: str, **metadata) -> Message:
        """Add a tool result message"""
        metadata["tool_name"] = tool_name
        return self.add_message(MessageRole.TOOL, content, **metadata)
    
    def get_messages(
        self,
        roles: Optional[List[MessageRole]] = None,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages, optionally filtered by role"""
        messages = self.messages
        
        if roles:
            messages = [m for m in messages if m.role in roles]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get the last message, optionally of a specific role"""
        messages = self.get_messages(roles=[role] if role else None)
        return messages[-1] if messages else None
    
    def get_user_messages(self) -> List[Message]:
        """Get all user messages"""
        return self.get_messages(roles=[MessageRole.USER])
    
    def get_assistant_messages(self) -> List[Message]:
        """Get all assistant/agent messages"""
        return self.get_messages(roles=[MessageRole.ASSISTANT, MessageRole.AGENT])
    
    def to_chat_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Convert to chat completion message format"""
        messages = []
        
        if include_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self.messages:
            messages.append(msg.to_chat_format())
        
        return messages
    
    def to_prompt_string(
        self,
        format: str = "chatml",
        include_system: bool = True,
    ) -> str:
        """Convert to a single prompt string"""
        if format == "chatml":
            parts = []
            
            if include_system and self.system_prompt:
                parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")
            
            for msg in self.messages:
                role = msg.role.value if msg.role != MessageRole.AGENT else "assistant"
                parts.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
            
            return "\n".join(parts)
        
        elif format == "simple":
            parts = []
            
            if include_system and self.system_prompt:
                parts.append(f"System: {self.system_prompt}")
            
            for msg in self.messages:
                role = msg.role.value.capitalize()
                if msg.agent_name:
                    role = f"{msg.agent_name}"
                parts.append(f"{role}: {msg.content}")
            
            return "\n\n".join(parts)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()
        self.updated_at = time.time()
    
    def truncate(self, keep_last: int) -> int:
        """Keep only the last N messages, return number removed"""
        if len(self.messages) <= keep_last:
            return 0
        
        removed = len(self.messages) - keep_last
        self.messages = self.messages[-keep_last:]
        self.updated_at = time.time()
        return removed
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "system_prompt": self.system_prompt,
            "is_active": self.is_active,
            "total_tokens": self.total_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Deserialize from dictionary"""
        conv = cls(
            conversation_id=data["conversation_id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
            system_prompt=data.get("system_prompt"),
            is_active=data.get("is_active", True),
            total_tokens=data.get("total_tokens", 0),
        )
        
        for msg_data in data.get("messages", []):
            conv.messages.append(Message.from_dict(msg_data))
        
        return conv


class ConversationManager:
    """
    Manager for multiple conversations.
    
    Features:
    - Multi-conversation handling
    - Conversation retrieval and search
    - Integration with LatentMAS system
    
    Example:
        manager = ConversationManager()
        conv = manager.create_conversation(system_prompt="You are helpful.")
        
        # Add messages
        conv.add_user_message("Hello!")
        response = system.run(conv.get_last_message().content)
        conv.add_assistant_message(response.final_answer)
        
        # Continue conversation
        conv.add_user_message("Tell me more")
        ...
    """
    
    def __init__(
        self,
        system=None,  # LatentMASSystem
        default_system_prompt: Optional[str] = None,
        max_conversations: int = 100,
    ):
        self.system = system
        self.default_system_prompt = default_system_prompt or "You are a helpful AI assistant."
        self.max_conversations = max_conversations
        
        self._conversations: Dict[str, Conversation] = {}
        self._active_conversation_id: Optional[str] = None
    
    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **metadata,
    ) -> Conversation:
        """Create a new conversation"""
        conv = Conversation(
            conversation_id=conversation_id or str(uuid.uuid4()),
            system_prompt=system_prompt or self.default_system_prompt,
            metadata=metadata,
        )
        
        self._conversations[conv.conversation_id] = conv
        self._active_conversation_id = conv.conversation_id
        
        # Clean up old conversations if needed
        if len(self._conversations) > self.max_conversations:
            self._cleanup_oldest()
        
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self._conversations.get(conversation_id)
    
    def get_active_conversation(self) -> Optional[Conversation]:
        """Get the currently active conversation"""
        if self._active_conversation_id:
            return self._conversations.get(self._active_conversation_id)
        return None
    
    def set_active(self, conversation_id: str) -> bool:
        """Set a conversation as active"""
        if conversation_id in self._conversations:
            self._active_conversation_id = conversation_id
            return True
        return False
    
    def list_conversations(
        self,
        active_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Conversation]:
        """List all conversations"""
        convs = list(self._conversations.values())
        
        if active_only:
            convs = [c for c in convs if c.is_active]
        
        # Sort by updated_at descending
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        
        if limit:
            convs = convs[:limit]
        
        return convs
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            if self._active_conversation_id == conversation_id:
                self._active_conversation_id = None
            return True
        return False
    
    def _cleanup_oldest(self) -> None:
        """Remove oldest inactive conversations"""
        convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
        )
        
        # Remove oldest until under limit
        while len(self._conversations) > self.max_conversations:
            conv = convs.pop(0)
            if conv.conversation_id != self._active_conversation_id:
                del self._conversations[conv.conversation_id]
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: User message
            conversation_id: Specific conversation (uses active if None)
            **kwargs: Arguments for system.run()
            
        Returns:
            Assistant response
        """
        # Get or create conversation
        if conversation_id:
            conv = self.get_conversation(conversation_id)
        else:
            conv = self.get_active_conversation()
        
        if conv is None:
            conv = self.create_conversation()
        
        # Add user message
        conv.add_user_message(message)
        
        # Run through system if available
        if self.system:
            # Build context from conversation history
            # Format conversation history for the model
            conversation_context = []
            for msg in conv.messages[:-1]:  # All messages except the one we just added
                if msg.role == MessageRole.USER:
                    conversation_context.append(f"User: {msg.content}")
                elif msg.role in [MessageRole.ASSISTANT, MessageRole.AGENT]:
                    agent_name = msg.agent_name or "Assistant"
                    conversation_context.append(f"{agent_name}: {msg.content}")
            
            # Prepend conversation history to current message
            if conversation_context:
                context_text = "\n\n".join(conversation_context)
                full_message = f"Previous conversation:\n{context_text}\n\nCurrent question:\n{message}"
            else:
                full_message = message
            
            # Use true_latent pipeline for faster latent-space collaboration
            result = self.system.run(
                full_message,
                pipeline="true_latent",
                max_new_tokens=800,
                temperature=0.7,
                **kwargs
            )
            response = result.final_answer
            
            # Track tokens
            conv.total_tokens += result.total_tokens
            
            # Add agent responses if multi-agent
            for agent_output in result.agent_outputs[:-1]:  # All but final
                conv.add_agent_message(
                    agent_output["output"][:500],  # Truncate for history
                    agent_name=agent_output["agent"],
                )
            
            # Add final response
            conv.add_assistant_message(response)
        else:
            response = "System not available. Please configure a LatentMASSystem."
            conv.add_assistant_message(response)
        
        return response
    
    def get_context_for_system(
        self,
        conversation_id: Optional[str] = None,
        max_messages: int = 10,
    ) -> str:
        """
        Get conversation context formatted for system input.
        
        Useful for prepending conversation history to new queries.
        """
        conv = self.get_conversation(conversation_id) if conversation_id else self.get_active_conversation()
        
        if conv is None:
            return ""
        
        # Get recent messages
        recent = conv.get_messages(limit=max_messages)
        
        # Format as context
        context_parts = []
        for msg in recent:
            if msg.role == MessageRole.USER:
                context_parts.append(f"User: {msg.content}")
            elif msg.role in [MessageRole.ASSISTANT, MessageRole.AGENT]:
                prefix = msg.agent_name if msg.agent_name else "Assistant"
                context_parts.append(f"{prefix}: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def search_conversations(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Conversation]:
        """Search conversations by content"""
        results = []
        query_lower = query.lower()
        
        for conv in self._conversations.values():
            for msg in conv.messages:
                if query_lower in msg.content.lower():
                    results.append(conv)
                    break
        
        return results[:limit]
    
    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
    ) -> str:
        """Export a conversation"""
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return ""
        
        if format == "json":
            import json
            return json.dumps(conv.to_dict(), indent=2)
        elif format == "text":
            return conv.to_prompt_string(format="simple")
        elif format == "markdown":
            lines = [f"# Conversation {conv.conversation_id}\n"]
            for msg in conv.messages:
                role = msg.role.value.capitalize()
                if msg.agent_name:
                    role = f"**{msg.agent_name}**"
                else:
                    role = f"**{role}**"
                lines.append(f"{role}:\n{msg.content}\n")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")
