"""
Conversation Continuity Module

Provides:
- Multi-turn conversation management
- Context window optimization
- Session persistence
"""

from .manager import ConversationManager, Conversation, Message, MessageRole
from .context import ContextWindow, ContextStrategy
from .session import SessionStore, Session

__all__ = [
    "ConversationManager",
    "Conversation",
    "Message",
    "MessageRole",
    "ContextWindow",
    "ContextStrategy",
    "SessionStore",
    "Session",
]
