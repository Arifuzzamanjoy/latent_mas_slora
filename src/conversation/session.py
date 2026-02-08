"""
Session Store - Persistent conversation storage

Features:
- Save/load sessions to disk
- Session metadata
- Multi-user session management
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading

from .manager import Conversation, Message


@dataclass
class Session:
    """
    A session containing one or more conversations.
    
    Sessions can be persisted to disk and restored later.
    """
    session_id: str
    user_id: Optional[str] = None
    conversations: Dict[str, Conversation] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Session state
    is_active: bool = True
    active_conversation_id: Optional[str] = None
    
    def add_conversation(self, conversation: Conversation) -> None:
        """Add a conversation to the session"""
        self.conversations[conversation.conversation_id] = conversation
        self.active_conversation_id = conversation.conversation_id
        self.updated_at = time.time()
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def get_active_conversation(self) -> Optional[Conversation]:
        """Get the active conversation"""
        if self.active_conversation_id:
            return self.conversations.get(self.active_conversation_id)
        return None
    
    def list_conversations(self) -> List[Conversation]:
        """List all conversations in the session"""
        return list(self.conversations.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversations": {
                conv_id: conv.to_dict()
                for conv_id, conv in self.conversations.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "is_active": self.is_active,
            "active_conversation_id": self.active_conversation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserialize session from dictionary"""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
            is_active=data.get("is_active", True),
            active_conversation_id=data.get("active_conversation_id"),
        )
        
        for conv_id, conv_data in data.get("conversations", {}).items():
            session.conversations[conv_id] = Conversation.from_dict(conv_data)
        
        return session


class SessionStore:
    """
    Persistent storage for sessions.
    
    Supports:
    - File-based storage (JSON)
    - In-memory storage
    - Session lifecycle management
    
    Example:
        store = SessionStore(storage_path="./sessions")
        
        # Create and save session
        session = store.create_session(user_id="user123")
        conv = Conversation()
        conv.add_user_message("Hello!")
        session.add_conversation(conv)
        store.save_session(session)
        
        # Load session later
        session = store.load_session(session.session_id)
    """
    
    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
        max_sessions: int = 1000,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_save = auto_save
        self.max_sessions = max_sessions
        
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        
        # Create storage directory if needed
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata,
    ) -> Session:
        """Create a new session"""
        import uuid
        
        session = Session(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            metadata=metadata,
        )
        
        with self._lock:
            self._sessions[session.session_id] = session
            
            # Cleanup if needed
            if len(self._sessions) > self.max_sessions:
                self._cleanup_old_sessions()
        
        if self.auto_save:
            self.save_session(session)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID (loads from disk if needed)"""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
        
        # Try loading from disk
        return self.load_session(session_id)
    
    def get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        **metadata,
    ) -> Session:
        """Get existing session or create new one"""
        session = self.get_session(session_id)
        if session:
            return session
        return self.create_session(session_id=session_id, user_id=user_id, **metadata)
    
    def save_session(self, session: Session) -> bool:
        """Save a session to storage"""
        if self.storage_path is None:
            return False
        
        try:
            session_file = self.storage_path / f"{session.session_id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"[Session] Error saving session {session.session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from storage"""
        if self.storage_path is None:
            return None
        
        session_file = self.storage_path / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = Session.from_dict(data)
            
            with self._lock:
                self._sessions[session_id] = session
            
            return session
            
        except Exception as e:
            print(f"[Session] Error loading session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
        
        if self.storage_path:
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                return True
        
        return False
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        active_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Session]:
        """List sessions with optional filtering"""
        # Load all sessions from disk if storage path exists
        if self.storage_path:
            for session_file in self.storage_path.glob("*.json"):
                session_id = session_file.stem
                if session_id not in self._sessions:
                    self.load_session(session_id)
        
        sessions = list(self._sessions.values())
        
        # Filter by user
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        # Filter by active status
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        
        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        if limit:
            sessions = sessions[:limit]
        
        return sessions
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user"""
        return self.list_sessions(user_id=user_id)
    
    def _cleanup_old_sessions(self) -> None:
        """Remove oldest inactive sessions"""
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
        )
        
        while len(self._sessions) > self.max_sessions:
            session = sessions.pop(0)
            if not session.is_active:
                del self._sessions[session.session_id]
                if self.storage_path:
                    session_file = self.storage_path / f"{session.session_id}.json"
                    if session_file.exists():
                        session_file.unlink()
    
    def save_all(self) -> int:
        """Save all sessions to storage"""
        count = 0
        for session in self._sessions.values():
            if self.save_session(session):
                count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_active),
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "auto_save": self.auto_save,
        }
    
    def export_session(
        self,
        session_id: str,
        output_path: Union[str, Path],
        format: str = "json",
    ) -> bool:
        """Export a session to a file"""
        session = self.get_session(session_id)
        if session is None:
            return False
        
        output_path = Path(output_path)
        
        try:
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format == "markdown":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Session: {session.session_id}\n\n")
                    f.write(f"User: {session.user_id or 'Anonymous'}\n")
                    f.write(f"Created: {time.ctime(session.created_at)}\n\n")
                    
                    for conv in session.list_conversations():
                        f.write(f"## Conversation: {conv.conversation_id}\n\n")
                        for msg in conv.messages:
                            role = msg.role.value.capitalize()
                            f.write(f"**{role}**: {msg.content}\n\n")
            
            elif format == "text":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for conv in session.list_conversations():
                        f.write(conv.to_prompt_string(format="simple"))
                        f.write("\n\n---\n\n")
            
            else:
                raise ValueError(f"Unknown format: {format}")
            
            return True
            
        except Exception as e:
            print(f"[Session] Error exporting session: {e}")
            return False
    
    def import_session(
        self,
        input_path: Union[str, Path],
    ) -> Optional[Session]:
        """Import a session from a file"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            return None
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = Session.from_dict(data)
            
            with self._lock:
                self._sessions[session.session_id] = session
            
            if self.auto_save:
                self.save_session(session)
            
            return session
            
        except Exception as e:
            print(f"[Session] Error importing session: {e}")
            return None
