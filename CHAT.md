# Interactive Chat Interface

Chat with LatentMAS using TRUE LatentMAS mode for ultra-fast latent-space collaboration.

**Features:**
- ⚡ **TRUE LatentMAS Mode** - 3-5x faster than normal (only final agent generates text)
- 🧠 **4 Agents in Latent Space** - Planner → Critic → Refiner → Judger
- 💬 **Conversation Continuity** - Remembers full conversation context
- 📚 **RAG** - Document intelligence (always enabled)
- 🔧 **Optional Tools** - Calculator, search, Python executor
- 💾 **Session Persistence** - Auto-save conversations

## Quick Start

```bash
# Basic chat (RAG enabled)
python examples/chat.py

# Load documents for RAG
python examples/chat.py --rag-docs ./docs

# Enable tools (calculator, search, etc.)
python examples/chat.py --enable-tools

# All features with documents
python examples/chat.py --enable-tools --rag-docs ./data
```

## Chat Commands

While chatting, you can use these commands:

- `/help` - Show help message
- `/clear` - Clear conversation history (start fresh)
- `/save` - Save conversation to markdown file
- `/exit` or `/quit` - Exit chat

## Examples

### Basic Conversation
```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...

You: Can you give me an example?
Assistant: [continues from previous context]
```

### With Tools Enabled
```
You: Calculate the compound interest on $10,000 at 5% for 3 years
Assistant: [Used 1 tools] Using the compound interest formula...
```

### With RAG Documents
```
You: What does our Q3 report say about revenue?
Assistant: According to the Q3 report, revenue grew by 15%...
```

**Note:** RAG is always enabled. Load documents with `--rag-docs` for context-aware responses.

## Options

```
--model MODEL              Model to use (default: Qwen/Qwen2.5-3B-Instruct)
--device DEVICE            Device (cuda/cpu, default: cuda)
--enable-tools             Enable tool use (calculator, search, etc.)
--rag-docs PATH            Load documents for RAG context
--system-prompt TEXT       Custom system prompt
--session-path PATH        Session storage path (default: ./chat_sessions)
```

## Notes

- **Full precision (bfloat16)** for best quality
- **RAG is always enabled** - load documents with `--rag-docs`
- Conversations are automatically saved to `./chat_sessions/`
- Use Ctrl+C or `/exit` to quit
- Use `/clear` to start a new conversation
- Tool use requires `--enable-tools` flag
