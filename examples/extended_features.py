"""
Extended Features Demo - RAG, Tools, Training, Conversations

This example demonstrates all four new features added to LatentMAS:
1. RAG for document intelligence
2. Custom LoRA training pipeline
3. Basic tool use
4. Conversation continuity

Run with: python examples/extended_features.py
"""

import os
import sys
import json
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Note: In production, you would import directly:
# from src import LatentMASSystem, AgentConfig, ...

# For this example, we'll show the usage patterns


def demo_rag():
    """
    Demo: RAG for Document Intelligence
    
    RAG (Retrieval-Augmented Generation) enables document-grounded responses.
    """
    print("\n" + "="*60)
    print("DEMO 1: RAG for Document Intelligence")
    print("="*60)
    
    # Example code (requires system initialization)
    example_code = '''
# Enable RAG capabilities
system.enable_rag(
    chunk_size=512,           # Size of document chunks
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    top_k=5                   # Documents to retrieve
)

# Load documents from various sources
system.load_documents("/path/to/docs")           # Directory
system.load_documents("/path/to/file.pdf")       # Single file
system.load_documents(["file1.txt", "file2.md"]) # Multiple files

# Query with document grounding
result = system.query_with_rag(
    "What does the Q3 report say about revenue growth?",
    include_citations=True
)

print(f"Answer: {result.answer}")
print(f"Sources: {result.retrieved_chunks}")
print(f"Context used: {result.context_used[:200]}...")

# Multi-hop reasoning for complex queries
result = system.rag.multi_hop_query(
    "Compare the marketing strategies mentioned in all quarterly reports",
    max_hops=3
)
'''
    print(example_code)
    
    # Create sample document store demo
    print("\nSample DocumentStore operations:")
    print("-" * 40)
    
    from src.rag import DocumentStore, RAGRetriever
    
    store = DocumentStore(chunk_size=256, chunk_overlap=32)
    
    # Add sample documents
    store.add_document(
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn from data. Deep learning uses neural networks with "
        "multiple layers to model complex patterns.",
        title="ML Overview",
        metadata={"category": "technology"}
    )
    
    store.add_document(
        "The company reported strong Q3 results with revenue growth of 15%. "
        "Operating margins improved due to cost optimization initiatives. "
        "Management raised full-year guidance.",
        title="Q3 Report",
        metadata={"category": "finance"}
    )
    
    print(f"Documents loaded: {store.get_stats()['num_documents']}")
    print(f"Total chunks: {store.get_stats()['num_chunks']}")
    
    # Demo retrieval
    retriever = RAGRetriever(store, device="cpu")
    retriever.build_index()
    
    results = retriever.retrieve("What is machine learning?", top_k=2)
    print(f"\nRetrieval results for 'What is machine learning?':")
    for chunk, score in zip(results.chunks, results.scores):
        print(f"  - [{score:.3f}] {chunk.text[:80]}...")


def demo_training():
    """
    Demo: Custom LoRA Training Pipeline
    
    Train domain-specific adapters for specialized agents.
    """
    print("\n" + "="*60)
    print("DEMO 2: Custom LoRA Training Pipeline")
    print("="*60)
    
    example_code = '''
from src.training import LoRATrainer, TrainingConfig, TrainingDataset

# Method 1: Quick training via system
result = system.train_adapter(
    train_data="domain_data.json",
    adapter_name="medical_expert",
    num_epochs=3,
    learning_rate=2e-4,
    lora_rank=32
)

print(f"Adapter saved to: {result.adapter_path}")
print(f"Final loss: {result.final_loss:.4f}")

# Method 2: Full control with trainer
config = TrainingConfig(
    lora_rank=48,
    lora_alpha=96,
    learning_rate=1e-4,
    num_epochs=5,
    batch_size=4,
    gradient_accumulation_steps=8,
    use_gradient_checkpointing=True,
    mixed_precision="bf16",
    output_dir="./my_adapter"
)

dataset = TrainingDataset.from_json("data.json", tokenizer)
# Or from HuggingFace:
# dataset = TrainingDataset.from_huggingface("dataset_name", tokenizer)

trainer = system.create_trainer()
result = trainer.train(dataset, config)

# Evaluate the adapter
from src.training import evaluate_lora
metrics = evaluate_lora(model, tokenizer, "test_data.json")
print(f"Exact match: {metrics['exact_match_rate']:.2%}")
'''
    print(example_code)
    
    # Show data format
    print("\nTraining data format (JSON):")
    print("-" * 40)
    sample_data = [
        {
            "instruction": "Explain the symptoms of diabetes",
            "input": "",
            "output": "Diabetes symptoms include increased thirst, frequent urination..."
        },
        {
            "instruction": "What medication treats hypertension?",
            "input": "Patient has stage 2 hypertension",
            "output": "First-line treatments include ACE inhibitors, ARBs..."
        }
    ]
    print(json.dumps(sample_data, indent=2))


def demo_tools():
    """
    Demo: Basic Tool Use
    
    Enable agents to perform actions via tool calls.
    """
    print("\n" + "="*60)
    print("DEMO 3: Basic Tool Use")
    print("="*60)
    
    example_code = '''
# Enable tools with defaults
system.enable_tools(register_defaults=True)

# Available tools: calculator, python_executor, search, read_file, web_fetch

# Run with tools (ReAct-style reasoning)
result = system.run_with_tools(
    "Calculate the compound interest on $10,000 at 5% for 3 years",
    max_tool_calls=5
)

print(f"Answer: {result['answer']}")
print(f"Steps: {result['steps']}")

# See the reasoning trace
for step in result['trace']:
    print(f"{step['type']}: {step.get('content', step.get('results', ''))}")

# Register a custom tool
from src.tools import Tool, tool

@tool("stock_price", "Get current stock price", category="finance")
def get_stock_price(symbol: str) -> str:
    # In production, call real API
    return f"Stock {symbol}: $150.25"

system.register_tool(get_stock_price)
'''
    print(example_code)
    
    # Demo built-in tools
    print("\nBuilt-in tool demo:")
    print("-" * 40)
    
    from src.tools import CalculatorTool, ToolRegistry, ToolExecutor
    
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    
    executor = ToolExecutor(registry)
    
    # Execute tool
    result = executor.execute("calculator", expression="sqrt(144) + 10")
    print(f"Calculator result: {result.output}")
    
    # Parse tool call from text
    text = '''
    I need to calculate this:
    <tool>calculator</tool>
    <args>{"expression": "2 ** 10"}</args>
    '''
    results = executor.execute_from_text(text)
    if results:
        print(f"Parsed and executed: {results[0].output}")


def demo_conversations():
    """
    Demo: Conversation Continuity
    
    Maintain context across multiple turns.
    """
    print("\n" + "="*60)
    print("DEMO 4: Conversation Continuity")
    print("="*60)
    
    example_code = '''
# Enable conversations with persistence
system.enable_conversations(
    session_path="./sessions",
    default_system_prompt="You are a helpful coding assistant."
)

# Simple chat interface (maintains context)
r1 = system.chat("I'm building a Python web scraper")
r2 = system.chat("How should I handle rate limiting?")
r3 = system.chat("What about the error handling we discussed?")  # Remembers context!

# Create specific conversation
conv = system.new_conversation(
    system_prompt="You are a medical expert.",
    specialty="cardiology"  # Custom metadata
)

# Chat in that conversation
response = system.chat(
    "What are the risk factors for heart disease?",
    conversation_id=conv.conversation_id
)

# Access conversation history
for msg in conv.messages:
    print(f"{msg.role.value}: {msg.content[:50]}...")

# Export conversation
system.conversations.export_conversation(
    conv.conversation_id,
    "conversation.md",
    format="markdown"
)

# Session persistence
session = system._session_store.create_session(user_id="user123")
session.add_conversation(conv)
system._session_store.save_session(session)

# Load later
session = system._session_store.load_session(session.session_id)
'''
    print(example_code)
    
    # Demo conversation management
    print("\nConversation demo:")
    print("-" * 40)
    
    from src.conversation import ConversationManager, Conversation, MessageRole
    
    # Create conversation
    conv = Conversation(system_prompt="You are helpful.")
    
    # Simulate multi-turn dialogue
    conv.add_user_message("What's the capital of France?")
    conv.add_assistant_message("The capital of France is Paris.")
    conv.add_user_message("What's its population?")
    conv.add_assistant_message("Paris has a population of about 2.1 million in the city proper.")
    
    print(f"Conversation has {len(conv)} messages")
    print("\nChat format:")
    for msg in conv.to_chat_messages():
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Demo context window
    from src.conversation import ContextWindow, ContextStrategy
    
    window = ContextWindow(max_tokens=100, strategy=ContextStrategy.SLIDING_WINDOW)
    window.add_from_conversation(conv)
    
    print(f"\nContext window stats:")
    stats = window.get_stats()
    print(f"  Messages: {stats['num_messages']}")
    print(f"  Tokens: {stats['total_tokens']}")
    print(f"  Remaining: {stats['remaining_tokens']}")


def demo_combined():
    """
    Demo: All Features Combined
    
    Show how features work together.
    """
    print("\n" + "="*60)
    print("DEMO 5: Combined Usage")
    print("="*60)
    
    combined_example = '''
# Full system with all features
system = LatentMASSystem(model_name="Qwen/Qwen2.5-3B-Instruct")
system.add_default_agents()

# Enable all extended features
system.enable_rag(top_k=5)
system.enable_tools()
system.enable_conversations(session_path="./sessions")

# Load company documents
system.load_documents("./docs/quarterly_reports/")
system.load_documents("./docs/product_specs/")

# Multi-turn conversation with RAG and tools
r1 = system.chat("What were our Q3 sales figures?")
# → RAG retrieves Q3 report, agents reason, tools calculate if needed

r2 = system.chat("Calculate the year-over-year growth rate")  
# → Remembers Q3 context, uses calculator tool

r3 = system.chat("Summarize this for an executive presentation")
# → Uses full conversation context, RAG for additional context

# Train a specialized adapter based on conversation patterns
system.train_adapter(
    train_data="./sales_qa_pairs.json",
    adapter_name="sales_analyst",
    num_epochs=3
)

# Export the conversation
system.conversations.export_conversation(
    system.conversations.get_active_conversation().conversation_id,
    "sales_analysis_session.md",
    format="markdown"
)
'''
    print(combined_example)


def main():
    """Run all demos"""
    print("="*60)
    print("LatentMAS Extended Features Demo")
    print("="*60)
    print("\nThis demo shows the four new features:")
    print("1. RAG for Document Intelligence")
    print("2. Custom LoRA Training Pipeline")
    print("3. Basic Tool Use")
    print("4. Conversation Continuity")
    
    demo_rag()
    demo_training()
    demo_tools()
    demo_conversations()
    demo_combined()
    
    print("\n" + "="*60)
    print("Demo complete! See the code examples above for usage patterns.")
    print("="*60)


if __name__ == "__main__":
    main()
