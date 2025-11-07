# MemoryInLangGraph - Multi-Tier Memory Systems for Agentic AI

## Overview

This folder contains a comprehensive exploration of **memory systems in LangGraph-based agents**, demonstrating three distinct memory architectures for building intelligent email management systems. Each module builds upon the previous one, showcasing different memory paradigms and their integration.

**Main Objective:** Understand how agents can leverage different types of memory (episodic, semantic, procedural) to become smarter, more personalized, and more efficient over time.

## Module Structure

### 📧 1. **Baseline** - Email Triage Foundation
**Directory:** `Baseline/`

**What it teaches:**
- Basic email classification (respond, ignore, notify)
- LangGraph state management fundamentals
- Simple ReAct agents with tool use
- Email drafting and meeting scheduling

**Key Components:**
- `Baseline_agent.ipynb` - Email router with triage logic
- `prompts.py` - System and user prompts
- `schemas.py` - Pydantic models for structured outputs
- Tools: write_email, schedule_meeting, check_calendar_availability

**Memory Used:** None (stateless classification)

**Use Case:** Starting point for understanding agentic workflows

```
Email → Triage Router → Classification → Tool Execution → Response
```

---

### 🧠 2. **SemanticMemory** - Context-Aware Intelligence
**Directory:** `SemanticMemory/`

**What it teaches:**
- Semantic memory for storing facts about contacts and topics
- InMemoryStore with vector embeddings (text-embedding-3-small)
- Memory search and management tools (langmem)
- User-scoped memory namespaces

**Key Components:**
- `Semantic_memory_agent.ipynb` - Agent with semantic memory tools
- `manage_memory_tool` - Store contact/discussion information
- `search_memory_tool` - Retrieve stored information
- Namespace: `("email_assistant", "{user_id}", "collection")`

**Memory Structure:**
```python
{
    "contact_name": "Alice Smith",
    "company": "TechCorp",
    "interaction_type": "technical_discussion",
    "details": "Asked about API authentication endpoints"
}
```

**Use Case:** Agents that understand context and relationships

```
Email → Triage → Search Memory for Context → Respond with Awareness → Store New Info
```

---

### 📚 3. **episodicMemory** - Few-Shot Learning
**Directory:** `episodicMemory/`

**What it teaches:**
- Episodic memory for storing labeled examples
- Few-shot learning with in-context learning
- Vector similarity search for example retrieval
- Dynamic prompt enhancement with relevant examples
- User-specific classifier training

**Key Components:**
- `epsiodicMemoryAgent.ipynb` - Few-shot learning email classifier
- `format_few_shot_examples()` - Format examples for prompts
- Example storage: `("email_assistant", "{user_id}", "examples")`
- Learns from user feedback to improve classification

**Memory Structure:**
```python
{
    "email": {
        "author": "Alice Smith <alice@company.com>",
        "subject": "API documentation question",
        "email_thread": "Hi John, I have questions about..."
    },
    "label": "respond"
}
```

**Key Feature:** Personalization
- User "lance" learns to ignore sales emails
- User "andrew" has different preferences
- Each user builds own example library

**Use Case:** Adaptive classifiers that learn from user feedback

```
Email → Find Similar Examples via Vector Search → 
Few-Shot Prompt with Examples → Better Classification → 
Add Example to Training Set → Improved Future Performance
```

---

### ⚙️ 4. **ProceduralMemory** - Learned Workflows
**Directory:** `ProceduralMemory/`

**What it teaches:**
- Procedural memory for learned action sequences
- Workflow optimization and skill development
- Performance metrics and effectiveness tracking
- Task-type based procedure retrieval
- Context-aware action planning

**Key Components:**
- `ProceduralMemoryAgents.ipynb` - Agent that learns procedures
- Procedure namespace: `("email_assistant", "{user_id}", "procedures")`
- Tracks success rates and execution metrics
- Stores optimal action sequences for common tasks

**Memory Structure:**
```python
{
    "task_type": "respond_to_technical_question",
    "action_sequence": [
        {
            "action": "search_memory",
            "query": "related API discussions",
            "success_rate": 0.92
        },
        {
            "action": "write_email",
            "style": "technical_detailed",
            "success_rate": 0.88
        }
    ],
    "total_executions": 12,
    "success_count": 11
}
```

**Key Feature:** Performance Optimization
- Agent learns which action sequences work best
- Procedures improve over time with use
- Can adapt based on context and success metrics

**Use Case:** Agents that become more efficient through experience

```
Email → Identify Task Type → Retrieve Learned Procedure → 
Execute Optimized Action Sequence → Track Outcome → 
Update Procedure Effectiveness → Better Performance Next Time
```

---

## Three-Tier Memory Architecture

### Memory Types Comparison

| Aspect | Episodic | Semantic | Procedural |
|--------|----------|----------|-----------|
| **What** | Specific labeled examples | General facts & relationships | Learned action sequences |
| **When** | Classification tasks | Context understanding | Workflow execution |
| **Learning** | Few-shot from examples | Search-based retrieval | Optimization from outcomes |
| **Scope** | Discrete events | Broad knowledge | Routine patterns |
| **Use in Email** | "Similar emails → class" | "Who is Alice? → context" | "How to respond? → procedure" |

### Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│              Complete Email Agent Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Email Input                                                 │
│     ↓                                                         │
│  [1] TRIAGE ROUTER                                           │
│     • Uses Episodic Memory (few-shot examples)               │
│     • Returns: classification (respond|ignore|notify)        │
│     ↓                                                         │
│  [2] RESPONSE AGENT (if "respond")                           │
│     • Uses Semantic Memory (facts about contacts)            │
│     • Searches for relevant context                          │
│     • Retrieves Procedural Memory (learned workflows)        │
│     • Executes optimized action sequences                    │
│     • Tools: write_email, schedule_meeting, etc.             │
│     ↓                                                         │
│  [3] MEMORY STORAGE                                          │
│     • Stores new examples (Episodic)                         │
│     • Stores contact facts (Semantic)                        │
│     • Updates procedure metrics (Procedural)                 │
│     ↓                                                         │
│  Response/Action Completed                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Learning Progression

### Module Sequence

1. **Start with Baseline**
   - Understand basic agent architecture
   - Learn tool usage and state management
   - Get comfortable with LangGraph fundamentals

2. **Move to SemanticMemory**
   - Add contextual awareness
   - Learn vector embeddings and search
   - Understand memory namespaces

3. **Progress to episodicMemory**
   - Implement few-shot learning
   - Learn from labeled examples
   - Build personalized classifiers

4. **Advance to ProceduralMemory**
   - Track and optimize procedures
   - Understand skill development
   - Build adaptive workflows

### Dependencies

Each module depends on previous concepts:

```
Baseline
   ↓
SemanticMemory (adds: memory tools, embeddings)
   ↓
episodicMemory (adds: few-shot learning, examples)
   ↓
ProceduralMemory (adds: procedure tracking, optimization)
```

## File Structure

```
MemoryInLangGraph/
├── README.md (this file)
│
├── Baseline/
│   ├── README.md (module-specific documentation)
│   ├── Baseline_agent.ipynb (main notebook)
│   ├── prompts.py
│   ├── schemas.py
│   ├── utils.py
│   ├── examples.py
│   └── requirements.txt
│
├── SemanticMemory/
│   ├── README.md
│   ├── Semantic_memory_agent.ipynb
│   ├── prompts.py
│   ├── schemas.py
│   ├── utils.py
│   ├── examples.py
│   └── requirements.txt
│
├── episodicMemory/
│   ├── README.md
│   ├── epsiodicMemoryAgent.ipynb
│   ├── prompts.py
│   ├── schemas.py
│   ├── utils.py
│   ├── examples.py
│   └── requirements.txt
│
└── ProceduralMemory/
    ├── README.md
    ├── ProceduralMemoryAgents.ipynb
    ├── prompts.py
    ├── schemas.py
    ├── utils.py
    ├── examples.py
    └── requirements.txt
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nitinsb/GenAI-Practice.git
cd GenAI-Practice/MemoryInLangGraph

# Install dependencies (choose the module)
pip install -r Baseline/requirements.txt
pip install -r SemanticMemory/requirements.txt
pip install -r episodicMemory/requirements.txt
pip install -r ProceduralMemory/requirements.txt
```

### Environment Setup

```bash
# Create .env file in project root
OPENAI_API_KEY=sk-proj-xxx...
```

### Running Notebooks

```bash
# Start with Baseline
jupyter notebook Baseline/Baseline_agent.ipynb

# Progress through modules
jupyter notebook SemanticMemory/Semantic_memory_agent.ipynb
jupyter notebook episodicMemory/epsiodicMemoryAgent.ipynb
jupyter notebook ProceduralMemory/ProceduralMemoryAgents.ipynb
```

## Key Concepts Explained

### 1. User-Scoped Namespaces

Each user has isolated memory:

```python
config = {"configurable": {"langgraph_user_id": "lance"}}

# Lance's examples
namespace_lance = ("email_assistant", "lance", "examples")

# Andrew's examples (independent)
namespace_andrew = ("email_assistant", "andrew", "examples")
```

### 2. Vector Embeddings for Similarity

Emails and examples are embedded for semantic search:

```python
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Find similar emails
results = store.search(
    namespace,
    query=new_email_text,  # Automatically embedded
    limit=3  # Top 3 most similar
)
```

### 3. Few-Shot Prompting

Retrieve examples and include in system prompt:

```python
examples = store.search(namespace, query=email)
formatted = format_few_shot_examples(examples)

system_prompt = f"""
You are an email classifier.

Here are similar emails and how they were classified:
{formatted}

Now classify this new email: {email}
"""
```

### 4. Action Sequences

Procedures are ordered steps with metadata:

```python
procedure = {
    "actions": [
        {"name": "search_memory", "params": {...}},
        {"name": "write_email", "params": {...}},
        {"name": "manage_memory", "params": {...}}
    ],
    "success_rate": 0.92,
    "avg_time_ms": 1500
}
```

## Common Patterns

### Pattern 1: Classification with Memory

```python
# Use few-shot examples to improve classification
examples = store.search(
    ("email_assistant", user_id, "examples"),
    query=email,
    limit=2
)
classification = classifier.invoke([
    {"role": "system", "content": few_shot_prompt(examples)},
    {"role": "user", "content": email}
])
```

### Pattern 2: Context-Aware Response

```python
# Get both facts and procedures
facts = store.search(
    ("email_assistant", user_id, "collection"),
    query=sender
)
procedure = store.search(
    ("email_assistant", user_id, "procedures"),
    query=task_type
)

response = agent.invoke(
    {"messages": [...], "facts": facts, "procedure": procedure},
    config=config
)
```

### Pattern 3: Learning from Feedback

```python
# User corrects classification
corrected_classification = "respond"

# Add to training set
store.put(
    ("email_assistant", user_id, "examples"),
    str(uuid.uuid4()),
    {"email": email, "label": corrected_classification}
)

# Future similar emails will use learned pattern
```

## Advanced Topics

### Transfer Learning

Share successful procedures between users:

```python
# Get top procedure from user A
best_procedure = get_best_procedure("user_a", "respond_to_question")

# Adapt and store for user B
adapted = adapt_procedure(best_procedure, "user_b")
store.put(("email_assistant", "user_b", "procedures"), ..., adapted)
```

### Hierarchical Procedures

Compose complex workflows from simpler procedures:

```python
main_procedure = {
    "steps": [
        "basic_research",      # Sub-procedure
        "draft_response",      # Sub-procedure
        "review_and_send"      # Sub-procedure
    ]
}
```

### Reinforcement Learning Integration

Optimize procedures with reward signals:

```python
# Track rewards for each action
rewards = {
    "search_memory": +0.3,
    "write_email": +1.0,
    "manage_memory": +0.2
}

# Update procedure effectiveness
procedure['score'] = sum(rewards.values())
```

## Performance Metrics

### Tracking Success

```python
# Each procedure tracks:
- total_executions: 47
- success_count: 44
- success_rate: 0.936
- avg_time_ms: 1250
- error_count: 3
- last_updated: "2025-11-06"
```

### Analytics Queries

```python
# Find best performing procedures
best = max(procedures, key=lambda p: p['success_rate'])

# Find slowest (optimization candidates)
slowest = max(procedures, key=lambda p: p['avg_time_ms'])

# Trending procedures (recently updated)
trending = sorted(procedures, key=lambda p: p['last_updated'])[:5]
```

## Troubleshooting Guide

### Issue: Agents not using stored memory

**Check:**
1. Verify memory store is passed to agent: `email_agent.compile(store=store)`
2. Confirm namespace format matches store.search() calls
3. Ensure user_id is consistent across config

### Issue: Poor few-shot examples retrieved

**Solutions:**
1. Increase example diversity in training set
2. Add more labeled examples
3. Adjust embedding model if similarity is off
4. Check query format matches stored format

### Issue: Procedures not improving

**Debug:**
1. Verify success_rate is being updated
2. Check execution traces are being recorded
3. Ensure new procedures are being stored
4. Review effectiveness calculation logic

## Best Practices

1. **Start Simple:** Begin with Baseline, add complexity gradually
2. **User Isolation:** Always use unique user IDs for multi-user systems
3. **Regular Updates:** Keep procedures and examples current
4. **Monitor Performance:** Track success rates and execution times
5. **Feedback Loop:** Collect user corrections to improve learning
6. **Test Thoroughly:** Verify behavior with edge cases
7. **Document Procedures:** Add comments to complex action sequences
8. **Clean Storage:** Periodically remove unused/deprecated examples

## Extending the System

### Add New Email Task Type

1. Define task in prompts.py
2. Create example labeled emails
3. Store as episodic examples
4. Agent learns procedure through use

### Add New Tool

1. Define tool function with @tool decorator
2. Add to tools list in agent creation
3. Tool automatically available to agent
4. Example usage stored in procedures

### Custom Memory Backend

Replace InMemoryStore with:
- PostgreSQL (persistent storage)
- Pinecone (managed vector DB)
- Weaviate (distributed graph DB)
- Redis (in-memory with persistence)

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Memory Modules](https://python.langchain.com/docs/modules/memory/)
- [Cognitive Psychology: Memory Types](https://en.wikipedia.org/wiki/Procedural_memory)
- [Few-Shot Learning](https://arxiv.org/abs/2005.14165)
- [Vector Embeddings](https://platform.openai.com/docs/guides/embeddings)

## Learning Outcomes

After working through all modules, you'll understand:

✅ How to build agentic email systems  
✅ How different memory types enhance AI agents  
✅ How to implement semantic memory with embeddings  
✅ How few-shot learning improves classification  
✅ How to track and optimize learned procedures  
✅ How to build user-scoped, personalized AI systems  
✅ How to combine multiple memory systems effectively  
✅ Best practices for production agent deployment  

## Contributing

Found issues or have improvements? Feel free to contribute:

1. Run tests on your changes
2. Update documentation
3. Submit PR with clear description
4. Include example use cases

## License

Part of GenAI-Practice educational repository.

---

**Happy Learning! 🚀**

Start with `Baseline/` and work your way through each module to build a comprehensive understanding of memory systems in agentic AI.
