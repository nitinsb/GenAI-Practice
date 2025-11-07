# Email Assistant with Episodic Memory

## Overview

This module extends the semantic memory email assistant with **episodic memory** capabilities using few-shot learning through episodic examples. The agent learns from human feedback on email classifications and applies these learned patterns to new emails with similar characteristics.

**Key Features:**
- ✅ Email triage and routing (ignore, respond, notify)
- ✅ Semantic and episodic memory integration
- ✅ Few-shot learning from past classification examples
- ✅ User-scoped memory namespaces
- ✅ Vector-based similarity search for example retrieval
- ✅ Email drafting and meeting scheduling
- ✅ Multi-user support with isolated memories

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Email Processing Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Email Input                                              │
│     └─> Email Metadata (author, subject, content)           │
│                                                               │
│  2. Triage Router Node                                       │
│     ├─> Retrieve episodic examples from store               │
│     ├─> Format few-shot examples                            │
│     ├─> Call LLM router with examples in prompt             │
│     └─> Classification: ignore | respond | notify           │
│                                                               │
│  3. Response Agent (if classification == "respond")          │
│     ├─> Access memory tools (search & manage)               │
│     ├─> Draft email response                                │
│     ├─> Schedule meetings if needed                         │
│     └─> Store new memories about contacts                   │
│                                                               │
│  4. Memory Store (InMemoryStore)                             │
│     ├─> Episodic Examples (namespace: examples)             │
│     └─> Semantic Memory (namespace: collection)             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Memory Structure

**InMemoryStore with OpenAI Embeddings (text-embedding-3-small)**

#### Episodic Memory Namespace: `("email_assistant", "{user_id}", "examples")`
Stores **labeled email examples** for few-shot learning:
```python
{
    "email": {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": "Hi John, I was reviewing the API documentation..."
    },
    "label": "respond"  # or "ignore", "notify"
}
```

#### Semantic Memory Namespace: `("email_assistant", "{user_id}", "collection")`
Stores **semantic information** about contacts and discussions for context:
```python
{
    "contact_name": "Alice Smith",
    "company": "Company Inc",
    "interaction_type": "technical_discussion",
    "details": "Asked about API authentication endpoints"
}
```

### User-Scoped Isolation

Each user has independent memory stores accessed via `langgraph_user_id`:
- User "lance": Examples and memories isolated to lance
- User "harrison": Examples and memories isolated to harrison
- User "andrew": Examples and memories isolated to andrew

This enables multi-user systems where each user can train their own personalized email classifier.

## State Schema

```python
class State(TypedDict):
    email_input: dict  # Email metadata: author, to, subject, email_thread
    messages: Annotated[list, add_messages]  # Conversation history
```

## Key Nodes

### 1. Triage Router Node

**Function:** `triage_router(state, config, store)`

**Responsibilities:**
- Extracts email details from `state['email_input']`
- Retrieves relevant episodic examples using vector search
- Formats examples as few-shot demonstrations
- Generates system prompt with examples
- Calls structured LLM router
- Returns routing decision: `response_agent`, `END`, or `notify`

**Example Flow:**
```
Input Email: "Hi John, need clarification on API docs"
                          ↓
Vector Search: Find similar historical emails
                          ↓
Few-shot Format: 
  "Here are similar emails and how they were handled:
   - Alice asking about API docs → RESPOND
   - Sarah notifying about deployment → IGNORE"
                          ↓
LLM Classification: RESPOND (based on similar examples)
                          ↓
Route to: response_agent
```

### 2. Response Agent Node

**Function:** ReAct agent with memory tools

**Tools Available:**
1. **write_email(to, subject, content)** - Draft and send responses
2. **schedule_meeting(attendees, subject, duration_minutes, preferred_day)** - Schedule meetings
3. **check_calendar_availability(day)** - Check availability
4. **manage_memory_tool** - Store contact/discussion information
5. **search_memory_tool** - Retrieve stored information

**Model:** OpenAI GPT-4o

**Capabilities:**
- Analyzes email content for meeting requests
- Drafts contextually appropriate responses
- Stores information about contacts and discussions in semantic memory
- Uses search_memory to reference past interactions

## Few-Shot Learning Process

### Adding Examples to Episodic Memory

```python
import uuid

# Example 1: Technical question requiring response
data = {
    "email": {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": "Hi John, I was reviewing API docs..."
    },
    "label": "respond"
}

store.put(
    ("email_assistant", "lance", "examples"),  # User-scoped namespace
    str(uuid.uuid4()),                          # Unique ID
    data                                        # Example data
)
```

### How Few-Shot Examples Improve Classification

1. **New Email Arrives:** "Hi John - want to buy documentation?"
2. **Vector Search:** Find semantically similar examples
3. **Example Retrieval:** System retrieves spam/sales emails marked as "ignore"
4. **Few-Shot Prompt:**
   ```
   Here are some examples:
   
   Email Subject: Want to buy documentation
   Email From: Tom Jones <tom.jones@bar.com>
   ...
   > Triage Result: ignore
   ```
5. **LLM Decision:** Uses examples to classify new email as "ignore"

### User-Specific Learning

Each user builds their own example library:

```python
# User "harrison" trains on his preferences
store.put(("email_assistant", "harrison", "examples"), ..., spam_example)

# User "andrew" sees no spam examples initially
response = email_agent.invoke(
    {"email_input": email},
    config={"configurable": {"langgraph_user_id": "andrew"}}
)
# May classify differently than harrison
```

## Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=sk-proj-xxx...
```

### Agent Configuration
```python
config = {"configurable": {"langgraph_user_id": "lance"}}
```

### LLM Settings
- **Router Model:** openai:gpt-4o-mini (structured output for classification)
- **Response Model:** openai:gpt-4o (full reasoning for email drafting)
- **Embeddings:** openai:text-embedding-3-small (for semantic search)

## Usage Examples

### Basic Email Triage

```python
# Initialize with episodic examples
email_agent = StateGraph(State)
email_agent.add_node("triage_router", triage_router)
email_agent.add_node("response_agent", response_agent)
email_agent.compile(store=store)

# Process an email
email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": "Hi John, I was reviewing the API..."
}

response = email_agent.invoke(
    {"email_input": email_input},
    config={"configurable": {"langgraph_user_id": "lance"}}
)
```

### Training with Feedback

```python
# User corrects a misclassification
incorrect_email = {
    "author": "Tom Jones <tom.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": "Hi John - want to buy documentation?"
}

# First run: might classify as "respond"
response = email_agent.invoke(
    {"email_input": incorrect_email},
    config={"configurable": {"langgraph_user_id": "harrison"}}
)

# Add corrected example to training set
data = {"email": incorrect_email, "label": "ignore"}
store.put(
    ("email_assistant", "harrison", "examples"),
    str(uuid.uuid4()),
    data
)

# Subsequent runs: now classifies as "ignore" using learned pattern
```

### Multi-User Scenarios

```python
# User "lance" vs "andrew" may have different email preferences
email = {...}  # Some sales-related email

# Lance ignores sales emails (has learned pattern)
response_lance = email_agent.invoke(
    {"email_input": email},
    config={"configurable": {"langgraph_user_id": "lance"}}
)  # Classification: "ignore" (has sales examples in memory)

# Andrew hasn't trained classifier yet (no examples)
response_andrew = email_agent.invoke(
    {"email_input": email},
    config={"configurable": {"langgraph_user_id": "andrew"}}
)  # Classification: might differ based on default behavior
```

## Key Files

| File | Purpose |
|------|---------|
| `epsiodicMemoryAgent.ipynb` | Main notebook with agent implementation and examples |
| `prompts.py` | System and user prompts for triage and response generation |
| `schemas.py` | Pydantic models for structured outputs (Router classification) |
| `utils.py` | Helper functions for email processing and formatting |
| `examples.py` | Pre-built example emails for demonstration |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Dependencies

```
langchain==0.3.18
langchain-openai==0.3.5
langgraph==0.2.72
langmem==0.0.8
openai
pydantic
python-dotenv
```

## Advanced Concepts

### Vector Embeddings for Example Retrieval

The system uses OpenAI's `text-embedding-3-small` to embed both new emails and historical examples. When a new email arrives, similar examples are retrieved using vector similarity:

```python
# Search for similar examples
results = store.search(
    ("email_assistant", "lance", "examples"),
    query=str({"email": new_email}),  # Embeds email content
    limit=2  # Top 2 most similar examples
)
```

This ensures the most relevant historical examples are used for few-shot learning, improving classification accuracy.

### Episodic vs. Semantic Memory

- **Episodic:** Specific labeled email examples with dates/contexts (used for few-shot learning)
- **Semantic:** General facts about contacts, topics, and relationships (used for context in email drafting)

### Improving Classification Over Time

1. **Initial State:** Generic classifier without user preferences
2. **User Feedback:** Each correction adds example to episodic memory
3. **Vector Search:** Similar emails retrieve relevant examples
4. **Improved Accuracy:** Subsequent similar emails use learned patterns
5. **Personalization:** Each user develops unique classifier based on preferences

## Troubleshooting

### Issue: Agent not using episodic examples

**Solution:** Verify examples are stored with correct namespace format:
```python
# Correct
namespace = ("email_assistant", config['configurable']['langgraph_user_id'], "examples")

# Common mistake: Using wrong user ID
namespace = ("email_assistant", "lance", "examples")  # ✅ Correct if langgraph_user_id="lance"
```

### Issue: Different users seeing same examples

**Solution:** Ensure each user has unique `langgraph_user_id` in config:
```python
# Good: Each user has isolated memory
config_lance = {"configurable": {"langgraph_user_id": "lance"}}
config_andrew = {"configurable": {"langgraph_user_id": "andrew"}}
```

### Issue: No improvement from adding examples

**Solution:** 
1. Check vector similarity threshold (adjust search `limit` parameter)
2. Verify example format matches email format exactly
3. Ensure LLM is receiving examples in prompt (check `format_few_shot_examples` output)

## Future Enhancements

- [ ] Persistent storage backend (PostgreSQL, Pinecone)
- [ ] Active learning for automatic example selection
- [ ] A/B testing different example sets
- [ ] Analytics on classification patterns over time
- [ ] Feedback loop for automatic retraining
- [ ] Human-in-the-loop approval step after triage
- [ ] Hierarchical examples (top-level categories + subcategories)

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangMem Memory Tools](https://github.com/langchain-ai/langmem)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [Few-Shot Learning Patterns](https://arxiv.org/abs/2005.14165)

## License

Part of GenAI-Practice educational repository.
