# Email Assistant with Procedural Memory

## Overview

This module extends the email assistant with **procedural memory** capabilities, enabling agents to learn and refine action sequences, workflows, and skill execution patterns over time. Unlike episodic memory (specific examples) and semantic memory (facts), procedural memory focuses on **learned behaviors and routines**.

**Key Features:**
- ✅ Email triage and routing with learned action patterns
- ✅ Procedural memory for learned workflows
- ✅ Action sequence learning from multi-step tasks
- ✅ Performance optimization based on historical outcomes
- ✅ Skill development tracking and refinement
- ✅ Context-aware action selection
- ✅ Multi-user procedural learning with isolation

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│              Procedural Memory Email Agent                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Email Input                                              │
│     └─> Email Metadata (author, subject, content)           │
│                                                               │
│  2. Triage Router Node                                       │
│     ├─> Classify email (ignore | respond | notify)          │
│     ├─> Retrieve learned procedures from store              │
│     └─> Route to appropriate handler                        │
│                                                               │
│  3. Response Agent (ReAct with Tools)                        │
│     ├─> Plan action sequence                                │
│     ├─> Execute tools: write_email, schedule_meeting, etc   │
│     ├─> Track action effectiveness                          │
│     └─> Store learned procedures                            │
│                                                               │
│  4. Memory Store (Multi-Tier)                                │
│     ├─> Episodic: Labeled email examples                    │
│     ├─> Semantic: Contact facts and relationships           │
│     └─> Procedural: Learned action sequences & workflows    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Three-Tier Memory System

#### 1. **Episodic Memory** (Baseline/SemanticMemory)
- **What:** Specific labeled email examples
- **Purpose:** Few-shot learning for classification
- **Namespace:** `("email_assistant", "{user_id}", "examples")`
- **Example:**
  ```python
  {
      "email": {...email data...},
      "label": "respond"
  }
  ```

#### 2. **Semantic Memory** (SemanticMemory)
- **What:** Facts about contacts, topics, relationships
- **Purpose:** Contextual understanding for responses
- **Namespace:** `("email_assistant", "{user_id}", "collection")`
- **Example:**
  ```python
  {
      "contact_name": "Alice Smith",
      "relationship": "team_member",
      "expertise": "API design"
  }
  ```

#### 3. **Procedural Memory** (NEW - This Module)
- **What:** Learned action sequences and workflows
- **Purpose:** Optimize routine execution and skill development
- **Namespace:** `("email_assistant", "{user_id}", "procedures")`
- **Structure:**
  ```python
  {
      "task_type": "respond_to_technical_question",
      "context": "question about API documentation",
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
          },
          {
              "action": "manage_memory",
              "store": "contact_expertise",
              "success_rate": 0.95
          }
      ],
      "total_executions": 12,
      "success_count": 11,
      "last_updated": "2025-11-06"
  }
  ```

## State Schema

```python
class State(TypedDict):
    email_input: dict  # Email metadata
    messages: Annotated[list, add_messages]  # Conversation history
    procedures: dict  # Current procedural context
    execution_trace: list  # Track actions taken
```

## Key Concepts

### 1. Procedural Learning

**Definition:** Automatic acquisition of action patterns through repeated execution.

**Process:**
```
Email arrives
    ↓
Identify task type (e.g., "respond_to_technical_question")
    ↓
Retrieve learned procedure from store
    ↓
Execute action sequence
    ↓
Track outcome (success/failure)
    ↓
Update procedure effectiveness scores
    ↓
Store refined procedure
```

### 2. Action Sequences

Procedures consist of ordered steps with metadata:

```python
[
    {
        "action": "search_memory",      # Tool/action name
        "params": {
            "query": "related discussions",
            "limit": 3
        },
        "success_rate": 0.92,           # Historical success
        "avg_time_ms": 250,             # Performance metric
        "prerequisite": None             # Dependencies
    },
    {
        "action": "write_email",
        "params": {
            "style": "technical_detailed",
            "include_examples": True
        },
        "success_rate": 0.88,
        "avg_time_ms": 1200,
        "prerequisite": "search_memory"  # Must follow search
    }
]
```

### 3. Success Metrics

Each action tracks:
- **Success Rate:** Percentage of successful executions
- **Execution Time:** Average time to complete action
- **Error Rate:** Frequency of failures
- **Context Effectiveness:** Performance in different scenarios

### 4. Task Types

Examples of learnable task patterns:

| Task Type | Description | Common Procedure |
|-----------|-------------|------------------|
| `respond_to_question` | Answer technical/non-technical questions | search_memory → write_email → manage_memory |
| `schedule_meeting` | Email contains meeting request | check_calendar → schedule_meeting → write_email |
| `urgent_response` | Requires immediate action | write_email → check_calendar → schedule_meeting |
| `delegation` | Forward to appropriate person | search_memory → write_email |
| `information_update` | Non-urgent informational email | manage_memory |

## Usage Examples

### Basic Setup

```python
from langgraph.store.memory import InMemoryStore

# Initialize with embeddings
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Create agent graph with procedural learning
email_agent = StateGraph(State)
email_agent.add_node("triage_router", triage_router)
email_agent.add_node("response_agent", response_agent)
email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(store=store)

config = {"configurable": {"langgraph_user_id": "lance"}}
```

### Processing an Email

```python
email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Questions about API documentation",
    "email_thread": "Hi John, I have questions about..."
}

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)

# Agent uses learned procedures to handle email
# Procedure effectiveness scores updated based on outcome
```

### Viewing Learned Procedures

```python
# Retrieve all procedures for user "lance"
procedures = store.search(
    ("email_assistant", "lance", "procedures"),
    limit=10
)

for proc in procedures:
    print(f"Task: {proc.value['task_type']}")
    print(f"Success Rate: {proc.value['success_rate']:.2%}")
    print(f"Actions: {len(proc.value['action_sequence'])}")
```

### Analyzing Procedure Effectiveness

```python
# Get most successful procedures
high_performers = store.search(
    ("email_assistant", "lance", "procedures"),
    query="high success rate",
    limit=5
)

# Get slowest procedures (optimization candidates)
slow_procedures = store.search(
    ("email_assistant", "lance", "procedures"),
    query="high execution time",
    limit=5
)
```

## Learning Process

### Phase 1: Observation
- Agent observes incoming emails
- Classifies email type (respond, ignore, notify)
- Retrieves similar emails to identify task pattern

### Phase 2: Execution
- Agent selects action sequence (learned or default)
- Executes each action with context awareness
- Records execution trace and outcomes

### Phase 3: Reflection
- Evaluates result quality
- Compares against learned baseline
- Updates procedure effectiveness scores

### Phase 4: Storage
- Stores refined procedure in memory
- Indexes by task type and context
- Makes available for future similar emails

## Advanced Features

### Adaptive Action Selection

```python
# Agent chooses between multiple learned procedures
procedures_for_task = store.search(
    ("email_assistant", "lance", "procedures"),
    query="respond_to_technical_question"
)

# Sort by success rate
best_procedure = max(
    procedures_for_task,
    key=lambda p: p.value['success_rate']
)

# Execute best-performing procedure
execute_procedure(best_procedure, email_input)
```

### Context-Aware Procedures

Procedures can vary based on context:

```python
procedure = {
    "task_type": "respond_to_technical_question",
    "contexts": {
        "API_design": {
            "style": "detailed_technical",
            "include_code_samples": True,
            "success_rate": 0.94
        },
        "deployment_issue": {
            "style": "urgent_concise",
            "include_code_samples": False,
            "success_rate": 0.87
        },
        "best_practices": {
            "style": "educational",
            "include_references": True,
            "success_rate": 0.91
        }
    }
}
```

### Performance Optimization

Track and optimize slowest steps:

```python
# Identify bottleneck actions
slow_actions = [
    action for action in procedure['action_sequence']
    if action['avg_time_ms'] > 1000
]

# Consider alternatives or optimizations
# Store optimized version with lower avg_time_ms
```

### User-Specific Learning

Each user develops unique procedures:

```python
# User "lance" specializes in technical responses
lance_tech_procedure = {
    "task_type": "respond_to_technical_question",
    "action_sequence": [...detailed technical handling...],
    "success_rate": 0.95,
    "executions": 47
}

# User "andrew" prefers delegation
andrew_procedure = {
    "task_type": "respond_to_technical_question",
    "action_sequence": [...delegation pattern...],
    "success_rate": 0.88,
    "executions": 23
}

# Each user develops their own optimal procedures
```

## Integration with Other Memory Types

### Combined Memory Usage

```python
# Step 1: Episodic Memory (classification)
examples = store.search(
    ("email_assistant", "lance", "examples"),
    query=email
)  # Get similar past examples for classification

# Step 2: Procedural Memory (action planning)
procedure = store.search(
    ("email_assistant", "lance", "procedures"),
    query=f"task_type: {classification}"
)  # Get learned action sequence

# Step 3: Semantic Memory (context)
context = store.search(
    ("email_assistant", "lance", "collection"),
    query=email['author']
)  # Get facts about sender

# Execute procedure with context awareness
result = execute_with_context(procedure, context)
```

## Metrics and Analytics

### Procedure Effectiveness Metrics

Track over time:
- Success rate per procedure
- Average execution time
- Error rates by action
- Context-specific performance
- User expertise development

### Example Analytics Query

```python
# Track improvement over time
procedures = store.search(
    ("email_assistant", "lance", "procedures"),
    limit=100
)

# Calculate average success rate
avg_success = sum(
    p.value['success_rate'] for p in procedures
) / len(procedures)

# Identify trending procedures
trending = sorted(
    procedures,
    key=lambda p: p.value['last_updated'],
    reverse=True
)[:5]
```

## Key Files

| File | Purpose |
|------|---------|
| `ProceduralMemoryAgents.ipynb` | Main notebook demonstrating procedural memory agent |
| `prompts.py` | System and user prompts for task planning |
| `schemas.py` | Pydantic models for procedures and execution traces |
| `utils.py` | Helper functions for procedure management |
| `examples.py` | Example procedures and learning patterns |
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

## Configuration

### Environment Setup

```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-proj-xxx..."

# Or use .env file
OPENAI_API_KEY=sk-proj-xxx...
```

### Model Configuration

```python
# Router: Quick classification
llm_router = init_chat_model("openai:gpt-4o-mini")

# Agent: Full reasoning and tool use
response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=[...],
    store=store
)

# Embeddings: For similarity search
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)
```

## Troubleshooting

### Issue: Procedures not improving over time

**Solution:**
1. Verify procedures are being stored correctly
2. Check that success metrics are being updated
3. Ensure execution traces are being recorded
4. Review procedure context matching logic

### Issue: Slow procedure execution

**Solution:**
1. Analyze execution trace to identify bottlenecks
2. Check for unnecessary tool calls
3. Optimize action sequences (remove redundant steps)
4. Consider parallel execution for independent actions

### Issue: Procedures too specific (not generalizing)

**Solution:**
1. Broaden task type definitions
2. Increase number of training examples
3. Add context variants to procedures
4. Use semantic search for approximate matching

## Future Enhancements

- [ ] Hierarchical procedures (main task → subtasks)
- [ ] Procedure composition (combining learned procedures)
- [ ] Transfer learning between users
- [ ] Automated procedure discovery
- [ ] A/B testing procedures
- [ ] Adaptive parameter tuning
- [ ] Skill ranking and expertise tracking
- [ ] Procedure versioning and rollback
- [ ] Batch execution optimization
- [ ] Machine-learned action ordering

## Advanced Topics

### Procedure Serialization

Store procedures in persistent database:

```python
import json

# Serialize procedure
procedure_json = json.dumps(procedure_dict)
database.save("procedures", procedure_json)

# Deserialize and execute
loaded_procedure = json.loads(procedure_json)
execute_procedure(loaded_procedure, email)
```

### Distributed Learning

Share procedures across agents:

```python
# Agent 1 learns procedure
agent1_procedure = execute_and_learn(email)

# Store in shared location
shared_store.put(agent1_procedure)

# Agent 2 uses learned procedure
agent2_uses_learned = shared_store.search(query)
```

### Reinforcement Learning Integration

Use RL rewards to optimize procedures:

```python
# Track rewards for each action
action_rewards = {
    "search_memory": +0.3,  # Helpful context
    "write_email": +1.0,    # Achieves goal
    "manage_memory": +0.2   # Useful for future
}

# Update procedure based on cumulative reward
procedure['effectiveness_score'] = sum(action_rewards.values())
```

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Procedural Memory in Cognitive Science](https://en.wikipedia.org/wiki/Procedural_memory)
- [Skill Learning and Automaticity](https://psycnet.apa.org/record/1986-30396-001)
- [Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1908.03963)

## License

Part of GenAI-Practice educational repository.
