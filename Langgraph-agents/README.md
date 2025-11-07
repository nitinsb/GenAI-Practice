# LangGraph Agents - Foundational Patterns and Examples

## Overview

This module contains foundational implementations of **LangGraph agent patterns**, demonstrating core concepts and advanced techniques for building intelligent, stateful agents with LangChain.

**Purpose:** Learn how to build agents that can reason about problems, take actions, and maintain persistent state across interactions.

## What is LangGraph?

**LangGraph** is a library for building stateful, multi-actor applications with LLMs. Key features:
- **State Management:** Persistent state across agent steps
- **Graph-Based Workflows:** Define agent behavior as directed acyclic graphs
- **Tool Integration:** Seamless integration with custom and built-in tools
- **Streaming Support:** Real-time output streaming for better UX
- **Persistence:** Save and restore agent state

## Module Contents

### 1. 🎯 **agent_re+act.ipynb** - ReAct Pattern Fundamentals

**What it teaches:**
- ReAct (Reasoning + Acting) pattern for agents
- Basic tool definition with @tool decorator
- Agent state management with LangGraph
- Tool execution and result handling

**Key Concepts:**
- `calculate` tool - Perform mathematical calculations
- `average_dog_weight` tool - Look up dog breed information
- Agent loop: Think → Act → Observe → Repeat

**Example Flow:**
```
User: "What is the average weight of a golden retriever?"
         ↓
Agent thinks: "I need to use average_dog_weight tool"
         ↓
Agent acts: Calls average_dog_weight("golden retriever")
         ↓
Agent observes: Returns "golden retriever weighs about 60 lbs"
         ↓
Agent responds: "A golden retriever weighs approximately 60 pounds"
```

**Use Cases:**
- Learning basic agent architecture
- Understanding ReAct reasoning pattern
- Implementing simple tool-using agents

---

### 2. 🏗️ **Langgraph_Components.ipynb** - Core LangGraph Components

**What it teaches:**
- LangGraph fundamentals (nodes, edges, graphs)
- StateGraph construction
- Tool schemas and definitions
- Streaming agent outputs
- TavilySearchResults integration for web search

**Key Components:**
- **Nodes:** Discrete processing steps in the graph
- **Edges:** Transitions between nodes
- **State:** Shared data structure across nodes
- **Compiled Graph:** Ready-to-run agent

**Example Architecture:**
```
START
  ↓
[Input Processing]
  ↓
[Tool Router Node]
  ├→ [Search Tool] ↘
  ├→ [Calculate Tool] → [Combine Results]
  └→ [Other Tools] ↗
  ↓
END
```

**Tools Demonstrated:**
- TavilySearchResults - Web search tool
- Custom tool schemas
- Tool output handling

---

### 3. 💾 **Persisance_and_streaming.ipynb** - State Persistence & Streaming

**What it teaches:**
- State persistence with SqliteSaver
- Asynchronous state persistence (AsyncSqliteSaver)
- Streaming agent outputs with astream_events
- Checkpointing and state recovery
- Interactive agent workflows

**Key Features:**

**Synchronous Persistence:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver()
graph = agent.compile(checkpointer=memory)

# Save state automatically
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke(input, config=config)

# Resume from saved state
result = graph.invoke(new_input, config=config)
```

**Asynchronous Persistence:**
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

memory = AsyncSqliteSaver()
# Use with async agents for better performance
```

**Streaming Events:**
```python
# Stream events in real-time
async for event in graph.astream_events(input, config):
    print(event)  # See agent thinking process
```

**Use Cases:**
- Long-running conversations
- Session management
- Agent history tracking
- Real-time monitoring

---

### 4. 🤝 **human_in_loop.ipynb** - Human-in-the-Loop Workflows

**What it teaches:**
- Human approval workflows
- Interrupting agent execution
- Human feedback integration
- Decision branching based on human input
- Interactive agent control

**Key Patterns:**

**Pattern 1: Approval Before Action**
```
Agent plans action
         ↓
[INTERRUPT] Pause execution
         ↓
Human reviews plan
         ↓
Human approves/rejects
         ↓
Agent continues or takes alternative action
```

**Pattern 2: Human Correction**
```
Agent makes decision
         ↓
[INTERRUPT] Get human feedback
         ↓
Human corrects/refines decision
         ↓
Agent learns and continues
```

**Use Cases:**
- Critical decision approval
- Quality control checkpoints
- User feedback loops
- Safety validation

---

### 5. 🔍 **search_tavily.ipynb** - Web Search Integration

**What it teaches:**
- Integrating Tavily search API
- Handling web search results
- Parsing and ranking search results
- Using search in agent decision-making
- Real-time information retrieval

**Key Capabilities:**
- **Query Formatting:** Optimize search queries
- **Result Parsing:** Extract relevant information
- **Ranking:** Prioritize best results
- **Caching:** Avoid redundant searches

**Example:**
```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-...")
results = client.search("latest LLM trends 2025")
# Returns: [{"title": "...", "url": "...", "content": "..."}, ...]
```

---

### 6. 📚 **Essay_agent/** - Multi-Step Essay Generation

**What it teaches:**
- Complex multi-step workflows
- Sub-agent orchestration
- Research and content generation
- Reflection and refinement loops
- Document assembly

**Architecture:**

```
┌────────────────────────────────────────────┐
│          Essay Generation Pipeline         │
├────────────────────────────────────────────┤
│                                             │
│ [1] PLANNER                                │
│     → Outline essay structure              │
│     → Define key sections                  │
│                                             │
│ [2] RESEARCHER                             │
│     → Search for sources                   │
│     → Gather information                   │
│                                             │
│ [3] WRITER                                 │
│     → Generate each section                │
│     → Integrate sources                    │
│     → Format content                       │
│                                             │
│ [4] REFLECTOR                              │
│     → Review quality                       │
│     → Check coherence                      │
│     → Suggest improvements                 │
│                                             │
│ [5] PUBLISHER                              │
│     → Final assembly                       │
│     → Output generation                    │
│                                             │
└────────────────────────────────────────────┘
```

**Sub-Agents:**
- **Planner Agent** - Creates essay outline
- **Research Agent** - Searches for relevant information
- **Generator Agent** - Writes content sections
- **Reflector Agent** - Reviews and provides feedback

**Workflow:**
```
User Input (topic, length)
    ↓
Planner creates outline
    ↓
Researcher gathers info for each section
    ↓
Generator writes sections
    ↓
Reflector reviews draft
    ↓
Refinement loop (if needed)
    ↓
Final essay output
```

**Key Files:**
- `agentLanggraph.ipynb` - Main implementation
- Supporting utilities for research and writing

---

## Key Concepts Across All Modules

### 1. Agent Loop Pattern

All agents follow this core pattern:

```
Initialize State
    ↓
While not done:
    ├─ Agent thinks (LLM reasoning)
    ├─ Agent chooses action
    ├─ Agent takes action (tool call)
    ├─ Agent observes result
    └─ Update state
```

### 2. Tool Use

Tools are functions agents can call:

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    # Implementation
    return results
```

### 3. State Management

State is shared across all agent steps:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Message history
    documents: list  # Retrieved documents
    result: str  # Final result
```

### 4. Graph Construction

Agents are constructed as directed graphs:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

# Add nodes
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("reviewer", reviewer_node)

# Add edges
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "reviewer")
graph.add_edge("reviewer", END)

# Compile
agent = graph.compile()
```

---

## Learning Progression

### Beginner Path
1. Start with `agent_re+act.ipynb`
   - Understand basic agent loop
   - Learn tool definition
   - Get comfortable with state management

2. Move to `Langgraph_Components.ipynb`
   - Understand graph construction
   - Learn node and edge concepts
   - Practice with real tools (Tavily)

### Intermediate Path
3. Explore `Persisance_and_streaming.ipynb`
   - Add state persistence
   - Implement streaming outputs
   - Build long-running agents

4. Learn `human_in_loop.ipynb`
   - Add human approval workflows
   - Implement interruption patterns
   - Build interactive agents

### Advanced Path
5. Study `search_tavily.ipynb`
   - Integrate web search
   - Build research capabilities
   - Combine multiple information sources

6. Master `Essay_agent/`
   - Build complex multi-step workflows
   - Orchestrate multiple sub-agents
   - Implement refinement loops

---

## Common Patterns

### Pattern 1: Chain of Thought

```python
def chain_of_thought_node(state):
    # Generate reasoning steps
    reasoning = llm.invoke("Think step by step: " + state['query'])
    # Generate action
    action = llm.invoke(f"Based on: {reasoning}, what tool to use?")
    return {"reasoning": reasoning, "action": action}
```

### Pattern 2: Self-Reflection

```python
def reflection_node(state):
    # Review previous output
    critique = llm.invoke(f"Review this work: {state['draft']}")
    # Decide if refinement needed
    if "needs improvement" in critique:
        return {"draft": refine(state['draft']), "cycles": state['cycles'] + 1}
    return {"final": state['draft']}
```

### Pattern 3: Tool Router

```python
def router_node(state):
    # Decide which tool to use
    tool_choice = llm.invoke(
        f"Choose tool for: {state['query']}. Options: {available_tools}"
    )
    
    if tool_choice == "search":
        return {"next_node": "search_node"}
    elif tool_choice == "calculate":
        return {"next_node": "calculator_node"}
    # ...
```

---

## Performance Considerations

### Streaming for Better UX
```python
# Instead of waiting for complete response
result = agent.invoke(input)  # Blocks until done

# Stream events in real-time
for event in agent.stream(input):
    print(event)  # See thinking process as it happens
```

### Async for Scalability
```python
# Async execution for better performance
async for output in agent.astream(input):
    process(output)
```

### Persistence for Reliability
```python
# Save state checkpoint
config = {"configurable": {"thread_id": "session_123"}}
result = agent.invoke(input, config=config)

# Resume from checkpoint
result = agent.invoke(new_input, config=config)
```

---

## Dependencies

```
langchain==0.3.18
langchain-openai==0.3.5
langgraph==0.2.72
python-dotenv
tavily-python
openai
```

---

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-proj-..."
export TAVILY_API_KEY="tvly-..."
```

---

## Quick Start Examples

### Simple Tool-Using Agent
```python
from langgraph.prebuilt import create_react_agent

tools = [calculate, search, lookup_breed]

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools
)

result = agent.invoke({"messages": [("user", "What is 2 + 2?")]})
```

### Stateful Multi-Step Agent
```python
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("reviewer", reviewer)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "reviewer")
graph.add_edge("reviewer", END)

agent = graph.compile()
result = agent.invoke({"query": "Write an essay about AI"})
```

---

## Troubleshooting

### Issue: Tool not being called
**Check:**
- Tool is in agents tool list
- Tool schema is correct
- Tool returns expected type

### Issue: Agent loops infinitely
**Solution:**
- Add max_iterations parameter
- Implement proper termination conditions
- Check for circular edges in graph

### Issue: State not persisting
**Solution:**
- Pass checkpointer to compile(): `graph.compile(checkpointer=memory)`
- Use consistent thread_id in config
- Verify database file exists and is writable

---

## Advanced Topics

### Custom State Reducers
```python
def add_messages_reducer(left, right):
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages_reducer]
```

### Conditional Edges
```python
def should_continue(state):
    if state["quality_score"] > 0.8:
        return "end"
    else:
        return "refine"

graph.add_conditional_edges("reviewer", should_continue, {
    "end": END,
    "refine": "generator"
})
```

### Sub-Graphs
```python
# Reusable sub-graph
research_graph = StateGraph(State)
research_graph.add_node(...)
compiled_research = research_graph.compile()

# Use in main graph
main_graph.add_node("research", compiled_research)
```

---

## References

- [LangGraph Official Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Agent Concepts](https://python.langchain.com/docs/concepts/agents)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Tavily API Documentation](https://tavily.com/api)

---

## Learning Outcomes

After completing this module, you'll understand:

✅ How to build agents with LangGraph  
✅ ReAct reasoning and acting pattern  
✅ Graph-based workflow design  
✅ Tool integration and usage  
✅ State management across steps  
✅ Persistence and checkpointing  
✅ Streaming for real-time output  
✅ Human-in-the-loop workflows  
✅ Web search integration  
✅ Multi-agent orchestration  
✅ Complex reasoning pipelines  
✅ Production-ready agent patterns  

---

**Happy Learning! 🚀**

Start with `agent_re+act.ipynb` and progress through each notebook to build a comprehensive understanding of LangGraph agents.
