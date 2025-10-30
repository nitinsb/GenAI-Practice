# Essay Agent - Multi-Agent Essay Writing System

## Overview

The **Essay Agent** is a sophisticated multi-agent system built with **LangGraph** that automates the essay writing process. It orchestrates multiple specialized agents to research, plan, write, and iteratively refine essays based on user input.

## 🎯 Key Features

- **Multi-Agent Architecture**: Coordinated agents for planning, research, writing, and reflection
- **Iterative Refinement**: Automatic feedback loops with revision limits
- **Tavily Search Integration**: Real-time web search for research support
- **State Persistence**: SQLite checkpointing for conversation memory and thread management
- **Gradio Interface**: User-friendly GUI for interactive essay generation
- **Structured Output**: Type-safe state management with Pydantic

## 📋 Agent Workflow

The essay generation pipeline follows this flow:

```
Planner → Research Plan → Generate → [Decision: Continue?]
                            ↓               ↓
                         Reflect       Reflect
                            ↓               ↓
                       Research Critique ← (loop back if revisions < max)
```

### Agents

1. **Planner Agent**
   - Creates a high-level outline for the essay
   - Analyzes user's topic/request
   - Provides structure and guidance for the writing process

2. **Research Agent (Plan Phase)**
   - Generates search queries based on the essay topic
   - Retrieves relevant content from web using Tavily
   - Accumulates research material for the writer

3. **Generation Agent**
   - Writes 5-paragraph essays based on the plan and research
   - Revises drafts based on feedback from the reflection agent
   - Tracks revision numbers

4. **Reflection Agent**
   - Reviews the draft essay
   - Provides detailed critique and recommendations
   - Suggests improvements for length, depth, and style

5. **Research Agent (Critique Phase)**
   - Generates additional search queries based on feedback
   - Retrieves supplementary research material
   - Supports the revision process

## 🚀 Quick Start

### Installation

```bash
pip install langgraph langchain-openai langchain-community tavily-python gradio pydantic
```

### Environment Setup

Ensure your `.env` file contains:
```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Basic Usage

```python
from agentLanggraph import graph

# Simple essay generation
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "Write about the impact of AI on education",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
```

### Using the Gradio Interface

```python
from helper import ewriter, writer_gui

MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()
```

This launches an interactive web interface where you can:
- Enter essay topics
- Set maximum revisions
- View the essay generation process in real-time
- Get final essays with feedback

## 📦 State Definition

The `AgentState` TypedDict manages the workflow:

```python
class AgentState(TypedDict):
    task: str                  # User's essay request
    plan: str                  # Generated outline
    draft: str                 # Current essay draft
    critique: str              # Feedback from reflection agent
    content: List[str]         # Accumulated research material
    revision_number: int       # Current revision count
    max_revisions: int         # Maximum allowed revisions
```

## 🔧 System Prompts

### Planning Prompt
Creates structured outlines for essay topics with relevant notes and instructions for each section.

### Writing Prompt
Generates excellent 5-paragraph essays incorporating research material and user feedback. Supports iterative revisions based on critique.

### Reflection Prompt
Acts as a teacher grading essays, providing detailed critique with specific recommendations for:
- Length and depth
- Writing style
- Argument structure
- Evidence support

### Research Prompts
Intelligently generates 3 search queries max for optimal information gathering during both initial research and revision phases.

## 🔄 Workflow Configuration

Key parameters:

- **max_revisions**: Controls the maximum number of essay revisions (default: 2)
- **revision_number**: Tracks current revision (auto-incremented)
- **max_results**: Tavily search results per query (default: 2)

The workflow terminates when:
- `revision_number > max_revisions`, OR
- User manually ends the session

## 📊 Graph Visualization

The LangGraph workflow can be visualized:

```python
from IPython.display import Image
Image(graph.get_graph().draw_png())
```

This shows the complete state machine with all nodes and conditional edges.

## 🧠 Advanced Features

### Thread Management
Thread IDs enable concurrent, independent essay generation sessions:

```python
thread_1 = {"configurable": {"thread_id": "1"}}
thread_2 = {"configurable": {"thread_id": "2"}}

# Different topics in parallel
graph.stream({"task": "Topic 1", ...}, thread_1)
graph.stream({"task": "Topic 2", ...}, thread_2)
```

### State Persistence
SQLite checkpointing preserves conversation history across sessions:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")
```

## 📝 Files

- **agentLanggraph.ipynb**: Core agent implementation with all nodes and workflow
- **helper.py**: Utility functions and Gradio interface builder
- **temp_test_gradio.ipynb**: Testing and experimentation notebook

## 🎓 Learning Outcomes

By studying this module, you'll understand:
- ✅ Multi-agent orchestration with LangGraph
- ✅ State machines and conditional edges
- ✅ Tool integration for research (Tavily)
- ✅ Iterative refinement patterns
- ✅ Structured output with Pydantic
- ✅ Thread management and persistence
- ✅ Building production-ready Gradio interfaces

## 🔗 Dependencies

- **langgraph**: Multi-agent orchestration
- **langchain-openai**: OpenAI model integration
- **langchain-community**: Community tools (Tavily)
- **tavily-python**: Web search API client
- **gradio**: Web interface framework
- **pydantic**: Data validation

## 🚀 Future Enhancements

- [ ] Support for different essay formats (research paper, blog post, etc.)
- [ ] Custom system prompts per agent
- [ ] Parallel research queries execution
- [ ] Integration with academic databases
- [ ] Citation management
- [ ] Export to different formats (PDF, DOCX, etc.)
- [ ] Streaming token output for real-time viewing

## 📚 Related Modules

- `agent_re+act.ipynb`: ReAct pattern for simple agent reasoning
- `Langgraph_Components.ipynb`: Core LangGraph concepts
- `Persisance_and_streaming.ipynb`: State management and streaming patterns
- `human_in_loop.ipynb`: Human-in-the-loop control patterns

---

**Created**: October 2025  
**Framework**: LangGraph v0.1+  
**Python**: 3.9+
