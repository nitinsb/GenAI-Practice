# MemoryInLangGraph - Baseline Email Agent

## 📧 Overview

The **Baseline Email Agent** is a foundational example of an intelligent email management system built with **LangGraph** and **LangChain**. This module demonstrates how to construct a multi-stage agent that automatically classifies, routes, and responds to emails using large language models (LLMs) and structured reasoning.

The system mimics a real-world email assistant for a senior software engineer, automating routine email triage and enabling smart response generation through tool integration.

## 🎯 Core Functionality

### Three-Stage Email Processing Pipeline

1. **Triage & Routing** 📊
   - Classifies incoming emails into three categories: `respond`, `ignore`, `notify`
   - Uses structured output with Pydantic models for reliable classification
   - Routes emails to appropriate handlers based on content analysis

2. **Tool-Enabled Response** 🛠️
   - Drafts intelligent email replies via `write_email` tool
   - Schedules meetings with `schedule_meeting` tool
   - Checks calendar availability with `check_calendar_availability` tool
   - All tools are optional—agent decides what to use based on context

3. **Multi-Agent Orchestration** 🎭
   - Combines triage and response agents in a unified workflow
   - Uses LangGraph's `StateGraph` for clear control flow
   - Leverages conditional edges to route emails appropriately

## 📁 Project Structure

```
Baseline/
├── Baseline_agent.ipynb          # Main notebook with agent implementation
├── prompts.py                    # System prompts for different agent roles
├── schemas.py                    # Pydantic models for structured outputs
├── helper.py                     # Utility functions and helpers
├── examples.py                   # Example test cases
├── utils.py                      # General utilities
├── requirements.txt              # Python dependencies
├── img/                          # Architecture diagrams and images
└── README.md                     # This file
```

## 🔧 Key Components

### 1. Router Schema (`schemas.py`)

```python
class Router(BaseModel):
    reasoning: str
    classification: Literal["ignore", "respond", "notify"]
```

The router uses **structured output** to ensure reliable email classification with explicit reasoning for each decision.

### 2. System Prompts (`prompts.py`)

- **`triage_system_prompt`**: Guides the LLM to classify emails based on user-defined rules
- **`triage_user_prompt`**: Formats the incoming email for analysis
- **`agent_system_prompt`**: Instructions for the response agent when handling emails

### 3. Tools Available

All tools are LangChain tools with full documentation:

```python
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""

@tool
def schedule_meeting(attendees: list[str], subject: str, duration_minutes: int, preferred_day: str) -> str:
    """Schedule a calendar meeting."""

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
```

### 4. Agent State

```python
class State(TypedDict):
    email_input: dict                    # Incoming email data
    messages: Annotated[list, add_messages]  # Conversation history
```

Uses `add_messages` reducer to maintain clean message history across turns.

## 📊 Workflow Architecture

```
START
  ↓
[Triage Router Node]
  ├─→ Classification: "respond"  → "response_agent" → [React Agent]
  │                                                        ↓
  │                                                   Tool Execution
  │                                                        ↓
  │                                                   END
  ├─→ Classification: "ignore"   → END
  └─→ Classification: "notify"   → END
```

### Email Classification Rules

**Ignore**: Marketing newsletters, spam emails, mass company announcements

**Notify**: Team member out sick, build system notifications, project status updates

**Respond**: Direct questions from team members, meeting requests, critical bug reports

## 🚀 Quick Start

### 1. Installation

```bash
cd MemoryInLangGraph/Baseline
pip install -r requirements.txt
```

**Requirements Include:**
- `langchain==0.3.18` - Core framework
- `langchain-openai==0.3.5` - OpenAI integration
- `langchain-anthropic==0.3.7` - Anthropic integration
- `langgraph==0.2.72` - Graph-based agent orchestration
- `langmem==0.0.8` - Language memory utilities
- `python-dotenv==1.0.1` - Environment variable management

### 2. Environment Setup

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Baseline Agent

```python
# In Jupyter/IPython, run all cells in Baseline_agent.ipynb

# Test with a marketing email (should be ignored)
email_input = {
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount!",
    "email_thread": "..."
}
response = email_agent.invoke({"email_input": email_input})

# Test with a direct question (should generate response)
email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": "..."
}
response = email_agent.invoke({"email_input": email_input})
```

## 🧑‍💼 User Profile Configuration

The agent is configured for a specific user persona:

```python
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers"
}
```

### Customization

Modify the profile to match your use case:
- Change `name` and `full_name` for different users
- Update `user_profile_background` with actual context
- Adjust `triage_rules` to match your specific classification needs
- Modify tool implementations to integrate with real systems

## 🔄 Agent Execution Flow

### Step 1: Email Input
```python
email_input = {
    "author": "sender@example.com",
    "to": "recipient@example.com", 
    "subject": "Topic",
    "email_thread": "Email body content"
}
```

### Step 2: Triage Router Analysis
- Analyzes email content against classification rules
- Returns structured output with `reasoning` and `classification`
- Determines routing decision

### Step 3: Conditional Routing
- **If "respond"**: Pass to ReAct agent for response generation
- **If "ignore" or "notify"**: Mark as processed and end

### Step 4: Response Generation (if needed)
- ReAct agent with tool access generates response
- May invoke multiple tools (write email, schedule meeting, check calendar)
- Returns final message to user

## 📚 Advanced Features

### 1. Structured Output with Pydantic

The agent uses Pydantic for guaranteed output format:
```python
llm_router = llm.with_structured_output(Router)
result = llm_router.invoke([...])
# Result is guaranteed to be a Router instance with valid classification
```

### 2. Tool Integration

Tools are automatically available to the ReAct agent:
```python
tools = [write_email, schedule_meeting, check_calendar_availability]
agent = create_react_agent("openai:gpt-4o", tools=tools, prompt=create_prompt)
```

### 3. State Management

Message history is preserved across multiple email processing cycles:
```python
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]  # Reducer combines messages
```

### 4. Custom Prompt Engineering

Prompts are dynamically formatted with user context:
```python
system_prompt = triage_system_prompt.format(
    full_name=profile["full_name"],
    name=profile["name"],
    user_profile_background=profile["user_profile_background"],
    triage_no=prompt_instructions["triage_rules"]["ignore"],
    triage_notify=prompt_instructions["triage_rules"]["notify"],
    triage_email=prompt_instructions["triage_rules"]["respond"],
    examples=None
)
```

## 🎓 Learning Outcomes

By studying this module, you'll understand:

✅ **Multi-Agent Patterns**: How to compose multiple agents for complex workflows  
✅ **Structured Output**: Using Pydantic with LLMs for reliable classification  
✅ **Tool Calling**: Enabling LLMs to use functions for extended capabilities  
✅ **State Management**: Maintaining conversation context across agent turns  
✅ **Conditional Routing**: Using LangGraph's conditional edges for decision logic  
✅ **Prompt Engineering**: Crafting effective system and user prompts  
✅ **ReAct Pattern**: Combining reasoning and acting for autonomous agent behavior  
✅ **LangGraph Integration**: Building production-ready agent workflows  

## 🔍 Testing & Examples

### Test Case 1: Marketing Email (Ignore)
```
Subject: 🔥 EXCLUSIVE OFFER: Limited Time Discount!
Expected: Classification = "ignore"
Result: 🚫 Classification: IGNORE
```

### Test Case 2: Technical Question (Respond)
```
Subject: Quick question about API documentation
Expected: Classification = "respond"
Result: 📧 Classification: RESPOND → [Agent generates reply]
```

### Test Case 3: Team Status (Notify)
```
Subject: Build system down for maintenance
Expected: Classification = "notify"
Result: 🔔 Classification: NOTIFY
```

## 🛠️ Module Dependencies

| Module | Purpose |
|--------|---------|
| `prompts.py` | System prompts for triage and response generation |
| `schemas.py` | Pydantic models for structured outputs |
| `helper.py` | Utility functions for common operations |
| `examples.py` | Example test cases and templates |
| `utils.py` | General utility functions |

## 🚀 Extensions & Improvements

### Possible Enhancements

1. **Persistence**: Add database storage for email history
2. **Real Tools**: Replace placeholder tools with actual email/calendar APIs
3. **Memory**: Integrate long-term memory of user preferences
4. **Multi-Model**: Support Claude, Gemini, or local models via LiteLLM
5. **Streaming**: Enable token-level streaming for real-time responses
6. **Custom Rules**: Learn classification rules from user feedback
7. **Attachment Handling**: Process email attachments
8. **Threading**: Handle email conversations and reply chains
9. **Batching**: Process multiple emails in parallel
10. **Analytics**: Track classification accuracy and tool usage

## 📝 Configuration

### User Settings
Modify these in the notebook to customize behavior:

```python
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}
```

## 🔐 Security Considerations

- API keys are loaded from `.env` via `python-dotenv`
- Never commit `.env` files to version control
- Tool implementations should validate inputs before execution
- Consider rate limiting for API calls
- Implement logging for audit trails

## 📖 Key Concepts

### Triage
Automatic classification of incoming emails into actionable categories without human intervention.

### Structured Output
Using Pydantic models to guarantee LLM outputs match expected schemas, enabling reliable downstream processing.

### ReAct Pattern
Combining "Reasoning" (thinking through steps) and "Acting" (using tools) in an agent loop.

### State Management
Maintaining context across multiple agent interactions using TypedDict and message reducers.

### Tool Calling
Giving LLMs access to functions they can invoke to gather information or take actions.

## 🤝 Integration with Memory Systems

This baseline agent is designed to be extended with memory capabilities:

- **Short-term**: Message history within a conversation
- **Long-term**: Persistent user preferences and email history
- **Semantic**: Remembering similar emails and responses

See related modules in `MemoryInLangGraph/` for advanced memory patterns.

## 📞 Support & Troubleshooting

### Common Issues

**ImportError: cannot import init_chat_model**
- Solution: Use `from langchain.chat_models import init_chat_model` (correct path)

**Missing API Key**
- Solution: Create `.env` file with `OPENAI_API_KEY=your_key`

**Tool Not Found**
- Solution: Ensure all tools are imported and passed to `create_react_agent`

## 🔗 Related Modules

- `MemoryInLangGraph/Advanced/`: Advanced memory patterns
- `Langgraph-agents/`: Additional LangGraph examples
- `AgenticAI/`: More agentic AI patterns

## 📄 License & Attribution

This module is part of the GenAI-Practice repository demonstrating LangChain and LangGraph capabilities.

---

**Last Updated**: November 2025  
**Framework**: LangChain 0.3.18 + LangGraph 0.2.72  
**Python**: 3.11+  
**Status**: Production-Ready Baseline
