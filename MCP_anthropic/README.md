# MCP Anthropic

This directory contains comprehensive experiments and implementations for using Anthropic's Claude models with the Model Context Protocol (MCP), including server development, client implementation, and advanced chatbot features.

---

## 📚 Lessons and Notebooks

### Lesson 3: Tool Use Fundamentals
**File:** `L3.ipynb`
- **Concepts:** Tool use with Anthropic Claude API
- **Topics:**
  - Tool schema definition
  - Tool execution through Claude
  - Basic chatbot with tool use
  - arXiv paper search integration

### Lesson 4: Building MCP Servers
**File:** `mcp_server.ipynb`
- **Concepts:** Building MCP servers using FastMCP framework
- **Topics:**
  - FastMCP server initialization
  - Tool definition and decoration
  - Low-level vs high-level MCP implementation
  - Server execution and testing

### Lesson 5: Building MCP Clients
**File:** `mcp_client.ipynb`
- **Concepts:** Creating MCP clients to interact with servers
- **Topics:**
  - Client-server architecture
  - Async communication patterns
  - MCP_ChatBot class design
  - Tool discovery and invocation
  - Interactive chat loop

### Lessons 6+: Advanced MCP Features
**Files:** `MCP_to_referrnce_Servers.ipynb`, `Prompts_and_resources.ipynb`
- **Concepts:** Advanced MCP capabilities beyond tools
- **Topics:**
  - MCP Prompts discovery and execution
  - MCP Resources access
  - Multiple server connections
  - Configuration management
  - Advanced chatbot features (resource browsing, prompt execution)

---

## 🛠️ Project Structure

### Core Implementation Files

#### `mcp_project/`
The main project directory for MCP server and client implementations:

- **`research_server.py`** - MCP server implementation
  - FastMCP server for arXiv paper research
  - Tool definitions for paper search and retrieval
  - Prompt definitions for research guidance
  - Resource endpoints for paper management

- **`mcp_chatbot.py`** - Enhanced MCP chatbot client
  - Multi-server connection support
  - Tool, Prompt, and Resource discovery
  - Unified chatbot interface
  - Support for:
    - Tool invocation: `Just ask a question`
    - Resource access: `@<topic>` or `@folders`
    - Prompt execution: `/prompt <name> <args>`
    - Prompt listing: `/prompts`

- **`main.py`** - Example server launcher

- **`server_config.json`** - MCP server configuration
  - Defines multiple MCP servers
  - Configures server parameters and environment
  - Example servers: research (arXiv), filesystem, GitHub

- **`pyproject.toml`** - Project metadata and dependencies

- **`Dockerfile`** - Container configuration for deployment

#### `papers/` - Generated Data
Directory for storing paper information JSON files organized by research topic.

#### `images/` - Documentation
Visual diagrams and screenshots for documentation.

---

## 🚀 Quick Start

### 1. Setup Python Environment

```bash
cd MCP_anthropic/mcp_project

# Option A: Using uv (recommended)
uv sync
uv add anthropic python-dotenv nest_asyncio

# Option B: Using pip
pip install -r ../requirements.txt
pip install anthropic python-dotenv nest_asyncio
```

### 2. Configure Environment

Create a `.env` file in the `mcp_project` directory:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

Never commit this file! It's in `.gitignore`.

### 3. Configure MCP Servers

Create `server_config.json` in the `mcp_project` directory:

```json
{
  "mcpServers": {
    "research": {
      "command": "uv",
      "args": ["run", "research_server.py"],
      "env": null
    }
  }
}
```

### 4. Run the Chatbot

```bash
# Using uv
uv run mcp_chatbot.py

# Or using Python directly
python mcp_chatbot.py
```

### 5. Interact with the Chatbot

Once running, you can:

**Ask questions to use tools:**
```
Query: Search for 2 papers on "LLM interpretability"
```

**Browse resources:**
```
Query: @folders          # List paper topics
Query: @computers        # Access papers on computers topic
```

**Execute prompts:**
```
Query: /prompts          # List available prompts
Query: /prompt name arg1=value1 arg2=value2
```

**Exit:**
```
Query: quit
```

---

## 📦 Dependencies

### Core Requirements
- `anthropic` - Anthropic Claude API
- `python-dotenv` - Environment variable management
- `mcp` - Model Context Protocol SDK
- `arxiv` - arXiv paper search
- `nest_asyncio` - Async context management

### Optional
- `uv` - Fast Python package manager
- `docker` - For containerized deployment

---

## 🔧 Configuration

### Environment Variables
- `ANTHROPIC_API_KEY` - Your Anthropic API key (required)

### Server Configuration (`server_config.json`)
```json
{
  "mcpServers": {
    "server_name": {
      "command": "executable",
      "args": ["arg1", "arg2"],
      "env": null or {"VAR": "value"}
    }
  }
}
```

---

## 📚 Advanced Features

### Multi-Server Support
The chatbot can connect to multiple MCP servers simultaneously:
- Tools from all servers are aggregated
- Prompts from all servers are available
- Resources from all servers are accessible

### Prompt System
- Discover available prompts: `/prompts`
- Execute prompts with arguments: `/prompt name arg=value`
- Prompts can generate content for tool use

### Resource Access
- Browse resources: `@resource_name`
- Support for papers-related resources
- Extensible to other resource types

---

## 🔒 Security Best Practices

1. **Never commit `.env` file** - It's in `.gitignore`
2. **Regenerate keys if exposed** - Use your Anthropic dashboard
3. **Use environment variables** - For all sensitive data
4. **Restrict file permissions** - On config files with credentials
5. **Review server_config.json** - Before running untrusted servers

---

## 📄 File Organization

```
MCP_anthropic/
├── L3.ipynb                           # Tool use fundamentals
├── mcp_server.ipynb                   # MCP server building
├── mcp_client.ipynb                   # MCP client building
├── MCP_to_referrnce_Servers.ipynb     # Advanced: multiple servers
├── Prompts_and_resources.ipynb        # Advanced: prompts & resources
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── papers/                            # Generated paper data
├── images/                            # Documentation images
└── mcp_project/
    ├── research_server.py             # MCP server implementation
    ├── mcp_chatbot.py                 # Enhanced chatbot client
    ├── main.py                        # Server launcher
    ├── server_config.json             # MCP server configuration
    ├── pyproject.toml                 # Project metadata
    ├── Dockerfile                     # Container config
    └── papers/                        # Local paper storage
```

---

## 🎯 Use Cases

1. **Research Assistant** - Search and retrieve academic papers
2. **Multi-Tool Chatbot** - Leverage tools from multiple MCP servers
3. **Resource Explorer** - Browse and access resources through chat
4. **Prompt-Powered Chat** - Use prompts to guide chatbot behavior
5. **Production Deployment** - Run in Docker with proper configuration

---

## 🐛 Troubleshooting

### Issue: "Tool not found"
- Verify `server_config.json` is correctly configured
- Check server is running: `uv run research_server.py`
- Verify tool names match between server and client

### Issue: "Connection refused"
- Ensure MCP server is running
- Check port and command in `server_config.json`
- Verify environment variables are set

### Issue: "ANTHROPIC_API_KEY not set"
- Create `.env` file in `mcp_project/` directory
- Add `ANTHROPIC_API_KEY=your-key-here`
- Run from the correct directory

### Issue: "Module not found"
- Install dependencies: `uv sync` or `pip install -r requirements.txt`
- Activate virtual environment if using one

---

## 📚 Resources

### Documentation
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)

### Example MCP Servers
- [Official MCP Examples](https://github.com/modelcontextprotocol/servers)
- [Simple Chatbot Client](https://github.com/modelcontextprotocol/python-sdk/tree/main/examples/clients)

---

## 💡 Key Concepts

### MCP (Model Context Protocol)
A standardized protocol for AI applications to:
- Discover available **Tools** (functions)
- Discover available **Prompts** (templates)
- Access **Resources** (data/files)

### FastMCP
High-level framework for building MCP servers:
- Decorator-based tool definition: `@mcp.tool()`
- Automatic schema generation
- Simplified server setup

### Async Patterns
The chatbot uses async/await for:
- Non-blocking I/O operations
- Concurrent server communication
- Responsive user interaction

---

## 📝 Notes

- Different runs may produce different Claude responses (probabilistic)
- Paper search results depend on arXiv availability
- MCP servers should be stateless for horizontal scaling
- Configuration is persistent in `server_config.json`

---

**Last Updated:** October 24, 2025

**Version:** 2.0 (Multi-server, Prompts, Resources support)
