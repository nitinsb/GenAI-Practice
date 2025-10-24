
# MCP Anthropic

This directory contains experiments and code for using Anthropic's Claude models and the Model Context Protocol (MCP), including building and running an MCP server with tool use.

---

## Setup Instructions

### 1. Python Environment
- Use Python 3.9+ (recommended: create a virtual environment)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Environment Variables
- Create a `.env` file in this directory with:
  ```
  ANTHROPIC_API_KEY=your-anthropic-api-key-here
  # Optional: other keys
  OPENAI_API_KEY=your-openai-api-key-here
  ```
- Never commit your `.env` file or API keys to version control.

### 3. Project Structure
- `L3.ipynb`: Main notebook for tool use and chatbot experiments
- `mcp_server.ipynb`: Guide and code for building an MCP server using FastMCP
- `mcp_project/`: Contains the actual MCP server code (e.g., `research_server.py`)
- `papers/`: Stores arXiv paper info
- `requirements.txt`: Python dependencies

### 4. Running the MCP Server
- The MCP server is implemented in `mcp_project/research_server.py` using FastMCP.
- To start the server, run:
  ```bash
  python mcp_project/research_server.py
  ```
- See `mcp_server.ipynb` for step-by-step instructions and explanations.

### 5. Running Notebooks
- Launch Jupyter and open `L3.ipynb` or `mcp_server.ipynb` for interactive experiments.

---

## Notes
- Anthropic Claude API is not free; you need a paid account and API key.
- For more details, see [Anthropic documentation](https://docs.anthropic.com/) and [MCP documentation](https://github.com/modelcontext/mcp).
