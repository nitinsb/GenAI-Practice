# MCP Anthropic

This directory contains experiments and code for using Anthropic's Claude models with MCP (Model Context Protocol) and related tools.

## Setup Instructions

### 1. Python Environment
- Use Python 3.9+ (recommended: create a virtual environment)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Environment Variables
Create a `.env` file in this directory with the following content:

```
# Anthropic API Key
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: other keys
OPENAI_API_KEY=your-openai-api-key-here
```

- Never commit your `.env` file or API keys to version control.
- Regenerate keys if exposed.

### 3. Usage
- Run notebooks or scripts as needed:
  ```bash
  jupyter notebook L3.ipynb
  ```

## Project Contents
- `L3.ipynb`: Main notebook for tool use and chatbot experiments
- `requirements.txt`: Python dependencies
- `papers/`: Directory for storing arXiv paper info

## Notes
- Anthropic Claude API is not free; you need a paid account and API key.
- For more details, see [Anthropic documentation](https://docs.anthropic.com/).
