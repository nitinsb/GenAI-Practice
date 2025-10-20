# Google ADK with Neo4j Integration

A project demonstrating the integration of Google Agent Development Kit (ADK) with Neo4j graph database and OpenAI's GPT models.

## 📋 Overview

This project combines:
- **Google ADK** - Google's Agent Development Kit for building AI agents
- **Neo4j** - Graph database for storing and querying connected data
- **LiteLLM** - Unified interface for multiple LLM providers (using OpenAI)
- **OpenAI GPT-4o** - Large language model for agent intelligence

## 🛠️ Prerequisites

- Python 3.12
- Conda environment manager
- OpenAI API key
- Neo4j Aura database instance (or local Neo4j)

## 📦 Installation

### 1. Create and Activate Conda Environment

```bash
conda create -n studies python=3.12
conda activate studies
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- `google-adk==1.5.0` - Google Agent Development Kit
- `neo4j==5.28.1` - Neo4j Python driver
- `litellm==1.73.6` - LiteLLM for OpenAI integration
- `python-dotenv` - Environment variable management

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Neo4j Aura Database Credentials
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
AURA_INSTANCEID=your-instance-id
AURA_INSTANCENAME=your-instance-name
```

### Getting Your Credentials

#### OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Create a new secret key
3. Copy and paste it into your `.env` file

#### Neo4j Aura Database
1. Visit https://console.neo4j.io
2. Create a new Aura instance or use existing one
3. Wait 60 seconds after creation for the instance to be available
4. Copy the connection details to your `.env` file

## 📁 Project Structure

```
googleADKandNeo4j/
├── googleadk.ipynb          # Main Jupyter notebook
├── neo4j_for_adk.py        # Neo4j integration module
├── requirements.txt         # Python dependencies
├── helper.py               # Helper utilities
└── README.md               # This file
```

## 🚀 Usage

### Starting the Notebook

1. Activate your conda environment:
   ```bash
   conda activate studies
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook googleadk.ipynb
   ```

### Running the Code

The notebook is organized into several cells:

#### Cell 1: Import Libraries
Imports all necessary libraries including Google ADK, LiteLLM, and Neo4j drivers.

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from neo4j_for_adk import graphdb
```

#### Cell 2: Load Environment Variables
Loads your API keys and database credentials from the `.env` file.

#### Cell 3: Test OpenAI Connection
Creates a LiteLLM instance and tests the connection to OpenAI's GPT-4o model.

```python
MODEL_GPT = "openai/gpt-4o"
llm = LiteLlm(model=MODEL_GPT)
```

#### Cell 4: Initialize Neo4j
Imports and initializes the Neo4j graph database connection using the custom `neo4j_for_adk` module.

## 🔑 Key Components

### 1. Google ADK Agent
Google's Agent Development Kit provides:
- Agent creation and management
- Session handling (InMemorySessionService)
- Runner for agent execution
- Tool integration capabilities

### 2. LiteLLM Integration
- Unified interface for OpenAI models
- Support for GPT-4o and other models
- Simplified API calls through Google ADK

### 3. Neo4j Integration
The `neo4j_for_adk.py` module provides:
- Connection management to Neo4j Aura
- Graph database operations
- Integration with Google ADK tools

Key class:
```python
class Neo4jForADK:
    def __init__(self):
        # Reads credentials from environment variables
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME") or "neo4j"
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE") or "neo4j"
        
        self._driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
```

## 🐛 Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'litellm'
```bash
pip install litellm==1.73.6
```

#### 2. AuthenticationError: Incorrect API key
- Verify your OpenAI API key in the `.env` file
- Ensure the key starts with `sk-proj-`
- Regenerate the key if it was exposed or expired

#### 3. Neo4j Connection Error
- Wait 60 seconds after creating a new Aura instance
- Verify the URI format: `neo4j+s://instance-id.databases.neo4j.io`
- Check username and password are correct
- Ensure your IP is whitelisted in Neo4j Aura

#### 4. ModuleNotFoundError: No module named 'requests'
The old version of requests (2.3.0) is incompatible with Python 3.12. Fix:
```bash
pip uninstall requests
pip install requests>=2.31.0
```

## 🔒 Security Notes

### Important Security Practices

1. **Never commit `.env` file to git**
   - Already added to `.gitignore`
   - Contains sensitive API keys and passwords

2. **Rotate API keys if exposed**
   - If your OpenAI key was committed to git, regenerate it immediately
   - Same applies to Neo4j passwords

3. **Use environment variables**
   - All credentials should be in `.env` file
   - Never hardcode credentials in code

## 📚 Dependencies

Full list of installed packages:

- `google-adk==1.5.0` - Google Agent Development Kit
- `neo4j==5.28.1` - Neo4j Python Driver
- `litellm==1.73.6` - LiteLLM for OpenAI
- `python-dotenv>=1.0.0` - Environment variables
- `requests>=2.31.0` - HTTP library (upgraded for Python 3.12)
- `anyio>=4.11.0` - Async I/O
- `authlib>=1.6.0` - OAuth library
- `opentelemetry-api>=1.38.0` - Telemetry
- `starlette>=0.48.0` - ASGI framework
- `websockets>=15.0.1` - WebSocket support
- `httpx>=0.28.1` - HTTP client

## 🔄 Version History

### Latest Setup (October 20, 2025)
- ✅ Installed Google ADK 1.5.0
- ✅ Configured Neo4j Aura integration
- ✅ Set up LiteLLM with OpenAI GPT-4o
- ✅ Fixed Python 3.12 compatibility issues
- ✅ Upgraded requests library to 2.32.5
- ✅ Configured environment variables
- ✅ Moved `.env` to project root
- ✅ Added Neo4j credentials to `.env`

## 📖 Additional Resources

- [Google ADK Documentation](https://github.com/google/adk)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## 🤝 Contributing

This is a practice/learning project. Feel free to:
- Experiment with different agent configurations
- Add new Neo4j tools and queries
- Integrate additional LLM providers
- Enhance the agent capabilities

## 📄 License

This project is for educational and practice purposes.

## 🙏 Acknowledgments

- Google for the Agent Development Kit
- Neo4j for the graph database
- OpenAI for GPT models
- LiteLLM for unified LLM interface

---

**Note:** Remember to regenerate your OpenAI API key if it was previously exposed in git history. The current `.env` setup ensures your credentials stay secure going forward.
