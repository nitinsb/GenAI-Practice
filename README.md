# GenAI-Practice

A comprehensive collection of practical implementations and experiments with Generative AI technologies, including advanced agent development, multi-tier memory systems, RAG implementations, and agent evaluation frameworks.

**Repository Owner:** [nitinsb](https://github.com/nitinsb)

## 📁 Project Structure

```
GenAI-Practice/
├── MemoryInLangGraph/           # Multi-tier memory systems with LangGraph
│   ├── README.md                # Parent documentation and learning guide
│   ├── Baseline/                # Email triage foundation
│   │   ├── Baseline_agent.ipynb
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── utils.py
│   │   ├── examples.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── SemanticMemory/          # Context-aware with semantic search
│   │   ├── Semantic_memory_agent.ipynb
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── utils.py
│   │   ├── examples.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── episodicMemory/          # Few-shot learning with examples
│   │   ├── epsiodicMemoryAgent.ipynb
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── utils.py
│   │   ├── examples.py
│   │   ├── requirements.txt
│   │   └── README.md
│   └── ProceduralMemory/        # Learned workflows and optimization
│       ├── ProceduralMemoryAgents.ipynb
│       ├── prompts.py
│       ├── schemas.py
│       ├── utils.py
│       ├── examples.py
│       ├── requirements.txt
│       └── README.md
│
├── evaluation/                  # Agent evaluation and building frameworks
│   ├── Building_agents.ipynb
│   ├── skill_evalutions_for_router.ipynb
│   ├── tracing_Your_agents.ipynb
│   ├── L9.ipynb, L11.ipynb
│   ├── helper.py, utils.py
│   └── data/
│       └── Store_Sales_Price_Elasticity_Promotions_Data.parquet
│
├── googleADKandNeo4j/           # Google ADK with Neo4j integration
│   ├── googleadk.ipynb          # Main implementation notebook
│   ├── neo4j_for_adk.py         # Neo4j integration module
│   ├── requirements.txt         # Project dependencies
│   ├── helper.py
│   ├── README.md                # Detailed project documentation
│   └── data/                    # CSV files for import
│
├── Pipeline/                    # Pipeline implementations
│   ├── C1_W1.pdf
│   ├── C1_W2.pdf
│   └── C1_W3.pdf
│
└── RAG/                         # Retrieval Augmented Generation
    └── L1_Overview_of_Multimodality.ipynb
```

## 🚀 Featured Projects

### 1. MemoryInLangGraph - Multi-Tier Memory Systems
**Location:** `MemoryInLangGraph/`

A comprehensive exploration of memory systems in LangGraph-based agents, demonstrating how agents learn and improve over time through different memory paradigms.

**Four-Module Progression:**

1. **Baseline** - Email triage foundation
   - Basic email classification (respond, ignore, notify)
   - LangGraph state management fundamentals
   - Simple ReAct agents with tool use
   - Tools: write_email, schedule_meeting, check_calendar_availability

2. **SemanticMemory** - Context-aware intelligence
   - Semantic memory for storing facts about contacts and topics
   - InMemoryStore with vector embeddings (text-embedding-3-small)
   - Memory search and management tools (langmem)
   - User-scoped memory namespaces

3. **episodicMemory** - Few-shot learning
   - Episodic memory for storing labeled examples
   - Few-shot learning with in-context learning
   - Vector similarity search for example retrieval
   - User-specific classifier training and personalization

4. **ProceduralMemory** - Learned workflows
   - Procedural memory for learned action sequences
   - Workflow optimization and skill development
   - Performance metrics and effectiveness tracking
   - Context-aware action planning

**Key Features:**
- Three-tier memory architecture (Episodic + Semantic + Procedural)
- User-scoped memory isolation for multi-user systems
- Vector embeddings for similarity-based retrieval
- Few-shot learning for classification improvement
- Procedure tracking and optimization
- Comprehensive documentation at each level

**Quick Start:**
```bash
cd MemoryInLangGraph
# Start with Baseline
jupyter notebook Baseline/Baseline_agent.ipynb
# Progress through modules
jupyter notebook SemanticMemory/Semantic_memory_agent.ipynb
jupyter notebook episodicMemory/epsiodicMemoryAgent.ipynb
jupyter notebook ProceduralMemory/ProceduralMemoryAgents.ipynb
```

See [MemoryInLangGraph/README.md](./MemoryInLangGraph/README.md) for comprehensive learning guide.

---

### 2. Google ADK with Neo4j Integration
**Location:** `googleADKandNeo4j/`

A comprehensive implementation combining:
- **Google Agent Development Kit (ADK)** - Build intelligent AI agents
- **Neo4j Graph Database** - Store and query connected data
- **OpenAI GPT-4o** - Power agents with state-of-the-art LLM
- **LiteLLM** - Unified interface for multiple LLM providers

**Key Features:**
- Agent creation with custom tools
- Neo4j Aura cloud database integration
- Cypher query execution through agents
- Session management and conversation handling
- Security best practices with environment variables

**Quick Start:**
```bash
cd googleADKandNeo4j
pip install -r requirements.txt
jupyter notebook googleadk.ipynb
```

See [googleADKandNeo4j/README.md](./googleADKandNeo4j/README.md) for detailed setup instructions.

---

### 3. Anthropic Claude with Model Context Protocol (MCP)
**Location:** `MCP_anthropic/`

Building intelligent chatbots with Anthropic Claude and MCP:
- **Claude Tool Use** - Claude models making decisions about tool use
- **MCP Server** - Building custom MCP servers with FastMCP
- **MCP Client** - Creating MCP clients to connect to servers
- **arXiv Integration** - Tool examples for searching academic papers
- **Async Communication** - Client-server architecture with async/await

**Key Files:**
- `L3.ipynb` - Tool use fundamentals with Claude
- `mcp_server.ipynb` - Building MCP servers using FastMCP
- `mcp_client.ipynb` - Creating MCP clients and chatbots
- `mcp_project/research_server.py` - MCP server implementation
- `mcp_project/mcp_chatbot.py` - MCP-based chatbot class

**Key Features:**
- Async tool invocation pattern
- arXiv paper search and retrieval
- Server-client communication pattern
- Tool schema generation
- Interactive chatbot loop

**Quick Start:**
```bash
cd MCP_anthropic/mcp_project
uv add anthropic python-dotenv nest_asyncio
uv run mcp_chatbot.py
```

See [MCP_anthropic/README.md](./MCP_anthropic/README.md) for detailed setup instructions.

---

### 4. Agent Evaluation Framework
**Location:** `evaluation/`

Tools and notebooks for:
- Building and testing AI agents
- Skill evaluation for routing logic
- Agent tracing and debugging
- Performance analysis with Phoenix
- Router skill evaluation

**Notebooks:**
- `Building_agents.ipynb` - Agent construction patterns
- `skill_evalutions_for_router.ipynb` - Router skill testing
- `tracing_Your_agents.ipynb` - Agent debugging and monitoring
- `L9.ipynb`, `L11.ipynb` - Learning modules

---

### 5. RAG Implementation
**Location:** `RAG/`

Multimodal Retrieval Augmented Generation:
- Overview of multimodality in RAG
- Vector embeddings and similarity search
- Document retrieval and generation

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.12** - Primary programming language
- **Jupyter Notebooks** - Interactive development environment
- **Conda** - Environment and package management

### AI/ML Frameworks & LLMs
- **LangGraph 0.2.72** - Graph-based agent orchestration
- **LangChain 0.3.18** - LLM framework and tools
- **LangMem 0.0.8** - Memory management for agents
- **OpenAI GPT-4o & GPT-4o-mini** - Advanced language models
- **Anthropic Claude 3.5 Sonnet** - Alternative LLM provider
- **Google ADK** - Agent Development Kit
- **LiteLLM** - Multi-provider LLM interface
- **Arize Phoenix** - Agent tracing and evaluation

### Databases & Storage
- **Neo4j 5.28.1** - Graph database
- **Neo4j Aura** - Cloud-hosted Neo4j
- **InMemoryStore** - Vector-based memory storage (LangGraph)

### Memory & Embeddings
- **OpenAI Embeddings (text-embedding-3-small)** - Vector embeddings
- **Langmem** - Episodic, semantic, and procedural memory tools
- **Vector Similarity Search** - Semantic retrieval

### Supporting Libraries
- `python-dotenv` - Environment variable management
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `pydantic` - Data validation and serialization
- `opentelemetry` - Observability and tracing
- `anthropic` - Anthropic Claude API
- `mcp` - Model Context Protocol
- `arxiv` - arXiv paper search
- `tavily-python` - Web search integration
- `requests` - HTTP client library

## ⚙️ Setup

### Prerequisites
- Python 3.12+
- Conda or Miniconda
- OpenAI API key
- Neo4j Aura account (for googleADKandNeo4j project)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nitinsb/GenAI-Practice.git
   cd GenAI-Practice
   ```

2. **Create environment variables:**
   
   Create a `.env` file in the project root:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your-openai-api-key-here
   
   # Neo4j Aura (for googleADKandNeo4j project)
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-password
   NEO4J_DATABASE=neo4j
   ```

3. **Install project dependencies:**
   ```bash
   # For Google ADK project
   cd googleADKandNeo4j
   pip install -r requirements.txt
   
   # For evaluation project
   cd ../evaluation
   pip install -r req.txt
   ```

## 🔒 Security

### Important Notes
- ✅ `.env` files are included in `.gitignore`
- ✅ Never commit API keys or passwords
- ✅ Regenerate exposed credentials immediately
- ✅ Use environment variables for all sensitive data

### Git History Cleanup
If you accidentally committed sensitive data:
1. Regenerate all exposed credentials
2. Use `git filter-branch` to remove from history
3. Force push to update remote repository

## 📚 Learning Resources

### Module Structure & Progression

**MemoryInLangGraph (Newest - Multi-Tier Memory Systems):**
1. Baseline - Email triage foundation
2. SemanticMemory - Context-aware intelligence
3. episodicMemory - Few-shot learning & personalization
4. ProceduralMemory - Learned workflows & optimization

**Other Completed Modules:**
- Agent building fundamentals
- Neo4j graph database integration
- LLM integration patterns
- Agent evaluation and tracing
- Skill-based routing
- Multimodal RAG systems
- Anthropic Claude tool use
- Model Context Protocol (MCP) servers and clients

### Key Learning Outcomes

After completing this repository, you'll understand:
- ✅ How to build agentic email systems with LangGraph
- ✅ How different memory types enhance AI agents
- ✅ How to implement semantic memory with embeddings
- ✅ How few-shot learning improves classification
- ✅ How to track and optimize learned procedures
- ✅ How to build user-scoped, personalized AI systems
- ✅ How to combine multiple memory systems effectively
- ✅ Best practices for production agent deployment
- ✅ Multi-provider LLM integration
- ✅ Graph database integration with AI agents
- ✅ Agent evaluation and monitoring techniques

## 🎯 Use Cases

This repository demonstrates practical implementations for:
- **Conversational AI Agents** - Building intelligent chatbots
- **Knowledge Graph Integration** - Connecting LLMs with graph databases
- **Agent Orchestration** - Managing multi-agent systems
- **RAG Systems** - Retrieval-augmented generation
- **Agent Evaluation** - Testing and monitoring agent performance

## 📊 Data

Sample datasets included:
- `Store_Sales_Price_Elasticity_Promotions_Data.parquet` - Sales analysis data

## 🤝 Contributing

This is a personal learning and practice repository. Feel free to:
- Fork and experiment
- Suggest improvements via issues
- Share your own implementations

## 📄 License

This project is for educational and practice purposes.

## 🙏 Acknowledgments

- Google for Agent Development Kit
- Neo4j for graph database technology
- OpenAI for GPT models
- Arize for Phoenix tracing tools
- The open-source community

## 📧 Contact

**Repository Owner:** nitinsb
**Repository:** [GenAI-Practice](https://github.com/nitinsb/GenAI-Practice)

---

**Last Updated:** November 6, 2025

### Recent Updates (November 2025)

#### Major: MemoryInLangGraph Module Launch
- **New:** Complete four-module progression for multi-tier memory systems
  - `Baseline/` - Email triage foundation
  - `SemanticMemory/` - Semantic memory with embeddings
  - `episodicMemory/` - Few-shot learning from examples
  - `ProceduralMemory/` - Learned workflows and optimization
- Added comprehensive parent README with learning guide
- Each module includes detailed documentation, example notebooks, and supporting utilities
- Demonstrates episodic, semantic, and procedural memory architectures
- Shows user-scoped memory isolation for multi-user systems
- Includes performance tracking and procedure optimization

#### Documentation Improvements
- Updated main README with full project overview
- Added MemoryInLangGraph learning progression guide
- Created module-specific READMEs with usage examples
- Documented three-tier memory integration patterns
- Added troubleshooting and best practices guides

#### Technology Updates
- Integrated **LangMem 0.0.8** for memory tool creation
- Integrated **LangGraph InMemoryStore** for vector-based storage
- Added **OpenAI text-embedding-3-small** for semantic similarity search
- Configured **GPT-4o-mini** for fast classification
- Configured **GPT-4o** for full reasoning in response generation

#### Previous Updates
- Added `MCP_anthropic/` directory with Anthropic Claude and Model Context Protocol implementations
- Implemented `mcp_server.ipynb`: FastMCP server building patterns
- Implemented `mcp_client.ipynb`: MCP client and chatbot class
- Added `googleADKandNeo4j/user_intent.ipynb`: User intent agent orchestration with Google ADK, Neo4j, and OpenAI GPT-4o
- Enhanced agent evaluation and tracing workflows in `evaluation/`

**Note:** This is an active learning repository. New projects and experiments are added regularly as learning progresses.