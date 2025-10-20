# GenAI-Practice

A collection of practical implementations and experiments with Generative AI technologies, including agent development, RAG systems, and evaluation frameworks.

## 📁 Project Structure

```
GenAI-Practice/
├── evaluation/              # Agent evaluation and building frameworks
│   ├── Building_agents.ipynb
│   ├── skill_evalutions_for_router.ipynb
│   ├── tracing_Your_agents.ipynb
│   ├── L9.ipynb, L11.ipynb
│   ├── helper.py, utils.py
│   └── data/
│       └── Store_Sales_Price_Elasticity_Promotions_Data.parquet
│
├── googleADKandNeo4j/      # Google ADK with Neo4j integration
│   ├── googleadk.ipynb     # Main implementation notebook
│   ├── neo4j_for_adk.py   # Neo4j integration module
│   ├── requirements.txt    # Project dependencies
│   ├── helper.py
│   └── README.md           # Detailed project documentation
│
├── Pipeline/               # Pipeline implementations
│   ├── C1_W1.pdf
│   ├── C1_W2.pdf
│   └── C1_W3.pdf
│
└── RAG/                    # Retrieval Augmented Generation
    └── L1_Overview_of_Multimodality.ipynb
```

## 🚀 Featured Projects

### 1. Google ADK with Neo4j Integration
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

### 2. Agent Evaluation Framework
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

### 3. RAG Implementation
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

### AI/ML Frameworks
- **Google ADK 1.5.0** - Agent Development Kit
- **OpenAI GPT-4o** - Large Language Model
- **LiteLLM 1.73.6** - Multi-provider LLM interface
- **Arize Phoenix** - Agent tracing and evaluation

### Databases
- **Neo4j 5.28.1** - Graph database
- **Neo4j Aura** - Cloud-hosted Neo4j

### Supporting Libraries
- `python-dotenv` - Environment variable management
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `opentelemetry` - Observability and tracing

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

### Completed Modules
- Agent building fundamentals
- Neo4j graph database integration
- LLM integration patterns
- Agent evaluation and tracing
- Skill-based routing
- Multimodal RAG systems

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

**Last Updated:** October 20, 2025

**Note:** This is an active learning repository. New projects and experiments are added regularly.