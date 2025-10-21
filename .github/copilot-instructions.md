# Copilot Instructions for GenAI-Practice

This repository contains practical experiments with Generative AI, agent development, RAG systems, and graph database integration. Follow these guidelines to maximize AI coding agent productivity in this codebase.

## 🏗️ Architecture Overview
- **Major Components:**
  - `googleADKandNeo4j/`: Google ADK agent orchestration, Neo4j graph integration, OpenAI GPT-4o via LiteLLM
  - `evaluation/`: Agent evaluation, skill routing, tracing, and debugging workflows
  - `RAG/`: Multimodal retrieval-augmented generation
- **Data Flow:** Agents interact with Neo4j via custom tools, using environment variables for credentials. Notebooks orchestrate agent workflows and evaluation.
- **Why:** Modular structure enables rapid prototyping, agent evaluation, and integration of new LLMs or graph tools.

## 🛠️ Developer Workflows
- **Environment Setup:**
  - Use Python 3.12+ and Conda. Create `.env` in project root for API keys and Neo4j credentials.
  - Install dependencies:
    - `pip install -r googleADKandNeo4j/requirements.txt`
    - `pip install -r evaluation/req.txt`
- **Running Notebooks:**
  - Launch Jupyter and open main notebooks (e.g., `googleadk.ipynb`, `Building_agents.ipynb`).
- **Testing & Debugging:**
  - Use tracing notebooks (`tracing_Your_agents.ipynb`) and Phoenix for agent performance analysis.
  - For Neo4j connection issues, verify `.env` and wait 60s after Aura instance creation.
- **Security:**
  - Never commit `.env` or credentials. Regenerate keys if exposed.

## 📦 Project-Specific Patterns
- **Agent Construction:**
  - Agents are defined using Google ADK's `LlmAgent` and `LoopAgent` classes, with custom tool lists and callback functions.
  - Example: See `schema.ipynb` for agent orchestration and refinement loop.
- **Neo4j Integration:**
  - Use `neo4j_for_adk.py` for all graph database operations. Credentials are loaded from environment variables.
- **Tooling:**
  - Custom tools (e.g., `search_file`, `propose_node_construction`) are defined in notebooks and Python modules for agent workflows.
- **Session State:**
  - Agents pass and mutate session state dictionaries for context and feedback.

## 🔗 Integration Points
- **External Dependencies:**
  - Google ADK, Neo4j Python driver, LiteLLM, OpenAI API, Arize Phoenix
- **Cross-Component Communication:**
  - Agents communicate via shared state and callback contexts. Data files (CSV, Parquet) are loaded from `data/` directories.

## 📝 Conventions
- **Environment variables** for all secrets
- **Python 3.12** compatibility (requests >=2.31.0 required)
- **Notebooks** are the main entry point for workflows and experiments
- **Modular agent tools** for extensibility

## 📚 Key Files & Directories
- `googleADKandNeo4j/googleadk.ipynb`: Main agent implementation
- `googleADKandNeo4j/neo4j_for_adk.py`: Neo4j integration
- `googleADKandNeo4j/schema.ipynb`: Agent orchestration and schema refinement
- `evaluation/Building_agents.ipynb`: Agent construction patterns
- `evaluation/tracing_Your_agents.ipynb`: Debugging and tracing

---

For unclear or incomplete sections, please provide feedback so instructions can be improved for future AI agents.