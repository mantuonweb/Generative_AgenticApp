# Simple Agentic App 🤖

A minimal AI agent using Ollama, LangChain, and FAISS - perfect for learning!

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama
- macOS/Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- Windows: Download from https://ollama.com/download

### 3. Pull AI Model
```bash
ollama pull llama2
```

### 4. Run the App
```bash
python app.py
```

## What's Included

- **app.py** - Basic chat agent with memory
- **agent_with_tools.py** - Agent that can use tools (calculator, time, search)
- **simple_rag.py** - Question answering with document search

## Requirements

- Python 3.8+
- 8GB RAM minimum
- Ollama installed

## Troubleshooting

**Ollama not found?**
```bash
ollama serve
```

**Import errors?**
```bash
pip install --upgrade langchain langchain-community
```

**Slow responses?**
Try a smaller model:
```bash
ollama pull llama2:7b
```

## Learn More

- LangChain: https://python.langchain.com/
- Ollama: https://ollama.com/
- FAISS: https://github.com/facebookresearch/faiss

Happy Learning! 🎉
```

## Package Versions Explained

```txt:requirements-detailed.txt
# LangChain - Framework for LLM apps
langchain==0.1.0              # Main framework
langchain-community==0.0.10   # Community integrations (Ollama, FAISS)

# Ollama - Local LLM runtime
ollama==0.1.6                 # Python client for Ollama

# Vector Database
faiss-cpu==1.7.4              # CPU version (use faiss-gpu for GPU)

# Utilities
python-dotenv==1.0.0          # Environment variables (optional)

# Alternative Vector Stores (optional)
# chromadb==0.4.22            # Alternative to FAISS
# pinecone-client==3.0.0      # Cloud vector DB
```

## Development Requirements

```txt:requirements-dev.txt
# Include main requirements
-r requirements.txt

# Development tools
jupyter==1.0.0                # For notebooks
ipython==8.12.0               # Better Python shell
black==23.12.0                # Code formatter
pytest==7.4.3                 # Testing
```

Now you have everything documented! Just run:

```bash
pip install -r requirements.txt