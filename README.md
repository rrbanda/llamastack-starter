# Agentic RAG System (FastAPI + LlamaStack)

A production-ready RAG (Retrieval Augmented Generation) system using LlamaStack agents for intelligent document ingestion and querying. Supports multiple ingestion methods including GitHub repositories, direct documents, and code examples.

## Features

- 🤖 **Agentic RAG** - Intelligent agents that autonomously search and synthesize responses
- 📚 **Multiple Ingestion Methods** - GitHub repos, direct documents, code snippets
- 🔍 **Semantic Search** - Vector-based document retrieval with embeddings
- 💬 **Streaming & Non-streaming** - Real-time and batch response modes
- 🛠️ **Code Analysis** - Specialized for analyzing code patterns and documentation
- ⚙️ **YAML Configuration** - Flexible configuration management

## Setup

### Prerequisites
- Python 3.10+
- Running LlamaStack server
- Access to embedding models

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Update `src/config.yaml` with your LlamaStack settings:

```yaml
llama:
  base_url: "http://your-llamastack-server.com"
  model_id: "meta-llama/Llama-3.1-8B-Instruct"
  instructions: "You are a helpful assistant."

vector_db:
  id: "simple_rag_vdb"
  provider: "faiss"
  embedding: "all-MiniLM-L6-v2"
  embedding_dimension: 384
  chunk_size: 512
```

## Run

### Start the Server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8096 --log-level info
```

### Health Check
```bash
curl -s http://localhost:8096/health
```

## API Endpoints

### 🔍 **Health Check**
```bash
GET /health
```

### 📥 **Document Ingestion**

#### Standard Documents (URLs, Files, Text)
```bash
curl -X POST "http://localhost:8096/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "url": "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/memory_optimizations.rst",
      "mime_type": "text/plain",
      "document_id": "pytorch-guide"
    }],
    "chunk_size_in_tokens": 512
  }'
```

#### GitHub Repository
```bash
curl -X POST "http://localhost:8096/ingest/github" \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/x2ansible/x2a-ui/tree/main/docs/iac",
    "file_extensions": [".md", ".py", ".yaml", ".yml", ".txt"],
    "chunk_size_in_tokens": 512
  }'
```

#### Direct Documents (Code, Pre-processed Content)
```bash
curl -X POST "http://localhost:8096/ingest/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "document_id": "python-optimization",
      "content": "def optimize_memory():\n    torch.utils.checkpoint.checkpoint(model, input)\n    return model.half()",
      "metadata": {
        "language": "python",
        "topic": "optimization"
      }
    }],
    "chunk_size_in_tokens": 256
  }'
```

### 💬 **Querying**

#### Ask Questions (Non-streaming)
```bash
curl -X POST "http://localhost:8096/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I optimize memory in PyTorch?"
  }'
```

#### Ask Questions (Streaming)
```bash
curl -N -X POST "http://localhost:8096/ask/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "question": "What is activation checkpointing?"
  }'
```

## Quick Start Example

### 1. Start the Server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8096
```

### 2. Ingest Sample Data
```bash
# Ingest PyTorch documentation
curl -X POST "http://localhost:8096/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "url": "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/memory_optimizations.rst",
      "mime_type": "text/plain"
    }]
  }'

# Ingest GitHub repository
curl -X POST "http://localhost:8096/ingest/github" \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/x2ansible/x2a-ui/tree/main/docs/iac"
  }'
```

### 3. Ask Questions
```bash
# Ask about PyTorch optimization
curl -X POST "http://localhost:8096/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key PyTorch memory optimization techniques?"
  }'

# Ask about code analysis
curl -X POST "http://localhost:8096/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I have this code: def train_model(model, data): loss.backward() - what optimizations can I apply?"
  }'
```

## Testing

Run the complete test suite:
```bash
chmod +x greenfield_deploy.sh
./greenfield_deploy.sh
```

## Architecture

### Components
- **FastAPI Server** - REST API endpoints
- **LlamaStack Client** - Integration with LlamaStack
- **RAG Agent** - Intelligent document search and response generation
- **Vector Database** - Semantic search with embeddings
- **Multiple Ingesters** - GitHub, direct, and URL-based document ingestion

### Workflow
1. **Document Ingestion** → RAG tool chunks and embeds content
2. **User Query** → Agent autonomously searches knowledge base
3. **Retrieval** → Semantic search finds relevant chunks
4. **Generation** → LLM synthesizes response from retrieved context

## Configuration

### Environment Variables
```bash
CONFIG_PATH=/path/to/config.yaml          # Config file location
LLAMA_BASE_URL=http://your-server.com     # Override LlamaStack URL
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct # Override model
VDB_ID=my_vector_db                       # Override vector DB ID
```

### YAML Configuration
See `src/config.yaml` for complete configuration options including:
- LlamaStack connection settings
- Vector database configuration
- Model sampling parameters
- Query templates

## Troubleshooting

### Common Issues

**Vector DB Connection Issues**
```bash
# Check health endpoint
curl http://localhost:8096/health
```

**Ingestion Failures**
- Ensure LlamaStack server is running
- Check GitHub URLs are accessible
- Verify file permissions for local files

**Query Failures**
- Check that documents are ingested successfully
- Ensure vector database has content
- Verify LlamaStack model is responding

### Logs
Server logs show detailed RAG workflow:
```
INFO:app:=== Agent Steps for Question: ... ===
INFO:app:Step 1: inference
INFO:app:Step 2: tool_execution
INFO:app:  → Tool called: knowledge_search
INFO:app:Step 3: inference
```

## Production Deployment

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8096"]
```

### Environment Setup
- Set up proper logging
- Configure health checks
- Use environment variables for sensitive data
- Monitor vector database storage

## License

MIT License - see LICENSE file for details.