# Agentic RAG (FastAPI + LlamaStack)

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run
export CONFIG_PATH="$(pwd)/src/config.yaml"
python -m uvicorn src.app:app --host 0.0.0.0 --port 8096 --log-level info

## Health
curl -s http://localhost:8096/healthz

## Ingest (RAG Tool)
curl -s -X POST http://localhost:8096/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {
        "url":"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/memory_optimizations.rst",
        "mime_type":"text/plain"
      }
    ],
    "chunk_size_in_tokens": 512
  }'

## Ask (agent will call knowledge_search automatically)
curl -s -X POST http://localhost:8096/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How to optimize memory in PyTorch?"}'
