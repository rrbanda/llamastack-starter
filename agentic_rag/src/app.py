# src/app.py
import os
import json
import uuid
import logging
import requests
from typing import List, Optional, Any, Dict
from pathlib import Path
import base64

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import AppConfig
from llama_stack_client import LlamaStackClient, Agent, APIConnectionError, AgentEventLogger
from llama_stack_client import RAGDocument
from llama_stack_client.types import UserMessage

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agentic RAG API", version="1.0.0")

# ---- Globals ----
_cfg: Optional[AppConfig] = None
_client: Optional[LlamaStackClient] = None
_agent: Optional[Agent] = None
_vector_db_ready: bool = False


# ---- Pydantic Schemas ----
class IngestDoc(BaseModel):
    url: str
    mime_type: str = "text/plain"
    document_id: Optional[str] = None
    metadata: Optional[dict] = None


class IngestRequest(BaseModel):
    documents: List[IngestDoc]
    chunk_size_in_tokens: Optional[int] = None


class DirectIngestRequest(BaseModel):
    documents: List[Dict[str, Any]]  # Changed from chunks to documents
    chunk_size_in_tokens: Optional[int] = None


class GitHubIngestRequest(BaseModel):
    github_url: str  # e.g., "https://github.com/x2ansible/x2a-ui/tree/main/docs/iac"
    file_extensions: List[str] = [".md", ".py", ".yaml", ".yml", ".txt", ".rst"]
    github_token: Optional[str] = None
    chunk_size_in_tokens: Optional[int] = None


class AskRequest(BaseModel):
    question: str


# ---- Helpers ----
def _sse(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _vector_db_exists(vector_db_id: str) -> Optional[bool]:
    try:
        if hasattr(_client.vector_dbs, "get"):
            _client.vector_dbs.get(vector_db_id=vector_db_id)
            return True
        lst = _client.vector_dbs.list()
        items = lst.get("data", lst) if isinstance(lst, dict) else lst
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    vid = it.get("vector_db_id") or it.get("id") or it.get("identifier")
                    if vid == vector_db_id:
                        return True
            return False
        return None
    except Exception as e:
        msg = str(e).lower()
        if "404" in msg or "not found" in msg:
            return False
        logger.warning("Vector DB existence check inconclusive: %s", e)
        return None


def _ensure_vector_db(cfg: AppConfig):
    global _vector_db_ready
    v = cfg.vector_db
    
    # Always try to register - LlamaStack will handle if it exists
    try:
        payload = {
            "vector_db_id": v.id,
            "embedding_model": v.embedding,
            "embedding_dimension": int(v.embedding_dimension),
            "provider_id": v.provider,
        }
        if v.provider_vector_db_id:
            payload["provider_vector_db_id"] = v.provider_vector_db_id
            
        _client.vector_dbs.register(**payload)
        logger.info("Vector DB '%s' registered successfully", v.id)
        _vector_db_ready = True
        
    except Exception as e:
        # Check if it's just because it already exists
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            logger.info("Vector DB '%s' already exists", v.id)
            _vector_db_ready = True
        else:
            logger.error("Failed to register Vector DB '%s': %s", v.id, e)
            raise


def _create_agent(cfg: AppConfig) -> Agent:
    """Create RAG agent with knowledge search capabilities."""
    instructions = """You are a helpful RAG assistant specialized in code and documentation analysis.

For technical questions, search the knowledge base using the knowledge_search tool, then provide a complete response.

Keep responses focused and concise. Always provide a helpful answer after tool calls."""

    tools = [
        {
            "name": "builtin::rag/knowledge_search",
            "args": {
                "vector_db_ids": [cfg.vector_db.id],
                "top_k": 3,  # Reduced from 5 to 3 for stability
            },
        }
    ]

    agent = Agent(
        client=_client,
        model=cfg.llama.model_id,
        instructions=instructions,
        tools=tools,
    )
    
    logger.info("Created RAG agent with model: %s", cfg.llama.model_id)
    return agent


def _extract_response_content(response) -> str:
    """Extract text content from agent response."""
    if hasattr(response, 'output_message'):
        output_msg = response.output_message
        if hasattr(output_msg, 'content') and output_msg.content:
            return output_msg.content
        elif isinstance(output_msg, str):
            return output_msg
    
    if hasattr(response, 'content') and response.content:
        return response.content
    
    if hasattr(response, 'message'):
        msg = response.message
        if hasattr(msg, 'content'):
            return msg.content
        elif isinstance(msg, str):
            return msg
    
    if isinstance(response, str):
        return response
    
    return str(response) if response else "No response received"


def _iter_stream_with_agent(agent: Agent, session_id: str, question: str):
    """Stream responses from the agent."""
    try:
        response_stream = agent.create_turn(
            messages=[{"role": "user", "content": question}],
            session_id=session_id,
            stream=True
        )

        accumulated_text = ""
        event_logger = AgentEventLogger()
        
        for event in event_logger.log(response_stream):
            try:
                event_content = None
                if hasattr(event, 'content'):
                    event_content = event.content
                elif hasattr(event, 'text'):
                    event_content = event.text
                
                if event_content and isinstance(event_content, str) and event_content.strip():
                    # Skip tool call JSON, send actual response text
                    if not (event_content.strip().startswith('{') and 'knowledge_search' in event_content):
                        accumulated_text += event_content
                        yield _sse({"type": "text", "content": event_content})

                # Check for completion
                event_type_name = type(event).__name__.lower()
                if 'turn' in event_type_name and 'complete' in event_type_name:
                    if accumulated_text.strip():
                        yield _sse({"type": "done", "full_response": accumulated_text.strip()})
                    else:
                        yield _sse({"type": "done", "full_response": "Response completed"})
                    return

            except Exception as e:
                logger.warning("Error processing event: %s", e)

        # Fallback completion
        if accumulated_text.strip():
            yield _sse({"type": "done", "full_response": accumulated_text.strip()})
        else:
            yield _sse({"type": "done", "full_response": "Response completed"})

    except Exception as e:
        logger.error("Error in agent streaming: %s", e, exc_info=True)
        yield _sse({"type": "error", "error": str(e)})


def _fetch_github_files(github_url: str, file_extensions: List[str], token: Optional[str] = None) -> List[Dict]:
    """Fetch files from GitHub repository and return as documents ready for RAG ingestion."""
    try:
        # Parse GitHub URL to extract owner, repo, and path
        # URL format: https://github.com/owner/repo/tree/branch/path
        parts = github_url.replace("https://github.com/", "").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL format")
        
        owner = parts[0]
        repo = parts[1]
        
        # Find branch and path
        if "tree" in parts:
            tree_index = parts.index("tree")
            branch = parts[tree_index + 1] if tree_index + 1 < len(parts) else "main"
            path = "/".join(parts[tree_index + 2:]) if tree_index + 2 < len(parts) else ""
        else:
            branch = "main"
            path = ""
        
        # GitHub API to get repository contents
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        if branch != "main":
            api_url += f"?ref={branch}"
        
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        contents = response.json()
        documents = []
        
        def process_item(item, current_path=""):
            if item["type"] == "file":
                file_path = f"{current_path}/{item['name']}" if current_path else item["name"]
                file_ext = Path(item["name"]).suffix.lower()
                
                if file_ext in file_extensions:
                    # Download file content
                    file_response = requests.get(item["download_url"])
                    if file_response.status_code == 200:
                        try:
                            content = file_response.text
                            documents.append({
                                "document_id": f"{owner}-{repo}-{file_path.replace('/', '-')}",
                                "content": content,
                                "mime_type": "text/plain",
                                "metadata": {
                                    "file_path": file_path,
                                    "file_name": item["name"],
                                    "file_extension": file_ext,
                                    "repository": f"{owner}/{repo}",
                                    "branch": branch,
                                    "github_url": item["html_url"],
                                    "size": item["size"],
                                    "source": "github"
                                }
                            })
                            logger.info(f"Fetched: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not decode {file_path}: {e}")
            
            elif item["type"] == "dir":
                # Recursively fetch directory contents
                dir_path = f"{current_path}/{item['name']}" if current_path else item["name"]
                dir_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}/{dir_path}"
                if branch != "main":
                    dir_api_url += f"?ref={branch}"
                
                try:
                    dir_response = requests.get(dir_api_url, headers=headers)
                    dir_response.raise_for_status()
                    dir_contents = dir_response.json()
                    
                    for sub_item in dir_contents:
                        process_item(sub_item, dir_path)
                except Exception as e:
                    logger.warning(f"Could not fetch directory {dir_path}: {e}")
        
        # Process all items
        if isinstance(contents, list):
            for item in contents:
                process_item(item)
        else:
            # Single file
            process_item(contents)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching GitHub files: {e}")
        raise


def _optimal_rag_insert(documents: List[Dict], chunk_size: Optional[int] = None) -> Dict:
    """
    PROPER RAG INGESTION: Let LlamaStack handle all metadata generation.
    Do not interfere with the RAG tool's internal metadata management.
    """
    try:
        # Convert to proper RAGDocument format - let RAG tool handle metadata
        rag_documents = []
        for doc in documents:
            # Only preserve user-provided metadata, let RAG tool generate technical metadata
            user_metadata = doc.get("metadata", {})
            
            rag_doc = RAGDocument(
                document_id=doc["document_id"],
                content=doc["content"],
                mime_type=doc.get("mime_type", "text/plain"),
                metadata=user_metadata  # Only user metadata, no technical fields
            )
            rag_documents.append(rag_doc)
        
        # Let RAG tool handle chunking, embedding, and ALL technical metadata
        _client.tool_runtime.rag_tool.insert(
            documents=rag_documents,
            vector_db_id=_cfg.vector_db.id,
            chunk_size_in_tokens=chunk_size or int(_cfg.vector_db.chunk_size)
        )
        
        return {
            "status": "success",
            "method": "rag_tool_insert",
            "documents_processed": len(rag_documents),
            "vector_db_id": _cfg.vector_db.id,
            "chunk_size": chunk_size or int(_cfg.vector_db.chunk_size)
        }
        
    except Exception as e:
        logger.error("RAG tool insertion error: %s", e, exc_info=True)
        raise


# ---- Lifecycle ----
@app.on_event("startup")
def startup():
    global _cfg, _client, _agent

    _cfg = AppConfig.load()
    _cfg.llama.base_url = _cfg.llama.base_url.rstrip("/")
    _client = LlamaStackClient(base_url=_cfg.llama.base_url)

    try:
        _ensure_vector_db(_cfg)
        _agent = _create_agent(_cfg)
        logger.info("RAG Agent initialized successfully")
    except Exception as e:
        logger.error("Startup error: %s", e, exc_info=True)
        _agent = None


@app.get("/health")
def health():
    """Health check endpoint."""
    ok = all([_cfg, _client, _agent, _vector_db_ready])
    return JSONResponse(
        {"status": "healthy" if ok else "unhealthy", "ready": ok}, 
        status_code=200 if ok else 503
    )


# ---- OPTIMAL INGESTION ENDPOINTS ----
@app.post("/ingest")
def ingest_documents(req: IngestRequest):
    """
    BEST PRACTICE: Ingest documents using RAG tool for optimal chunking and embeddings.
    Supports URLs, local files, and direct text content.
    """
    if not all([_client, _agent, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        documents = []
        for d in req.documents:
            doc_id = d.document_id or f"doc-{uuid.uuid4()}"
            
            # Handle different content types
            if d.url.startswith(('http://', 'https://')):
                content = d.url
            elif Path(d.url).exists():
                content = Path(d.url).read_text(encoding="utf-8")
            else:
                content = d.url  # Treat as direct text content
            
            documents.append({
                "document_id": doc_id,
                "content": content,
                "mime_type": d.mime_type,
                "metadata": d.metadata or {}
            })

        # Use optimal RAG insertion
        result = _optimal_rag_insert(documents, req.chunk_size_in_tokens)
        result["ingested"] = len(documents)
        
        logger.info("Successfully ingested %d documents using RAG tool", len(documents))
        return result
        
    except Exception as e:
        logger.error("Document ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/ingest/direct")
def ingest_direct_documents(req: DirectIngestRequest):
    """
    BEST PRACTICE: Ingest pre-prepared documents using RAG tool for optimal chunking.
    Perfect for code files, documentation, or any structured content.
    """
    if not all([_client, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Validate and prepare documents
        documents = []
        for i, doc in enumerate(req.documents):
            if "content" not in doc:
                raise HTTPException(status_code=400, detail=f"Document {i} missing 'content' field")
            
            # Ensure required fields
            document = {
                "document_id": doc.get("document_id", f"direct-doc-{i}"),
                "content": doc["content"],
                "mime_type": doc.get("mime_type", "text/plain"),
                "metadata": doc.get("metadata", {})
            }
            
            # Add source info to metadata
            document["metadata"]["source"] = "direct_upload"
            documents.append(document)

        # Use optimal RAG insertion
        result = _optimal_rag_insert(documents, req.chunk_size_in_tokens)
        result["ingested"] = len(documents)
        
        logger.info("Successfully ingested %d direct documents using RAG tool", len(documents))
        return result
        
    except Exception as e:
        logger.error("Direct document ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Direct ingestion failed: {e}")


@app.post("/ingest/github")
def ingest_github_repository(req: GitHubIngestRequest):
    """
    BEST PRACTICE: Ingest GitHub repository using RAG tool for optimal chunking.
    Fetches files and processes them with intelligent chunking and embedding.
    """
    if not all([_client, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Fetch files from GitHub
        logger.info(f"Fetching files from GitHub: {req.github_url}")
        github_documents = _fetch_github_files(req.github_url, req.file_extensions, req.github_token)
        
        if not github_documents:
            return {
                "status": "success",
                "message": "No files found matching the specified extensions",
                "ingested": 0
            }
        
        # Use optimal RAG insertion
        result = _optimal_rag_insert(github_documents, req.chunk_size_in_tokens)
        result["ingested"] = len(github_documents)
        result["source"] = "github"
        result["repository"] = req.github_url
        result["files"] = [doc["metadata"]["file_path"] for doc in github_documents]
        
        logger.info(f"Successfully ingested {len(github_documents)} GitHub files using RAG tool")
        return result
        
    except Exception as e:
        logger.error("GitHub ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"GitHub ingestion failed: {e}")


# ---- QUERY ENDPOINTS ----
@app.post("/ask")
def ask_question(req: AskRequest):
    """Ask a question to the RAG agent."""
    if not all([_client, _agent, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        session_id = _agent.create_session(f"session-{uuid.uuid4()}")
        
        response = _agent.create_turn(
            messages=[{"role": "user", "content": req.question}],
            session_id=session_id,
            stream=False
        )
        
        # Use built-in LlamaStack logging to show steps
        logger.info("=== Agent Steps for Question: %s ===", req.question)
        if hasattr(response, 'steps') and response.steps:
            # Log each step to show RAG tool usage
            for i, step in enumerate(response.steps):
                logger.info("Step %d: %s", i+1, getattr(step, 'step_type', 'unknown'))
                if hasattr(step, 'tool_calls') and step.tool_calls:
                    for tool_call in step.tool_calls:
                        tool_name = getattr(tool_call, 'tool_name', 'unknown')
                        logger.info("  → Tool called: %s", tool_name)
                        if hasattr(tool_call, 'result') and tool_call.result:
                            result_preview = str(tool_call.result)[:100] + "..." if len(str(tool_call.result)) > 100 else str(tool_call.result)
                            logger.info("  → Tool result preview: %s", result_preview)
        
        response_text = _extract_response_content(response)
        
        return {
            "question": req.question,
            "answer": response_text,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error("Question processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.post("/ask/stream")
def ask_question_stream(req: AskRequest):
    """Ask a question to the RAG agent with streaming response."""
    if not all([_client, _agent, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        session_id = _agent.create_session(f"stream-session-{uuid.uuid4()}")
        
        return StreamingResponse(
            _iter_stream_with_agent(_agent, session_id, req.question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        logger.error("Streaming error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming failed: {e}")


@app.post("/fix/metadata")
def fix_existing_metadata():
    """
    Fix existing data that's missing token_count metadata.
    This is a one-time migration function.
    """
    if not all([_client, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # This is a workaround - we can't directly update vector DB metadata
        # The cleanest solution is to clear and re-ingest
        return {
            "status": "migration_required",
            "message": "Please clear vector DB and re-ingest data with fixed metadata",
            "instructions": [
                "1. Clear/restart your vector database",
                "2. Re-run your ingestion scripts",
                "3. New data will have proper metadata"
            ]
        }
    except Exception as e:
        logger.error("Metadata fix error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metadata fix failed: {e}")