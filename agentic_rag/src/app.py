# src/app.py
import os
import json
import uuid
import logging
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import AppConfig
from llama_stack_client import LlamaStackClient, APIConnectionError
from llama_stack_client.types import UserMessage

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agentic RAG (FastAPI + LlamaStack, low-level API)", version="2.0.0")

# ---- Globals ----
_cfg: Optional[AppConfig] = None
_client: Optional[LlamaStackClient] = None
_agent_id: Optional[str] = None  # existing or newly created
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


class AskRequest(BaseModel):
    question: str


# ---- Helpers ----
def _sse(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _extract_text_blocks(blocks) -> str:
    out = []
    if isinstance(blocks, list):
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, dict):
                    out.append(t.get("data") or t.get("value") or "")
                elif isinstance(t, str):
                    out.append(t)
    return "".join([x for x in out if x])


def _deep_find_text(node) -> Optional[str]:
    # direct str
    if isinstance(node, str):
        s = node.strip()
        return s if s else None

    # blocks at top-level
    txt = _extract_text_blocks(node) if isinstance(node, list) else None
    if txt:
        return txt

    if isinstance(node, dict):
        # Prefer delta
        d = node.get("delta")
        if isinstance(d, dict):
            t = d.get("text") or d.get("output_text")
            if isinstance(t, str) and t.strip():
                return t
            txt = _extract_text_blocks(d.get("content"))
            if txt:
                return txt

        # message snapshot
        m = node.get("message")
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
            txt = _extract_text_blocks(c)
            if txt:
                return txt

        # generic textual keys
        for k in ("text", "output_text", "data", "value", "content"):
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v
            txt = _extract_text_blocks(v) if isinstance(v, list) else None
            if txt:
                return txt
            if isinstance(v, (dict, list)):
                found = _deep_find_text(v)
                if found:
                    return found

        # fallback recurse
        for v in node.values():
            found = _deep_find_text(v)
            if found:
                return found

    if isinstance(node, list):
        for v in node:
            found = _deep_find_text(v)
            if found:
                return found

    return None


def _vector_db_exists(vector_db_id: str) -> Optional[bool]:
    try:
        # Prefer a direct get()
        if hasattr(_client.vector_dbs, "get"):
            _client.vector_dbs.get(vector_db_id=vector_db_id)
            return True
        # Fall back to list()
        lst = _client.vector_dbs.list()
        items = lst.get("data", lst) if isinstance(lst, dict) else lst
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    vid = it.get("vector_db_id") or it.get("id")
                    if vid == vector_db_id:
                        return True
            return False
        return None
    except Exception as e:
        msg = str(e).lower()
        if "404" in msg or "not found" in msg:
            return False
        if "400" in msg or "bad request" in msg:
            return None
        logger.warning("Vector DB existence check inconclusive: %s", e)
        return None


def _ensure_vector_db(cfg: AppConfig):
    global _vector_db_ready
    v = cfg.vector_db
    exists = _vector_db_exists(v.id)
    if exists is True:
        logger.info("Vector DB '%s' already exists; skipping register.", v.id)
        _vector_db_ready = True
        return
    try:
        payload = {
            "vector_db_id": v.id,
            "embedding_model": v.embedding,
            "embedding_dimension": int(v.embedding_dimension),
            "provider_id": v.provider,  # e.g., 'faiss'
        }
        if v.provider_vector_db_id:
            payload["provider_vector_db_id"] = v.provider_vector_db_id
        _client.vector_dbs.register(**payload)
        logger.info(
            "Registered Vector DB '%s' (provider=%s, embedding=%s, dim=%s).",
            v.id, v.provider, v.embedding, v.embedding_dimension
        )
        _vector_db_ready = True
    except Exception as e:
        logger.error("Failed to register Vector DB '%s': %s", v.id, e)
        raise


def _ensure_agent(cfg: AppConfig) -> str:
    """
    Use existing agent if AGENT_ID is set; otherwise create one
    with builtin::rag/knowledge_search configured for our vector DB.
    """
    agent_id = os.getenv("AGENT_ID")
    if agent_id:
        logger.info("Using existing agent: %s", agent_id)
        return agent_id

    # Create a minimal agent that always calls knowledge_search first
    tools = [
        {
            "name": "builtin::rag/knowledge_search",
            "args": {
                "vector_db_ids": [cfg.vector_db.id],
                "query_config": {
                    "chunk_size_in_tokens": int(cfg.vector_db.chunk_size),
                    "chunk_overlap_in_tokens": 0,
                    "chunk_template": cfg.query.chunk_template,
                },
            },
        }
    ]
    instructions = (
        (cfg.llama.instructions or "You are a helpful assistant.")
        + "\n\nCRITICAL:\n"
        "- Always call the tool `builtin::rag/knowledge_search` BEFORE answering a user question.\n"
        "- Use the retrieved context to ground your answer. If nothing is retrieved, say so clearly.\n"
        "- Keep answers concise and faithful to the context."
    )

    # sampling params at agent level
    s = cfg.sampling
    if float(s.temperature) > 0:
        strategy = {"type": "top_p", "temperature": float(s.temperature), "top_p": float(s.top_p)}
    else:
        strategy = {"type": "greedy"}
    sampling_params = {"strategy": strategy, "max_tokens": int(s.max_tokens)}

    resp = _client.agents.create(
        agent_name=f"rag-agent-{uuid.uuid4()}",
        model=cfg.llama.model_id,
        instructions=instructions,
        tools=tools,
        sampling_params=sampling_params,
    )
    created_id = resp.agent_id if hasattr(resp, "agent_id") else getattr(resp, "id", None)
    if not created_id:
        raise RuntimeError("Could not determine created agent_id from response.")
    logger.info("Created agent: %s", created_id)
    return created_id


def _create_session(agent_id: str, name_prefix: str = "rag-session") -> str:
    sess = _client.agents.session.create(
        agent_id=agent_id,
        session_name=f"{name_prefix}-{uuid.uuid4()}",
    )
    sid = getattr(sess, "session_id", None)
    if not sid:
        raise RuntimeError("No session_id in agents.session.create() response.")
    return sid


def _iter_stream(agent_id: str, session_id: str, question: str, debug: bool):
    """
    Stream events directly from agents.turn.create(stream=True).
    Works with event payloads shaped like:
      - event.payload.delta.text / output_text / content blocks
      - event.payload.message.content
      - event.payload.event_type in {step_complete, turn_complete}
    """
    try:
        gen = _client.agents.turn.create(
            agent_id=agent_id,
            session_id=session_id,
            messages=[UserMessage(role="user", content=question)],
            stream=True,
        )

        dumped_keys = False
        final_emitted = False

        for chunk in gen:
            ev = getattr(chunk, "event", None)
            if not ev:
                # dict fallback if any
                if isinstance(chunk, dict):
                    d_ev = chunk.get("event", {})
                    payload = d_ev.get("payload") if isinstance(d_ev, dict) else None
                else:
                    yield _sse({"type": "raw", "hint": "unknown_chunk"})
                    continue
            else:
                payload = getattr(ev, "payload", None)

            # Optionally show top-level payload keys once for debugging
            if debug and not dumped_keys and isinstance(payload, dict):
                keys = list(payload.keys())[:16]
                yield _sse({"type": "debug_payload_keys", "keys": keys})
                dumped_keys = True

            # 1) Try to extract incremental text
            text = _deep_find_text(payload)
            if text:
                yield _sse({"type": "model_delta", "text": text})

            # 2) Handle lifecycle events
            etype = None
            if isinstance(payload, dict):
                etype = payload.get("event_type") or payload.get("type")
            if etype == "step_complete":
                yield _sse({"type": "step_complete"})
            elif etype == "turn_complete":
                # Try to emit the final LLM message if present
                turn = payload.get("turn") if isinstance(payload, dict) else None
                if isinstance(turn, dict):
                    # Various places where final content may live
                    final_text = (
                        _deep_find_text(turn.get("output_message")) or
                        _deep_find_text(turn.get("message")) or
                        _deep_find_text(turn.get("content")) or
                        _deep_find_text(turn)
                    )
                    if final_text and not final_emitted:
                        yield _sse({"type": "final", "text": final_text})
                        final_emitted = True
                yield _sse({"type": "done"})
                return

        # If we reach here without explicit 'turn_complete'
        yield _sse({"type": "done"})

    except Exception as e:
        yield _sse({"type": "error", "error": str(e)})


# ---- Lifecycle ----
@app.on_event("startup")
def startup():
    global _cfg, _client, _agent_id

    # Optional: disable local OpenTelemetry noise if auto-instrumented
    # export OTEL_SDK_DISABLED=1

    _cfg = AppConfig.load()
    _cfg.llama.base_url = _cfg.llama.base_url.rstrip("/")
    _client = LlamaStackClient(base_url=_cfg.llama.base_url, provider_data=None)

    try:
        _ensure_vector_db(_cfg)
        _agent_id = _ensure_agent(_cfg)
        # warm a short session so the stack lazily initializes tools
        sid = _create_session(_agent_id, "warmup")
        logger.info("Connected to LlamaStack at %s (agent_id=%s, warmup_session=%s)",
                    _cfg.llama.base_url, _agent_id, sid)
    except APIConnectionError as e:
        logger.error("Could not connect to LlamaStack at %s: %s", _cfg.llama.base_url, e)
        _agent_id = None


@app.get("/healthz")
def healthz():
    ok = all([_cfg, _client, _agent_id, _vector_db_ready])
    return JSONResponse({"status": "ok" if ok else "not-ready"}, status_code=200 if ok else 503)


# ---- Endpoints ----
@app.post("/rag/ingest")
def rag_ingest(req: IngestRequest):
    if not all([_client, _agent_id, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Server not ready")

    docs = []
    for d in req.documents:
        docs.append(
            {
                "document_id": d.document_id or f"doc-{uuid.uuid4()}",
                "content": d.url,
                "mime_type": d.mime_type,
                "metadata": d.metadata or {},
            }
        )

    try:
        _client.tool_runtime.rag_tool.insert(
            documents=docs,  # accepts plain dicts too; RAGDocument compatible
            vector_db_id=_cfg.vector_db.id,
            chunk_size_in_tokens=req.chunk_size_in_tokens or int(_cfg.vector_db.chunk_size),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG insert failed: {e}")

    return {"status": "ok", "ingested": len(docs), "vector_db_id": _cfg.vector_db.id}


@app.post("/rag/ask/stream")
def rag_ask_stream(req: AskRequest, debug: bool = Query(False, description="Emit one compact payload key preview")):
    if not all([_client, _agent_id, _vector_db_ready]):
        raise HTTPException(status_code=503, detail="Server not ready")

    # Create a fresh session per question (like your ContextAgent)
    sid = _create_session(_agent_id, "rag-query")
    return StreamingResponse(
        _iter_stream(_agent_id, sid, req.question, debug),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
