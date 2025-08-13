import os
import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict


class AppMeta(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    session_name: str = "rag_session"


class SamplingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 2048


class QueryConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    chunk_template: str = "Result {index}\nContent: {chunk.content}\nMetadata: {metadata}\n"


class VectorDBConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str = "simple_rag_vdb"
    provider: str = "faiss"                # engine: faiss/milvus/pgvector/etc.
    embedding: str = "all-MiniLM-L6-v2"    # embedding model id (dim = 384)
    embedding_dimension: int = 384
    chunk_size: int = 512
    provider_vector_db_id: str | None = None  # optional bind to an existing provider-native index


class LlamaConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    base_url: str = "https://lss-metric.apps.tsisodia-dev.51ty.p1.openshiftapps.com"
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    instructions: str = "You are a helpful assistant."


class AppConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    app: AppMeta = Field(default_factory=AppMeta)
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)

    @staticmethod
    def _env_override(cfg: "AppConfig") -> "AppConfig":
        cfg.app.session_name = os.getenv("SESSION_NAME", cfg.app.session_name)

        cfg.llama.base_url = os.getenv("LLAMA_BASE_URL", cfg.llama.base_url).rstrip("/")
        cfg.llama.model_id = os.getenv("MODEL_ID", cfg.llama.model_id)
        cfg.llama.instructions = os.getenv("LLAMA_INSTRUCTIONS", cfg.llama.instructions)

        cfg.vector_db.id = os.getenv("VDB_ID", cfg.vector_db.id)
        cfg.vector_db.provider = os.getenv("VDB_PROVIDER", cfg.vector_db.provider)
        cfg.vector_db.embedding = os.getenv("VDB_EMBEDDING", cfg.vector_db.embedding)
        cfg.vector_db.embedding_dimension = int(os.getenv("VDB_EMBEDDING_DIMENSION", cfg.vector_db.embedding_dimension))
        cfg.vector_db.chunk_size = int(os.getenv("VECTOR_DB_CHUNK_SIZE", cfg.vector_db.chunk_size))
        cfg.vector_db.provider_vector_db_id = os.getenv(
            "PROVIDER_VECTOR_DB_ID", cfg.vector_db.provider_vector_db_id or ""
        ) or None

        cfg.sampling.temperature = float(os.getenv("TEMPERATURE", cfg.sampling.temperature))
        cfg.sampling.top_p = float(os.getenv("TOP_P", cfg.sampling.top_p))
        cfg.sampling.max_tokens = int(os.getenv("MAX_TOKENS", cfg.sampling.max_tokens))

        cfg.query.chunk_template = os.getenv("QUERY_CHUNK_TEMPLATE", cfg.query.chunk_template)
        return cfg

    @classmethod
    def load(cls, path: str | None = None) -> "AppConfig":
        path = path or os.getenv("CONFIG_PATH", os.path.join(os.getcwd(), "src", "config.yaml"))
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        try:
            cfg = cls(**data)
        except ValidationError as e:
            raise RuntimeError(f"Invalid config at {path}: {e}")
        return cls._env_override(cfg)
