import os
import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Optional


class AppMeta(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    session_name: str


class SamplingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    temperature: float
    top_p: float
    max_tokens: int


class QueryConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    chunk_template: str


class VectorDBConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    provider: str
    embedding: str
    embedding_dimension: int
    chunk_size: int
    provider_vector_db_id: Optional[str] = None


class LlamaConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    base_url: str
    model_id: str
    instructions: str


class AppConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    app: AppMeta
    llama: LlamaConfig
    vector_db: VectorDBConfig
    sampling: SamplingConfig
    query: QueryConfig

    @staticmethod
    def _env_override(cfg: "AppConfig") -> "AppConfig":
        """Override config values with environment variables if present."""
        cfg.app.session_name = os.getenv("SESSION_NAME", cfg.app.session_name)

        cfg.llama.base_url = os.getenv("LLAMA_BASE_URL", cfg.llama.base_url).rstrip("/")
        cfg.llama.model_id = os.getenv("MODEL_ID", cfg.llama.model_id)
        cfg.llama.instructions = os.getenv("LLAMA_INSTRUCTIONS", cfg.llama.instructions)

        cfg.vector_db.id = os.getenv("VDB_ID", cfg.vector_db.id)
        cfg.vector_db.provider = os.getenv("VDB_PROVIDER", cfg.vector_db.provider)
        cfg.vector_db.embedding = os.getenv("VDB_EMBEDDING", cfg.vector_db.embedding)
        cfg.vector_db.embedding_dimension = int(os.getenv("VDB_EMBEDDING_DIMENSION", str(cfg.vector_db.embedding_dimension)))
        cfg.vector_db.chunk_size = int(os.getenv("VECTOR_DB_CHUNK_SIZE", str(cfg.vector_db.chunk_size)))
        cfg.vector_db.provider_vector_db_id = os.getenv("PROVIDER_VECTOR_DB_ID", cfg.vector_db.provider_vector_db_id) or None

        cfg.sampling.temperature = float(os.getenv("TEMPERATURE", str(cfg.sampling.temperature)))
        cfg.sampling.top_p = float(os.getenv("TOP_P", str(cfg.sampling.top_p)))
        cfg.sampling.max_tokens = int(os.getenv("MAX_TOKENS", str(cfg.sampling.max_tokens)))

        cfg.query.chunk_template = os.getenv("QUERY_CHUNK_TEMPLATE", cfg.query.chunk_template)
        return cfg

    @classmethod
    def load(cls, path: str | None = None) -> "AppConfig":
        """Load configuration from YAML file, then apply environment variable overrides."""
        if path is None:
            # Try multiple possible locations
            possible_paths = [
                os.getenv("CONFIG_PATH"),
                os.path.join(os.getcwd(), "src", "config.yaml"),
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                "src/config.yaml",
                "config.yaml"
            ]
            
            path = None
            for p in possible_paths:
                if p and os.path.exists(p):
                    path = p
                    break
        
        if not path or not os.path.exists(path):
            raise RuntimeError(f"Config file not found. Tried locations: {possible_paths}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise RuntimeError(f"Config file is empty at {path}")
        
        try:
            cfg = cls(**data)
        except ValidationError as e:
            raise RuntimeError(f"Invalid config at {path}: {e}")
        
        return cls._env_override(cfg)