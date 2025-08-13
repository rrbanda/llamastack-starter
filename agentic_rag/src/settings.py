import os
from pydantic import BaseModel

class Settings(BaseModel):
    # LlamaStack
    LLAMA_BASE_URL: str = os.getenv("LLAMA_BASE_URL", "http://localhost:8000")  # your local LlamaStack
    MODEL_ID: str = os.getenv("MODEL_ID", "granite32-8b")

    # Sampling
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # Tools / providers
    TAVILY_SEARCH_API_KEY: str | None = os.getenv("TAVILY_SEARCH_API_KEY") or None

    # App
    SESSION_NAME: str = os.getenv("SESSION_NAME", "web-session")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
