from pydantic import BaseModel, Field

from app.config import settings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(settings.default_top_k, ge=1, le=settings.max_top_k)


class Source(BaseModel):
    source: str
    page: int
    text: str
    score: float
    similarity: float
    chunk_id: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    mode: str = "hybrid"
    model: str | None = None


class HealthResponse(BaseModel):
    status: str            # "ok" or "no_documents"
    documents: list[str]
    chunks: int
    llm_model: str | None = None
    embed_model: str | None = None
    llm_configured: bool = False