import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.generator import NOT_FOUND_MESSAGE, is_not_found
from app.schemas import HealthResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

LLM_ERROR = "The language model service failed. Check your API key and try again."


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _retrieve(request: Request, body: QueryRequest):
    """Shared by both endpoints: check state, search, apply the relevance gate.
    Returns (sources, is_relevant, retrieval_ms)."""
    state = request.app.state
    if state.retriever is None:
        raise HTTPException(status_code=503, detail="No documents indexed. Add PDFs to 'documents/'.")

    t0 = time.perf_counter()
    sources = state.retriever.search(body.question, top_k=body.top_k)
    retrieval_ms = _ms(t0)

    # Relevance gate: skip the LLM when nothing relevant was found
    relevant = max(s["similarity"] for s in sources) >= settings.min_similarity
    if not relevant:
        logger.info("Gate: no relevant passage for %r", body.question)
    elif not state.generator.configured:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set. Add it to your .env file.")
    return sources, relevant, retrieval_ms


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    state = request.app.state
    return HealthResponse(
        status="ok" if state.retriever is not None else "no_documents",
        documents=state.documents,
        chunks=state.chunk_count,
        llm_model=state.generator.model,
        embed_model=settings.embed_model,
        llm_configured=state.generator.configured,
    )


@router.post("/api/v1/query", response_model=QueryResponse)
def handle_query(body: QueryRequest, request: Request):
    started = time.perf_counter()
    state = request.app.state
    sources, relevant, retrieval_ms = _retrieve(request, body)

    if not relevant:
        return QueryResponse(answer=NOT_FOUND_MESSAGE, sources=[], latency_ms=_ms(started),
                             retrieval_ms=retrieval_ms, model=state.generator.model)

    t0 = time.perf_counter()
    try:
        answer = state.generator.generate_answer(body.question, sources)
    except Exception as exc:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=LLM_ERROR) from exc

    # If the model says the answer is not in the context, don't show unrelated sources
    if is_not_found(answer):
        sources = []

    return QueryResponse(answer=answer, sources=sources, latency_ms=_ms(started),
                         retrieval_ms=retrieval_ms, generation_ms=_ms(t0),
                         model=state.generator.model)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/api/v1/query/stream")
def handle_query_stream(body: QueryRequest, request: Request):
    """Server-Sent Events: sends sources first, then answer tokens, then timing metadata."""
    started = time.perf_counter()
    state = request.app.state
    sources, relevant, retrieval_ms = _retrieve(request, body)

    def events():
        yield _sse("sources", {"sources": sources if relevant else []})
        t0 = time.perf_counter()
        if not relevant:
            yield _sse("token", {"text": NOT_FOUND_MESSAGE})
        else:
            try:
                for piece in state.generator.stream_answer(body.question, sources):
                    yield _sse("token", {"text": piece})
            except Exception:
                logger.exception("LLM stream failed")
                yield _sse("error", {"detail": LLM_ERROR})
                return
        yield _sse("done", {
            "latency_ms": _ms(started), "retrieval_ms": retrieval_ms, "generation_ms": _ms(t0),
            "mode": "hybrid", "model": state.generator.model,
        })

    return StreamingResponse(events(), media_type="text/event-stream")