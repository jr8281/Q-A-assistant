import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.embedding import PyTorchEmbedder
from app.generator import LLMGenerator
from app.indexing import load_corpus
from app.logging_config import setup_logging
from app.retriever import HybridRetriever

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = None
    app.state.documents = []
    app.state.chunk_count = 0
    app.state.generator = LLMGenerator()

    names, chunks = load_corpus()
    if not chunks:
        logger.warning("No PDFs found in 'documents/'. Add PDFs and restart.")
    else:
        app.state.retriever = HybridRetriever(chunks, PyTorchEmbedder())
        app.state.documents = names
        app.state.chunk_count = len(chunks)
        logger.info("Indexed %d PDF(s), %d chunks", len(names), len(chunks))
    yield


app = FastAPI(
    title="PDF Q&A Assistant",
    description="Ask questions about PDFs and get answers with page-level citations. "
                "Hybrid search (BM25 + embeddings) feeds an LLM that answers only from the retrieved passages.",
    version="1.2.0",
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
    logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(settings.static_dir / "index.html"))