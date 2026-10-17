from pathlib import Path

from app.config import settings
from app.pdf_processor import PDFProcessor


def load_corpus(docs_dir: Path | None = None) -> tuple[list[str], list[dict]]:
    """Read every PDF in the folder. Returns (file names, chunks with globally unique ids)."""
    docs_dir = docs_dir or settings.docs_dir
    docs_dir.mkdir(exist_ok=True)
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    chunks: list[dict] = []
    for pdf in pdf_files:
        chunks.extend(PDFProcessor.extract_and_chunk(pdf))

    # chunk_id restarts at 0 for every PDF inside the processor; make it unique across all files
    for new_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = new_id

    return [p.name for p in pdf_files], chunks