import logging
from pathlib import Path

from pypdf import PdfReader

from app.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Reads a PDF and splits each page into overlapping word chunks."""

    @staticmethod
    def extract_and_chunk(
        pdf_path: Path,
        chunk_size: int = settings.chunk_words,
        overlap: int = settings.chunk_overlap,
    ) -> list[dict]:
        reader = PdfReader(pdf_path)
        chunks: list[dict] = []
        step = chunk_size - overlap

        for page_no, page in enumerate(reader.pages, start=1):
            words = (page.extract_text() or "").split()
            for i in range(0, len(words), step):
                piece = words[i:i + chunk_size]
                if len(piece) < 20:      # skip tiny fragments (page numbers, headers)
                    continue
                chunks.append({
                    "text": " ".join(piece),
                    "source": pdf_path.name,
                    "page": page_no,
                    "chunk_id": len(chunks),
                })

        if not chunks:
            logger.warning("%s: no extractable text (scanned PDF? it would need OCR)", pdf_path.name)
        return chunks