from pathlib import Path

from app import pdf_processor


class FakePage:
    def __init__(self, text): self.text = text
    def extract_text(self): return self.text


def make_reader(text):
    class FakeReader:
        def __init__(self, path): self.pages = [FakePage(text)]
    return FakeReader


def test_chunks_keep_page_source_and_size_limit(monkeypatch):
    monkeypatch.setattr(pdf_processor, "PdfReader", make_reader("word " * 450))
    chunks = pdf_processor.PDFProcessor.extract_and_chunk(Path("sample.pdf"))

    assert len(chunks) == 3
    assert all(c["page"] == 1 and c["source"] == "sample.pdf" for c in chunks)
    assert all(len(c["text"].split()) <= 200 for c in chunks)


def test_tiny_pages_are_skipped(monkeypatch):
    monkeypatch.setattr(pdf_processor, "PdfReader", make_reader("page 3"))
    assert pdf_processor.PDFProcessor.extract_and_chunk(Path("sample.pdf")) == []