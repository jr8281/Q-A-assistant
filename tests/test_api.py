from fastapi.testclient import TestClient

from app.main import app


class FakeRetriever:
    def __init__(self, similarity):
        self.similarity = similarity

    def search(self, query, top_k=3, mode="hybrid", alpha=None):
        return [{"chunk_id": 0, "text": "some text", "source": "a.pdf", "page": 1,
                 "score": 1.0, "similarity": self.similarity}]


class FakeGenerator:
    configured = True
    model = "fake-llm"
    calls = 0

    def generate_answer(self, query, chunks):
        self.calls += 1
        return "The answer [a.pdf, p.1]"

    def stream_answer(self, query, chunks):
        self.calls += 1
        yield "Hello "
        yield "world"


class BrokenGenerator(FakeGenerator):
    def generate_answer(self, query, chunks):
        raise RuntimeError("boom")


def make_client(retriever, generator):
    # No `with` block, so the lifespan (model loading) does not run
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.documents = ["a.pdf"]
    app.state.chunk_count = 1
    return TestClient(app)


QUESTION = {"question": "What is in the document?"}


def test_health_reports_documents():
    body = make_client(FakeRetriever(0.9), FakeGenerator()).get("/health").json()
    assert body["status"] == "ok" and body["documents"] == ["a.pdf"] and body["llm_model"] == "fake-llm"


def test_no_documents_returns_503():
    assert make_client(None, FakeGenerator()).post("/api/v1/query", json=QUESTION).status_code == 503


def test_short_question_is_rejected():
    assert make_client(FakeRetriever(0.9), FakeGenerator()).post("/api/v1/query", json={"question": "hi"}).status_code == 422


def test_relevant_question_returns_answer_and_sources():
    body = make_client(FakeRetriever(0.9), FakeGenerator()).post("/api/v1/query", json=QUESTION).json()
    assert "a.pdf" in body["answer"] and len(body["sources"]) == 1 and body["model"] == "fake-llm"


def test_gate_skips_the_llm_when_nothing_is_relevant():
    generator = FakeGenerator()
    body = make_client(FakeRetriever(0.05), generator).post("/api/v1/query", json=QUESTION).json()
    assert "not found" in body["answer"] and body["sources"] == []
    assert generator.calls == 0


def test_llm_failure_returns_502():
    assert make_client(FakeRetriever(0.9), BrokenGenerator()).post("/api/v1/query", json=QUESTION).status_code == 502


def test_stream_endpoint_sends_sources_tokens_and_done():
    text = make_client(FakeRetriever(0.9), FakeGenerator()).post("/api/v1/query/stream", json=QUESTION).text
    assert "event: sources" in text and text.count("event: token") == 2 and "event: done" in text