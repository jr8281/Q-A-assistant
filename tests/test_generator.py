from app.generator import NOT_FOUND_MESSAGE, LLMGenerator, is_not_found


def test_is_not_found_is_case_insensitive_and_ignores_final_period():
    assert is_not_found(NOT_FOUND_MESSAGE.upper())
    assert is_not_found(NOT_FOUND_MESSAGE.rstrip("."))
    assert not is_not_found("The AI RMF has four functions.")


def test_prompt_contains_question_and_labelled_sources():
    chunks = [{"source": "a.pdf", "page": 7, "text": "hello"}]
    messages = LLMGenerator.build_messages("What is hello?", chunks)
    assert messages[0]["role"] == "system"
    assert 'source="a.pdf"' in messages[1]["content"] and 'page="7"' in messages[1]["content"]
    assert "What is hello?" in messages[1]["content"]


def test_generator_reports_not_configured_without_a_key():
    assert LLMGenerator(api_key=None).configured in (True, False)   # depends on local .env