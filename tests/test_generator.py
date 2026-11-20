from types import SimpleNamespace

from src.generator import NOT_FOUND, answer, build_prompt

CHUNKS = [{"text": "Appeals are due within 10 days.", "source": "policy.pdf", "page": 3, "distance": 0.2}]


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Within 10 days [policy.pdf, p. 3].")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def test_prompt_contains_source_page_and_question():
    prompt = build_prompt("When are appeals due?", CHUNKS)
    assert 'source="policy.pdf"' in prompt and 'page="3"' in prompt
    assert "When are appeals due?" in prompt


def test_answer_returns_text_sources_and_usage():
    client = FakeClient()
    result = answer("When are appeals due?", CHUNKS, client=client)
    assert "10 days" in result["answer"]
    assert result["sources"] == CHUNKS
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_no_chunks_skips_the_model_call():
    client = FakeClient()
    result = answer("Anything?", [], client=client)
    assert result["answer"] == NOT_FOUND
    assert client.calls == 0
