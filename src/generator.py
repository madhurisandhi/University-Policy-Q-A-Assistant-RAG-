"""Ask the LLM to answer using only the retrieved passages, with citations."""
import anthropic

from src.config import ANTHROPIC_MODEL

NOT_FOUND = "I couldn't find that in the provided documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions about university policies.

Rules:
- Use ONLY the information inside the <document> passages provided by the user.
- If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
- Cite every claim with its source in the form [file name, p. page].
- Be concise and do not invent rules, dates, or numbers.
- Remind the user to confirm important decisions with the university when the answer affects them."""


def build_prompt(question, chunks):
    """Put the retrieved passages and the question into one prompt."""
    blocks = [
        f'<document source="{c["source"]}" page="{c["page"]}">\n{c["text"]}\n</document>'
        for c in chunks
    ]
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(question, chunks, client=None, model=ANTHROPIC_MODEL):
    """Return {"answer": str, "sources": chunks, "usage": {...}}."""
    if not chunks:
        return {"answer": NOT_FOUND, "sources": [], "usage": None}

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return {
        "answer": text,
        "sources": chunks,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
