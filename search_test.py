"""Quick check that retrieval works (no LLM needed).

Run from the project root:
    python -m src.search_test "How do I appeal a grade?"
"""
import sys

from src.retriever import retrieve

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the academic integrity policy?"
    print(f"Question: {question}\n")
    for rank, c in enumerate(retrieve(question), start=1):
        print(f"[{rank}] {c['source']} (page {c['page']}) | distance {c['distance']:.3f}")
        print(c["text"][:300].replace("\n", " ") + "...\n")
