"""Measure retrieval quality for different chunk sizes and top-k values.

For each question in questions.json, a retrieval is a HIT if at least one of the
top-k chunks contains ALL of the question's expected_keywords (case-insensitive).

Run from the project root:
    python -m evaluation.run_eval
"""
import json
import sys
from pathlib import Path

import chromadb

from src.config import CHUNK_OVERLAP_WORDS, DB_DIR
from src.ingest import build_index
from src.retriever import retrieve

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.md"
CHUNK_SIZES = [200, 350, 500]
K_VALUES = [1, 3, 5]


def is_hit(chunks, keywords):
    return any(all(k.lower() in c["text"].lower() for k in keywords) for c in chunks)


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())
    if any("REPLACE" in q["question"] for q in questions):
        print("Edit evaluation/questions.json first: replace the example entries with real questions.")
        sys.exit(1)

    rows = []
    for size in CHUNK_SIZES:
        name = f"eval_chunk{size}"
        n_chunks = build_index(size, min(CHUNK_OVERLAP_WORDS, size // 4), collection_name=name, verbose=False)
        hits = {k: 0 for k in K_VALUES}
        for q in questions:
            retrieved = retrieve(q["question"], k=max(K_VALUES), collection_name=name)
            for k in K_VALUES:
                hits[k] += is_hit(retrieved[:k], q["expected_keywords"])
        rows.append((size, n_chunks, {k: hits[k] / len(questions) for k in K_VALUES}))
        chromadb.PersistentClient(path=str(DB_DIR)).delete_collection(name)

    header = "| Chunk size (words) | Chunks | " + " | ".join(f"Hit@{k}" for k in K_VALUES) + " |"
    divider = "|---|---|" + "---|" * len(K_VALUES)
    lines = [header, divider]
    for size, n_chunks, rates in rows:
        lines.append(f"| {size} | {n_chunks} | " + " | ".join(f"{rates[k]:.0%}" for k in K_VALUES) + " |")
    table = "\n".join(lines)

    summary = f"Questions evaluated: {len(questions)}\n\n{table}\n"
    RESULTS_FILE.write_text(summary)
    print(summary)
    print(f"Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
