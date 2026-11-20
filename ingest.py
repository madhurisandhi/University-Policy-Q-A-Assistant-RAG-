"""Load PDFs, split them into overlapping chunks, embed them, and store them in Chroma.

Run from the project root:
    python -m src.ingest
"""
from pathlib import Path

import chromadb
from pypdf import PdfReader

from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    COLLECTION_NAME,
    DATA_DIR,
    DB_DIR,
)
from src.embeddings import embed


def load_pdf(path):
    """Return a list of (page_number, text) for every page that has text."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping chunks of roughly `size` words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def build_index(
    chunk_size=CHUNK_SIZE_WORDS,
    overlap=CHUNK_OVERLAP_WORDS,
    collection_name=COLLECTION_NAME,
    data_dir=DATA_DIR,
    db_dir=DB_DIR,
    verbose=True,
):
    """(Re)build the vector index from every PDF in `data_dir`. Returns the chunk count."""
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        if verbose:
            print(f"No PDFs found in {data_dir}. Add some and run again.")
        return 0

    client = chromadb.PersistentClient(path=str(db_dir))
    try:  # start fresh so re-running never creates duplicates
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    for pdf in pdf_files:
        ids, docs, metas = [], [], []
        for page_num, text in load_pdf(pdf):
            for j, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                ids.append(f"{pdf.stem}-p{page_num}-c{j}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_num})

        if not docs:
            if verbose:
                print(f"Skipped {pdf.name}: no extractable text (scanned PDF?)")
            continue

        for i in range(0, len(docs), 64):  # batches keep memory use small
            collection.add(
                ids=ids[i:i + 64],
                documents=docs[i:i + 64],
                embeddings=embed(docs[i:i + 64]),
                metadatas=metas[i:i + 64],
            )
        total_chunks += len(docs)
        if verbose:
            print(f"Indexed {pdf.name}: {len(docs)} chunks")

    if verbose:
        print(f"Done. {total_chunks} chunks from {len(pdf_files)} file(s) stored in {db_dir}")
    return total_chunks


if __name__ == "__main__":
    build_index()
