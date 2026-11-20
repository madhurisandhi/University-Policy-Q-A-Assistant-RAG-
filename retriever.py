"""Find the passages most relevant to a question."""
import chromadb

from src.config import COLLECTION_NAME, DB_DIR, TOP_K
from src.embeddings import embed


def index_size(collection_name=COLLECTION_NAME, db_dir=DB_DIR):
    """Number of chunks in the index (0 if it has not been built yet)."""
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        return client.get_collection(collection_name).count()
    except Exception:
        return 0


def retrieve(question, k=TOP_K, collection_name=COLLECTION_NAME, db_dir=DB_DIR):
    """Return the top-k chunks as dicts: text, source, page, distance (lower = closer)."""
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            "Index not found. Add PDFs to data/ and run `python -m src.ingest` first."
        ) from exc

    count = collection.count()
    if count == 0:
        return []
    result = collection.query(
        query_embeddings=embed([question]), n_results=min(k, count)
    )
    return [
        {"text": text, "source": meta["source"], "page": meta["page"], "distance": dist}
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
