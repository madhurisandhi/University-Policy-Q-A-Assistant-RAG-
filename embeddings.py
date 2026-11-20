"""Sentence-transformer embeddings (the model is loaded once, on first use)."""
from functools import lru_cache

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts):
    """Turn a list of strings into a list of embedding vectors."""
    return _get_model().encode(list(texts), show_progress_bar=False).tolist()
