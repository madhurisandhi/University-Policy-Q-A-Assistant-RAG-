"""Central settings for the RAG document assistant."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"          # put your PDF files here
DB_DIR = ROOT_DIR / "chroma_db"       # vector store is saved here (git-ignored)

COLLECTION_NAME = "university_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Chunking (measured in words; ~350 words is roughly 450-500 tokens)
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50

# Retrieval
TOP_K = 4
