"""Configuration for the hybrid retriever and vector store."""

from __future__ import annotations

from pathlib import Path

from ..storage import STORAGE_DIR

# ChromaDB persist directory
CHROMA_PERSIST_DIR: Path = STORAGE_DIR / "chroma_db"

# Embedding model
# all-MiniLM-L6-v2: 384-dim, fast, 80MB, good general purpose
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Cross-encoder reranker (lightweight)
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search defaults
DEFAULT_TOP_K: int = 5
DEFAULT_CACHE_TOP_K: int = 3

# Collection names
COLLECTION_FIGURES: str = "figures"
COLLECTION_CACHE: str = "cache"
