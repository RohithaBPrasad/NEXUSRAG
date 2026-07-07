"""
Central configuration for the RAG14 backend.
Every value can be overridden with an environment variable of the same
name (see backend/.env.example), so you don't have to edit source to
point at different Ollama models or a different persistence directory.
"""

from __future__ import annotations

import os
from pathlib import Path

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")          # main generation model
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # embedding model
VISION_MODEL = os.getenv("VISION_MODEL", "llava")           # used only by Multimodal RAG (mode 7)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./rag14_chroma_db")   # persisted vector store on disk
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag14_docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # characters of overlap between chunks
TOP_K = int(os.getenv("TOP_K", "4"))                    # default number of chunks to retrieve

OLLAMA_TIMEOUT_RETRIES = int(os.getenv("OLLAMA_TIMEOUT_RETRIES", "2"))

# Where uploaded files (from the React frontend's drag-and-drop / file picker)
# get saved before being ingested into the store.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Comma-separated list of origins allowed to call this API (your React dev
# server / production domain). "*" is convenient for local dev only.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
