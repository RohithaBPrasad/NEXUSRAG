"""
RAG14 API — FastAPI backend for the React frontend.

Run with:
    uvicorn app.main:app --reload --port 8000
(from inside the backend/ folder, with your virtualenv active)
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .memory import Memory
from .modes import MODE_LABELS, chunks_contain_code, resolve_mode, run_mode
from .store import Store, cite_sources

app = FastAPI(title="RAG14 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A single shared store + conversation memory, same design as the original
# CLI/REPL tool. Good for a personal/local deployment; if you need per-user
# isolation, key these by session/user id instead.
store = Store()
memory = Memory()


# ---------------------------------------------------------------- schemas --

class IngestPathRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    mode: str = "simple"
    query: str
    detailed: bool = False
    k: int = config.TOP_K
    image_path: Optional[str] = None       # for multimodal mode, if the image is already on the server
    retriever: Optional[str] = None        # for modular mode
    reranker: Optional[str] = None         # for modular mode


class IngestResponse(BaseModel):
    files: int
    chunks: int
    errors: List[str]
    total_chunks: int
    sources: Dict[str, int]


class QueryResponse(BaseModel):
    answer: str
    sources: str
    notes: str
    is_code_mode: bool
    chunks_count: int
    chunks_raw: List[Dict[str, str]]


# ------------------------------------------------------------- endpoints --

@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/modes")
def get_modes() -> Dict[str, Any]:
    """All 14 RAG strategies, for populating the mode selector dropdown."""
    return {
        "modes": [{"key": key, "label": label} for key, label in MODE_LABELS],
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "vision_model": config.VISION_MODEL,
    }


@app.get("/api/stats")
def get_stats() -> Dict[str, Any]:
    return store.stats()


@app.post("/api/ingest", response_model=IngestResponse)
def ingest_path(req: IngestPathRequest) -> Dict[str, Any]:
    """Ingest a file or folder that already exists on the server's filesystem."""
    if not req.path.strip():
        raise HTTPException(status_code=400, detail="No path provided.")
    try:
        results = store.ingest_path(req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    stats = store.stats()
    return {
        "files": results["files"],
        "chunks": results["chunks"],
        "errors": results["errors"],
        "total_chunks": stats["total_chunks"],
        "sources": stats["sources"],
    }


@app.post("/api/upload", response_model=IngestResponse)
async def upload_and_ingest(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Upload one or more files from the React frontend (drag-and-drop / file
    picker) and ingest them straight into the knowledge base."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    batch_dir = config.UPLOAD_DIR / str(uuid.uuid4())
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        dest = batch_dir / Path(f.filename).name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="No files could be saved.")

    try:
        results = store.ingest_path(str(batch_dir))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

    stats = store.stats()
    return {
        "files": results["files"],
        "chunks": results["chunks"],
        "errors": results["errors"],
        "total_chunks": stats["total_chunks"],
        "sources": stats["sources"],
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> Dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="No query provided.")
    try:
        resolve_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    kwargs: Dict[str, Any] = {"k": req.k, "detailed": req.detailed}
    if req.image_path:
        kwargs["query_image_path"] = req.image_path
    if req.retriever:
        kwargs["retriever"] = req.retriever
    if req.reranker:
        kwargs["reranker"] = req.reranker

    try:
        result = run_mode(req.mode, store, req.query, memory, **kwargs)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

    memory.add(req.query, result["answer"])
    chunks_used = result.get("chunks_used") or []
    return {
        "answer": result["answer"],
        "sources": cite_sources(chunks_used),
        "notes": result.get("notes", ""),
        "is_code_mode": chunks_contain_code(chunks_used),
        "chunks_count": len(chunks_used),
        "chunks_raw": [{"source": c.source, "text": c.text[:400]} for c in chunks_used],
    }


@app.post("/api/memory/clear")
def clear_memory() -> Dict[str, str]:
    memory.clear()
    return {"status": "cleared"}
