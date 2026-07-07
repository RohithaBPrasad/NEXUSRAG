from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    sys.exit("Missing dependency 'chromadb'. Run: pip install chromadb")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("Missing dependency 'rank_bm25'. Run: pip install rank_bm25")

try:
    import networkx as nx
except ImportError:
    sys.exit("Missing dependency 'networkx'. Run: pip install networkx")

from . import config
from .loaders import discover_files, load_document
from .utils import chunk_code, chunk_text, safe_ollama_embed


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    doc_type: str


class Store:
    def __init__(self, persist_dir: str = None, collection_name: str = None):
        persist_dir = persist_dir or config.CHROMA_DIR
        collection_name = collection_name or config.COLLECTION_NAME
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self._chunks: Dict[str, Chunk] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self.graph = nx.DiGraph()
        self._load_existing_into_memory()

    # -- bookkeeping -----------------------------------------------------
    def _load_existing_into_memory(self) -> None:
        try:
            data = self.collection.get(include=["documents", "metadatas"])
        except Exception:  # noqa: BLE001
            return
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for cid, doc, meta in zip(ids, docs, metas):
            self._chunks[cid] = Chunk(
                id=cid, text=doc,
                source=(meta or {}).get("source", "unknown"),
                doc_type=(meta or {}).get("doc_type", "text"),
            )
        if self._chunks:
            self._rebuild_bm25()
            self._rebuild_graph()

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self._chunks.keys())
        tokenized = [self._chunks[cid].text.lower().split() for cid in self._bm25_ids]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _rebuild_graph(self) -> None:
        """Very lightweight entity graph: capitalised multi-word phrases and
        proper nouns per chunk become nodes; chunks sharing an entity get an edge."""
        self.graph = nx.DiGraph()
        entity_pattern = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b")
        for cid, chunk in self._chunks.items():
            entities = set(entity_pattern.findall(chunk.text))
            entities = {e for e in entities if len(e) > 2}
            self.graph.add_node(cid, type="chunk", source=chunk.source)
            for ent in entities:
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent, type="entity")
                self.graph.add_edge(ent, cid)
                self.graph.add_edge(cid, ent)

    # -- ingestion ---------------------------------------------------------
    def add_document(self, path: Path) -> int:
        """Loads, chunks, embeds, and stores one file. Returns #chunks added."""
        raw_text, doc_type = load_document(path)
        pieces = chunk_code(raw_text) if doc_type == "code" else chunk_text(raw_text)
        if not pieces:
            return 0
        ids, embeddings, metadatas, documents = [], [], [], []
        for piece in pieces:
            cid = str(uuid.uuid4())
            emb = safe_ollama_embed(piece)
            ids.append(cid)
            embeddings.append(emb)
            metadatas.append({"source": str(path.name), "doc_type": doc_type})
            documents.append(piece)
            self._chunks[cid] = Chunk(id=cid, text=piece, source=str(path.name), doc_type=doc_type)
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        self._rebuild_bm25()
        self._rebuild_graph()
        return len(pieces)

    def ingest_path(self, root: str) -> Dict[str, Any]:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Path does not exist: {root}")
        files = discover_files(root_path)
        if not files:
            raise ValueError(f"No supported files found under: {root}")
        results = {"files": 0, "chunks": 0, "errors": []}
        for f in files:
            try:
                n = self.add_document(f)
                results["files"] += 1
                results["chunks"] += n
            except Exception as e:  # noqa: BLE001
                results["errors"].append(f"{f.name}: {e}")
        return results

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def stats(self) -> Dict[str, Any]:
        sources = {}
        for c in self._chunks.values():
            sources[c.source] = sources.get(c.source, 0) + 1
        return {"total_chunks": len(self._chunks), "sources": sources}

    # -- retrieval -----------------------------------------------------
    def vector_search(self, query: str, k: int = None, query_embedding: Optional[List[float]] = None) -> List[Chunk]:
        k = k if k is not None else config.TOP_K
        if self.is_empty():
            return []
        emb = query_embedding if query_embedding is not None else safe_ollama_embed(query)
        res = self.collection.query(query_embeddings=[emb], n_results=min(k, len(self._chunks)))
        ids = res.get("ids", [[]])[0]
        return [self._chunks[i] for i in ids if i in self._chunks]

    def keyword_search(self, query: str, k: int = None) -> List[Chunk]:
        k = k if k is not None else config.TOP_K
        if self._bm25 is None or self.is_empty():
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return [self._chunks[cid] for cid, score in ranked[:k] if score > 0]

    def hybrid_search(self, query: str, k: int = None) -> List[Chunk]:
        """Merge vector + keyword results, de-duplicated, vector-first ranking."""
        k = k if k is not None else config.TOP_K
        vec_results = self.vector_search(query, k=k)
        kw_results = self.keyword_search(query, k=k)
        seen = set()
        merged = []
        for c in vec_results + kw_results:
            if c.id not in seen:
                seen.add(c.id)
                merged.append(c)
        return merged[:k]

    def graph_search(self, query: str, k: int = None) -> List[Chunk]:
        """Entity-anchored traversal: find entities mentioned in the query that
        exist in the graph, then pull chunks connected to those entities
        (1-hop), falling back to vector search to fill any remaining slots."""
        k = k if k is not None else config.TOP_K
        if self.graph.number_of_nodes() == 0:
            return self.vector_search(query, k=k)
        query_words = set(re.findall(r"\b[A-Za-z]+\b", query))
        matched_entities = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "entity" and any(w.lower() in n.lower() for w in query_words if len(w) > 2)
        ]
        chunk_ids: List[str] = []
        for ent in matched_entities:
            for neighbor in self.graph.successors(ent):
                if neighbor in self._chunks and neighbor not in chunk_ids:
                    chunk_ids.append(neighbor)
        results = [self._chunks[cid] for cid in chunk_ids[:k]]
        if len(results) < k:
            for c in self.vector_search(query, k=k):
                if c.id not in chunk_ids:
                    results.append(c)
                if len(results) >= k:
                    break
        return results[:k]


def format_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return "(no relevant context found)"
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c.source})\n{c.text}")
    return "\n\n".join(parts)


def cite_sources(chunks: List[Chunk]) -> str:
    if not chunks:
        return "none"
    seen = []
    for c in chunks:
        if c.source not in seen:
            seen.append(c.source)
    return ", ".join(seen)
