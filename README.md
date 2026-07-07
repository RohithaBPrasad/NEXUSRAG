# RAG14 — 14 RAG Architectures, One Engine (React + FastAPI)

This is a full React frontend + FastAPI backend rewrite of the original
`rag14.py` single-file tool. All 14 RAG strategies (Simple, Memory, Agentic,
Graph, Self-RAG, Branched, Multimodal, Adaptive, Speculative, Corrective,
Modular, Naive, Advanced, HyDE) run exactly as before — same Ollama models,
same ChromaDB store, same BM25/entity-graph retrieval — just exposed over a
clean REST API and driven from a proper React UI instead of the CLI/REPL and
embedded HTML page.

```
rag14-app/
├── backend/            FastAPI app (the RAG engine)
│   ├── app/
│   │   ├── config.py    tunable constants (models, chunk size, CORS, ...)
│   │   ├── utils.py      chunking + safe Ollama wrappers
│   │   ├── loaders.py     file loaders (txt/md/pdf/docx/code/images)
│   │   ├── store.py        Chroma + BM25 + entity graph
│   │   ├── memory.py        conversation memory
│   │   ├── modes.py          all 14 RAG strategies + registry
│   │   └── main.py            FastAPI routes
│   ├── requirements.txt
│   └── .env.example
└── frontend/           React (Vite) app
    ├── src/
    │   ├── api.js         API client
    │   ├── modeMeta.js      per-mode pipeline metadata (for the UI)
    │   ├── App.jsx
    │   └── components/
    ├── package.json
    └── .env.example
```

## 1. Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **[Ollama](https://ollama.com)** installed and running locally (or reachable
  over the network), with the models this project uses pulled:

  ```bash
  ollama pull llama3.2          # chat model
  ollama pull nomic-embed-text  # embedding model
  ollama pull llava             # vision model (only needed for Multimodal RAG)
  ```

  You can swap any of these for other Ollama models via environment
  variables — see `backend/.env.example`.

## 2. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit if you need different models/CORS

uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Interactive API docs (from
FastAPI/OpenAPI) are at `http://localhost:8000/docs`.

### Backend API reference

| Method | Path                | Body / notes                                                        |
|--------|----------------------|----------------------------------------------------------------------|
| GET    | `/api/health`        | liveness check                                                       |
| GET    | `/api/modes`         | list of the 14 strategies + which Ollama models are configured      |
| GET    | `/api/stats`         | `{ total_chunks, sources }` for the current knowledge base           |
| POST   | `/api/ingest`        | `{ path }` — ingest a file/folder already on the server's disk       |
| POST   | `/api/upload`        | multipart `files[]` — upload + ingest files from the browser         |
| POST   | `/api/query`         | `{ mode, query, detailed, k, retriever?, reranker?, image_path? }`    |
| POST   | `/api/memory/clear`  | clears the rolling conversation memory                               |

`/api/query` returns:

```json
{
  "answer": "...",
  "sources": "doc1.pdf, doc2.md",
  "notes": "mode-specific notes (e.g. rewritten query, agent steps)",
  "is_code_mode": false,
  "chunks_count": 4,
  "chunks_raw": [{ "source": "doc1.pdf", "text": "..." }]
}
```

## 3. Frontend setup

In a second terminal:

```bash
cd frontend
npm install
cp .env.example .env    # points at http://localhost:8000 by default
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). You should see
the console: a strategy list with a live "pipeline strip" showing each
architecture's actual internal stages, a sidebar to drag-and-drop files or
ingest a server-side path, and a query box that streams answers with their
sources, notes, and retrieved chunks.

## 4. Using it

1. Drag files into the sidebar dropzone (or type a path already on the
   server and click **Ingest path**) — supports `.txt` `.md` `.pdf` `.docx`,
   images (described via the vision model), and most source code files.
2. Pick a strategy — the pipeline strip shows exactly what that mode does
   internally before you run it.
3. Ask a question. Toggle **detailed answer** for long-form output, adjust
   **k** for how many chunks are retrieved, and (for Modular RAG) pick a
   retriever/reranker module.
4. Click **show retrieved chunks** on any answer to see exactly what context
   the model was given.

## 5. Building for production

```bash
cd frontend
npm run build     # outputs static files to frontend/dist
```

Serve `frontend/dist` with any static host (Nginx, Vercel, Netlify, S3 +
CloudFront, etc.), and run the backend behind a process manager (systemd,
Docker, Render, Railway, Fly.io, etc.). Make sure to:

- Set `CORS_ORIGINS` in the backend's `.env` to your deployed frontend's
  origin(s) (comma-separated).
- Set `VITE_API_URL` in the frontend's `.env` to your deployed backend's URL
  **before** running `npm run build` (Vite bakes env vars in at build time).
- Point `CHROMA_DIR` / `UPLOAD_DIR` at a persistent volume if you're
  deploying on a platform with an ephemeral filesystem.
- Ensure the Ollama models are reachable from wherever the backend runs
  (either Ollama installed on the same host, or `OLLAMA_HOST` pointed at a
  remote Ollama instance — see the [Ollama docs](https://github.com/ollama/ollama)).

## Notes / limitations

- The backend keeps **one shared** `Store` and `Memory` instance, matching
  the original tool's design (a personal/local RAG playground). If you need
  per-user knowledge bases or isolated conversation histories, key those by
  a session/user id instead of using module-level globals in `app/main.py`.
- There's no authentication layer. If you deploy this publicly, put it
  behind your own auth (reverse proxy basic auth, an API gateway, etc.) —
  anyone who can reach `/api/ingest` and `/api/upload` can write to your
  knowledge base.
- Multimodal RAG's `image_path` field expects a path already on the server;
  wiring up an image-upload variant of `/api/upload` for query-time images
  is a natural next step if you need that from the browser instead of the
  filesystem.
