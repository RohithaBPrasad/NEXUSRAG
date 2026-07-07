const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function handle(res) {
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`)
  }
  return data
}

export async function fetchModes() {
  const res = await fetch(`${API_URL}/api/modes`)
  return handle(res)
}

export async function fetchStats() {
  const res = await fetch(`${API_URL}/api/stats`)
  return handle(res)
}

export async function ingestPath(path) {
  const res = await fetch(`${API_URL}/api/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  return handle(res)
}

export async function uploadFiles(fileList) {
  const form = new FormData()
  for (const f of fileList) form.append('files', f)
  const res = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    body: form,
  })
  return handle(res)
}

export async function runQuery({ mode, query, detailed, k, retriever, reranker, imagePath }) {
  const res = await fetch(`${API_URL}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode,
      query,
      detailed,
      k,
      retriever,
      reranker,
      image_path: imagePath || null,
    }),
  })
  return handle(res)
}

export async function clearMemory() {
  const res = await fetch(`${API_URL}/api/memory/clear`, { method: 'POST' })
  return handle(res)
}
