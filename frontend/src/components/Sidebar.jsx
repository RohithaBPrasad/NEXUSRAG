import React, { useRef, useState } from 'react'
import { ingestPath, uploadFiles } from '../api.js'

export default function Sidebar({ modelsInfo, stats, onStatsChange }) {
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null) // { type: 'error'|'success', text }
  const [path, setPath] = useState('')
  const fileInputRef = useRef(null)

  const sourceEntries = stats?.sources ? Object.entries(stats.sources) : []

  async function handleFiles(fileList) {
    if (!fileList || fileList.length === 0) return
    setBusy(true)
    setMessage(null)
    try {
      const res = await uploadFiles(fileList)
      onStatsChange({ total_chunks: res.total_chunks, sources: res.sources })
      setMessage({
        type: res.errors?.length ? 'error' : 'success',
        text: res.errors?.length
          ? `Ingested ${res.files} file(s), ${res.errors.length} failed: ${res.errors[0]}`
          : `Ingested ${res.files} file(s) → ${res.chunks} chunks.`,
      })
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  async function handleIngestPath() {
    if (!path.trim()) return
    setBusy(true)
    setMessage(null)
    try {
      const res = await ingestPath(path.trim())
      onStatsChange({ total_chunks: res.total_chunks, sources: res.sources })
      setMessage({
        type: res.errors?.length ? 'error' : 'success',
        text: res.errors?.length
          ? `Ingested ${res.files} file(s), ${res.errors.length} failed: ${res.errors[0]}`
          : `Ingested ${res.files} file(s) → ${res.chunks} chunks.`,
      })
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">NexusRAG</span>
        <h1 className="brand-title">14 architectures,<br />one engine</h1>
        <p className="brand-sub">
          Ingest a knowledge base once, then compare 14 different RAG
          pipelines against it side by side.
        </p>
      </div>

      {modelsInfo && (
        <div className="panel-section">
          <span className="section-label">Models</span>
          <div className="model-row"><span>chat</span><span>{modelsInfo.chat_model}</span></div>
          <div className="model-row"><span>embed</span><span>{modelsInfo.embed_model}</span></div>
          <div className="model-row"><span>vision</span><span>{modelsInfo.vision_model}</span></div>
        </div>
      )}

      <div className="panel-section">
        <span className="section-label">Knowledge base</span>
        <div className="stat-box">
          <div className="stat-number">{stats?.total_chunks ?? 0}</div>
          <div className="stat-caption">chunks indexed</div>
          {sourceEntries.length > 0 && (
            <div className="source-list">
              {sourceEntries.map(([name, count]) => (
                <div className="source-item" key={name}>
                  <span title={name}>{name}</span>
                  <span>{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="panel-section">
        <span className="section-label">Ingest documents</span>
        <div
          className={`dropzone ${dragging ? 'dragging' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            handleFiles(e.dataTransfer.files)
          }}
        >
          {busy ? <span className="spinner" /> : 'Drop files here, or click to browse'}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        <span className="small-note">— or ingest a path already on the server —</span>
        <input
          className="text-input"
          placeholder="/path/to/docs or file.pdf"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleIngestPath()}
        />
        <button className="btn" onClick={handleIngestPath} disabled={busy || !path.trim()}>
          Ingest path
        </button>

        {message && (
          <div className={message.type === 'error' ? 'error-banner' : 'success-banner'}>
            {message.text}
          </div>
        )}

        <span className="small-note">
          Supports .txt .md .pdf .docx, images (.png/.jpg — described via a
          vision model), and most source code files.
        </span>
      </div>
    </aside>
  )
}


