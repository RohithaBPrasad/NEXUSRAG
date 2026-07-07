import React, { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ModeSelector from './components/ModeSelector.jsx'
import QueryPanel from './components/QueryPanel.jsx'
import { fetchModes, fetchStats, runQuery } from './api.js'

export default function App() {
  const [modes, setModes] = useState([])
  const [modelsInfo, setModelsInfo] = useState(null)
  const [mode, setMode] = useState('1')
  const [stats, setStats] = useState({ total_chunks: 0, sources: {} })
  const [results, setResults] = useState([])
  const [asking, setAsking] = useState(false)
  const [loadError, setLoadError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const [modesRes, statsRes] = await Promise.all([fetchModes(), fetchStats()])
        setModes(modesRes.modes)
        setModelsInfo({
          chat_model: modesRes.chat_model,
          embed_model: modesRes.embed_model,
          vision_model: modesRes.vision_model,
        })
        setStats(statsRes)
      } catch (e) {
        setLoadError(e.message)
      }
    }
    load()
  }, [])

  const currentModeLabel = modes.find((m) => m.key === mode)?.label || mode

  async function handleAsk({ query, detailed, k, retriever, reranker }) {
    const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now())
    const placeholder = { id, query, modeLabel: currentModeLabel, loading: true }
    setResults((prev) => [placeholder, ...prev])
    setAsking(true)
    try {
      const res = await runQuery({ mode, query, detailed, k, retriever, reranker })
      setResults((prev) =>
        prev.map((r) => (r.id === id ? { ...r, ...res, modeLabel: currentModeLabel, loading: false } : r))
      )
    } catch (e) {
      setResults((prev) =>
        prev.map((r) => (r.id === id ? { ...r, error: e.message, loading: false } : r))
      )
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        modelsInfo={modelsInfo}
        stats={stats}
        onStatsChange={(s) => setStats((prev) => ({ ...prev, ...s }))}
      />
      <main className="main">
        <div className="top-bar">
          <h1>Query console</h1>
          <span className="top-bar-sub">
            {stats.total_chunks} chunk(s) indexed · mode: {currentModeLabel}
          </span>
        </div>

        {loadError && (
          <div className="error-banner" style={{ marginBottom: 16 }}>
            Could not reach the backend at the configured API URL: {loadError}
          </div>
        )}

        <ModeSelector modes={modes} mode={mode} onModeChange={setMode} />

        <div style={{ height: 20 }} />

        <QueryPanel
          mode={mode}
          modeLabel={currentModeLabel}
          onAsk={handleAsk}
          results={results}
          asking={asking}
        />
      </main>
    </div>
  )
}
