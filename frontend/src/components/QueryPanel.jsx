import React, { useState } from 'react'
import AnswerCard from './AnswerCard.jsx'

export default function QueryPanel({ mode, modeLabel, onAsk, results, asking }) {
  const [query, setQuery] = useState('')
  const [detailed, setDetailed] = useState(false)
  const [k, setK] = useState(4)
  const [retriever, setRetriever] = useState('hybrid')
  const [reranker, setReranker] = useState('llm')

  const isModular = mode === '11'

  function submit() {
    if (!query.trim() || asking) return
    onAsk({ query: query.trim(), detailed, k: Number(k) || 4, retriever, reranker })
    setQuery('')
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="query-box">
      <textarea
        className="query-textarea"
        placeholder="Ask a question about your ingested documents…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
      />

      <div className="controls-row">
        <div className="control-group">
          <label htmlFor="k-input">k</label>
          <input id="k-input" type="number" min="1" max="20" value={k} onChange={(e) => setK(e.target.value)} />
        </div>

        <label className="toggle-label">
          <input type="checkbox" checked={detailed} onChange={(e) => setDetailed(e.target.checked)} />
          detailed answer
        </label>

        {isModular && (
          <>
            <div className="control-group">
              <label htmlFor="retriever-select">retriever</label>
              <select id="retriever-select" value={retriever} onChange={(e) => setRetriever(e.target.value)}>
                <option value="vector">vector</option>
                <option value="keyword">keyword</option>
                <option value="hybrid">hybrid</option>
                <option value="graph">graph</option>
              </select>
            </div>
            <div className="control-group">
              <label htmlFor="reranker-select">reranker</label>
              <select id="reranker-select" value={reranker} onChange={(e) => setReranker(e.target.value)}>
                <option value="none">none</option>
                <option value="llm">llm</option>
              </select>
            </div>
          </>
        )}

        <button className="btn btn-primary" onClick={submit} disabled={asking || !query.trim()}>
          {asking ? <span className="spinner" /> : 'Ask'}
        </button>

        <span className="small-note" style={{ marginLeft: 'auto' }}>Enter to ask · Shift+Enter for newline</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 8 }}>
        {results.length === 0 ? (
          <div className="empty-state">
            <div className="glyph">⚡</div>
            Ingest documents in the sidebar, pick a strategy, then ask a question.
          </div>
        ) : (
          results.map((r) => <AnswerCard key={r.id} item={r} />)
        )}
      </div>
    </div>
  )
}
