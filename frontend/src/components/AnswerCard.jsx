import React, { useState } from 'react'

export default function AnswerCard({ item }) {
  const [showChunks, setShowChunks] = useState(false)

  if (item.loading) {
    return (
      <div className="answer-card">
        {item.query && <div className="answer-question">&gt; {item.query}</div>}
        <div className="answer-header">
          <span className="spinner" /> running {item.modeLabel}…
        </div>
      </div>
    )
  }

  if (item.error) {
    return (
      <div className="answer-card">
        {item.query && <div className="answer-question">&gt; {item.query}</div>}
        <div className="error-banner">{item.error}</div>
      </div>
    )
  }

  return (
    <div className="answer-card">
      {item.query && <div className="answer-question">&gt; {item.query}</div>}

      <div className="answer-header">
        <span className="mode-tag">{item.modeLabel}</span>
        <span>· {item.chunks_count} chunk(s)</span>
      </div>

      <div className={`answer-body ${item.is_code_mode ? 'code-mode' : ''}`}>
        {item.answer}
      </div>

      <div className="answer-meta">
        {item.sources && item.sources !== 'none' && (
          <span className="meta-chip">sources: {item.sources}</span>
        )}
      </div>

      {item.notes && <div className="answer-notes">{item.notes}</div>}

      {item.chunks_raw?.length > 0 && (
        <div>
          <button className="chunks-toggle" onClick={() => setShowChunks((v) => !v)}>
            {showChunks ? '▾ hide retrieved chunks' : '▸ show retrieved chunks'}
          </button>
          {showChunks && item.chunks_raw.map((c, i) => (
            <div className="chunk-preview" key={i}>
              <span className="chunk-source">[{i + 1}] {c.source}</span>
              {c.text}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
