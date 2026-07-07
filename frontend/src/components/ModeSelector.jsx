import React from 'react'
import { MODE_PIPELINES, MODE_DESCRIPTIONS } from '../modeMeta.js'

export default function ModeSelector({ modes, mode, onModeChange }) {
  const pipeline = MODE_PIPELINES[mode] || []
  const description = MODE_DESCRIPTIONS[mode]

  return (
    <div className="panel-section">
      <span className="section-label">Strategy</span>
      <div className="mode-grid">
        {modes.map((m) => (
          <div
            key={m.key}
            className={`mode-item ${m.key === mode ? 'active' : ''}`}
            onClick={() => onModeChange(m.key)}
          >
            <span className="mode-num">{m.key.padStart(2, '0')}</span>
            <span>{m.label}</span>
          </div>
        ))}
      </div>

      {pipeline.length > 0 && (
        <>
          {description && <p className="small-note">{description}</p>}
          <div className="pipeline-strip">
            {pipeline.map((stage, i) => (
              <React.Fragment key={i}>
                <span className={`pipeline-stage ${stage.type}`}>{stage.label}</span>
                {i < pipeline.length - 1 && <span className="pipeline-arrow">→</span>}
              </React.Fragment>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
