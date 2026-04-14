import { useState, useRef } from 'react'

// ─── Constants ────────────────────────────────────────────────────────────────

const SCANNER_NODES = ['github_scanner', 'leetcode_scanner', 'linkedin_scanner', 'google_docs_scanner']
const ANALYSIS_NODES = ['gap_analysis', 'plan_generator', 'resume_tuner']

const NODE_META = {
  github_scanner:      { label: 'GitHub',      icon: '⌥' },
  leetcode_scanner:    { label: 'LeetCode',    icon: '⚡' },
  linkedin_scanner:    { label: 'LinkedIn',    icon: '◈' },
  google_docs_scanner: { label: 'Google Docs', icon: '◻' },
  gap_analysis:        { label: 'Gap Analysis',    icon: '◎' },
  plan_generator:      { label: 'Learning Plan',   icon: '◐' },
  resume_tuner:        { label: 'Resume Tuner',    icon: '◑' },
}

const SAMPLE_JD = `Senior Machine Learning Engineer — FinTech AI Platform

Requirements:
- 5+ years of software engineering, 3+ years of production ML
- Python (NumPy, Pandas, Scikit-learn, PyTorch or TensorFlow)
- MLOps: ML pipelines, model serving, monitoring (MLflow, Kubeflow)
- Docker and Kubernetes
- Distributed data processing (Spark or Dask)
- System design and distributed systems
- PostgreSQL, Redis
- CI/CD pipelines (GitHub Actions, Jenkins)

Nice to have:
- LLMs and LangChain/LangGraph experience
- Financial domain (fraud detection, risk modelling)
- Kafka / streaming systems
- AWS or GCP`

// ─── Small reusable components ────────────────────────────────────────────────

function StatusDot({ status }) {
  if (status === 'done')    return <span className="dot dot-done">✓</span>
  if (status === 'running') return <span className="dot dot-running" />
  if (status === 'error')   return <span className="dot dot-error">✕</span>
  return <span className="dot dot-idle" />
}

function Badge({ children, variant = 'default' }) {
  return <span className={`badge badge-${variant}`}>{children}</span>
}

function ScoreBar({ score }) {
  const pct = Math.round((score || 0) * 100)
  const color = pct >= 70 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444'
  return (
    <div className="score-wrap">
      <div className="score-bar-bg">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="score-label" style={{ color }}>{pct}% match</span>
    </div>
  )
}

// ─── Progress panel ───────────────────────────────────────────────────────────

function ProgressPanel({ nodeStatus }) {
  return (
    <div className="card progress-card">
      <h3 className="section-title">Pipeline</h3>

      <div className="progress-row">
        {SCANNER_NODES.map(node => (
          <div key={node} className={`progress-item ${nodeStatus[node] || 'idle'}`}>
            <StatusDot status={nodeStatus[node]} />
            <span>{NODE_META[node].label}</span>
          </div>
        ))}
      </div>

      <div className="progress-divider" />

      <div className="analysis-steps">
        {ANALYSIS_NODES.map(node => (
          <div key={node} className={`analysis-step ${nodeStatus[node] || 'idle'}`}>
            <StatusDot status={nodeStatus[node]} />
            <span>{NODE_META[node].label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Results tabs ─────────────────────────────────────────────────────────────

function GapTab({ gap }) {
  if (!gap) return <div className="empty">Gap analysis not available.</div>
  return (
    <div className="tab-content">
      <ScoreBar score={gap.gap_score} />

      {gap.experience_gap && (
        <div className="info-box">{gap.experience_gap}</div>
      )}

      <div className="skill-grid">
        <div className="skill-col">
          <h4 className="skill-col-title green">Strong Matches ({gap.strong_matches?.length || 0})</h4>
          <div className="skill-chips">
            {gap.strong_matches?.map(s => <Badge key={s} variant="green">{s}</Badge>)}
          </div>
        </div>
        <div className="skill-col">
          <h4 className="skill-col-title red">Missing ({gap.missing_skills?.length || 0})</h4>
          <div className="skill-chips">
            {gap.missing_skills?.map(s => <Badge key={s} variant="red">{s}</Badge>)}
          </div>
        </div>
        <div className="skill-col">
          <h4 className="skill-col-title amber">Partial ({gap.partial_skills?.length || 0})</h4>
          <div className="skill-chips">
            {gap.partial_skills?.map(s => <Badge key={s} variant="amber">{s}</Badge>)}
          </div>
        </div>
      </div>

      {gap.analysis_summary && (
        <div className="summary-box">
          <h4>Analysis</h4>
          <p>{gap.analysis_summary}</p>
        </div>
      )}
    </div>
  )
}

function PlanTab({ plan }) {
  if (!plan) return <div className="empty">Learning plan not available.</div>
  return (
    <div className="tab-content">
      {plan.plan_summary && (
        <div className="summary-box">
          <p>{plan.plan_summary}</p>
        </div>
      )}

      <div className="plan-meta">
        <span><strong>{plan.total_weeks}</strong> weeks total</span>
        {plan.priority_order?.length > 0 && (
          <span>Priority: {plan.priority_order.slice(0, 4).join(' → ')}</span>
        )}
      </div>

      <div className="weeks-list">
        {plan.weekly_plan?.map(week => (
          <div key={week.week} className="week-card">
            <div className="week-header">
              <span className="week-num">Week {week.week}</span>
              <span className="week-hours">{week.daily_commitment_hours}h/day</span>
            </div>
            <div className="week-skills">
              {week.focus_skills?.map(s => <Badge key={s} variant="blue">{s}</Badge>)}
            </div>
            {week.milestones?.length > 0 && (
              <ul className="milestones">
                {week.milestones.map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            )}
            {week.project && (
              <div className="week-project">Project: {week.project}</div>
            )}
          </div>
        ))}
      </div>

      {plan.recommended_resources?.length > 0 && (
        <div className="resources-section">
          <h4>Recommended Resources</h4>
          <div className="resources-list">
            {plan.recommended_resources.map((r, i) => (
              <div key={i} className="resource-item">
                <span className="resource-skill">{r.skill}</span>
                <span className="resource-title">{r.title}</span>
                <span className="resource-platform">{r.platform}</span>
                {r.estimated_hours && (
                  <span className="resource-hours">{r.estimated_hours}h</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ResumeTab({ resume }) {
  if (!resume) return <div className="empty">Tuned resume not available.</div>
  const sections = resume.tuned_sections || {}
  return (
    <div className="tab-content">
      {resume.ats_keywords_added?.length > 0 && (
        <div className="ats-section">
          <h4>ATS Keywords Added</h4>
          <div className="skill-chips">
            {resume.ats_keywords_added.map(k => <Badge key={k} variant="blue">{k}</Badge>)}
          </div>
        </div>
      )}

      {sections.summary && (
        <div className="resume-section">
          <h4>Professional Summary</h4>
          <p>{sections.summary}</p>
        </div>
      )}

      {sections.skills && (
        <div className="resume-section">
          <h4>Skills</h4>
          <p className="mono">{sections.skills}</p>
        </div>
      )}

      {sections.experience && Array.isArray(sections.experience) && (
        <div className="resume-section">
          <h4>Experience</h4>
          {sections.experience.map((exp, i) => (
            <div key={i} className="exp-block">
              <div className="exp-header">
                <strong>{exp.title}</strong>
                <span>{exp.company}</span>
                <span className="exp-duration">{exp.duration}</span>
              </div>
              <ul className="exp-bullets">
                {exp.bullets?.map((b, j) => <li key={j}>{b}</li>)}
              </ul>
            </div>
          ))}
        </div>
      )}

      {resume.cover_letter_snippet && (
        <div className="resume-section">
          <h4>Cover Letter Opening</h4>
          <p className="cover-letter">{resume.cover_letter_snippet}</p>
        </div>
      )}

      {resume.tuning_notes && (
        <div className="summary-box">
          <h4>What Was Changed</h4>
          <p>{resume.tuning_notes}</p>
        </div>
      )}
    </div>
  )
}

// ─── Input form ───────────────────────────────────────────────────────────────

function InputForm({ form, setForm, onSubmit, loading }) {
  const handleChange = (field) => (e) => setForm(prev => ({ ...prev, [field]: e.target.value }))

  return (
    <form onSubmit={onSubmit} className="input-form">
      <div className="field">
        <label>Job Description</label>
        <textarea
          rows={8}
          placeholder="Paste the full job description here…"
          value={form.job_description}
          onChange={handleChange('job_description')}
          required
          disabled={loading}
        />
      </div>

      <div className="field-row">
        <div className="field">
          <label>GitHub Username</label>
          <input
            type="text"
            placeholder="e.g. torvalds"
            value={form.github_username}
            onChange={handleChange('github_username')}
            required
            disabled={loading}
          />
        </div>
        <div className="field">
          <label>LeetCode Username</label>
          <input
            type="text"
            placeholder="e.g. neal_wu"
            value={form.leetcode_username}
            onChange={handleChange('leetcode_username')}
            required
            disabled={loading}
          />
        </div>
      </div>

      <div className="field">
        <label>LinkedIn Profile URL</label>
        <input
          type="url"
          placeholder="https://www.linkedin.com/in/your-profile/"
          value={form.linkedin_url}
          onChange={handleChange('linkedin_url')}
          required
          disabled={loading}
        />
      </div>

      <div className="field">
        <label>
          Google Docs Resume ID
          <span className="label-hint"> — from the URL: /document/d/<em>ID</em>/edit</span>
        </label>
        <input
          type="text"
          placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
          value={form.google_docs_id}
          onChange={handleChange('google_docs_id')}
          disabled={loading}
        />
      </div>

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? (
          <><span className="spinner" /> Analysing…</>
        ) : (
          'Analyse My Profile'
        )}
      </button>
    </form>
  )
}

// ─── Root App ────────────────────────────────────────────────────────────────

export default function App() {
  const [form, setForm] = useState({
    job_description:   SAMPLE_JD,
    github_username:   '',
    leetcode_username: '',
    linkedin_url:      '',
    google_docs_id:    '',
  })

  const [status, setStatus]           = useState('idle')   // idle | running | done | error
  const [nodeStatus, setNodeStatus]   = useState({})        // node → 'running' | 'done' | 'error'
  const [results, setResults]         = useState({})        // accumulated state patches
  const [errorMsg, setErrorMsg]       = useState('')
  const [activeTab, setActiveTab]     = useState('gap')
  const readerRef = useRef(null)

  const handleEvent = (event) => {
    if (event.type === 'start') {
      // Mark all scanners as running
      const running = {}
      SCANNER_NODES.forEach(n => { running[n] = 'running' })
      setNodeStatus(running)
      return
    }

    if (event.type === 'node_done') {
      const { node, data } = event
      setNodeStatus(prev => ({ ...prev, [node]: data?.errors?.length ? 'error' : 'done' }))

      // After scanners done, mark analysis nodes running in sequence
      if (node === 'aggregate') {
        setNodeStatus(prev => ({ ...prev, gap_analysis: 'running' }))
      }
      if (node === 'gap_analysis') {
        setNodeStatus(prev => ({ ...prev, plan_generator: 'running' }))
      }
      if (node === 'plan_generator') {
        setNodeStatus(prev => ({ ...prev, resume_tuner: 'running' }))
      }

      // Accumulate results
      setResults(prev => ({ ...prev, ...data }))
      return
    }

    if (event.type === 'done') {
      setStatus('done')
      return
    }

    if (event.type === 'error') {
      setStatus('error')
      setErrorMsg(event.message || 'Unknown error')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setStatus('running')
    setNodeStatus({})
    setResults({})
    setErrorMsg('')

    try {
      const resp = await fetch('/api/analyze/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(form),
      })

      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`Server error ${resp.status}: ${text}`)
      }

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      readerRef.current = reader
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // SSE messages are separated by double newlines
        const parts = buffer.split('\n\n')
        buffer = parts.pop()   // keep incomplete trailing chunk

        for (const part of parts) {
          const line = part.trim()
          if (line.startsWith('data: ')) {
            try {
              handleEvent(JSON.parse(line.slice(6)))
            } catch (_) {}
          }
        }
      }

      setStatus(prev => prev === 'running' ? 'done' : prev)
    } catch (err) {
      setStatus('error')
      setErrorMsg(err.message)
    }
  }

  const gap    = results.skill_gap
  const plan   = results.learning_plan
  const resume = results.tuned_resume
  const errors = results.errors || []

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">DeepJobAgent</span>
          </div>
          <span className="logo-tagline">Career gap analyser · powered by LangGraph</span>
        </div>
      </header>

      <main className="main">
        {/* Left column: input */}
        <section className="col-left">
          <div className="card">
            <h2 className="card-title">Your Profile</h2>
            <InputForm
              form={form}
              setForm={setForm}
              onSubmit={handleSubmit}
              loading={status === 'running'}
            />
          </div>
        </section>

        {/* Right column: progress + results */}
        <section className="col-right">
          {status === 'idle' && (
            <div className="idle-state">
              <div className="idle-icon">◎</div>
              <p>Fill in your profile details and paste a job description to get started.</p>
            </div>
          )}

          {(status === 'running' || status === 'done' || status === 'error') && (
            <ProgressPanel nodeStatus={nodeStatus} />
          )}

          {status === 'error' && (
            <div className="card error-card">
              <strong>Error</strong>
              <p>{errorMsg}</p>
            </div>
          )}

          {errors.length > 0 && (
            <div className="card warning-card">
              <strong>Warnings ({errors.length})</strong>
              <ul className="warning-list">
                {errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            </div>
          )}

          {status === 'done' && (
            <div className="card results-card">
              {/* Tab bar */}
              <div className="tab-bar">
                {[
                  { key: 'gap',    label: 'Gap Analysis' },
                  { key: 'plan',   label: 'Learning Plan' },
                  { key: 'resume', label: 'Tuned Resume' },
                ].map(tab => (
                  <button
                    key={tab.key}
                    className={`tab-btn ${activeTab === tab.key ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {activeTab === 'gap'    && <GapTab    gap={gap} />}
              {activeTab === 'plan'   && <PlanTab   plan={plan} />}
              {activeTab === 'resume' && <ResumeTab resume={resume} />}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
