import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Map a raw top-level error string to a human-friendly message. */
function friendlyError(raw) {
  if (!raw) return 'Something went wrong. Please try again.'
  const e = raw.toLowerCase()
  if (e.includes('pdf upload'))
    return 'PDF upload failed — make sure the file is a valid PDF under 10 MB.'
  if (e.includes('api key') || e.includes('invalid api') || e.includes('401') || e.includes('authentication'))
    return 'Your OpenAI API key is invalid or missing. Check your .env file and restart the server.'
  if (e.includes('connection refused') || e.includes('failed to fetch') || e.includes('networkerror'))
    return "Can't reach the backend — make sure uvicorn is running on port 8001."
  if (e.includes('timeout'))
    return 'The analysis timed out. Try again — large profiles can take a while.'
  if (e.includes('server error 500'))
    return 'The server hit an unexpected error. Check the uvicorn terminal for details.'
  return 'Something went wrong. Please try again.'
}

/** Convert the raw pipeline errors array into deduplicated, user-friendly strings. */
function friendlyWarnings(errors) {
  if (!errors?.length) return []
  const seen = new Set()
  const out = []
  const add = (msg) => { if (!seen.has(msg)) { seen.add(msg); out.push(msg) } }

  for (const raw of errors) {
    const e = (raw || '').toLowerCase()
    // Silently ignore "no documents provided" — user chose not to add them
    if (e.includes('no documents provided') || e.includes('no pdf or google')) continue
    if (e.includes('linkedin'))
      add("LinkedIn couldn't be scraped — LinkedIn blocks automated access. Your other sources were still analysed.")
    else if (e.includes('rate limit') || (e.includes('github') && e.includes('403')))
      add('GitHub rate limit reached. Add a GITHUB_TOKEN to your .env file to fix this.')
    else if (e.includes('github') && (e.includes('not found') || e.includes('404')))
      add("GitHub profile not found — double-check your username.")
    else if (e.includes('github'))
      add("GitHub profile couldn't be fully loaded.")
    else if (e.includes('leetcode'))
      add("LeetCode stats couldn't be loaded — your profile may be set to private.")
    else if (e.includes('pdf'))
      add("Couldn't read your PDF — make sure it's a text-based PDF, not a scanned image.")
    else if (e.includes('google doc') || e.includes("doc '"))
      add("Couldn't access a Google Doc — ensure it's shared (View access) with your service account.")
    else if (e.includes('missing or errored from'))
      add("Some sources couldn't be scanned — analysis was based on available data.")
    else if (e.includes('json_parse') || e.includes('parse_failed'))
      add('An analysis step had a formatting hiccup — results may be incomplete. Try again if they look off.')
    // All other internal messages are suppressed silently
  }
  return out
}

// ─── Constants ────────────────────────────────────────────────────────────────

const SCANNER_NODES  = ['github_scanner', 'leetcode_scanner', 'linkedin_scanner', 'google_docs_scanner']
const ANALYSIS_NODES = ['gap_analysis', 'plan_generator', 'resume_tuner']

const NODE_META = {
  github_scanner:      { label: 'GitHub',       icon: '⌥' },
  leetcode_scanner:    { label: 'LeetCode',     icon: '⚡' },
  linkedin_scanner:    { label: 'LinkedIn',     icon: '◈' },
  google_docs_scanner: { label: 'Documents',    icon: '◻' },
  gap_analysis:        { label: 'Gap Analysis', icon: '◎' },
  plan_generator:      { label: 'Learning Plan',icon: '◐' },
  resume_tuner:        { label: 'Resume Tuner', icon: '◑' },
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

const CHAT_SUGGESTIONS = [
  'Explain my biggest gaps',
  'What should I learn first?',
  'Shorten the plan to 4 weeks',
  'How long to close the full gap?',
  'What roles match my current skills?',
]

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
  const pct   = Math.round((score || 0) * 100)
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
      {gap.experience_gap && <div className="info-box">{gap.experience_gap}</div>}
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
      {plan.plan_summary && <div className="summary-box"><p>{plan.plan_summary}</p></div>}
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
            {week.project && <div className="week-project">Project: {week.project}</div>}
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
                {r.estimated_hours && <span className="resource-hours">{r.estimated_hours}h</span>}
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

// ─── Chat tab ─────────────────────────────────────────────────────────────────

function ChatTab({ analysisContext }) {
  const gapPct = Math.round((analysisContext?.skill_gap?.gap_score || 0) * 100)
  const greeting = gapPct > 0
    ? `I've finished analysing your profile — you're a **${gapPct}% match** for this role. Ask me anything: why a skill is missing, how to tweak the learning plan, what to tackle first, or anything else.`
    : `Analysis complete. Ask me anything about your gaps, learning plan, or how to improve your profile for this role.`

  const [messages, setMessages] = useState([{ role: 'assistant', content: greeting }])
  const [input, setInput]       = useState('')
  const [streaming, setStreaming] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(true)
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text) => {
    const trimmed = text.trim()
    if (!trimmed || streaming) return

    setShowSuggestions(false)
    setInput('')
    setStreaming(true)

    const history = messages.slice(1) // exclude greeting
    setMessages(prev => [...prev, { role: 'user', content: trimmed }])
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      const resp = await fetch('/api/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, history, context: analysisContext }),
      })

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(line.slice(6))
            if (evt.type === 'token') {
              setMessages(prev => {
                const next = [...prev]
                next[next.length - 1] = {
                  role: 'assistant',
                  content: next[next.length - 1].content + evt.content,
                }
                return next
              })
            }
          } catch (_) {}
        }
      }
    } catch {
      setMessages(prev => {
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', content: "Something went wrong. Please try again." }
        return next
      })
    }

    setStreaming(false)
    inputRef.current?.focus()
  }

  return (
    <div className="chat-tab">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
            {msg.role === 'assistant' && <span className="chat-avatar">◈</span>}
            <div className="chat-bubble">
              {msg.content
                ? msg.role === 'assistant'
                  ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  : msg.content
                : <span className="chat-cursor">▋</span>
              }
            </div>
          </div>
        ))}

        {showSuggestions && (
          <div className="chat-suggestions">
            {CHAT_SUGGESTIONS.map(s => (
              <button key={s} className="suggestion-chip" onClick={() => send(s)} disabled={streaming}>
                {s}
              </button>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          ref={inputRef}
          type="text"
          placeholder="Ask about your gaps, tweak the learning plan…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) send(input) }}
          disabled={streaming}
        />
        <button
          className="chat-send-btn"
          onClick={() => send(input)}
          disabled={!input.trim() || streaming}
        >
          {streaming ? <span className="spinner" /> : '↑'}
        </button>
      </div>
    </div>
  )
}

// ─── PDF Upload field ─────────────────────────────────────────────────────────

function PdfUploadField({ pdfFile, setPdfFile, loading }) {
  const fileRef = useRef(null)

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type === 'application/pdf') setPdfFile(file)
  }

  return (
    <div className="field">
      <label>Resume PDF <span className="label-hint">— optional</span></label>
      <div
        className={`pdf-upload-area ${pdfFile ? 'has-file' : ''} ${loading ? 'disabled' : ''}`}
        onClick={() => !loading && fileRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={handleDrop}
      >
        {pdfFile ? (
          <div className="pdf-file-info">
            <span className="pdf-icon">⬜</span>
            <span className="pdf-name">{pdfFile.name}</span>
            <span className="pdf-size">{formatBytes(pdfFile.size)}</span>
            <button
              type="button"
              className="pdf-remove"
              onClick={e => { e.stopPropagation(); setPdfFile(null) }}
              disabled={loading}
            >✕</button>
          </div>
        ) : (
          <div className="pdf-placeholder">
            <span className="pdf-upload-icon">↑</span>
            <span>Click or drag to upload PDF</span>
          </div>
        )}
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={e => setPdfFile(e.target.files[0] || null)}
      />
    </div>
  )
}

// ─── Google Docs URL list ─────────────────────────────────────────────────────

function GoogleDocsField({ urls, setUrls, loading }) {
  const handleChange = (i, val) => setUrls(prev => prev.map((u, j) => j === i ? val : u))
  const addRow    = () => setUrls(prev => [...prev, ''])
  const removeRow = (i) => setUrls(prev => prev.filter((_, j) => j !== i))

  return (
    <div className="field">
      <label>
        Google Docs
        <span className="label-hint"> — paste full URLs or doc IDs, optional</span>
      </label>
      <div className="gdocs-list">
        {urls.map((url, i) => (
          <div key={i} className="gdocs-row">
            <input
              type="text"
              placeholder="https://docs.google.com/document/d/…"
              value={url}
              onChange={e => handleChange(i, e.target.value)}
              disabled={loading}
            />
            {urls.length > 1 && (
              <button type="button" className="gdocs-remove" onClick={() => removeRow(i)} disabled={loading}>
                ✕
              </button>
            )}
          </div>
        ))}
      </div>
      <button type="button" className="gdocs-add-btn" onClick={addRow} disabled={loading}>
        + Add another doc
      </button>
    </div>
  )
}

// ─── Input form ───────────────────────────────────────────────────────────────

function InputForm({ form, setForm, pdfFile, setPdfFile, googleDocsUrls, setGoogleDocsUrls, onSubmit, loading }) {
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

      <PdfUploadField pdfFile={pdfFile} setPdfFile={setPdfFile} loading={loading} />
      <GoogleDocsField urls={googleDocsUrls} setUrls={setGoogleDocsUrls} loading={loading} />

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? <><span className="spinner" /> Analysing…</> : 'Analyse My Profile'}
      </button>
    </form>
  )
}

// ─── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [form, setForm] = useState({
    job_description:   SAMPLE_JD,
    github_username:   '',
    leetcode_username: '',
    linkedin_url:      '',
  })
  const [pdfFile, setPdfFile]               = useState(null)
  const [googleDocsUrls, setGoogleDocsUrls] = useState([''])

  const [status, setStatus]         = useState('idle')
  const [nodeStatus, setNodeStatus] = useState({})
  const [results, setResults]       = useState({})
  const [errorMsg, setErrorMsg]     = useState('')
  const [activeTab, setActiveTab]   = useState('gap')
  const readerRef = useRef(null)

  const handleEvent = (event) => {
    if (event.type === 'start') {
      const running = {}
      SCANNER_NODES.forEach(n => { running[n] = 'running' })
      setNodeStatus(running)
      return
    }

    if (event.type === 'node_done') {
      const { node, data } = event
      setNodeStatus(prev => ({ ...prev, [node]: data?.errors?.length ? 'error' : 'done' }))
      if (node === 'aggregate')    setNodeStatus(prev => ({ ...prev, gap_analysis:  'running' }))
      if (node === 'gap_analysis') setNodeStatus(prev => ({ ...prev, plan_generator:'running' }))
      if (node === 'plan_generator') setNodeStatus(prev => ({ ...prev, resume_tuner:'running' }))
      setResults(prev => ({ ...prev, ...data }))
      return
    }

    if (event.type === 'done') { setStatus('done'); return }
    if (event.type === 'error') { setStatus('error'); setErrorMsg(event.message || '') }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setStatus('running')
    setNodeStatus({})
    setResults({})
    setErrorMsg('')
    setActiveTab('gap')

    try {
      // Step 1 — upload PDF if selected
      let pdf_path = ''
      if (pdfFile) {
        const fd = new FormData()
        fd.append('file', pdfFile)
        const uploadResp = await fetch('/api/upload-pdf', { method: 'POST', body: fd })
        if (!uploadResp.ok) throw new Error('PDF upload failed')
        pdf_path = (await uploadResp.json()).pdf_path
      }

      // Step 2 — filter blank Google Docs entries
      const google_docs_ids = googleDocsUrls.map(u => u.trim()).filter(Boolean)

      // Step 3 — stream analysis
      const resp = await fetch('/api/analyze/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ...form, pdf_path, google_docs_ids }),
      })
      if (!resp.ok) throw new Error(`server error ${resp.status}`)

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      readerRef.current = reader
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()
        for (const part of parts) {
          const line = part.trim()
          if (line.startsWith('data: ')) {
            try { handleEvent(JSON.parse(line.slice(6))) } catch (_) {}
          }
        }
      }

      setStatus(prev => prev === 'running' ? 'done' : prev)
    } catch (err) {
      setStatus('error')
      setErrorMsg(err.message)
    }
  }

  const gap     = results.skill_gap
  const plan    = results.learning_plan
  const resume  = results.tuned_resume
  const warnings = friendlyWarnings(results.errors)

  const TABS = [
    { key: 'gap',    label: 'Gap Analysis' },
    { key: 'plan',   label: 'Learning Plan' },
    { key: 'resume', label: 'Tuned Resume' },
    { key: 'chat',   label: '◈ Ask AI' },
  ]

  return (
    <div className="app">
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
        {/* Left column */}
        <section className="col-left">
          <div className="card">
            <h2 className="card-title">Your Profile</h2>
            <InputForm
              form={form}
              setForm={setForm}
              pdfFile={pdfFile}
              setPdfFile={setPdfFile}
              googleDocsUrls={googleDocsUrls}
              setGoogleDocsUrls={setGoogleDocsUrls}
              onSubmit={handleSubmit}
              loading={status === 'running'}
            />
          </div>
        </section>

        {/* Right column */}
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
              <strong>Couldn't complete the analysis</strong>
              <p>{friendlyError(errorMsg)}</p>
            </div>
          )}

          {warnings.length > 0 && status !== 'error' && (
            <div className="card notice-card">
              <ul className="notice-list">
                {warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}

          {status === 'done' && (
            <div className="card results-card">
              <div className="tab-bar">
                {TABS.map(tab => (
                  <button
                    key={tab.key}
                    className={`tab-btn ${activeTab === tab.key ? 'active' : ''} ${tab.key === 'chat' ? 'tab-btn-chat' : ''}`}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {activeTab === 'gap'    && <GapTab    gap={gap} />}
              {activeTab === 'plan'   && <PlanTab   plan={plan} />}
              {activeTab === 'resume' && <ResumeTab resume={resume} />}
              {activeTab === 'chat'   && <ChatTab   analysisContext={results} />}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
