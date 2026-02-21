import { useState } from 'react'
import Plot from 'react-plotly.js'

const tabs = ['Upload', 'Schema & Variables', 'EDA', 'Modeling']

const chartLayout = (title, y) => ({
  title,
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  margin: { l: 60, r: 24, t: 56, b: 56 },
  xaxis: { title: 'year_month' },
  yaxis: { title: y },
  legend: { orientation: 'h', y: -0.2 },
})

const DataTable = ({ rows }) => {
  if (!rows?.length) return <p className="muted">No rows.</p>
  const cols = Object.keys(rows[0])
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{String(r[c])}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}

export default function App() {
  const [active, setActive] = useState('Upload')
  const [edaTab, setEdaTab] = useState('Quantitative')
  const [file, setFile] = useState(null)
  const [fileId, setFileId] = useState('')
  const [schema, setSchema] = useState(null)
  const [checklist, setChecklist] = useState([])
  const [eda, setEda] = useState(null)
  const [qualVar, setQualVar] = useState('')
  const [covConf, setCovConf] = useState('')
  const [summary, setSummary] = useState(null)
  const [multiplier, setMultiplier] = useState(1)
  const [channel, setChannel] = useState('')
  const [error, setError] = useState('')

  const upload = async () => {
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('/upload', { method: 'POST', body: form })
    const payload = await res.json()
    if (!res.ok) return setError(payload.detail || 'Upload failed')
    setError('')
    setFileId(payload.file_id)
    setSchema(payload.schema)
    setActive('Schema & Variables')
  }

  const loadChecklist = async () => {
    if (!fileId) return
    const res = await fetch(`/checklist/${fileId}`)
    const payload = await res.json()
    if (!res.ok) return setError(payload.detail || 'Checklist failed')
    setChecklist(payload.items || [])
  }

  const loadEda = async () => {
    if (!fileId) return
    const res = await fetch(`/eda/${fileId}?metric_group=Suggestions`)
    const payload = await res.json()
    if (!res.ok) return setError(payload.detail || 'EDA failed')
    setEda(payload)
    setQualVar(payload.categorical_variables?.[0] || '')
    setCovConf(payload.quantitative_covariate_confounders?.[0] || '')
  }

  const runModel = async () => {
    if (!fileId) return
    const res = await fetch(`/run/${fileId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_multiplier: Number(multiplier), isolate_channel: channel || null }),
    })
    const payload = await res.json()
    if (!res.ok) return setError(payload.detail || 'Run failed')
    setSummary(payload.summary)
  }

  const stacked = (rows) => {
    if (!rows?.length) return []
    const x = rows.map((r) => r.month)
    return Object.keys(rows[0]).filter((k) => k !== 'month').map((k) => ({ type: 'bar', x, y: rows.map((r) => r[k]), name: k }))
  }

  const outcomes = eda?.monthly_counts_by_type?.outcomes || []
  const trxTrace = outcomes[0]?.trx !== undefined ? [{ type: 'bar', x: outcomes.map((r) => r.month), y: outcomes.map((r) => r.trx), name: 'trx', marker: { color: '#0ea5a4' } }] : []
  const nbrxTrace = outcomes[0]?.nbrx !== undefined ? [{ type: 'bar', x: outcomes.map((r) => r.month), y: outcomes.map((r) => r.nbrx), name: 'nbrx', marker: { color: '#7c3aed' } }] : []

  const covRows = eda?.monthly_covariate_trends?.[covConf] || []
  const covTrace = covRows.length ? [{ type: 'scatter', mode: 'lines+markers', x: covRows.map((r) => r.month), y: covRows.map((r) => r[covConf]), name: covConf }] : []

  const corr = eda?.quantitative_correlation_matrix || {}
  const vars = Object.keys(corr)
  const corrTrace = vars.length ? [{ type: 'heatmap', x: vars, y: vars, z: vars.map((r) => vars.map((c) => corr[r][c])), zmin: -1, zmax: 1, colorscale: 'RdBu' }] : []

  return (
    <div className="shell">
      <header className="hero">
        <h1>Pharma Causal Lift Model</h1>
        <p>Modern causal analytics workspace with polished EDA and model diagnostics.</p>
      </header>

      <div className="tabs">{tabs.map((t) => <button key={t} className={`tab ${active === t ? 'active' : ''}`} onClick={() => setActive(t)}>{t}</button>)}</div>
      {error && <div className="card danger">{error}</div>}

      {active === 'Upload' && <section className="card"><h3>Upload Dataset</h3><div className="grid"><input type="file" accept=".xlsx,.xls" onChange={(e) => setFile(e.target.files[0])} /><button onClick={upload} className="primary">Upload & Detect Schema</button></div>{fileId && <p>Loaded file_id: <b>{fileId}</b></p>}</section>}

      {active === 'Schema & Variables' && <section className="card"><h3>Variable Checklist</h3><button onClick={loadChecklist}>Refresh</button>{checklist.map((c, i) => <p key={i}><b>{c.status.toUpperCase()}</b> — {c.label}: {c.detail}</p>)}<h4>Schema</h4><pre>{JSON.stringify(schema, null, 2)}</pre></section>}

      {active === 'EDA' && <>
        <section className="card"><h3>EDA Summary</h3><button className="primary" onClick={loadEda}>Load EDA</button></section>
        {eda && <>
          <div className="subtabs">{['Quantitative', 'Qualitative', 'Correlation'].map((t) => <button key={t} className={`tab ${edaTab === t ? 'active' : ''}`} onClick={() => setEdaTab(t)}>{t}</button>)}</div>

          {edaTab === 'Quantitative' && <>
            <section className="card"><Plot data={stacked(eda.monthly_counts_by_type?.actions)} layout={{ ...chartLayout('Action Type by Action Count', 'action_count'), barmode: 'stack' }} useResizeHandler style={{ width: '100%', height: 380 }} /></section>
            <section className="card"><Plot data={stacked(eda.monthly_counts_by_type?.suggestions)} layout={{ ...chartLayout('Suggestion Type by Suggestion Count', 'suggestion_count'), barmode: 'stack' }} useResizeHandler style={{ width: '100%', height: 380 }} /></section>
            <section className="card"><Plot data={trxTrace} layout={chartLayout('Outcome Type = trx', 'outcome_count')} useResizeHandler style={{ width: '100%', height: 340 }} /></section>
            <section className="card"><Plot data={nbrxTrace} layout={chartLayout('Outcome Type = nbrx', 'outcome_count')} useResizeHandler style={{ width: '100%', height: 340 }} /></section>
            <section className="card"><h4>Covariate & Confound (quantitative only)</h4><select value={covConf} onChange={(e) => setCovConf(e.target.value)}>{(eda.quantitative_covariate_confounders || []).map((c) => <option key={c} value={c}>{c}</option>)}</select><Plot data={covTrace} layout={chartLayout(`${covConf} by year_month`, covConf)} useResizeHandler style={{ width: '100%', height: 340 }} /></section>
          </>}

          {edaTab === 'Qualitative' && <section className="card"><h4>Qualitative Summary</h4><select value={qualVar} onChange={(e) => setQualVar(e.target.value)}>{(eda.categorical_variables || []).map((c) => <option key={c} value={c}>{c}</option>)}</select><DataTable rows={eda.categorical_unique_counts?.[qualVar] || []} /></section>}

          {edaTab === 'Correlation' && <>
            <section className="card"><Plot data={corrTrace} layout={{ title: 'Correlation Matrix (Quantitative Variables)', paper_bgcolor: '#fff', plot_bgcolor: '#fff', margin: { l: 70, r: 30, t: 56, b: 60 } }} useResizeHandler style={{ width: '100%', height: 680 }} /></section>
            <section className="card"><h4>Highly Correlated Insights</h4><DataTable rows={eda.high_correlation_pairs || []} /></section>
            <section className="card"><h4>Variance & Sparsity</h4><DataTable rows={eda.quantitative_diagnostics || []} /></section>
          </>}
        </>}
      </>}

      {active === 'Modeling' && <section className="card"><h3>Modeling</h3><div className="grid"><input type="number" value={multiplier} onChange={(e) => setMultiplier(e.target.value)} step="0.1" /><input value={channel} onChange={(e) => setChannel(e.target.value)} placeholder="Isolate channel" /><button className="primary" onClick={runModel}>Run</button></div>{summary && <><h4>Path A</h4><DataTable rows={summary.path_a?.monthly_rollup || []} /><h4>Path B</h4><DataTable rows={summary.path_b?.monthly_rollup || []} /></>}</section>}
    </div>
  )
}
