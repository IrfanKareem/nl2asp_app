<script>
  import { onMount } from 'svelte'

  // ── State ─────────────────────────────────────────────────────────────────
  let selectedModel  = 'llama'
  let pipelineMode   = 'direct'
  let activeTab      = 'single'
  let modelSearch    = ''
  let modelDropOpen  = false

  let nlInput    = ''
  let cnlOutput  = ''
  let aspOutput  = ''
  let cnlStatus  = ''
  let aspStatus  = ''

  // Input history (last 10 queries)
  let history      = []
  let historyOpen  = false

  // ── Dashboard (multi-model comparison) ───────────────────────────────────
  let dashModels       = {}          // { modelKey: true/false } — selected models
  let dashInputMode    = 'single'    // 'single' | 'batch'
  let dashNLInput      = ''          // single NL input
  let dashFile         = null
  let dashFileName     = ''
  let dashRecords      = []          // parsed JSON records
  let dashResults      = []          // [{id, category, nl, gold_cnl, gold_asp, models:{key:{cnl,asp,syntax_valid,compiled}}}]
  let dashProgress     = 0
  let dashTotal        = 0
  let dashBatchLimit   = 0           // 0 = all records; slider max = dashRecords.length
  let dashRunning       = false
  let dashStopped       = false
  let dashDone          = false
  let dashResumeIndex   = 0
  let dashParsedRecords = []
  let dashFilter       = 'all'
  let dashSearch       = ''
  let dashExpandedRow  = null
  let dashErrorMsg     = ''

  // Batch
  let batchFile     = null
  let batchFileName = ''
  let batchResults  = []
  let batchProgress = 0
  let batchTotal    = 0
  let batchStopped      = false
  let batchResumeIndex  = 0       // index to resume from after stop
  let batchParsedRecords = []     // keep parsed records so resume works
  let batchFilter   = 'all'   // 'all' | 'valid' | 'invalid' | 'compiled' | 'error'
  let batchSearch   = ''
  let expandedRow   = null

  // Loading
  let loadingNL2CNL  = false
  let loadingCNL2ASP = false
  let loadingDirect  = false
  let loadingBatch   = false
  let serverStatus   = 'unknown'
  let loadedModels   = []
  let gpuName        = ''
  let errorMsg       = ''
  let ensembleSource = ''
  let foldDetails    = []

  // Copy feedback
  let copiedNL  = false
  let copiedCNL = false
  let copiedASP = false

  const API = 'http://localhost:8000'

  const MODEL_INFO = {
    // ── LLaMA ──────────────────────────────────────────────────────────────────
    llama:            { label: 'LLaMA-3.1 8B',       sub: 'Fine-tuned · LoRA',        tag: '8B',    icon: '🦙', group: 'LLaMA' },
    llama_HP:         { label: 'LLaMA-3.1 8B HP',    sub: 'Hyperparameter-tuned',      tag: '8B',    icon: '🦙', group: 'LLaMA' },
    // ── Qwen3 ──────────────────────────────────────────────────────────────────
    qwen3:            { label: 'Qwen3-8B',            sub: 'Fine-tuned · LoRA · bfloat16', tag: '8B', icon: '🌐', group: 'Qwen3' },
    // ── T5 ─────────────────────────────────────────────────────────────────────
    t5small:          { label: 'T5-Small',            sub: 'Seq2Seq',                  tag: '60M',   icon: '🔬', group: 'T5 Models' },
    t5large:          { label: 'T5-Large',            sub: 'Seq2Seq',                  tag: '770M',  icon: '🔬', group: 'T5 Models' },
    t53b:             { label: 'T5-3B',               sub: 'Seq2Seq',                  tag: '3B',    icon: '🔬', group: 'T5 Models' },
    // ── KFold ensemble ─────────────────────────────────────────────────────────
    t5small_ensemble: { label: 'KFold Ensemble',      sub: 'Majority vote · 5 folds',  tag: '5×60M', icon: '🎯', group: 'KFold T5-Small' },
    t5small_fold1:    { label: 'T5-Small Fold 1',     sub: 'KFold best checkpoint',    tag: 'F1',    icon: '🔬', group: 'KFold T5-Small' },
    t5small_fold2:    { label: 'T5-Small Fold 2',     sub: 'KFold best checkpoint',    tag: 'F2',    icon: '🔬', group: 'KFold T5-Small' },
    t5small_fold3:    { label: 'T5-Small Fold 3',     sub: 'KFold best checkpoint',    tag: 'F3',    icon: '🔬', group: 'KFold T5-Small' },
    t5small_fold4:    { label: 'T5-Small Fold 4',     sub: 'KFold best checkpoint',    tag: 'F4',    icon: '🔬', group: 'KFold T5-Small' },
    t5small_fold5:    { label: 'T5-Small Fold 5',     sub: 'KFold best checkpoint',    tag: 'F5',    icon: '🔬', group: 'KFold T5-Small' },
  }

  $: filteredModels = Object.entries(MODEL_INFO).filter(([key, info]) => {
    const q = modelSearch.toLowerCase()
    return !q || info.label.toLowerCase().includes(q) || info.group.toLowerCase().includes(q) || key.includes(q)
  })
  $: groupedModels = filteredModels.reduce((acc, [key, info]) => {
    if (!acc[info.group]) acc[info.group] = []
    acc[info.group].push([key, info])
    return acc
  }, {})
  $: selectedInfo = MODEL_INFO[selectedModel]

  // Batch filtered rows
  $: filteredBatch = batchResults.filter(r => {
    const matchFilter =
      batchFilter === 'all'      ? true :
      batchFilter === 'valid'    ? r.syntax_valid :
      batchFilter === 'invalid'  ? !r.syntax_valid :
      batchFilter === 'compiled' ? r.compiled :
      batchFilter === 'error'    ? !r.compiled : true
    const matchSearch = !batchSearch || r.nl.toLowerCase().includes(batchSearch.toLowerCase()) ||
      r.predicted_cnl.toLowerCase().includes(batchSearch.toLowerCase())
    return matchFilter && matchSearch
  })

  // ── Health check ──────────────────────────────────────────────────────────
  onMount(async () => {
    // Load history from sessionStorage
    try { history = JSON.parse(sessionStorage.getItem('nl2asp_history') || '[]') } catch {}
    try {
      const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(5000) })
      if (r.ok) {
        const d      = await r.json()
        serverStatus = 'online'
        loadedModels = d.loaded_models ?? []
        gpuName      = d.gpu ?? ''
        if (loadedModels.length && !loadedModels.includes(selectedModel))
          selectedModel = loadedModels[0]
      } else { serverStatus = 'offline' }
    } catch { serverStatus = 'offline' }
  })

  function handleOutsideClick(e) {
    if (!e.target.closest('.model-selector')) modelDropOpen = false
    if (!e.target.closest('.history-panel') && !e.target.closest('.btn-history')) historyOpen = false
  }

  // ── History ───────────────────────────────────────────────────────────────
  function saveToHistory(nl, cnl, asp, model, syntaxValid, compiled) {
    const entry = {
      id: Date.now(),
      ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      nl, cnl, asp, model,
      syntaxValid, compiled,
    }
    history = [entry, ...history].slice(0, 10)
    try { sessionStorage.setItem('nl2asp_history', JSON.stringify(history)) } catch {}
  }

  function loadFromHistory(entry) {
    nlInput   = entry.nl
    cnlOutput = entry.cnl
    aspOutput = entry.asp
    cnlStatus = entry.syntaxValid ? 'valid' : 'invalid'
    aspStatus = entry.compiled    ? 'compiled' : 'error'
    historyOpen = false
  }

  function clearHistory() {
    history = []
    try { sessionStorage.removeItem('nl2asp_history') } catch {}
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function reset() {
    cnlOutput = ''; aspOutput = ''
    cnlStatus = ''; aspStatus = ''
    errorMsg  = ''
    ensembleSource = ''; foldDetails = []
  }
  function fullReset() { nlInput = ''; reset() }

  async function copyText(text, which) {
    await navigator.clipboard.writeText(text)
    if (which === 'nl')  { copiedNL  = true; setTimeout(() => copiedNL  = false, 1500) }
    if (which === 'cnl') { copiedCNL = true; setTimeout(() => copiedCNL = false, 1500) }
    if (which === 'asp') { copiedASP = true; setTimeout(() => copiedASP = false, 1500) }
  }

  async function postJSON(path, body) {
    const r = await fetch(`${API}${path}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!r.ok) throw new Error(`Server error ${r.status}: ${await r.text()}`)
    return r.json()
  }

  async function runNL2CNL() {
    if (!nlInput.trim()) return
    reset(); loadingNL2CNL = true
    try {
      const modelKey = selectedModel === 't5small_ensemble' ? 't5small_fold1' : selectedModel
      const d = await postJSON('/api/nl2cnl', { nl: nlInput, model: modelKey })
      cnlOutput = d.cnl; cnlStatus = d.syntax_valid ? 'valid' : 'invalid'
    } catch (e) { errorMsg = e.message }
    finally { loadingNL2CNL = false }
  }

  async function runCNL2ASP() {
    if (!cnlOutput.trim()) return
    aspOutput = ''; aspStatus = ''; loadingCNL2ASP = true
    try {
      const d = await postJSON('/api/cnl2asp', { cnl: cnlOutput })
      aspOutput = d.asp; aspStatus = d.compiled ? 'compiled' : 'error'
    } catch (e) { errorMsg = e.message }
    finally { loadingCNL2ASP = false }
  }

  async function runDirect() {
    if (!nlInput.trim()) return
    reset(); loadingDirect = true
    try {
      let d
      if (selectedModel === 't5small_ensemble') {
        d = await postJSON('/api/nl2asp_kfold_ensemble', { nl: nlInput })
      } else {
        d = await postJSON('/api/nl2asp', { nl: nlInput, model: selectedModel })
      }
      cnlOutput = d.cnl; cnlStatus = d.syntax_valid ? 'valid' : 'invalid'
      aspOutput = d.asp; aspStatus = d.compiled     ? 'compiled' : 'error'
      if (d.cnl_source)   ensembleSource = d.cnl_source
      if (d.fold_details) foldDetails    = d.fold_details
      // Save to history
      saveToHistory(nlInput, d.cnl, d.asp, selectedModel, d.syntax_valid, d.compiled)
    } catch (e) { errorMsg = e.message }
    finally { loadingDirect = false }
  }

  // ── Export single result as plain text ───────────────────────────────────
  function exportResult() {
    const lines = [
      '═══════════════════════════════════════════════',
      '  NL2ASP Pipeline Result',
      `  Model : ${selectedInfo?.label ?? selectedModel}`,
      `  Time  : ${new Date().toLocaleString()}`,
      '═══════════════════════════════════════════════',
      '',
      '── Natural Language Input ──────────────────────',
      nlInput,
      '',
      `── CNL Output  [${cnlStatus.toUpperCase()}] ──────────────────────`,
      cnlOutput || '(empty)',
      '',
      `── ASP Output  [${aspStatus.toUpperCase()}] ──────────────────────`,
      aspOutput || '(empty)',
      '',
    ]
    if (ensembleSource) {
      lines.push('── Ensemble Details ────────────────────────────')
      lines.push(`Strategy: ${ensembleSource}`)
      foldDetails.forEach(f => lines.push(`  ${f.fold}: ${f.syntax_valid ? '✓' : '✗'} — ${f.cnl}`))
      lines.push('')
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `nl2asp_result_${Date.now()}.txt`
    a.click()
  }

  // ── Batch ─────────────────────────────────────────────────────────────────
  function onFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    batchFile = f; batchFileName = f.name
    batchResults = []; batchProgress = 0; batchTotal = 0; batchStopped = false
  }

  async function runBatch() {
    if (!batchFile) return
    // fresh run — clear everything and re-parse
    batchResults = []; batchProgress = 0; batchResumeIndex = 0
    loadingBatch = true; errorMsg = ''; batchStopped = false
    try {
      const text = await batchFile.text()
      const json = JSON.parse(text)
      batchParsedRecords = Array.isArray(json) ? json : Array.isArray(json.data_dict) ? json.data_dict : []
      if (!batchParsedRecords.length) throw new Error('No records found in JSON file.')
      batchTotal = batchParsedRecords.length
      await _runBatchFrom(0)
    } catch (e) { errorMsg = e.message }
    finally { loadingBatch = false }
  }

  async function resumeBatch() {
    if (!batchParsedRecords.length) return
    loadingBatch = true; batchStopped = false; errorMsg = ''
    try {
      await _runBatchFrom(batchResumeIndex)
    } catch (e) { errorMsg = e.message }
    finally { loadingBatch = false }
  }

  async function _runBatchFrom(startIdx) {
    for (let i = startIdx; i < batchParsedRecords.length; i++) {
      if (batchStopped) { batchResumeIndex = i; break }
      const rec = batchParsedRecords[i]
      const nl  = rec.NL_V2 ?? rec.nl ?? rec.NL ?? ''
      let result = { nl, gold_cnl: rec.CNL_V2 ?? rec.cnl ?? '', gold_asp: rec.ASP ?? rec.asp ?? '',
                     predicted_cnl: '', asp: '', syntax_valid: false, compiled: false, error: false }
      try {
        const endpoint = selectedModel === 't5small_ensemble' ? '/api/nl2asp_kfold_ensemble' : '/api/nl2asp'
        const body     = selectedModel === 't5small_ensemble' ? { nl } : { nl, model: selectedModel }
        const d = await postJSON(endpoint, body)
        result.predicted_cnl = d.cnl; result.asp = d.asp
        result.syntax_valid  = d.syntax_valid; result.compiled = d.compiled
      } catch { result.predicted_cnl = 'ERROR'; result.asp = 'ERROR'; result.error = true }
      batchResults = [...batchResults, result]
      batchProgress = i + 1
    }
  }

  function stopBatch()  { batchStopped = true }
  function resetBatch() {
    batchResults = []; batchProgress = 0; batchResumeIndex = 0; batchTotal = 0
    batchStopped = false; batchParsedRecords = []; batchFile = null; batchFileName = ''
    errorMsg = ''
  }

  function downloadCSV() {
    const header = ['NL','Predicted CNL','Syntax Valid','Generated ASP','Compiled','Gold CNL','Gold ASP']
    const rows   = batchResults.map(r => [
      `"${r.nl.replace(/"/g,'""')}"`, `"${r.predicted_cnl.replace(/"/g,'""')}"`, r.syntax_valid,
      `"${r.asp.replace(/"/g,'""')}"`, r.compiled,
      `"${r.gold_cnl.replace(/"/g,'""')}"`, `"${r.gold_asp.replace(/"/g,'""')}"`,
    ])
    const csv  = [header, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const a    = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = 'nl2asp_results.csv'; a.click()
  }

  $: syntaxAcc  = batchResults.length ? ((batchResults.filter(r => r.syntax_valid).length / batchResults.length) * 100).toFixed(1) : null
  $: compileAcc = batchResults.length ? ((batchResults.filter(r => r.compiled).length  / batchResults.length) * 100).toFixed(1) : null
  $: errorCount = batchResults.filter(r => r.error).length
  $: isLoading  = loadingNL2CNL || loadingCNL2ASP || loadingDirect || loadingBatch
  $: hasResult  = cnlOutput || aspOutput

  // ── Dashboard helpers ────────────────────────────────────────────────────
  $: dashSelectedKeys = Object.entries(dashModels).filter(([,v])=>v).map(([k])=>k)

  $: dashFilteredResults = dashResults.filter(r => {
    const matchSearch = !dashSearch ||
      r.nl.toLowerCase().includes(dashSearch.toLowerCase())
    if (!matchSearch) return false
    if (dashFilter === 'all') return true
    return dashSelectedKeys.some(k => {
      const m = r.models[k]
      if (!m) return false
      if (dashFilter === 'any_valid')    return m.syntax_valid
      if (dashFilter === 'any_compiled') return m.compiled
      if (dashFilter === 'all_valid')    return dashSelectedKeys.every(k2 => r.models[k2]?.syntax_valid)
      if (dashFilter === 'all_compiled') return dashSelectedKeys.every(k2 => r.models[k2]?.compiled)
      return true
    })
  })

  // Per-model summary stats
  $: dashStats = dashSelectedKeys.map(k => {
    const rows = dashResults.filter(r => r.models[k])
    const n    = rows.length
    return {
      key:      k,
      label:    MODEL_INFO[k]?.label ?? k,
      icon:     MODEL_INFO[k]?.icon  ?? '🤖',
      tag:      MODEL_INFO[k]?.tag   ?? '',
      total:    n,
      syntaxOk: rows.filter(r => r.models[k].syntax_valid).length,
      compiled: rows.filter(r => r.models[k].compiled).length,
      syntaxPct: n ? ((rows.filter(r => r.models[k].syntax_valid).length/n)*100).toFixed(1) : '—',
      compilePct: n ? ((rows.filter(r => r.models[k].compiled).length/n)*100).toFixed(1) : '—',
    }
  })

  function toggleDashModel(key) {
    dashModels = { ...dashModels, [key]: !dashModels[key] }
  }

  function selectAllDashModels() {
    const allLoaded = loadedModels
    const newMap = {}
    allLoaded.forEach(k => newMap[k] = true)
    // also add ensemble key if folds exist
    if (allLoaded.some(k=>k.startsWith('t5small_fold'))) newMap['t5small_ensemble'] = true
    dashModels = newMap
  }

  function deselectAllDashModels() { dashModels = {} }

  function onDashFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    dashFile = f; dashFileName = f.name
    dashRecords = []; dashResults = []; dashProgress = 0; dashTotal = 0
    dashDone = false; dashBatchLimit = 0; dashErrorMsg = ''
    f.text().then(txt => {
      try {
        const j = JSON.parse(txt)
        dashRecords = Array.isArray(j) ? j : Array.isArray(j.data_dict) ? j.data_dict : []
        dashTotal = dashRecords.length
        dashBatchLimit = dashRecords.length   // default = run all
      } catch { dashErrorMsg = 'Invalid JSON file' }
    })
  }

  async function runDashboard() {
    if (dashSelectedKeys.length === 0) { dashErrorMsg = 'Select at least one model.'; return }
    dashErrorMsg = ''; dashResults = []; dashProgress = 0
    dashResumeIndex = 0; dashDone = false
    dashRunning = true; dashStopped = false

    if (dashInputMode === 'single') {
      if (!dashNLInput.trim()) { dashErrorMsg = 'Enter a NL sentence.'; dashRunning = false; return }
      dashParsedRecords = [{ NL_V2: dashNLInput, CNL_V2: '', ASP: '' }]
    } else {
      if (!dashRecords.length) { dashErrorMsg = 'Load a JSON file first.'; dashRunning = false; return }
      const limit = (dashBatchLimit > 0 && dashBatchLimit < dashRecords.length)
        ? dashBatchLimit : dashRecords.length
      dashParsedRecords = dashRecords.slice(0, limit)
    }
    dashTotal = dashParsedRecords.length
    await _runDashFrom(0)
    dashRunning = false
    dashDone = !dashStopped
  }

  async function resumeDashboard() {
    if (!dashParsedRecords.length) return
    dashRunning = true; dashStopped = false; dashDone = false; dashErrorMsg = ''
    await _runDashFrom(dashResumeIndex)
    dashRunning = false
    dashDone = !dashStopped
  }

  async function _runDashFrom(startIdx) {
    for (let i = startIdx; i < dashParsedRecords.length; i++) {
      if (dashStopped) { dashResumeIndex = i; break }
      const rec = dashParsedRecords[i]
      const nl  = rec.NL_V2 ?? rec.nl ?? rec.NL ?? ''
      const row = {
        id:       rec.id       ?? rec.ID       ?? '',
        category: rec.category ?? rec.Category ?? rec.cat ?? '',
        nl,
        gold_cnl: rec.CNL_V2 ?? rec.cnl ?? '',
        gold_asp: rec.ASP    ?? rec.asp  ?? '',
        models: {}
      }
      for (const key of dashSelectedKeys) {
        if (dashStopped) break
        try {
          const endpoint = key === 't5small_ensemble' ? '/api/nl2asp_kfold_ensemble' : '/api/nl2asp'
          const body     = key === 't5small_ensemble' ? { nl } : { nl, model: key }
          const d = await postJSON(endpoint, body)
          row.models[key] = { cnl: d.cnl, asp: d.asp, syntax_valid: d.syntax_valid, compiled: d.compiled }
        } catch(e) {
          row.models[key] = { cnl: 'ERROR', asp: 'ERROR', syntax_valid: false, compiled: false }
        }
      }
      dashResults = [...dashResults, row]
      dashProgress = i + 1
    }
  }

  function stopDashboard()  { dashStopped = true; dashRunning = false }
  function resetDashboard() {
    dashResults = []; dashProgress = 0; dashTotal = 0; dashDone = false
    dashStopped = false; dashResumeIndex = 0; dashParsedRecords = []
    dashFile = null; dashFileName = ''; dashRecords = []
    dashBatchLimit = 0; dashErrorMsg = ''
  }

  function downloadDashCSV() {
    const modelCols = dashSelectedKeys.flatMap(k => {
      const lbl = MODEL_INFO[k]?.label ?? k
      return [`${lbl} CNL`, `${lbl} CNL Valid`, `${lbl} ASP`, `${lbl} Compiled`]
    })
    const header = ['ID', 'Category', 'NL', 'Gold CNL', 'Gold ASP', ...modelCols]
    const rows = dashResults.map(r => {
      const base = [
        `"${String(r.id ?? '').replace(/"/g,'""')}"`,
        `"${String(r.category ?? '').replace(/"/g,'""')}"`,
        `"${r.nl.replace(/"/g,'""')}"`,
        `"${r.gold_cnl.replace(/"/g,'""')}"`,
        `"${r.gold_asp.replace(/"/g,'""')}"`,
      ]
      const modelData = dashSelectedKeys.flatMap(k => {
        const m = r.models[k] ?? { cnl:'', asp:'', syntax_valid:false, compiled:false }
        return [
          `"${m.cnl.replace(/"/g,'""')}"`,
          m.syntax_valid,
          `"${m.asp.replace(/"/g,'""')}"`,
          m.compiled,
        ]
      })
      return [...base, ...modelData]
    })
    const csv  = [header, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const a    = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `nl2asp_comparison_${Date.now()}.csv`
    a.click()
  }
</script>

<svelte:window on:click={handleOutsideClick} />

<!-- ════════════════════════════ STYLES ════════════════════════════ -->
<style>
  :global(*, *::before, *::after) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    font-family: 'DM Sans', 'Syne', system-ui, sans-serif;
    background: #f0f2f5;
    color: #1a1f2e;
    min-height: 100vh;
  }

  /* ── Layout ── */
  .app  { display: grid; grid-template-rows: 54px 1fr; height: 100vh; overflow: hidden; }
  .main { display: grid; grid-template-columns: 268px 1fr; overflow: hidden; }

  /* ── Header ── */
  header {
    background: #ffffff;
    border-bottom: 1px solid #e8ecf0;
    display: flex; align-items: center;
    padding: 0 1.25rem; gap: 1rem;
    z-index: 20; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .brand { display: flex; align-items: center; gap: 0.6rem; }
  .brand-icon {
    width: 32px; height: 32px; border-radius: 9px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
  }
  .brand-name { font-weight: 700; font-size: 0.95rem; color: #0f172a; letter-spacing: -0.02em; }
  .brand-sub  { font-size: 0.58rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
  .hdivider { width: 1px; height: 22px; background: #e8ecf0; }

  .tabs { display: flex; gap: 3px; background: #f0f2f5; padding: 3px; border-radius: 8px; }
  .tab-btn {
    padding: 0.28rem 0.85rem; border-radius: 6px; border: none;
    background: none; color: #64748b; font-size: 0.75rem; font-weight: 600;
    cursor: pointer; transition: all 0.15s; font-family: inherit; letter-spacing: 0.01em;
  }
  .tab-btn:hover { color: #1e293b; }
  .tab-btn.active { background: #ffffff; color: #1e293b; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

  .header-right { margin-left: auto; display: flex; align-items: center; gap: 0.75rem; }
  .gpu-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; font-weight: 500;
    color: #059669; background: #ecfdf5; border: 1px solid #a7f3d0;
    padding: 3px 10px; border-radius: 20px; white-space: nowrap;
  }
  .status-pill { display: flex; align-items: center; gap: 0.4rem; font-size: 0.65rem; color: #94a3b8; font-weight: 500; }
  .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .dot.online  { background: #22c55e; box-shadow: 0 0 0 2px #dcfce7; }
  .dot.offline { background: #ef4444; box-shadow: 0 0 0 2px #fee2e2; }
  .dot.unknown { background: #f59e0b; box-shadow: 0 0 0 2px #fef3c7; }

  /* ── Sidebar ── */
  aside {
    background: #ffffff; border-right: 1px solid #e8ecf0;
    display: flex; flex-direction: column; overflow: hidden;
  }
  .sidebar-inner {
    flex: 1; overflow-y: auto; padding: 1.1rem 0.9rem;
    display: flex; flex-direction: column; gap: 1.4rem;
  }
  .sidebar-inner::-webkit-scrollbar { width: 3px; }
  .sidebar-inner::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 2px; }

  .s-label {
    font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #94a3b8; margin-bottom: 0.55rem;
  }

  /* Model selector */
  .model-selector { position: relative; }
  .model-trigger {
    width: 100%; display: flex; align-items: center; gap: 0.5rem;
    background: #f8fafc; border: 1.5px solid #e8ecf0; border-radius: 10px;
    padding: 0.65rem 0.8rem; cursor: pointer; transition: all 0.15s; text-align: left;
  }
  .model-trigger:hover { border-color: #cbd5e1; background: #f1f5f9; }
  .model-trigger.open  { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); background: #ffffff; }
  .mt-icon  { font-size: 1.1rem; flex-shrink: 0; }
  .mt-text  { flex: 1; min-width: 0; }
  .mt-label { font-size: 0.8rem; font-weight: 600; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .mt-sub   { font-size: 0.6rem; color: #94a3b8; margin-top: 1px; }
  .mt-tag   { font-family: 'JetBrains Mono', monospace; font-size: 0.57rem; font-weight: 600;
              color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe;
              padding: 2px 7px; border-radius: 5px; flex-shrink: 0; }
  .mt-caret { color: #cbd5e1; font-size: 0.6rem; flex-shrink: 0; transition: transform 0.15s; }
  .mt-caret.open { transform: rotate(180deg); }

  .model-dropdown {
    position: absolute; top: calc(100% + 5px); left: 0; right: 0; z-index: 100;
    background: #ffffff; border: 1.5px solid #e8ecf0; border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12); overflow: hidden;
  }
  .msearch-wrap { padding: 0.5rem; border-bottom: 1px solid #f1f5f9; }
  .msearch {
    width: 100%; background: #f8fafc; border: 1px solid #e8ecf0; border-radius: 7px;
    padding: 0.4rem 0.7rem 0.4rem 2rem; color: #1e293b; font-size: 0.75rem;
    outline: none; font-family: inherit; transition: border-color 0.15s;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: 0.55rem center;
  }
  .msearch:focus { border-color: #2563eb; background-color: #ffffff; }
  .msearch::placeholder { color: #cbd5e1; }
  .model-list { max-height: 265px; overflow-y: auto; }
  .model-list::-webkit-scrollbar { width: 3px; }
  .model-list::-webkit-scrollbar-thumb { background: #e2e8f0; }
  .mgroup-label {
    padding: 0.5rem 0.7rem 0.2rem;
    font-size: 0.55rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #cbd5e1;
  }
  .moption {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 0.7rem; cursor: pointer; transition: background 0.08s;
  }
  .moption:hover   { background: #f8fafc; }
  .moption.sel     { background: #eff6ff; }
  .moption.dis     { opacity: 0.3; cursor: not-allowed; pointer-events: none; }
  .mo-icon  { font-size: 0.9rem; flex-shrink: 0; }
  .mo-text  { flex: 1; min-width: 0; }
  .mo-label { font-size: 0.75rem; font-weight: 500; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .mo-sub   { font-size: 0.58rem; color: #94a3b8; }
  .mo-tag   { font-family: 'JetBrains Mono', monospace; font-size: 0.54rem; color: #94a3b8;
              background: #f1f5f9; padding: 1px 5px; border-radius: 3px; flex-shrink: 0; border: 1px solid #e8ecf0; }
  .moption.sel .mo-tag { color: #2563eb; background: #eff6ff; border-color: #bfdbfe; }
  .mo-check { font-size: 0.65rem; color: #2563eb; flex-shrink: 0; font-weight: 700; }
  .mno-result { padding: 0.85rem; text-align: center; font-size: 0.7rem; color: #cbd5e1; }

  /* Pipeline mode */
  .mode-row { display: flex; gap: 0.4rem; }
  .mode-btn {
    flex: 1; padding: 0.55rem 0.3rem; border-radius: 9px;
    border: 1.5px solid #e8ecf0; background: #f8fafc;
    color: #94a3b8; font-family: inherit; font-size: 0.68rem;
    text-align: center; cursor: pointer; transition: all 0.15s; line-height: 1.3;
  }
  .mode-btn:hover { border-color: #cbd5e1; color: #475569; background: #f1f5f9; }
  .mode-btn.active {
    background: #ffffff; border-color: #2563eb; color: #1e293b;
    box-shadow: 0 2px 6px rgba(37,99,235,0.12);
  }
  .mode-icon  { font-size: 1rem; display: block; margin-bottom: 2px; }
  .mode-label { font-weight: 700; font-size: 0.68rem; display: block; }
  .mode-sub   { font-size: 0.56rem; color: #94a3b8; display: block; }
  .mode-btn.active .mode-sub { color: #64748b; }

  /* Server info box */
  .server-box {
    margin-top: auto; padding: 0.8rem; background: #f8fafc;
    border: 1px solid #e8ecf0; border-radius: 10px;
  }
  .srow { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }
  .slabel { font-size: 0.58rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .sval   { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #64748b; }
  .smodels { font-family: 'JetBrains Mono', monospace; font-size: 0.57rem; color: #94a3b8; margin-top: 0.4rem; line-height: 1.7; word-break: break-all; }

  /* ── Content ── */
  .content { display: flex; flex-direction: column; overflow: hidden; background: #f0f2f5; }
  .panel   { flex: 1; overflow-y: auto; padding: 1.25rem 1.4rem; }
  .panel::-webkit-scrollbar { width: 5px; }
  .panel::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 3px; }

  /* Pipeline breadcrumb */
  .pipe-bar {
    display: flex; align-items: center; gap: 0.5rem;
    background: #ffffff; border: 1px solid #e8ecf0; border-radius: 10px;
    padding: 0.6rem 1rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .pnode {
    padding: 3px 11px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.63rem; font-weight: 700; letter-spacing: 0.04em;
  }
  .pnode.nl  { background: #eff6ff;  color: #2563eb; border: 1px solid #bfdbfe; }
  .pnode.cnl { background: #f5f3ff;  color: #7c3aed; border: 1px solid #ddd6fe; }
  .pnode.asp { background: #ecfdf5;  color: #059669; border: 1px solid #a7f3d0; }
  .parr { color: #cbd5e1; font-size: 0.9rem; }
  .pipe-model { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #94a3b8; font-weight: 500; }

  /* Cards */
  .card {
    background: #ffffff; border: 1.5px solid #e8ecf0; border-radius: 12px;
    margin-bottom: 0.9rem; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .card:focus-within { border-color: #93c5fd; box-shadow: 0 0 0 3px rgba(37,99,235,0.07), 0 1px 4px rgba(0,0,0,0.04); }
  .card-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1rem; border-bottom: 1px solid #f1f5f9;
    background: #fafbfc;
  }
  .card-title {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #94a3b8;
  }
  .card-body { padding: 1rem; }

  textarea {
    width: 100%; background: none; border: none; outline: none;
    color: #1a1f2e; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; line-height: 1.8; resize: vertical; min-height: 96px;
  }
  textarea::placeholder { color: #e2e8f0; }

  .code-out {
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    line-height: 1.8; white-space: pre-wrap; min-height: 48px;
  }
  .code-out.cnl-color { color: #5b21b6; }
  .code-out.asp-color { color: #065f46; }
  .ph { color: #d1d5db; font-style: italic; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }

  /* Buttons */
  .btn {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.35rem 0.85rem; border-radius: 7px; border: none;
    font-family: inherit; font-size: 0.72rem; font-weight: 600;
    cursor: pointer; transition: all 0.15s; white-space: nowrap;
  }
  .btn:disabled { opacity: 0.38; cursor: not-allowed; }
  .btn-primary   { background: #2563eb; color: #fff; box-shadow: 0 1px 4px rgba(37,99,235,0.25); }
  .btn-primary:hover:not(:disabled)   { background: #1d4ed8; }
  .btn-secondary { background: #f8fafc; color: #64748b; border: 1.5px solid #e8ecf0; }
  .btn-secondary:hover:not(:disabled) { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }
  .btn-success   { background: #ecfdf5; color: #059669; border: 1.5px solid #a7f3d0; font-weight: 600; }
  .btn-success:hover:not(:disabled)   { background: #d1fae5; }
  .btn-danger    { background: #fef2f2; color: #dc2626; border: 1.5px solid #fecaca; }
  .btn-danger:hover:not(:disabled)    { background: #fee2e2; }
  .btn-warning   { background: #fff7ed; color: #c2410c; border: 1.5px solid #fed7aa; }
  .btn-warning:hover:not(:disabled)   { background: #ffedd5; }
  .btn-ghost-red { background: transparent; color: #ef4444; border: 1.5px solid #fecaca; }
  .btn-ghost-red:hover:not(:disabled) { background: #fee2e2; }
  .btn-copy {
    background: none; border: 1.5px solid transparent; color: #94a3b8;
    padding: 0.3rem 0.65rem; border-radius: 6px; font-size: 0.68rem; font-weight: 600;
    transition: all 0.12s;
  }
  .btn-copy:hover  { background: #f1f5f9; border-color: #e2e8f0; color: #64748b; }
  .btn-copy.copied { color: #059669; border-color: #a7f3d0; background: #ecfdf5; }
  .btn-history {
    background: none; border: 1.5px solid #e8ecf0; color: #64748b;
    padding: 0.3rem 0.65rem; border-radius: 6px; font-size: 0.72rem; font-weight: 600;
    cursor: pointer; transition: all 0.12s; position: relative;
    font-family: inherit;
  }
  .btn-history:hover { background: #f1f5f9; }
  .btn-history.has-items { border-color: #bfdbfe; color: #2563eb; background: #eff6ff; }
  .hist-count {
    position: absolute; top: -5px; right: -5px; background: #2563eb; color: #fff;
    font-size: 0.5rem; font-weight: 700; width: 15px; height: 15px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }
  .row { display: flex; gap: 0.35rem; align-items: center; flex-wrap: wrap; }

  /* Badges */
  .badge {
    display: inline-flex; align-items: center; gap: 0.2rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; font-weight: 700;
    padding: 2px 8px; border-radius: 20px; letter-spacing: 0.02em;
  }
  .badge.valid    { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
  .badge.invalid  { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
  .badge.compiled { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
  .badge.error    { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }

  /* Error bar */
  .err-bar {
    display: flex; align-items: flex-start; gap: 0.6rem;
    background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 1rem;
    color: #dc2626; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
  }

  /* Spinner */
  .spin {
    display: inline-block; width: 10px; height: 10px;
    border: 2px solid rgba(255,255,255,0.35); border-top-color: #fff;
    border-radius: 50%; animation: spin .55s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Ensemble bar */
  .ensemble-bar {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    background: #f5f3ff; border: 1px solid #ddd6fe;
    border-radius: 10px; padding: 0.55rem 1rem; margin-bottom: 1rem;
    font-size: 0.62rem; font-family: 'JetBrains Mono', monospace; color: #7c3aed;
  }
  .fold-chip {
    display: inline-flex; align-items: center; gap: 2px;
    background: #fff; border: 1px solid #e8ecf0;
    border-radius: 5px; padding: 1px 7px; font-size: 0.58rem; color: #94a3b8;
  }
  .fold-chip.valid   { border-color: #a7f3d0; color: #059669; background: #ecfdf5; }
  .fold-chip.invalid { border-color: #fecaca; color: #dc2626; background: #fef2f2; }

  /* History panel */
  .history-panel {
    position: fixed; right: 1rem; top: 62px; width: 340px; z-index: 200;
    background: #ffffff; border: 1.5px solid #e8ecf0; border-radius: 12px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.12); overflow: hidden;
  }
  .hp-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 1rem; border-bottom: 1px solid #f1f5f9; background: #fafbfc;
  }
  .hp-title { font-size: 0.72rem; font-weight: 700; color: #1e293b; }
  .hp-list  { max-height: 400px; overflow-y: auto; }
  .hp-list::-webkit-scrollbar { width: 3px; }
  .hp-list::-webkit-scrollbar-thumb { background: #e2e8f0; }
  .hp-item {
    padding: 0.7rem 1rem; border-bottom: 1px solid #f8fafc;
    cursor: pointer; transition: background 0.08s;
  }
  .hp-item:hover { background: #f8fafc; }
  .hp-item:last-child { border-bottom: none; }
  .hp-nl  { font-size: 0.75rem; color: #1e293b; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
  .hp-meta { display: flex; align-items: center; gap: 0.5rem; }
  .hp-time { font-size: 0.6rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
  .hp-model{ font-size: 0.58rem; color: #94a3b8; background: #f1f5f9; padding: 1px 6px; border-radius: 3px; }
  .hp-empty { padding: 1.5rem; text-align: center; font-size: 0.72rem; color: #94a3b8; }

  /* Progress */
  .prog-wrap  { margin-bottom: 1rem; }
  .prog-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
  .prog-label { font-size: 0.65rem; font-weight: 600; color: #64748b; }
  .prog-pct   { font-family: 'JetBrains Mono', monospace; font-size: 0.63rem; color: #94a3b8; }
  .prog-bg    { height: 6px; background: #e8ecf0; border-radius: 3px; overflow: hidden; }
  .prog-fill  { height: 100%; background: linear-gradient(90deg,#2563eb,#7c3aed); transition: width .3s; border-radius: 3px; }

  /* Stats row */
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.65rem; margin-bottom: 1rem; }
  .stat  { background: #ffffff; border: 1px solid #e8ecf0; border-radius: 10px; padding: 0.75rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .sv    { font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; }
  .sv.b  { color: #2563eb; } .sv.g { color: #059669; } .sv.p { color: #7c3aed; } .sv.r { color: #dc2626; }
  .sl    { font-size: 0.57rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }

  /* Batch toolbar */
  .batch-toolbar {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    margin-bottom: 0.85rem;
  }
  .filter-btn {
    padding: 0.28rem 0.75rem; border-radius: 20px; border: 1.5px solid #e8ecf0;
    background: #fff; color: #94a3b8; font-size: 0.68rem; font-weight: 600;
    cursor: pointer; transition: all 0.12s; font-family: inherit;
  }
  .filter-btn:hover { border-color: #cbd5e1; color: #64748b; }
  .filter-btn.active { background: #eff6ff; border-color: #bfdbfe; color: #2563eb; }
  .batch-search {
    flex: 1; min-width: 150px; background: #fff; border: 1.5px solid #e8ecf0;
    border-radius: 7px; padding: 0.3rem 0.7rem 0.3rem 2rem; color: #1e293b;
    font-size: 0.72rem; outline: none; font-family: inherit; transition: border-color 0.12s;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: 0.55rem center;
  }
  .batch-search:focus { border-color: #2563eb; }
  .batch-search::placeholder { color: #cbd5e1; }

  /* Drop zone */
  .drop {
    border: 2px dashed #e8ecf0; border-radius: 10px;
    padding: 2.2rem; text-align: center; cursor: pointer;
    position: relative; transition: all .15s; background: #fafbfc;
  }
  .drop:hover { border-color: #2563eb; background: #f0f7ff; }
  .drop input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .drop-icon  { font-size: 1.8rem; margin-bottom: 0.5rem; }
  .drop-lbl   { font-size: 0.8rem; color: #64748b; font-weight: 500; }
  .drop-sub   { font-size: 0.62rem; color: #94a3b8; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

  /* Table */
  .tbl-wrap { border: 1px solid #e8ecf0; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
  table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  th {
    text-align: left; padding: 0.55rem 0.85rem;
    background: #fafbfc; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace; font-size: 0.57rem;
    text-transform: uppercase; letter-spacing: .07em; font-weight: 700;
    border-bottom: 1px solid #e8ecf0; white-space: nowrap;
  }
  td {
    padding: 0.5rem 0.85rem; border-bottom: 1px solid #f8fafc;
    font-family: 'JetBrains Mono', monospace; color: #64748b;
    max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f8fafc; }
  tr.expanded td { background: #f0f7ff; white-space: normal; max-width: none; }
  .expand-btn {
    background: none; border: none; cursor: pointer; color: #94a3b8;
    font-size: 0.65rem; padding: 2px 5px; border-radius: 4px; transition: all 0.1s;
    font-family: inherit;
  }
  .expand-btn:hover { background: #f1f5f9; color: #64748b; }
  .expanded-content {
    padding: 0.75rem 0.85rem; background: #f8fafc;
    border-top: 1px solid #e8ecf0;
  }
  .exp-row { margin-bottom: 0.5rem; }
  .exp-label { font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 2px; }
  .exp-val { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #1e293b;
             white-space: pre-wrap; word-break: break-word; }
  .exp-val.cnl { color: #5b21b6; } .exp-val.asp { color: #065f46; }

  /* ── Dashboard ── */
  .dash-layout { display: grid; grid-template-columns: 260px 1fr; gap: 1rem; align-items: start; }
  .dash-sidebar {
    background: #ffffff; border: 1.5px solid #e8ecf0; border-radius: 12px;
    overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    position: sticky; top: 0;
  }
  .dash-sb-header {
    padding: 0.7rem 1rem; background: #fafbfc; border-bottom: 1px solid #f1f5f9;
    display: flex; align-items: center; justify-content: space-between;
  }
  .dash-sb-title { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; }
  .dash-model-list { padding: 0.5rem; max-height: 340px; overflow-y: auto; }
  .dash-model-list::-webkit-scrollbar { width: 3px; }
  .dash-model-list::-webkit-scrollbar-thumb { background: #e2e8f0; }
  .dash-group-label { font-size: 0.55rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #cbd5e1; padding: 0.45rem 0.5rem 0.2rem; }
  .dash-model-item {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.45rem 0.6rem; border-radius: 8px; cursor: pointer;
    transition: background 0.1s; user-select: none;
  }
  .dash-model-item:hover { background: #f8fafc; }
  .dash-model-item.selected { background: #eff6ff; }
  .dash-model-item.unavailable { opacity: 0.3; cursor: not-allowed; }
  .dm-check {
    width: 16px; height: 16px; border-radius: 4px; flex-shrink: 0;
    border: 1.5px solid #e2e8f0; background: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-weight: 800; color: #2563eb;
    transition: all 0.1s;
  }
  .dash-model-item.selected .dm-check { background: #2563eb; border-color: #2563eb; color: #fff; }
  .dm-icon  { font-size: 0.85rem; flex-shrink: 0; }
  .dm-text  { flex: 1; min-width: 0; }
  .dm-label { font-size: 0.72rem; font-weight: 500; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dm-sub   { font-size: 0.57rem; color: #94a3b8; }
  .dm-tag   { font-family: 'JetBrains Mono', monospace; font-size: 0.54rem; color: #94a3b8;
    background: #f1f5f9; padding: 1px 5px; border-radius: 3px; border: 1px solid #e8ecf0; flex-shrink: 0; }
  .dash-model-item.selected .dm-tag { color: #2563eb; background: #eff6ff; border-color: #bfdbfe; }

  /* Input mode toggle */
  .input-mode-row { display: flex; gap: 0.4rem; margin-bottom: 0.85rem; }
  .imode-btn {
    flex: 1; padding: 0.45rem; border-radius: 8px; border: 1.5px solid #e8ecf0;
    background: #f8fafc; color: #94a3b8; font-family: inherit; font-size: 0.7rem;
    font-weight: 600; cursor: pointer; text-align: center; transition: all 0.12s;
  }
  .imode-btn.active { background: #fff; border-color: #2563eb; color: #1e293b; box-shadow: 0 1px 4px rgba(37,99,235,0.1); }

  /* Summary cards — horizontal scroll strip */
  .dash-summary {
    display: flex; gap: 0.55rem; margin-bottom: 1rem;
    overflow-x: auto; padding-bottom: 4px;
    scrollbar-width: thin; scrollbar-color: #e2e8f0 transparent;
  }
  .dash-summary::-webkit-scrollbar { height: 4px; }
  .dash-summary::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 2px; }
  .dash-model-card {
    flex: 0 0 148px;                     /* fixed width — no stretching */
    background: #fff; border: 1.5px solid #e8ecf0; border-radius: 10px;
    padding: 0.65rem 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .dmc-header { display: flex; align-items: center; gap: 0.3rem; margin-bottom: 0.5rem; }
  .dmc-icon   { font-size: 0.85rem; flex-shrink: 0; }
  .dmc-label  { font-size: 0.65rem; font-weight: 700; color: #1e293b; flex: 1; min-width: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dmc-tag    { font-family: 'JetBrains Mono', monospace; font-size: 0.5rem; color: #2563eb;
    background: #eff6ff; border: 1px solid #bfdbfe; padding: 1px 5px; border-radius: 3px; flex-shrink: 0; }
  .dmc-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem; }
  .dmc-metric  { background: #f8fafc; border-radius: 6px; padding: 0.4rem 0.45rem; }
  .dmc-val     { font-size: 1rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; }
  .dmc-val.g   { color: #059669; } .dmc-val.b { color: #2563eb; }
  .dmc-lbl     { font-size: 0.5rem; color: #94a3b8; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
  .dmc-bar-wrap { margin-top: 0.4rem; }
  .dmc-bar-bg   { height: 3px; background: #f1f5f9; border-radius: 2px; overflow: hidden; }
  .dmc-bar-fill { height: 100%; border-radius: 2px; transition: width 0.4s; }
  .dmc-bar-fill.g { background: #22c55e; }
  .dmc-bar-fill.b { background: #2563eb; }
  .dmc-counts   { margin-top: 0.35rem; font-size: 0.52rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }

  /* Comparison table */
  .cmp-table-wrap { border: 1px solid #e8ecf0; border-radius: 10px; overflow-x: auto; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
  .cmp-table { border-collapse: collapse; font-size: 0.68rem; table-layout: auto; }
  .cmp-table th {
    text-align: left; padding: 0.38rem 0.55rem; background: #fafbfc;
    color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.52rem;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 700;
    border-bottom: 1px solid #e8ecf0; white-space: nowrap;
  }
  .cmp-table th.model-header {
    background: #eff6ff; color: #2563eb;
    border-left: 2px solid #bfdbfe; text-align: center;
  }
  .cmp-table td {
    padding: 0.35rem 0.55rem; border-bottom: 1px solid #f8fafc;
    font-family: 'JetBrains Mono', monospace; color: #64748b;
    max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    vertical-align: middle;
  }
  .cmp-table td.model-cell {
    border-left: 2px solid #f1f5f9; text-align: center;
    padding: 0.3rem 0.4rem;
  }
  .cmp-table tr:last-child td { border-bottom: none; }
  .cmp-table tr:hover td { background: #f8fafc; }
  .cmp-table tr.cmp-expanded td { background: #f0f7ff; white-space: normal; max-width: none; }
  /* Sticky first columns so NL stays visible when scrolling right */
  .cmp-table th.col-sticky, .cmp-table td.col-sticky {
    position: sticky; left: 0; z-index: 2; background: #fafbfc;
  }
  .cmp-table td.col-sticky { background: #ffffff; }
  .cmp-table tr:hover td.col-sticky { background: #f8fafc; }
  .cmp-expanded-content { padding: 0.75rem; background: #f8fafc; border-top: 1px solid #e8ecf0; }
  .cmp-model-sections { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap: 0.75rem; }
  .cmp-model-section  { background: #fff; border: 1px solid #e8ecf0; border-radius: 8px; padding: 0.7rem; }
  .cms-title { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: #64748b; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.35rem; }
  .cms-row   { margin-bottom: 0.4rem; }
  .cms-label { font-size: 0.55rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
  .cms-val   { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #1e293b; white-space: pre-wrap; word-break: break-word; }
  .cms-val.cnl { color: #5b21b6; } .cms-val.asp { color: #065f46; }

  /* Progress bar (dashboard) */
  .dash-prog { margin-bottom: 1rem; }
  .dash-prog-header { display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
  .dash-prog-label { font-size: 0.65rem; font-weight: 600; color: #64748b; }
  .dash-prog-pct   { font-family: 'JetBrains Mono', monospace; font-size: 0.63rem; color: #94a3b8; }
  .dash-prog-bg    { height: 6px; background: #e8ecf0; border-radius: 3px; overflow: hidden; }
  .dash-prog-fill  { height: 100%; border-radius: 3px; background: linear-gradient(90deg,#2563eb,#7c3aed); transition: width .3s; }

  .dash-filter-row { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.85rem; }
  .no-data { text-align: center; padding: 3rem; color: #94a3b8; font-size: 0.8rem; }
  .no-data-icon { font-size: 2rem; margin-bottom: 0.5rem; }

  /* Batch size slider */
  .batch-limit-wrap { margin-top: 0.85rem; padding: 0.75rem 0.9rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e8ecf0; }
  .batch-limit-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
  .batch-limit-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
  .batch-limit-val { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 20px; }
  .batch-slider { width: 100%; accent-color: #2563eb; cursor: pointer; height: 4px; }
  .batch-slider-ticks { display: flex; justify-content: space-between; margin-top: 4px; }
  .batch-slider-ticks span { font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; color: #cbd5e1; }

  /* Category pill */
  .cat-pill { display: inline-block; font-size: 0.58rem; font-weight: 600; padding: 2px 7px; border-radius: 20px; background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 85px; }
</style>

<!-- ════════════════════════════ MARKUP ════════════════════════════ -->
<div class="app">

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <div class="brand-name">NL2ASP</div>
        <div class="brand-sub">NL → CNL → ASP</div>
      </div>
    </div>
    <div class="hdivider"></div>
    <div class="tabs">
      <button class="tab-btn" class:active={activeTab==='single'}    on:click={() => activeTab='single'}>Single</button>
      <button class="tab-btn" class:active={activeTab==='batch'}     on:click={() => activeTab='batch'}>Batch</button>
      <button class="tab-btn" class:active={activeTab==='dashboard'} on:click={() => activeTab='dashboard'}>📊 Dashboard</button>
    </div>
    <div class="header-right">
      {#if gpuName}
        <span class="gpu-pill">⬡ {gpuName}</span>
      {/if}
      <span class="status-pill">
        <span class="dot {serverStatus}"></span>
        {serverStatus === 'online' ? 'Online' : serverStatus === 'offline' ? 'Offline' : 'Connecting…'}
      </span>
    </div>
  </header>

  <div class="main">
    <!-- ── Sidebar ── -->
    <aside>
      <div class="sidebar-inner">

        <!-- Model selector -->
        <div>
          <div class="s-label">Model</div>
          <div class="model-selector">
            <button class="model-trigger" class:open={modelDropOpen}
              on:click|stopPropagation={() => modelDropOpen = !modelDropOpen}>
              <span class="mt-icon">{selectedInfo?.icon ?? '🤖'}</span>
              <span class="mt-text">
                <span class="mt-label">{selectedInfo?.label ?? selectedModel}</span>
                <span class="mt-sub">{selectedInfo?.sub ?? ''}</span>
              </span>
              <span class="mt-tag">{selectedInfo?.tag ?? ''}</span>
              <span class="mt-caret" class:open={modelDropOpen}>▾</span>
            </button>
            {#if modelDropOpen}
              <div class="model-dropdown" on:click|stopPropagation>
                <div class="msearch-wrap">
                  <input class="msearch" type="text" placeholder="Search models…"
                    bind:value={modelSearch} autofocus />
                </div>
                <div class="model-list">
                  {#each Object.entries(groupedModels) as [group, items]}
                    <div class="mgroup-label">{group}</div>
                    {#each items as [key, info]}
                      {@const isLoaded = key === 't5small_ensemble'
                        ? loadedModels.some(m => m.startsWith('t5small_fold'))
                        : loadedModels.includes(key)}
                      <div class="moption"
                        class:sel={selectedModel === key}
                        class:dis={loadedModels.length > 0 && !isLoaded}
                        on:click={() => { if (!loadedModels.length || isLoaded) { selectedModel = key; modelDropOpen = false; modelSearch = '' } }}
                      >
                        <span class="mo-icon">{info.icon}</span>
                        <span class="mo-text">
                          <span class="mo-label">{info.label}</span>
                          <span class="mo-sub">{info.sub}</span>
                        </span>
                        <span class="mo-tag">{info.tag}</span>
                        {#if selectedModel === key}<span class="mo-check">✓</span>{/if}
                      </div>
                    {/each}
                  {/each}
                  {#if Object.keys(groupedModels).length === 0}
                    <div class="mno-result">No models match "{modelSearch}"</div>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        </div>

        <!-- Pipeline mode -->
        <div>
          <div class="s-label">Pipeline Mode</div>
          <div class="mode-row">
            <button class="mode-btn" class:active={pipelineMode==='direct'} on:click={() => pipelineMode='direct'}>
              <span class="mode-icon">⚡</span>
              <span class="mode-label">Direct</span>
              <span class="mode-sub">NL → ASP</span>
            </button>
            <button class="mode-btn" class:active={pipelineMode==='stepwise'} on:click={() => pipelineMode='stepwise'}>
              <span class="mode-icon">🔁</span>
              <span class="mode-label">Stepwise</span>
              <span class="mode-sub">NL→CNL→ASP</span>
            </button>
          </div>
        </div>

        <!-- Server info -->
        <div class="server-box" style="margin-top:auto">
          <div class="srow">
            <span class="slabel">API</span>
            <span class="sval">{API}</span>
          </div>
          <div class="srow">
            <span class="slabel">Models</span>
            <span class="sval">{loadedModels.length} loaded</span>
          </div>
          {#if loadedModels.length}
            <div class="smodels">{loadedModels.join(' · ')}</div>
          {/if}
        </div>

      </div>
    </aside>

    <!-- ── Content ── -->
    <div class="content">
      <div class="panel">

        <!-- Error -->
        {#if errorMsg}
          <div class="err-bar">⚠ {errorMsg}</div>
        {/if}

        <!-- Ensemble info -->
        {#if ensembleSource && foldDetails.length}
          <div class="ensemble-bar">
            <span>🎯 Ensemble</span>
            <span style="color:#c4b5fd">·</span>
            <span>{ensembleSource.replace(/_/g,' ')}</span>
            <span style="color:#c4b5fd">·</span>
            {#each foldDetails as fd}
              <span class="fold-chip" class:valid={fd.syntax_valid} class:invalid={!fd.syntax_valid} title={fd.cnl}>
                {fd.fold.replace('t5small_','')} {fd.syntax_valid ? '✓' : '✗'}
              </span>
            {/each}
          </div>
        {/if}

        <!-- ══ SINGLE TAB ══ -->
        {#if activeTab === 'single'}

          <!-- Pipeline breadcrumb -->
          <div class="pipe-bar">
            <span class="pnode nl">NL</span>
            <span class="parr">→</span>
            <span class="pnode cnl">CNL</span>
            <span class="parr">→</span>
            <span class="pnode asp">ASP</span>
            <span class="pipe-model">{selectedInfo?.label ?? selectedModel}</span>
          </div>

          <!-- NL Input card -->
          <div class="card">
            <div class="card-header">
              <span class="card-title"><span class="pnode nl">NL</span> Natural Language Input</span>
              <div class="row">
                <!-- History button -->
                <button class="btn-history" class:has-items={history.length > 0}
                  on:click|stopPropagation={() => historyOpen = !historyOpen}>
                  🕐 History
                  {#if history.length > 0}<span class="hist-count">{history.length}</span>{/if}
                </button>
                {#if nlInput.trim()}
                  <button class="btn btn-copy" class:copied={copiedNL} on:click={() => copyText(nlInput,'nl')}>
                    {copiedNL ? '✓ Copied' : '⎘ Copy'}
                  </button>
                {/if}
                {#if pipelineMode === 'stepwise'}
                  <button class="btn btn-secondary" on:click={runNL2CNL}
                    disabled={loadingNL2CNL || !nlInput.trim()}>
                    {#if loadingNL2CNL}<span class="spin"></span>{/if} NL → CNL
                  </button>
                {/if}
                <button class="btn btn-primary" on:click={runDirect}
                  disabled={loadingDirect || !nlInput.trim()}>
                  {#if loadingDirect}<span class="spin"></span>{/if}
                  {pipelineMode === 'direct' ? '⚡ Run' : '⚡ Full Pipeline'}
                </button>
                <button class="btn btn-secondary" on:click={fullReset} title="Clear all">↺</button>
              </div>
            </div>
            <div class="card-body">
              <textarea bind:value={nlInput} rows="4"
                placeholder="Enter natural language specification…&#10;e.g. Every node must be reachable from the source vertex."></textarea>
            </div>
          </div>

          <!-- CNL Output card -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">
                <span class="pnode cnl">CNL</span> Controlled Natural Language
                {#if cnlStatus}<span class="badge {cnlStatus}">{cnlStatus === 'valid' ? '✓ Valid' : '✗ Invalid'}</span>{/if}
              </span>
              <div class="row">
                {#if pipelineMode === 'stepwise' && cnlOutput}
                  <button class="btn btn-success" on:click={runCNL2ASP}
                    disabled={loadingCNL2ASP || !cnlOutput.trim()}>
                    {#if loadingCNL2ASP}<span class="spin"></span>{/if} Compile → ASP
                  </button>
                {/if}
                {#if cnlOutput}
                  <button class="btn btn-copy" class:copied={copiedCNL} on:click={() => copyText(cnlOutput,'cnl')}>
                    {copiedCNL ? '✓ Copied' : '⎘ Copy'}
                  </button>
                {/if}
              </div>
            </div>
            <div class="card-body">
              {#if loadingNL2CNL || loadingDirect}
                <p class="ph">Generating CNL…</p>
              {:else if cnlOutput}
                <div class="code-out cnl-color">{cnlOutput}</div>
              {:else}
                <p class="ph">CNL output will appear here…</p>
              {/if}
            </div>
          </div>

          <!-- ASP Output card -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">
                <span class="pnode asp">ASP</span> Answer Set Program
                {#if aspStatus}<span class="badge {aspStatus}">{aspStatus === 'compiled' ? '✓ Compiled' : '✗ Error'}</span>{/if}
              </span>
              <div class="row">
                {#if aspOutput}
                  <button class="btn btn-copy" class:copied={copiedASP} on:click={() => copyText(aspOutput,'asp')}>
                    {copiedASP ? '✓ Copied' : '⎘ Copy'}
                  </button>
                {/if}
                {#if hasResult}
                  <button class="btn btn-secondary" on:click={exportResult} title="Export result as .txt">
                    ↓ Export
                  </button>
                {/if}
              </div>
            </div>
            <div class="card-body">
              {#if loadingCNL2ASP || loadingDirect}
                <p class="ph">Compiling ASP…</p>
              {:else if aspOutput}
                <div class="code-out asp-color">{aspOutput}</div>
              {:else}
                <p class="ph">ASP output will appear here…</p>
              {/if}
            </div>
          </div>

        <!-- ══ BATCH TAB ══ -->
        {:else if activeTab === 'batch'}

          <!-- File upload card -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">📁 Dataset File</span>
              <div class="row">
                {#if !loadingBatch}
                  {#if batchFile}
                    {#if batchStopped && batchResumeIndex > 0}
                      <button class="btn btn-warning" on:click={resumeBatch}>
                        ↩ Resume · {batchResumeIndex}/{batchTotal}
                      </button>
                    {:else}
                      <button class="btn btn-primary" on:click={runBatch}>
                        ▶ Run · {batchTotal || '?'} samples
                      </button>
                    {/if}
                  {/if}
                  {#if batchResults.length}
                    <button class="btn btn-ghost-red" on:click={resetBatch} title="Clear all results and file">↺ Reset</button>
                  {/if}
                {:else}
                  <button class="btn btn-danger" on:click={stopBatch}>⏹ Stop</button>
                {/if}
              </div>
            </div>
            <div class="card-body">
              <div class="drop">
                <input type="file" accept=".json" on:change={onFileChange} />
                <div class="drop-icon">{batchFileName ? '📄' : '📂'}</div>
                {#if batchFileName}
                  <div class="drop-lbl" style="color:#2563eb">{batchFileName}</div>
                  <div class="drop-sub">click to change file</div>
                {:else}
                  <div class="drop-lbl">Drop JSON dataset or click to browse</div>
                  <div class="drop-sub">{"{ data_dict: [{NL_V2, CNL_V2, ASP}] }"}</div>
                {/if}
              </div>
            </div>
          </div>

          <!-- Progress -->
          {#if loadingBatch || batchProgress > 0}
            <div class="prog-wrap">
              <div class="prog-header">
                <span class="prog-label">
                  {#if loadingBatch}Processing…{:else if batchStopped}Stopped at {batchResumeIndex} — click Resume to continue{:else}Complete ✓{/if}
                </span>
                <span class="prog-pct">{batchProgress} / {batchTotal}
                  ({batchTotal ? ((batchProgress/batchTotal)*100).toFixed(0) : 0}%)</span>
              </div>
              <div class="prog-bg">
                <div class="prog-fill" style="width:{batchTotal ? (batchProgress/batchTotal*100) : 0}%"></div>
              </div>
            </div>
          {/if}

          <!-- Stats -->
          {#if batchResults.length}
            <div class="stats">
              <div class="stat"><div class="sv b">{batchResults.length}</div><div class="sl">Total</div></div>
              <div class="stat"><div class="sv g">{syntaxAcc}%</div><div class="sl">Syntax Acc.</div></div>
              <div class="stat"><div class="sv p">{compileAcc}%</div><div class="sl">Compile Rate</div></div>
              <div class="stat"><div class="sv r">{errorCount}</div><div class="sl">Errors</div></div>
            </div>

            <!-- Toolbar: filter + search + export -->
            <div class="batch-toolbar">
              {#each [['all','All'],['valid','✓ Valid CNL'],['invalid','✗ Invalid CNL'],['compiled','✓ Compiled'],['error','✗ Error']] as [f, label]}
                <button class="filter-btn" class:active={batchFilter===f}
                  on:click={() => batchFilter = f}>{label}</button>
              {/each}
              <input class="batch-search" type="text" placeholder="Search NL or CNL…" bind:value={batchSearch} />
              <button class="btn btn-secondary" on:click={downloadCSV}>⬇ CSV</button>
            </div>

            <!-- Results table -->
            <div class="tbl-wrap">
              <table>
                <thead>
                  <tr>
                    <th></th><th>#</th><th>Natural Language</th>
                    <th>Predicted CNL</th><th>Syntax</th>
                    <th>Generated ASP</th><th>Compiled</th>
                  </tr>
                </thead>
                <tbody>
                  {#each filteredBatch as r, i}
                    {@const idx = batchResults.indexOf(r)}
                    <tr class:expanded={expandedRow === idx}>
                      <td style="width:32px;padding:0.4rem 0.5rem;">
                        <button class="expand-btn"
                          on:click={() => expandedRow = expandedRow === idx ? null : idx}>
                          {expandedRow === idx ? '▲' : '▼'}
                        </button>
                      </td>
                      <td style="color:#cbd5e1;width:36px">{idx+1}</td>
                      <td title={r.nl} style="font-weight:500;color:#1e293b">{r.nl}</td>
                      <td title={r.predicted_cnl} style="color:#5b21b6">{r.predicted_cnl}</td>
                      <td><span class="badge {r.syntax_valid?'valid':'invalid'}">{r.syntax_valid?'✓ Valid':'✗ Fail'}</span></td>
                      <td title={r.asp} style="color:#065f46">{r.asp}</td>
                      <td><span class="badge {r.compiled?'compiled':'error'}">{r.compiled?'✓':'✗'}</span></td>
                    </tr>
                    {#if expandedRow === idx}
                      <tr>
                        <td colspan="7" style="padding:0;border-bottom:1px solid #e8ecf0;">
                          <div class="expanded-content">
                            <div class="exp-row">
                              <div class="exp-label">Natural Language</div>
                              <div class="exp-val">{r.nl}</div>
                            </div>
                            <div class="exp-row">
                              <div class="exp-label">Predicted CNL</div>
                              <div class="exp-val cnl">{r.predicted_cnl || '—'}</div>
                            </div>
                            <div class="exp-row">
                              <div class="exp-label">Generated ASP</div>
                              <div class="exp-val asp">{r.asp || '—'}</div>
                            </div>
                            {#if r.gold_cnl}
                              <div class="exp-row">
                                <div class="exp-label">Gold CNL</div>
                                <div class="exp-val">{r.gold_cnl}</div>
                              </div>
                            {/if}
                            {#if r.gold_asp}
                              <div class="exp-row">
                                <div class="exp-label">Gold ASP</div>
                                <div class="exp-val">{r.gold_asp}</div>
                              </div>
                            {/if}
                          </div>
                        </td>
                      </tr>
                    {/if}
                  {/each}
                  {#if filteredBatch.length === 0}
                    <tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:1.5rem;font-size:0.72rem;">No results match current filter</td></tr>
                  {/if}
                </tbody>
              </table>
            </div>
          {/if}

        <!-- ══ DASHBOARD TAB ══ -->
        {:else if activeTab === 'dashboard'}

          <div class="dash-layout">

            <!-- Left: model selector sidebar -->
            <div class="dash-sidebar">
              <div class="dash-sb-header">
                <span class="dash-sb-title">Select Models</span>
                <div class="row">
                  <button class="btn btn-secondary" style="font-size:0.62rem;padding:0.2rem 0.55rem"
                    on:click={selectAllDashModels}>All</button>
                  <button class="btn btn-secondary" style="font-size:0.62rem;padding:0.2rem 0.55rem"
                    on:click={deselectAllDashModels}>None</button>
                </div>
              </div>
              <div class="dash-model-list">
                {#each Object.entries(groupedModels) as [group, items]}
                  <div class="dash-group-label">{group}</div>
                  {#each items as [key, info]}
                    {@const avail = key==='t5small_ensemble'
                      ? loadedModels.some(m=>m.startsWith('t5small_fold'))
                      : loadedModels.includes(key)}
                    <div class="dash-model-item"
                      class:selected={dashModels[key]}
                      class:unavailable={!avail}
                      on:click={() => avail && toggleDashModel(key)}>
                      <div class="dm-check">{dashModels[key] ? '✓' : ''}</div>
                      <span class="dm-icon">{info.icon}</span>
                      <span class="dm-text">
                        <span class="dm-label">{info.label}</span>
                        <span class="dm-sub">{info.sub}</span>
                      </span>
                      <span class="dm-tag">{info.tag}</span>
                    </div>
                  {/each}
                {/each}
              </div>

              <!-- Selected count -->
              <div style="padding:0.6rem 1rem;border-top:1px solid #f1f5f9;font-size:0.65rem;color:#94a3b8">
                {dashSelectedKeys.length} model{dashSelectedKeys.length !== 1 ? 's' : ''} selected
              </div>
            </div>

            <!-- Right: input + results -->
            <div>

              <!-- Input mode toggle -->
              <div class="input-mode-row">
                <button class="imode-btn" class:active={dashInputMode==='single'}
                  on:click={() => { dashInputMode='single'; dashResults=[]; dashProgress=0; dashDone=false; dashBatchLimit=0 }}>✏ Single Sentence</button>
                <button class="imode-btn" class:active={dashInputMode==='batch'}
                  on:click={() => { dashInputMode='batch'; dashResults=[]; dashProgress=0; dashDone=false }}>📁 JSON Dataset</button>
              </div>

              <!-- Input card -->
              <div class="card" style="margin-bottom:0.85rem">
                <div class="card-header">
                  <span class="card-title">
                    {#if dashInputMode==='single'}✏ Natural Language Input{:else}📁 Dataset File{/if}
                  </span>
                  <div class="row">
                    {#if !dashRunning}
                      {#if dashStopped && dashResumeIndex > 0}
                        <button class="btn btn-warning" on:click={resumeDashboard}
                          disabled={dashSelectedKeys.length===0}>
                          ↩ Resume · {dashResumeIndex}/{dashTotal}
                        </button>
                        <button class="btn btn-primary" on:click={runDashboard}
                          disabled={dashSelectedKeys.length===0}>
                          ▶ Run Fresh
                        </button>
                      {:else}
                        <button class="btn btn-primary" on:click={runDashboard}
                          disabled={dashSelectedKeys.length===0}>
                          ▶ Run Comparison
                          {#if dashSelectedKeys.length > 0}· {dashSelectedKeys.length} models{/if}
                        </button>
                      {/if}
                      {#if dashResults.length}
                        <button class="btn btn-ghost-red" on:click={resetDashboard} title="Clear all results">↺ Reset</button>
                      {/if}
                    {:else}
                      <button class="btn btn-danger" on:click={stopDashboard}>⏹ Stop</button>
                    {/if}
                    {#if dashResults.length}
                      <button class="btn btn-secondary" on:click={downloadDashCSV}>⬇ CSV</button>
                    {/if}
                  </div>
                </div>
                <div class="card-body">
                  {#if dashInputMode === 'single'}
                    <textarea bind:value={dashNLInput} rows="3"
                      placeholder="Enter natural language sentence to compare across models…&#10;e.g. Every node must be reachable from the source vertex."></textarea>
                  {:else}
                    <div class="drop">
                      <input type="file" accept=".json" on:change={onDashFileChange} />
                      <div class="drop-icon">{dashFileName ? '📄' : '📂'}</div>
                      {#if dashFileName}
                        <div class="drop-lbl" style="color:#2563eb">{dashFileName}</div>
                        <div class="drop-sub">{dashTotal} records loaded · click to change</div>
                      {:else}
                        <div class="drop-lbl">Drop JSON dataset or click to browse</div>
                        <div class="drop-sub">{"{ data_dict: [{NL_V2, CNL_V2, ASP, id, category}] }"}</div>
                      {/if}
                    </div>
                    {#if dashTotal > 0}
                      <div class="batch-limit-wrap">
                        <div class="batch-limit-header">
                          <span class="batch-limit-label">Batch Size</span>
                          <span class="batch-limit-val">
                            {dashBatchLimit === dashTotal ? 'All' : dashBatchLimit}
                            / {dashTotal} records
                          </span>
                        </div>
                        <input class="batch-slider" type="range"
                          min="1" max={dashTotal} bind:value={dashBatchLimit} />
                        <div class="batch-slider-ticks">
                          <span>1</span>
                          <span>{Math.round(dashTotal/4)}</span>
                          <span>{Math.round(dashTotal/2)}</span>
                          <span>{Math.round(dashTotal*3/4)}</span>
                          <span>{dashTotal}</span>
                        </div>
                      </div>
                    {/if}
                  {/if}
                  {#if dashErrorMsg}
                    <div class="err-bar" style="margin-top:0.6rem;margin-bottom:0">⚠ {dashErrorMsg}</div>
                  {/if}
                </div>
              </div>

              <!-- Progress -->
              {#if dashRunning || dashProgress > 0}
                <div class="dash-prog">
                  <div class="dash-prog-header">
                    <span class="dash-prog-label">
                      {#if dashRunning}Running {dashSelectedKeys.length} models…{:else if dashStopped}Stopped at {dashResumeIndex} — click Resume to continue{:else}Complete ✓{/if}
                    </span>
                    <span class="dash-prog-pct">{dashProgress} / {dashTotal} rows
                      ({dashTotal ? ((dashProgress/dashTotal)*100).toFixed(0) : 0}%)</span>
                  </div>
                  <div class="dash-prog-bg">
                    <div class="dash-prog-fill" style="width:{dashTotal ? (dashProgress/dashTotal*100) : 0}%"></div>
                  </div>
                </div>
              {/if}

              <!-- Per-model summary cards -->
              {#if dashStats.length && dashResults.length}
                <div class="dash-summary">
                  {#each dashStats as s}
                    <div class="dash-model-card">
                      <div class="dmc-header">
                        <span class="dmc-icon">{s.icon}</span>
                        <span class="dmc-label">{s.label}</span>
                        <span class="dmc-tag">{s.tag}</span>
                      </div>
                      <div class="dmc-metrics">
                        <div class="dmc-metric">
                          <div class="dmc-val g">{s.syntaxPct}%</div>
                          <div class="dmc-lbl">CNL Valid</div>
                          <div class="dmc-bar-wrap">
                            <div class="dmc-bar-bg">
                              <div class="dmc-bar-fill g" style="width:{s.syntaxPct}%"></div>
                            </div>
                          </div>
                        </div>
                        <div class="dmc-metric">
                          <div class="dmc-val b">{s.compilePct}%</div>
                          <div class="dmc-lbl">Compiled</div>
                          <div class="dmc-bar-wrap">
                            <div class="dmc-bar-bg">
                              <div class="dmc-bar-fill b" style="width:{s.compilePct}%"></div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="dmc-counts">{s.syntaxOk}/{s.total} valid · {s.compiled}/{s.total} compiled</div>
                    </div>
                  {/each}
                </div>

                <!-- Filter toolbar -->
                <div class="dash-filter-row">
                  {#each [['all','All'],['any_valid','Any CNL Valid'],['all_valid','All CNL Valid'],['any_compiled','Any Compiled'],['all_compiled','All Compiled']] as [f,lbl]}
                    <button class="filter-btn" class:active={dashFilter===f}
                      on:click={() => dashFilter=f}>{lbl}</button>
                  {/each}
                  <input class="batch-search" style="flex:1;min-width:140px" type="text"
                    placeholder="Search NL…" bind:value={dashSearch} />
                  <span style="font-size:0.65rem;color:#94a3b8;font-weight:600;white-space:nowrap">
                    {dashFilteredResults.length} / {dashResults.length} rows
                  </span>
                </div>

                <!-- Comparison table -->
                <div class="cmp-table-wrap">
                  <table class="cmp-table">
                    <thead>
                      <tr>
                        <th style="width:28px"></th>
                        <th style="width:30px">#</th>
                        {#if dashInputMode === 'batch'}
                          <th style="width:60px">ID</th>
                          <th style="width:90px">Category</th>
                        {/if}
                        <th class="col-sticky" style="min-width:160px;max-width:200px">Natural Language</th>
                        {#if dashInputMode === 'batch'}
                          <th>Gold CNL</th>
                          <th>Gold ASP</th>
                        {/if}
                        {#each dashSelectedKeys as k}
                          <th class="model-header" colspan="2" style="min-width:110px">
                            <span style="overflow:hidden;text-overflow:ellipsis;display:block;max-width:130px;margin:0 auto">
                              {MODEL_INFO[k]?.icon} {MODEL_INFO[k]?.label ?? k}
                            </span>
                          </th>
                        {/each}
                      </tr>
                      <tr>
                        <th colspan={dashInputMode === 'batch' ? 7 : 3} style="height:0;padding:0;border:none"></th>
                        {#each dashSelectedKeys as k}
                          <th class="model-header" style="font-size:0.5rem">CNL Valid</th>
                          <th class="model-header" style="font-size:0.5rem">Compiled</th>
                        {/each}
                      </tr>
                    </thead>
                    <tbody>
                      {#each dashFilteredResults as r, i}
                        {@const idx = dashResults.indexOf(r)}
                        <tr class:cmp-expanded={dashExpandedRow===idx}>
                          <td style="padding:0.35rem 0.4rem">
                            <button class="expand-btn"
                              on:click={() => dashExpandedRow = dashExpandedRow===idx ? null : idx}>
                              {dashExpandedRow===idx ? '▲' : '▼'}
                            </button>
                          </td>
                          <td style="color:#cbd5e1">{idx+1}</td>
                          {#if dashInputMode === 'batch'}
                            <td style="color:#94a3b8;font-size:0.65rem" title={r.id}>{r.id || '—'}</td>
                            <td title={r.category} style="max-width:90px">
                              {#if r.category}
                                <span class="cat-pill">{r.category}</span>
                              {:else}—{/if}
                            </td>
                          {/if}
                          <td title={r.nl} class="col-sticky" style="font-weight:500;color:#1e293b;min-width:160px;max-width:200px">{r.nl}</td>
                          {#if dashInputMode === 'batch'}
                            <td title={r.gold_cnl} style="color:#5b21b6;max-width:130px">{r.gold_cnl || '—'}</td>
                            <td title={r.gold_asp} style="color:#065f46;max-width:130px">{r.gold_asp || '—'}</td>
                          {/if}
                          {#each dashSelectedKeys as k}
                            {@const m = r.models[k]}
                            <td class="model-cell">
                              {#if m}
                                <span class="badge {m.syntax_valid?'valid':'invalid'}">{m.syntax_valid?'✓':'✗'}</span>
                              {:else}<span style="color:#e2e8f0">—</span>{/if}
                            </td>
                            <td class="model-cell">
                              {#if m}
                                <span class="badge {m.compiled?'compiled':'error'}">{m.compiled?'✓':'✗'}</span>
                              {:else}<span style="color:#e2e8f0">—</span>{/if}
                            </td>
                          {/each}
                        </tr>

                        <!-- Expanded detail row -->
                        {#if dashExpandedRow === idx}
                          <tr>
                            <td colspan={(dashInputMode === 'batch' ? 7 : 3) + dashSelectedKeys.length * 2} style="padding:0;border-bottom:1px solid #e2e8f0">
                              <div class="cmp-expanded-content">
                                <!-- Source row -->
                                <div style="margin-bottom:0.75rem;padding:0.6rem 0.75rem;background:#f8fafc;border-radius:8px;border:1px solid #e8ecf0">
                                  <div style="font-size:0.6rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Natural Language</div>
                                  <div style="font-size:0.8rem;font-weight:500;color:#1e293b">{r.nl}</div>
                                  {#if dashInputMode === 'batch'}
                                    {#if r.gold_cnl}
                                      <div style="font-size:0.6rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin:6px 0 2px">Gold CNL</div>
                                      <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#5b21b6;white-space:pre-wrap">{r.gold_cnl}</div>
                                    {/if}
                                    {#if r.gold_asp}
                                      <div style="font-size:0.6rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin:6px 0 2px">Gold ASP</div>
                                      <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#065f46;white-space:pre-wrap">{r.gold_asp}</div>
                                    {/if}
                                  {/if}
                                </div>
                                <!-- Per-model output grid -->
                                <div class="cmp-model-sections">
                                  {#each dashSelectedKeys as k}
                                    {@const m = r.models[k]}
                                    {@const info = MODEL_INFO[k]}
                                    <div class="cmp-model-section">
                                      <div class="cms-title">
                                        <span>{info?.icon}</span>
                                        {info?.label ?? k}
                                        {#if m}
                                          <span class="badge {m.syntax_valid?'valid':'invalid'}" style="margin-left:auto">{m.syntax_valid?'✓ CNL':'✗ CNL'}</span>
                                          <span class="badge {m.compiled?'compiled':'error'}">{m.compiled?'✓ ASP':'✗ ASP'}</span>
                                        {/if}
                                      </div>
                                      {#if m}
                                        <div class="cms-row">
                                          <div class="cms-label">Predicted CNL</div>
                                          <div class="cms-val cnl">{m.cnl || '—'}</div>
                                        </div>
                                        <div class="cms-row">
                                          <div class="cms-label">Generated ASP</div>
                                          <div class="cms-val asp">{m.asp || '—'}</div>
                                        </div>
                                      {:else}
                                        <div style="color:#e2e8f0;font-size:0.72rem">Not run</div>
                                      {/if}
                                    </div>
                                  {/each}
                                </div>
                              </div>
                            </td>
                          </tr>
                        {/if}
                      {/each}
                      {#if dashFilteredResults.length === 0 && dashResults.length > 0}
                        <tr><td colspan={5+dashSelectedKeys.length*2} style="text-align:center;color:#94a3b8;padding:1.5rem;font-size:0.72rem">No rows match current filter</td></tr>
                      {/if}
                    </tbody>
                  </table>
                </div>

              {:else if !dashRunning}
                <div class="no-data">
                  <div class="no-data-icon">📊</div>
                  <div>Select models, enter input, and click <strong>Run Comparison</strong></div>
                  <div style="font-size:0.68rem;margin-top:0.4rem;color:#cbd5e1">Results and per-model stats will appear here</div>
                </div>
              {/if}

            </div>
          </div>

        {/if}
      </div>
    </div>
  </div>
</div>

<!-- History panel (floating) -->
{#if historyOpen}
  <div class="history-panel">
    <div class="hp-header">
      <span class="hp-title">🕐 Recent Queries</span>
      <div class="row">
        {#if history.length}
          <button class="btn btn-secondary" style="font-size:0.65rem;padding:0.22rem 0.6rem"
            on:click={clearHistory}>Clear</button>
        {/if}
        <button class="btn btn-secondary" style="font-size:0.65rem;padding:0.22rem 0.6rem"
          on:click={() => historyOpen = false}>✕</button>
      </div>
    </div>
    <div class="hp-list">
      {#if history.length === 0}
        <div class="hp-empty">No queries yet.<br>Run a translation to save history.</div>
      {:else}
        {#each history as entry}
          <div class="hp-item" on:click={() => loadFromHistory(entry)}>
            <div class="hp-nl">{entry.nl}</div>
            <div class="hp-meta">
              <span class="hp-time">{entry.ts}</span>
              <span class="hp-model">{MODEL_INFO[entry.model]?.label ?? entry.model}</span>
              <span class="badge {entry.syntaxValid ? 'valid' : 'invalid'}" style="font-size:0.52rem;padding:1px 5px">
                {entry.syntaxValid ? '✓ CNL' : '✗ CNL'}
              </span>
              <span class="badge {entry.compiled ? 'compiled' : 'error'}" style="font-size:0.52rem;padding:1px 5px">
                {entry.compiled ? '✓ ASP' : '✗ ASP'}
              </span>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  </div>
{/if}
