<script>
  import { onMount } from 'svelte'

  // ── State ──────────────────────────────────────────────────────────────────
  let selectedModel = 'llama'
  let pipelineMode  = 'direct'    // 'direct' | 'stepwise'
  let activeTab     = 'single'

  let nlInput    = ''
  let cnlOutput  = ''
  let aspOutput  = ''
  let cnlStatus  = ''
  let aspStatus  = ''

  let batchFile     = null
  let batchFileName = ''
  let batchResults  = []
  let batchProgress = 0
  let batchTotal    = 0

  let loadingNL2CNL  = false
  let loadingCNL2ASP = false
  let loadingDirect  = false
  let loadingBatch   = false
  let serverStatus   = 'unknown'
  let loadedModels   = []
  let gpuName        = ''
  let errorMsg       = ''

  // Point this to your ROCm server IP:port
  // If you open the browser ON the server itself, keep localhost
  // If you access from another machine, replace with server IP
  const API = 'http://localhost:8000'

  const MODEL_INFO = {
    llama:    { label: 'LLaMA-3.1 8B (Fine-tuned)', tag: '8B',   icon: '🦙' },
    t5small:  { label: 'T5-Small',                   tag: '60M',  icon: '🔬' },
    bartbase: { label: 'BART-Base',                   tag: '139M', icon: '📊' },
  }

  // ── Health check ────────────────────────────────────────────────────────────
  onMount(async () => {
    try {
      const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(5000) })
      if (r.ok) {
        const d     = await r.json()
        serverStatus  = 'online'
        loadedModels  = d.loaded_models ?? []
        gpuName       = d.gpu ?? ''
        if (loadedModels.length && !loadedModels.includes(selectedModel)) {
          selectedModel = loadedModels[0]
        }
      } else {
        serverStatus = 'offline'
      }
    } catch {
      serverStatus = 'offline'
    }
  })

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function reset() {
    cnlOutput = ''; aspOutput = ''
    cnlStatus = ''; aspStatus = ''
    errorMsg  = ''
  }

  async function postJSON(path, body) {
    const r = await fetch(`${API}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!r.ok) throw new Error(`Server error ${r.status}: ${await r.text()}`)
    return r.json()
  }

  async function runNL2CNL() {
    if (!nlInput.trim()) return
    reset(); loadingNL2CNL = true
    try {
      const d   = await postJSON('/api/nl2cnl', { nl: nlInput, model: selectedModel })
      cnlOutput = d.cnl
      cnlStatus = d.syntax_valid ? 'valid' : 'invalid'
    } catch (e) { errorMsg = e.message }
    finally { loadingNL2CNL = false }
  }

  async function runCNL2ASP() {
    if (!cnlOutput.trim()) return
    aspOutput = ''; aspStatus = ''; loadingCNL2ASP = true
    try {
      const d   = await postJSON('/api/cnl2asp', { cnl: cnlOutput })
      aspOutput = d.asp
      aspStatus = d.compiled ? 'compiled' : 'error'
    } catch (e) { errorMsg = e.message }
    finally { loadingCNL2ASP = false }
  }

  async function runDirect() {
    if (!nlInput.trim()) return
    reset(); loadingDirect = true
    try {
      const d   = await postJSON('/api/nl2asp', { nl: nlInput, model: selectedModel })
      cnlOutput = d.cnl; cnlStatus = d.syntax_valid ? 'valid' : 'invalid'
      aspOutput = d.asp; aspStatus = d.compiled     ? 'compiled' : 'error'
    } catch (e) { errorMsg = e.message }
    finally { loadingDirect = false }
  }

  function onFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    batchFile = f; batchFileName = f.name
    batchResults = []; batchProgress = 0; batchTotal = 0
  }

  async function runBatch() {
    if (!batchFile) return
    batchResults = []; batchProgress = 0; loadingBatch = true; errorMsg = ''
    try {
      const text    = await batchFile.text()
      const json    = JSON.parse(text)
      const records = Array.isArray(json) ? json
        : Array.isArray(json.data_dict)   ? json.data_dict : []
      if (!records.length) throw new Error('No records found in JSON file.')
      batchTotal = records.length
      for (let i = 0; i < records.length; i++) {
        const rec = records[i]
        const nl  = rec.NL_V2 ?? rec.nl ?? rec.NL ?? ''
        let result = {
          nl,
          gold_cnl: rec.CNL_V2 ?? rec.cnl ?? '',
          gold_asp: rec.ASP    ?? rec.asp ?? '',
          predicted_cnl: '', asp: '',
          syntax_valid: false, compiled: false,
        }
        try {
          const d              = await postJSON('/api/nl2asp', { nl, model: selectedModel })
          result.predicted_cnl = d.cnl
          result.asp           = d.asp
          result.syntax_valid  = d.syntax_valid
          result.compiled      = d.compiled
        } catch {
          result.predicted_cnl = 'ERROR'; result.asp = 'ERROR'
        }
        batchResults = [...batchResults, result]
        batchProgress = i + 1
      }
    } catch (e) { errorMsg = e.message }
    finally { loadingBatch = false }
  }

  function downloadCSV() {
    const header = ['NL','Predicted CNL','Syntax Valid','Generated ASP','Compiled','Gold CNL','Gold ASP']
    const rows   = batchResults.map(r => [
      `"${r.nl.replace(/"/g,'""')}"`,
      `"${r.predicted_cnl.replace(/"/g,'""')}"`,
      r.syntax_valid,
      `"${r.asp.replace(/"/g,'""')}"`,
      r.compiled,
      `"${r.gold_cnl.replace(/"/g,'""')}"`,
      `"${r.gold_asp.replace(/"/g,'""')}"`,
    ])
    const csv  = [header, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const a    = document.createElement('a')
    a.href     = URL.createObjectURL(blob)
    a.download = 'nl2asp_results.csv'
    a.click()
  }

  $: syntaxAcc  = batchResults.length
    ? ((batchResults.filter(r => r.syntax_valid).length / batchResults.length) * 100).toFixed(1)
    : null
  $: compileAcc = batchResults.length
    ? ((batchResults.filter(r => r.compiled).length / batchResults.length) * 100).toFixed(1)
    : null

  $: isLoading = loadingNL2CNL || loadingCNL2ASP || loadingDirect || loadingBatch
</script>

<!-- ═══════════════════════════════ STYLES ═══════════════════════════════ -->
<style>
  :global(*, *::before, *::after) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    font-family: 'Syne', sans-serif;
    background: #080b12;
    color: #dde3ed;
    min-height: 100vh;
  }

  /* Layout */
  .app { display: grid; grid-template-rows: 56px 1fr; height: 100vh; overflow: hidden; }

  /* Header */
  header {
    background: #0c1018;
    border-bottom: 1px solid #1a2235;
    display: flex; align-items: center;
    padding: 0 1.5rem; gap: 1.5rem;
  }
  .brand { display: flex; align-items: center; gap: 0.7rem; }
  .brand-icon {
    width: 32px; height: 32px; border-radius: 7px;
    background: linear-gradient(135deg, #3b82f6 0%, #7c3aed 100%);
    display: flex; align-items: center; justify-content: center; font-size: 16px;
  }
  .brand-name { font-weight: 800; font-size: 1rem; letter-spacing: -0.02em; }
  .brand-sub  { font-size: 0.62rem; color: #4b5a72; font-family: 'JetBrains Mono', monospace; }

  .header-mid { display: flex; gap: 0.25rem; margin-left: 1rem; }
  .tab-btn {
    padding: 0.35rem 0.85rem; border-radius: 6px;
    border: 1px solid transparent;
    background: none; color: #4b5a72;
    font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 600;
    cursor: pointer; transition: all 0.12s;
  }
  .tab-btn:hover { color: #94a3b8; }
  .tab-btn.active { background: #1a2235; border-color: #253047; color: #dde3ed; }

  .header-right { margin-left: auto; display: flex; align-items: center; gap: 1rem; }
  .gpu-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    color: #4ade80; background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    padding: 2px 8px; border-radius: 20px;
  }
  .status-pill {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #4b5a72;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; }
  .dot.online  { background: #22c55e; box-shadow: 0 0 5px #22c55e88; }
  .dot.offline { background: #ef4444; }
  .dot.unknown { background: #f59e0b; }

  /* Main */
  .main { display: grid; grid-template-columns: 230px 1fr; overflow: hidden; }

  /* Sidebar */
  aside {
    background: #0c1018; border-right: 1px solid #1a2235;
    padding: 1.25rem 0.9rem; overflow-y: auto;
    display: flex; flex-direction: column; gap: 1.5rem;
  }
  .s-label {
    font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #2d3a52; margin-bottom: 0.6rem; font-weight: 700;
  }
  .model-btn {
    width: 100%; padding: 0.55rem 0.75rem; border-radius: 7px;
    border: 1px solid #1a2235; background: none;
    color: #4b5a72; font-family: 'Syne', sans-serif;
    font-size: 0.78rem; text-align: left; cursor: pointer;
    display: flex; align-items: center; gap: 0.5rem;
    transition: all 0.12s; margin-bottom: 0.3rem;
    position: relative;
  }
  .model-btn:hover { background: #131c2e; color: #94a3b8; }
  .model-btn.active {
    background: linear-gradient(135deg,rgba(59,130,246,.12),rgba(124,58,237,.12));
    border-color: #2d4a80; color: #dde3ed;
  }
  .model-btn.disabled-model { opacity: 0.35; cursor: not-allowed; }
  .m-tag {
    margin-left: auto; font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; padding: 1px 5px; border-radius: 4px;
    background: #1a2235; color: #4b5a72;
  }
  .model-btn.active .m-tag { background: rgba(59,130,246,0.15); color: #60a5fa; }

  .mode-btn {
    width: 100%; padding: 0.55rem 0.75rem; border-radius: 7px;
    border: 1px solid #1a2235; background: none;
    color: #4b5a72; font-family: 'Syne', sans-serif;
    font-size: 0.78rem; text-align: left; cursor: pointer;
    transition: all 0.12s; margin-bottom: 0.3rem;
  }
  .mode-btn:hover { background: #131c2e; color: #94a3b8; }
  .mode-btn.active { background: #131c2e; border-color: #253047; color: #dde3ed; }
  .mode-sub { font-size: 0.6rem; color: #2d3a52; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

  /* Content */
  .content { display: flex; flex-direction: column; overflow: hidden; }
  .panel   { flex: 1; overflow-y: auto; padding: 1.25rem 1.5rem; }

  /* Cards */
  .card { background: #0c1018; border: 1px solid #1a2235; border-radius: 10px; margin-bottom: 0.85rem; overflow: hidden; }
  .card-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 0.9rem; border-bottom: 1px solid #1a2235;
    background: #0f1520;
  }
  .card-title {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #4b5a72;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .card-body { padding: 0.9rem; }

  /* Pipeline bar */
  .pipe-bar {
    display: flex; align-items: center; gap: 0.5rem;
    background: #0f1520; border: 1px solid #1a2235;
    border-radius: 8px; padding: 0.6rem 0.9rem;
    margin-bottom: 0.85rem; overflow-x: auto;
  }
  .pnode {
    padding: 3px 10px; border-radius: 5px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600;
  }
  .pnode.nl  { background: rgba(59,130,246,.12); color: #93c5fd; border: 1px solid #1e3a5f; }
  .pnode.cnl { background: rgba(124,58,237,.12);  color: #c4b5fd; border: 1px solid #3b1f6e; }
  .pnode.asp { background: rgba(34,197,94,.1);    color: #86efac; border: 1px solid #14532d; }
  .parr { color: #253047; }
  .pipe-model { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #2d3a52; }

  /* Inputs / outputs */
  textarea {
    width: 100%; background: none; border: none; outline: none;
    color: #dde3ed; font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; line-height: 1.7; resize: vertical; min-height: 80px;
  }
  .code-out {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    line-height: 1.7; white-space: pre-wrap; min-height: 50px;
  }
  .code-out.cnl-color { color: #c4b5fd; }
  .code-out.asp-color { color: #86efac; }
  .ph { color: #253047; font-style: italic; font-size: 0.78rem; }

  /* Buttons */
  .btn {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.38rem 0.9rem; border-radius: 7px; border: none;
    font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700;
    cursor: pointer; transition: all 0.12s; white-space: nowrap;
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary  { background: linear-gradient(135deg,#3b82f6,#6366f1); color: #fff; }
  .btn-primary:hover:not(:disabled) { filter: brightness(1.1); }
  .btn-ghost    { background: #1a2235; color: #64748b; border: 1px solid #253047; }
  .btn-ghost:hover:not(:disabled) { background: #1e2a40; color: #94a3b8; }
  .btn-green    { background: rgba(34,197,94,.12); color: #86efac; border: 1px solid #14532d; }
  .btn-green:hover:not(:disabled) { background: rgba(34,197,94,.2); }
  .row { display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap; }

  /* Badges */
  .badge {
    display: inline-flex; align-items: center; gap: 0.25rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    padding: 2px 7px; border-radius: 20px; font-weight: 600;
  }
  .badge.valid    { background: rgba(34,197,94,.1);  color: #86efac; border: 1px solid #14532d; }
  .badge.invalid  { background: rgba(239,68,68,.1);  color: #fca5a5; border: 1px solid #7f1d1d; }
  .badge.compiled { background: rgba(34,197,94,.1);  color: #86efac; border: 1px solid #14532d; }
  .badge.error    { background: rgba(239,68,68,.1);  color: #fca5a5; border: 1px solid #7f1d1d; }

  /* Error */
  .err {
    background: rgba(239,68,68,.07); border: 1px solid #7f1d1d;
    border-radius: 7px; padding: 0.65rem 0.9rem;
    color: #fca5a5; font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; margin-bottom: 0.85rem;
  }

  /* Spinner */
  .spin {
    display: inline-block; width: 11px; height: 11px;
    border: 2px solid rgba(255,255,255,0.2);
    border-top-color: #fff; border-radius: 50%;
    animation: s .65s linear infinite;
  }
  @keyframes s { to { transform: rotate(360deg); } }

  /* Progress */
  .prog-wrap { margin-bottom: 0.85rem; }
  .prog-bg   { height: 5px; background: #1a2235; border-radius: 3px; overflow: hidden; }
  .prog-fill { height: 100%; background: linear-gradient(90deg,#3b82f6,#7c3aed); border-radius: 3px; transition: width .3s; }
  .prog-lbl  { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #4b5a72; margin-top: 3px; }

  /* Stats */
  .stats { display: flex; gap: 0.65rem; margin-bottom: 0.85rem; flex-wrap: wrap; }
  .stat  { flex: 1; min-width: 100px; background: #0f1520; border: 1px solid #1a2235; border-radius: 8px; padding: 0.65rem 0.85rem; }
  .sv    { font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
  .sv.b  { color: #60a5fa; } .sv.g { color: #4ade80; } .sv.p { color: #a78bfa; }
  .sl    { font-size: 0.6rem; color: #2d3a52; margin-top: 2px; text-transform: uppercase; letter-spacing: .05em; }

  /* Drop zone */
  .drop {
    border: 2px dashed #1a2235; border-radius: 8px;
    padding: 1.75rem; text-align: center; cursor: pointer;
    position: relative; transition: all .15s;
  }
  .drop:hover { border-color: #3b82f6; background: rgba(59,130,246,.03); }
  .drop input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .drop-icon  { font-size: 1.75rem; margin-bottom: 0.4rem; }
  .drop-lbl   { font-size: 0.8rem; color: #4b5a72; }
  .drop-sub   { font-size: 0.65rem; color: #253047; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }

  /* Table */
  .tbl-wrap { border: 1px solid #1a2235; border-radius: 8px; overflow-x: auto; }
  table     { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  th {
    text-align: left; padding: 0.55rem 0.75rem;
    background: #0f1520; color: #2d3a52;
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    text-transform: uppercase; letter-spacing: .06em;
    border-bottom: 1px solid #1a2235; white-space: nowrap;
  }
  td {
    padding: 0.5rem 0.75rem; border-bottom: 1px solid #0f1520;
    font-family: 'JetBrains Mono', monospace; color: #64748b;
    max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #0f1520; }
</style>

<!-- ═══════════════════════════════ MARKUP ═══════════════════════════════ -->
<div class="app">

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <div class="brand-name">NL2ASP</div>
        <div class="brand-sub">NL → CNL → ASP pipeline</div>
      </div>
    </div>

    <div class="header-mid">
      <button class="tab-btn" class:active={activeTab==='single'} on:click={() => activeTab='single'}>Single</button>
      <button class="tab-btn" class:active={activeTab==='batch'}  on:click={() => activeTab='batch'}>Batch</button>
    </div>

    <div class="header-right">
      {#if gpuName}
        <span class="gpu-tag">🎮 {gpuName}</span>
      {/if}
      <span class="status-pill">
        <span class="dot {serverStatus}"></span>
        Backend {serverStatus === 'online' ? 'online' : serverStatus === 'offline' ? 'offline' : '...'}
      </span>
    </div>
  </header>

  <div class="main">

    <!-- Sidebar -->
    <aside>
      <div>
        <div class="s-label">Model</div>
        {#each Object.entries(MODEL_INFO) as [key, info]}
          <button
            class="model-btn"
            class:active={selectedModel === key}
            class:disabled-model={loadedModels.length > 0 && !loadedModels.includes(key)}
            on:click={() => loadedModels.includes(key) && (selectedModel = key)}
            title={!loadedModels.includes(key) ? 'Not loaded on server' : ''}
          >
            {info.icon} {info.label}
            <span class="m-tag">{info.tag}</span>
          </button>
        {/each}
      </div>

      <div>
        <div class="s-label">Pipeline Mode</div>
        <button class="mode-btn" class:active={pipelineMode==='direct'} on:click={() => pipelineMode='direct'}>
          ⚡ Direct NL → ASP
          <div class="mode-sub">one click end-to-end</div>
        </button>
        <button class="mode-btn" class:active={pipelineMode==='stepwise'} on:click={() => pipelineMode='stepwise'}>
          🔁 Step-by-step
          <div class="mode-sub">NL → CNL → ASP separately</div>
        </button>
      </div>

      <div style="margin-top:auto">
        <div class="s-label">Server</div>
        <div style="font-size:0.65rem;color:#2d3a52;line-height:1.7;font-family:'JetBrains Mono',monospace">
          {API}<br>
          {#if loadedModels.length}
            Loaded: {loadedModels.join(', ')}
          {:else}
            No models loaded
          {/if}
        </div>
      </div>
    </aside>

    <!-- Content -->
    <div class="content">
      <div class="panel">

        {#if errorMsg}
          <div class="err">⚠ {errorMsg}</div>
        {/if}

        <!-- ══ SINGLE TAB ══ -->
        {#if activeTab === 'single'}

          <div class="pipe-bar">
            <span class="pnode nl">NL</span>
            <span class="parr">→</span>
            <span class="pnode cnl">CNL</span>
            <span class="parr">→</span>
            <span class="pnode asp">ASP</span>
            <span class="pipe-model">{MODEL_INFO[selectedModel]?.label}</span>
          </div>

          <!-- NL Input -->
          <div class="card">
            <div class="card-header">
              <span class="card-title"><span class="pnode nl">NL</span> Natural Language</span>
              <div class="row">
                {#if pipelineMode === 'stepwise'}
                  <button class="btn btn-ghost" on:click={runNL2CNL}
                    disabled={loadingNL2CNL || !nlInput.trim()}>
                    {#if loadingNL2CNL}<span class="spin"></span>{/if}
                    NL → CNL
                  </button>
                {/if}
                <button class="btn btn-primary" on:click={runDirect}
                  disabled={loadingDirect || !nlInput.trim()}>
                  {#if loadingDirect}<span class="spin"></span>{/if}
                  {pipelineMode === 'direct' ? '⚡ NL → ASP' : '⚡ Full Pipeline'}
                </button>
                <button class="btn btn-ghost" on:click={reset}>✕</button>
              </div>
            </div>
            <div class="card-body">
              <textarea bind:value={nlInput} rows="4"
                placeholder="Enter natural language specification…&#10;e.g. Every node must be reachable from the source vertex.">
              </textarea>
            </div>
          </div>

          <!-- CNL Output -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">
                <span class="pnode cnl">CNL</span> Controlled Natural Language
                {#if cnlStatus}
                  <span class="badge {cnlStatus}">{cnlStatus === 'valid' ? '✓ Valid' : '✗ Invalid'}</span>
                {/if}
              </span>
              <div class="row">
                {#if pipelineMode === 'stepwise'}
                  <button class="btn btn-green" on:click={runCNL2ASP}
                    disabled={loadingCNL2ASP || !cnlOutput.trim()}>
                    {#if loadingCNL2ASP}<span class="spin"></span>{/if}
                    Compile → ASP
                  </button>
                {/if}
                {#if cnlOutput}
                  <button class="btn btn-ghost" on:click={() => navigator.clipboard.writeText(cnlOutput)}>📋</button>
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

          <!-- ASP Output -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">
                <span class="pnode asp">ASP</span> Answer Set Program
                {#if aspStatus}
                  <span class="badge {aspStatus}">{aspStatus === 'compiled' ? '✓ Compiled' : '✗ Error'}</span>
                {/if}
              </span>
              {#if aspOutput}
                <button class="btn btn-ghost" on:click={() => navigator.clipboard.writeText(aspOutput)}>📋 Copy</button>
              {/if}
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
        {:else}

          <div class="card">
            <div class="card-header">
              <span class="card-title">📁 Dataset File</span>
              {#if batchFile}
                <button class="btn btn-primary" on:click={runBatch} disabled={loadingBatch}>
                  {#if loadingBatch}<span class="spin"></span>{/if}
                  Run Batch ({batchTotal || '?'} samples)
                </button>
              {/if}
            </div>
            <div class="card-body">
              <div class="drop">
                <input type="file" accept=".json" on:change={onFileChange} />
                <div class="drop-icon">📂</div>
                {#if batchFileName}
                  <div class="drop-lbl" style="color:#60a5fa">{batchFileName}</div>
                  <div class="drop-sub">click to change</div>
                {:else}
                  <div class="drop-lbl">Drop JSON dataset or click to browse</div>
                  <div class="drop-sub">{"{ data_dict: [{NL_V2, CNL_V2, ASP}] }"}</div>
                {/if}
              </div>
            </div>
          </div>

          {#if loadingBatch || batchProgress > 0}
            <div class="prog-wrap">
              <div class="prog-bg">
                <div class="prog-fill" style="width:{batchTotal ? (batchProgress/batchTotal*100) : 0}%"></div>
              </div>
              <div class="prog-lbl">{batchProgress} / {batchTotal} processed</div>
            </div>
          {/if}

          {#if batchResults.length}
            <div class="stats">
              <div class="stat"><div class="sv b">{batchResults.length}</div><div class="sl">Total</div></div>
              <div class="stat"><div class="sv g">{syntaxAcc}%</div><div class="sl">Syntax Acc.</div></div>
              <div class="stat"><div class="sv p">{compileAcc}%</div><div class="sl">Compile Rate</div></div>
              <div class="stat" style="flex:0 0 auto">
                <button class="btn btn-ghost" on:click={downloadCSV} style="margin-top:4px">⬇ CSV</button>
              </div>
            </div>

            <div class="tbl-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th><th>Natural Language</th><th>Predicted CNL</th>
                    <th>Syntax</th><th>Generated ASP</th><th>Compiled</th>
                  </tr>
                </thead>
                <tbody>
                  {#each batchResults as r, i}
                    <tr>
                      <td style="color:#253047">{i+1}</td>
                      <td title={r.nl}>{r.nl}</td>
                      <td title={r.predicted_cnl} style="color:#c4b5fd">{r.predicted_cnl}</td>
                      <td><span class="badge {r.syntax_valid?'valid':'invalid'}">{r.syntax_valid?'✓':'✗'}</span></td>
                      <td title={r.asp} style="color:#86efac">{r.asp}</td>
                      <td><span class="badge {r.compiled?'compiled':'error'}">{r.compiled?'✓':'✗'}</span></td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

        {/if}
      </div>
    </div>
  </div>
</div>
