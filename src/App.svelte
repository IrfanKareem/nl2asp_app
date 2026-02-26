<script>
  import { onMount } from "svelte";

  // ── State ──────────────────────────────────────────────────────────────────
  let selectedModel = "llama";
  let pipelineMode = "direct";
  let activeTab = "single";
  let modelSearch = "";
  let modelDropOpen = false;

  let nlInput = "";
  let cnlOutput = "";
  let aspOutput = "";
  let cnlStatus = "";
  let aspStatus = "";

  let batchFile = null;
  let batchFileName = "";
  let batchResults = [];
  let batchProgress = 0;
  let batchTotal = 0;

  let loadingNL2CNL = false;
  let loadingCNL2ASP = false;
  let loadingDirect = false;
  let loadingBatch = false;
  let serverStatus = "unknown";
  let loadedModels = [];
  let gpuName = "";
  let errorMsg = "";
  let ensembleSource = "";
  let foldDetails = [];

  // copy feedback per panel
  let copiedNL = false;
  let copiedCNL = false;
  let copiedASP = false;

  const API = "http://localhost:8000";

  const MODEL_INFO = {
    llama: {
      label: "LLaMA-3.1 8B",
      sub: "Fine-tuned · LoRA",
      tag: "8B",
      icon: "🦙",
      group: "Base Models",
    },
    t5small: {
      label: "T5-Small",
      sub: "Seq2Seq",
      tag: "60M",
      icon: "🔬",
      group: "Base Models",
    },
    bartbase: {
      label: "BART-Base",
      sub: "Seq2Seq",
      tag: "139M",
      icon: "📊",
      group: "Base Models",
    },
    t5small_ensemble: {
      label: "KFold Ensemble",
      sub: "Majority vote · 5 folds",
      tag: "5×60M",
      icon: "🎯",
      group: "KFold T5-Small",
    },
    t5small_fold1: {
      label: "T5-Small Fold 1",
      sub: "KFold best checkpoint",
      tag: "F1",
      icon: "🔬",
      group: "KFold T5-Small",
    },
    t5small_fold2: {
      label: "T5-Small Fold 2",
      sub: "KFold best checkpoint",
      tag: "F2",
      icon: "🔬",
      group: "KFold T5-Small",
    },
    t5small_fold3: {
      label: "T5-Small Fold 3",
      sub: "KFold best checkpoint",
      tag: "F3",
      icon: "🔬",
      group: "KFold T5-Small",
    },
    t5small_fold4: {
      label: "T5-Small Fold 4",
      sub: "KFold best checkpoint",
      tag: "F4",
      icon: "🔬",
      group: "KFold T5-Small",
    },
    t5small_fold5: {
      label: "T5-Small Fold 5",
      sub: "KFold best checkpoint",
      tag: "F5",
      icon: "🔬",
      group: "KFold T5-Small",
    },
  };

  // Filtered model list for search
  $: filteredModels = Object.entries(MODEL_INFO).filter(([key, info]) => {
    const q = modelSearch.toLowerCase();
    return (
      !q ||
      info.label.toLowerCase().includes(q) ||
      info.group.toLowerCase().includes(q) ||
      key.includes(q)
    );
  });

  // Group the filtered list
  $: groupedModels = filteredModels.reduce((acc, [key, info]) => {
    if (!acc[info.group]) acc[info.group] = [];
    acc[info.group].push([key, info]);
    return acc;
  }, {});

  $: selectedInfo = MODEL_INFO[selectedModel];

  // ── Health check ──────────────────────────────────────────────────────────
  onMount(async () => {
    try {
      const r = await fetch(`${API}/api/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (r.ok) {
        const d = await r.json();
        serverStatus = "online";
        loadedModels = d.loaded_models ?? [];
        gpuName = d.gpu ?? "";
        if (loadedModels.length && !loadedModels.includes(selectedModel))
          selectedModel = loadedModels[0];
      } else {
        serverStatus = "offline";
      }
    } catch {
      serverStatus = "offline";
    }
  });

  // Close dropdown when clicking outside
  function handleOutsideClick(e) {
    if (!e.target.closest(".model-selector")) modelDropOpen = false;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function reset() {
    cnlOutput = "";
    aspOutput = "";
    cnlStatus = "";
    aspStatus = "";
    errorMsg = "";
    ensembleSource = "";
    foldDetails = [];
  }

  function fullReset() {
    nlInput = "";
    reset();
  }

  async function copyText(text, which) {
    await navigator.clipboard.writeText(text);
    if (which === "nl") {
      copiedNL = true;
      setTimeout(() => (copiedNL = false), 1500);
    }
    if (which === "cnl") {
      copiedCNL = true;
      setTimeout(() => (copiedCNL = false), 1500);
    }
    if (which === "asp") {
      copiedASP = true;
      setTimeout(() => (copiedASP = false), 1500);
    }
  }

  async function postJSON(path, body) {
    const r = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`Server error ${r.status}: ${await r.text()}`);
    return r.json();
  }

  async function runNL2CNL() {
    if (!nlInput.trim()) return;
    reset();
    loadingNL2CNL = true;
    try {
      const modelKey =
        selectedModel === "t5small_ensemble" ? "t5small_fold1" : selectedModel;
      const d = await postJSON("/api/nl2cnl", { nl: nlInput, model: modelKey });
      cnlOutput = d.cnl;
      cnlStatus = d.syntax_valid ? "valid" : "invalid";
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loadingNL2CNL = false;
    }
  }

  async function runCNL2ASP() {
    if (!cnlOutput.trim()) return;
    aspOutput = "";
    aspStatus = "";
    loadingCNL2ASP = true;
    try {
      const d = await postJSON("/api/cnl2asp", { cnl: cnlOutput });
      aspOutput = d.asp;
      aspStatus = d.compiled ? "compiled" : "error";
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loadingCNL2ASP = false;
    }
  }

  async function runDirect() {
    if (!nlInput.trim()) return;
    reset();
    loadingDirect = true;
    try {
      let d;
      if (selectedModel === "t5small_ensemble") {
        d = await postJSON("/api/nl2asp_kfold_ensemble", { nl: nlInput });
      } else {
        d = await postJSON("/api/nl2asp", {
          nl: nlInput,
          model: selectedModel,
        });
      }
      cnlOutput = d.cnl;
      cnlStatus = d.syntax_valid ? "valid" : "invalid";
      aspOutput = d.asp;
      aspStatus = d.compiled ? "compiled" : "error";
      if (d.cnl_source) ensembleSource = d.cnl_source;
      if (d.fold_details) foldDetails = d.fold_details;
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loadingDirect = false;
    }
  }

  function onFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    batchFile = f;
    batchFileName = f.name;
    batchResults = [];
    batchProgress = 0;
    batchTotal = 0;
  }

  async function runBatch() {
    if (!batchFile) return;
    batchResults = [];
    batchProgress = 0;
    loadingBatch = true;
    errorMsg = "";
    try {
      const text = await batchFile.text();
      const json = JSON.parse(text);
      const records = Array.isArray(json)
        ? json
        : Array.isArray(json.data_dict)
          ? json.data_dict
          : [];
      if (!records.length) throw new Error("No records found in JSON file.");
      batchTotal = records.length;
      for (let i = 0; i < records.length; i++) {
        const rec = records[i];
        const nl = rec.NL_V2 ?? rec.nl ?? rec.NL ?? "";
        let result = {
          nl,
          gold_cnl: rec.CNL_V2 ?? rec.cnl ?? "",
          gold_asp: rec.ASP ?? rec.asp ?? "",
          predicted_cnl: "",
          asp: "",
          syntax_valid: false,
          compiled: false,
        };
        try {
          const endpoint =
            selectedModel === "t5small_ensemble"
              ? "/api/nl2asp_kfold_ensemble"
              : "/api/nl2asp";
          const body =
            selectedModel === "t5small_ensemble"
              ? { nl }
              : { nl, model: selectedModel };
          const d = await postJSON(endpoint, body);
          result.predicted_cnl = d.cnl;
          result.asp = d.asp;
          result.syntax_valid = d.syntax_valid;
          result.compiled = d.compiled;
        } catch {
          result.predicted_cnl = "ERROR";
          result.asp = "ERROR";
        }
        batchResults = [...batchResults, result];
        batchProgress = i + 1;
      }
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loadingBatch = false;
    }
  }

  function downloadCSV() {
    const header = [
      "NL",
      "Predicted CNL",
      "Syntax Valid",
      "Generated ASP",
      "Compiled",
      "Gold CNL",
      "Gold ASP",
    ];
    const rows = batchResults.map((r) => [
      `"${r.nl.replace(/"/g, '""')}"`,
      `"${r.predicted_cnl.replace(/"/g, '""')}"`,
      r.syntax_valid,
      `"${r.asp.replace(/"/g, '""')}"`,
      r.compiled,
      `"${r.gold_cnl.replace(/"/g, '""')}"`,
      `"${r.gold_asp.replace(/"/g, '""')}"`,
    ]);
    const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "nl2asp_results.csv";
    a.click();
  }

  $: syntaxAcc = batchResults.length
    ? (
        (batchResults.filter((r) => r.syntax_valid).length /
          batchResults.length) *
        100
      ).toFixed(1)
    : null;
  $: compileAcc = batchResults.length
    ? (
        (batchResults.filter((r) => r.compiled).length / batchResults.length) *
        100
      ).toFixed(1)
    : null;
  $: isLoading =
    loadingNL2CNL || loadingCNL2ASP || loadingDirect || loadingBatch;
</script>

<svelte:window on:click={handleOutsideClick} />

<!-- ═══════════════════════════ MARKUP ═══════════════════════════ -->
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

    <div class="header-divider"></div>

    <div class="tabs">
      <button
        class="tab-btn"
        class:active={activeTab === "single"}
        on:click={() => (activeTab = "single")}>Single</button
      >
      <button
        class="tab-btn"
        class:active={activeTab === "batch"}
        on:click={() => (activeTab = "batch")}>Batch</button
      >
    </div>

    <div class="header-right">
      {#if gpuName}
        <span class="gpu-pill">⬡ {gpuName}</span>
      {/if}
      <span class="status-pill">
        <span class="dot {serverStatus}"></span>
        {serverStatus === "online"
          ? "Backend online"
          : serverStatus === "offline"
            ? "Backend offline"
            : "Connecting…"}
      </span>
    </div>
  </header>

  <div class="main">
    <!-- Sidebar -->
    <aside>
      <div class="sidebar-inner">
        <!-- Model Selector -->
        <div>
          <div class="s-section-label">Model</div>
          <div class="model-selector">
            <!-- Trigger -->
            <button
              class="model-trigger"
              class:open={modelDropOpen}
              on:click|stopPropagation={() => (modelDropOpen = !modelDropOpen)}
            >
              <span class="mt-icon">{selectedInfo?.icon ?? "🤖"}</span>
              <span class="mt-text">
                <span class="mt-label"
                  >{selectedInfo?.label ?? selectedModel}</span
                >
                <span class="mt-sub">{selectedInfo?.sub ?? ""}</span>
              </span>
              <span class="mt-tag">{selectedInfo?.tag ?? ""}</span>
              <span class="mt-caret" class:open={modelDropOpen}>▾</span>
            </button>

            <!-- Dropdown -->
            {#if modelDropOpen}
              <div class="model-dropdown" on:click|stopPropagation>
                <div class="model-search-wrap">
                  <input
                    class="model-search"
                    type="text"
                    placeholder="Search models…"
                    bind:value={modelSearch}
                    autofocus
                  />
                </div>
                <div class="model-list">
                  {#each Object.entries(groupedModels) as [group, items]}
                    <div class="model-group-label">{group}</div>
                    {#each items as [key, info]}
                      {@const isLoaded =
                        key === "t5small_ensemble"
                          ? loadedModels.some((m) =>
                              m.startsWith("t5small_fold"),
                            )
                          : loadedModels.includes(key)}
                      <div
                        class="model-option"
                        class:selected={selectedModel === key}
                        class:disabled={loadedModels.length > 0 && !isLoaded}
                        on:click={() => {
                          if (isLoaded || !loadedModels.length) {
                            selectedModel = key;
                            modelDropOpen = false;
                            modelSearch = "";
                          }
                        }}
                      >
                        <span class="mo-icon">{info.icon}</span>
                        <span class="mo-text">
                          <span class="mo-label">{info.label}</span>
                          <span class="mo-sub">{info.sub}</span>
                        </span>
                        <span class="mo-tag">{info.tag}</span>
                        {#if selectedModel === key}<span class="mo-check"
                            >✓</span
                          >{/if}
                      </div>
                    {/each}
                  {/each}
                  {#if Object.keys(groupedModels).length === 0}
                    <div
                      style="padding:0.75rem;text-align:center;font-size:0.7rem;color:#334155"
                    >
                      No matches
                    </div>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        </div>

        <!-- Pipeline Mode -->
        <div>
          <div class="s-section-label">Pipeline Mode</div>
          <div class="mode-row">
            <button
              class="mode-btn"
              class:active={pipelineMode === "direct"}
              on:click={() => (pipelineMode = "direct")}
            >
              <span class="mode-icon">⚡</span>
              <span class="mode-label">Direct</span>
              <span class="mode-sub">NL → ASP</span>
            </button>
            <button
              class="mode-btn"
              class:active={pipelineMode === "stepwise"}
              on:click={() => (pipelineMode = "stepwise")}
            >
              <span class="mode-icon">🔁</span>
              <span class="mode-label">Stepwise</span>
              <span class="mode-sub">NL→CNL→ASP</span>
            </button>
          </div>
        </div>

        <!-- Server Info -->
        <div class="server-info" style="margin-top:auto">
          <div class="server-row">
            <span class="server-label">API</span>
            <span class="server-val">{API}</span>
          </div>
          <div class="server-row">
            <span class="server-label">Models</span>
            <span class="server-val">{loadedModels.length} loaded</span>
          </div>
          {#if loadedModels.length}
            <div class="server-models">{loadedModels.join(" · ")}</div>
          {/if}
        </div>
      </div>
    </aside>

    <!-- Content -->
    <div class="content">
      <div class="panel">
        {#if errorMsg}
          <div class="err-bar">⚠ {errorMsg}</div>
        {/if}

        {#if ensembleSource && foldDetails.length}
          <div class="ensemble-bar">
            <span>🎯 Ensemble</span>
            <span style="color:#334155">·</span>
            <span>{ensembleSource.replace(/_/g, " ")}</span>
            <span style="color:#334155">·</span>
            {#each foldDetails as fd}
              <span
                class="fold-chip"
                class:valid={fd.syntax_valid}
                class:invalid={!fd.syntax_valid}
                title={fd.cnl}
              >
                {fd.fold.replace("t5small_", "")}
                {fd.syntax_valid ? "✓" : "✗"}
              </span>
            {/each}
          </div>
        {/if}

        <!-- ══ SINGLE TAB ══ -->
        {#if activeTab === "single"}
          <div class="pipe-bar">
            <span class="pnode nl">NL</span>
            <span class="parr">→</span>
            <span class="pnode cnl">CNL</span>
            <span class="parr">→</span>
            <span class="pnode asp">ASP</span>
            <span class="pipe-model"
              >{selectedInfo?.label ?? selectedModel}</span
            >
          </div>

          <!-- NL Input -->
          <div class="card">
            <div class="card-header">
              <span class="card-title"
                ><span class="pnode nl">NL</span> Natural Language Input</span
              >
              <div class="row">
                {#if nlInput.trim()}
                  <button
                    class="btn btn-copy"
                    class:copied={copiedNL}
                    on:click={() => copyText(nlInput, "nl")}
                  >
                    {copiedNL ? "✓ Copied" : "⎘ Copy"}
                  </button>
                {/if}
                {#if pipelineMode === "stepwise"}
                  <button
                    class="btn btn-secondary"
                    on:click={runNL2CNL}
                    disabled={loadingNL2CNL || !nlInput.trim()}
                  >
                    {#if loadingNL2CNL}<span class="spin"></span>{/if}
                    NL → CNL
                  </button>
                {/if}
                <button
                  class="btn btn-primary"
                  on:click={runDirect}
                  disabled={loadingDirect || !nlInput.trim()}
                >
                  {#if loadingDirect}<span class="spin"></span>{/if}
                  {pipelineMode === "direct" ? "⚡ Run" : "⚡ Full Pipeline"}
                </button>
                <button
                  class="btn btn-secondary"
                  on:click={fullReset}
                  title="Clear all">↺</button
                >
              </div>
            </div>
            <div class="card-body">
              <textarea
                bind:value={nlInput}
                rows="4"
                placeholder="Enter natural language specification…&#10;e.g. Every node must be reachable from the source vertex."
              ></textarea>
            </div>
          </div>

          <!-- CNL Output -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">
                <span class="pnode cnl">CNL</span> Controlled Natural Language
                {#if cnlStatus}
                  <span class="badge {cnlStatus}"
                    >{cnlStatus === "valid" ? "✓ Valid" : "✗ Invalid"}</span
                  >
                {/if}
              </span>
              <div class="row">
                {#if pipelineMode === "stepwise" && cnlOutput}
                  <button
                    class="btn btn-success"
                    on:click={runCNL2ASP}
                    disabled={loadingCNL2ASP || !cnlOutput.trim()}
                  >
                    {#if loadingCNL2ASP}<span class="spin"></span>{/if}
                    Compile → ASP
                  </button>
                {/if}
                {#if cnlOutput}
                  <button
                    class="btn btn-copy"
                    class:copied={copiedCNL}
                    on:click={() => copyText(cnlOutput, "cnl")}
                  >
                    {copiedCNL ? "✓ Copied" : "⎘ Copy"}
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

          <!-- ASP Output -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">
                <span class="pnode asp">ASP</span> Answer Set Program
                {#if aspStatus}
                  <span class="badge {aspStatus}"
                    >{aspStatus === "compiled" ? "✓ Compiled" : "✗ Error"}</span
                  >
                {/if}
              </span>
              {#if aspOutput}
                <button
                  class="btn btn-copy"
                  class:copied={copiedASP}
                  on:click={() => copyText(aspOutput, "asp")}
                >
                  {copiedASP ? "✓ Copied" : "⎘ Copy"}
                </button>
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
                <button
                  class="btn btn-primary"
                  on:click={runBatch}
                  disabled={loadingBatch}
                >
                  {#if loadingBatch}<span class="spin"></span>{/if}
                  Run Batch · {batchTotal || "?"} samples
                </button>
              {/if}
            </div>
            <div class="card-body">
              <div class="drop">
                <input type="file" accept=".json" on:change={onFileChange} />
                <div class="drop-icon">📂</div>
                {#if batchFileName}
                  <div class="drop-lbl" style="color:#60a5fa">
                    {batchFileName}
                  </div>
                  <div class="drop-sub">click to change file</div>
                {:else}
                  <div class="drop-lbl">
                    Drop JSON dataset or click to browse
                  </div>
                  <div class="drop-sub">
                    {"{ data_dict: [{NL_V2, CNL_V2, ASP}] }"}
                  </div>
                {/if}
              </div>
            </div>
          </div>

          {#if loadingBatch || batchProgress > 0}
            <div class="prog-wrap">
              <div class="prog-bg">
                <div
                  class="prog-fill"
                  style="width:{batchTotal
                    ? (batchProgress / batchTotal) * 100
                    : 0}%"
                ></div>
              </div>
              <div class="prog-label">
                {batchProgress} / {batchTotal} processed
              </div>
            </div>
          {/if}

          {#if batchResults.length}
            <div class="stats">
              <div class="stat">
                <div class="sv b">{batchResults.length}</div>
                <div class="sl">Total</div>
              </div>
              <div class="stat">
                <div class="sv g">{syntaxAcc}%</div>
                <div class="sl">Syntax Acc.</div>
              </div>
              <div class="stat">
                <div class="sv p">{compileAcc}%</div>
                <div class="sl">Compile Rate</div>
              </div>
              <div
                class="stat"
                style="flex:0 0 auto; display:flex; align-items:center;"
              >
                <button class="btn btn-secondary" on:click={downloadCSV}
                  >⬇ Export CSV</button
                >
              </div>
            </div>

            <div class="tbl-wrap">
              <table>
                <thead>
                  <tr
                    ><th>#</th><th>Natural Language</th><th>Predicted CNL</th
                    ><th>Syntax</th><th>Generated ASP</th><th>Compiled</th></tr
                  >
                </thead>
                <tbody>
                  {#each batchResults as r, i}
                    <tr>
                      <td style="color:#1e2a3a">{i + 1}</td>
                      <td title={r.nl}>{r.nl}</td>
                      <td title={r.predicted_cnl} style="color:#c4b5fd"
                        >{r.predicted_cnl}</td
                      >
                      <td
                        ><span
                          class="badge {r.syntax_valid ? 'valid' : 'invalid'}"
                          >{r.syntax_valid ? "✓" : "✗"}</span
                        ></td
                      >
                      <td title={r.asp} style="color:#6ee7b7">{r.asp}</td>
                      <td
                        ><span class="badge {r.compiled ? 'compiled' : 'error'}"
                          >{r.compiled ? "✓" : "✗"}</span
                        ></td
                      >
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

<!-- ═══════════════════════════ STYLES ═══════════════════════════ -->
<style>
  /* ── Reset & Base ── */
  :global(*, *::before, *::after) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  :global(body) {
    font-family: "Inter", "Syne", system-ui, sans-serif;
    background: #f1f5f9;
    color: #1e293b;
    min-height: 100vh;
  }

  /* ── Layout ── */
  .app {
    display: grid;
    grid-template-rows: 52px 1fr;
    height: 100vh;
    overflow: hidden;
  }
  .main {
    display: grid;
    grid-template-columns: 260px 1fr;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    padding: 0 1.25rem;
    gap: 1rem;
    z-index: 10;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .brand-icon {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }
  .brand-name {
    font-weight: 700;
    font-size: 0.9rem;
    color: #0f172a;
    letter-spacing: -0.01em;
  }
  .brand-sub {
    font-size: 0.6rem;
    color: #94a3b8;
    font-family: "JetBrains Mono", monospace;
    margin-top: 1px;
  }
  .header-divider {
    width: 1px;
    height: 20px;
    background: #e2e8f0;
    margin: 0 0.25rem;
  }

  .tabs {
    display: flex;
    gap: 2px;
  }
  .tab-btn {
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    border: none;
    background: none;
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.1s;
    font-family: inherit;
  }
  .tab-btn:hover {
    color: #475569;
    background: #f1f5f9;
  }
  .tab-btn.active {
    background: #f1f5f9;
    color: #1e293b;
    border: 1px solid #e2e8f0;
  }

  .header-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .gpu-pill {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.6rem;
    color: #059669;
    background: rgba(5, 150, 105, 0.07);
    border: 1px solid rgba(5, 150, 105, 0.2);
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.65rem;
    color: #94a3b8;
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot.online {
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e66;
  }
  .dot.offline {
    background: #ef4444;
  }
  .dot.unknown {
    background: #f59e0b;
  }

  /* ── Sidebar ── */
  aside {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-inner {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  .sidebar-inner::-webkit-scrollbar {
    width: 4px;
  }
  .sidebar-inner::-webkit-scrollbar-track {
    background: transparent;
  }
  .sidebar-inner::-webkit-scrollbar-thumb {
    background: #e2e8f0;
    border-radius: 2px;
  }

  .s-section-label {
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
  }

  /* Model selector dropdown */
  .model-selector {
    position: relative;
  }
  .model-trigger {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    cursor: pointer;
    transition: border-color 0.1s;
    text-align: left;
  }
  .model-trigger:hover {
    border-color: #cbd5e1;
    background: #f1f5f9;
  }
  .model-trigger.open {
    border-color: #2563eb;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    background: #f8fafc;
  }
  .mt-icon {
    font-size: 1rem;
    flex-shrink: 0;
  }
  .mt-text {
    flex: 1;
    min-width: 0;
  }
  .mt-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #1e293b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mt-sub {
    font-size: 0.6rem;
    color: #94a3b8;
    margin-top: 1px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mt-tag {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    color: #2563eb;
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.2);
    padding: 1px 6px;
    border-radius: 4px;
    flex-shrink: 0;
  }
  .mt-caret {
    color: #cbd5e1;
    font-size: 0.65rem;
    flex-shrink: 0;
    transition: transform 0.15s;
  }
  .mt-caret.open {
    transform: rotate(180deg);
  }

  /* Dropdown panel */
  .model-dropdown {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    z-index: 100;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    overflow: hidden;
  }
  .model-search-wrap {
    padding: 0.5rem;
    border-bottom: 1px solid #f1f5f9;
  }
  .model-search {
    width: 100%;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.38rem 0.6rem;
    color: #1e293b;
    font-size: 0.75rem;
    outline: none;
    font-family: inherit;
    transition: border-color 0.1s;
  }
  .model-search:focus {
    border-color: #2563eb;
    background: #ffffff;
  }
  .model-search::placeholder {
    color: #cbd5e1;
  }

  .model-list {
    max-height: 260px;
    overflow-y: auto;
  }
  .model-list::-webkit-scrollbar {
    width: 3px;
  }
  .model-list::-webkit-scrollbar-thumb {
    background: #e2e8f0;
  }
  .model-group-label {
    padding: 0.45rem 0.65rem 0.2rem;
    font-size: 0.56rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #cbd5e1;
  }
  .model-option {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.65rem;
    cursor: pointer;
    transition: background 0.08s;
  }
  .model-option:hover {
    background: #f8fafc;
  }
  .model-option.selected {
    background: rgba(37, 99, 235, 0.06);
  }
  .model-option.disabled {
    opacity: 0.35;
    cursor: not-allowed;
    pointer-events: none;
  }
  .mo-icon {
    font-size: 0.85rem;
    flex-shrink: 0;
  }
  .mo-text {
    flex: 1;
    min-width: 0;
  }
  .mo-label {
    font-size: 0.75rem;
    color: #1e293b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mo-sub {
    font-size: 0.6rem;
    color: #94a3b8;
  }
  .mo-tag {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.56rem;
    color: #94a3b8;
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 3px;
    flex-shrink: 0;
    border: 1px solid #e2e8f0;
  }
  .model-option.selected .mo-tag {
    color: #2563eb;
    background: rgba(37, 99, 235, 0.08);
    border-color: rgba(37, 99, 235, 0.2);
  }
  .mo-check {
    font-size: 0.65rem;
    color: #2563eb;
    flex-shrink: 0;
  }

  /* Pipeline mode */
  .mode-row {
    display: flex;
    gap: 0.4rem;
  }
  .mode-btn {
    flex: 1;
    padding: 0.5rem 0.4rem;
    border-radius: 7px;
    border: 1px solid #e2e8f0;
    background: #f8fafc;
    color: #94a3b8;
    font-family: inherit;
    font-size: 0.7rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.1s;
    line-height: 1.3;
  }
  .mode-btn:hover {
    background: #f1f5f9;
    color: #64748b;
    border-color: #cbd5e1;
  }
  .mode-btn.active {
    background: #ffffff;
    border-color: #2563eb;
    color: #1e293b;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  }
  .mode-icon {
    font-size: 0.9rem;
    display: block;
    margin-bottom: 2px;
  }
  .mode-label {
    font-weight: 600;
    font-size: 0.68rem;
    display: block;
  }
  .mode-sub {
    font-size: 0.58rem;
    color: #94a3b8;
    display: block;
  }

  /* Server info */
  .server-info {
    margin-top: auto;
    padding: 0.75rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
  }
  .server-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
  }
  .server-label {
    font-size: 0.6rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
  }
  .server-val {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.6rem;
    color: #64748b;
  }
  .server-models {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    line-height: 1.6;
  }

  /* ── Content area ── */
  .content {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #f1f5f9;
  }
  .panel {
    flex: 1;
    overflow-y: auto;
    padding: 1.25rem;
  }
  .panel::-webkit-scrollbar {
    width: 5px;
  }
  .panel::-webkit-scrollbar-track {
    background: transparent;
  }
  .panel::-webkit-scrollbar-thumb {
    background: #e2e8f0;
    border-radius: 3px;
  }

  /* Pipeline bar */
  .pipe-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }
  .pnode {
    padding: 3px 10px;
    border-radius: 5px;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.65rem;
    font-weight: 600;
  }
  .pnode.nl {
    background: rgba(37, 99, 235, 0.08);
    color: #2563eb;
    border: 1px solid rgba(37, 99, 235, 0.2);
  }
  .pnode.cnl {
    background: rgba(124, 58, 237, 0.08);
    color: #7c3aed;
    border: 1px solid rgba(124, 58, 237, 0.2);
  }
  .pnode.asp {
    background: rgba(5, 150, 105, 0.08);
    color: #059669;
    border: 1px solid rgba(5, 150, 105, 0.2);
  }
  .parr {
    color: #cbd5e1;
    font-size: 0.8rem;
  }
  .pipe-model {
    margin-left: auto;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.6rem;
    color: #94a3b8;
  }

  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-bottom: 0.85rem;
    overflow: hidden;
    transition: border-color 0.15s;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }
  .card:focus-within {
    border-color: #93c5fd;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.06);
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid #f1f5f9;
    background: #fafafa;
  }
  .card-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
  }
  .card-body {
    padding: 0.85rem;
  }

  /* Textarea */
  textarea {
    width: 100%;
    background: none;
    border: none;
    outline: none;
    color: #1e293b;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.8rem;
    line-height: 1.75;
    resize: vertical;
    min-height: 90px;
  }
  textarea::placeholder {
    color: #cbd5e1;
  }

  /* Output */
  .code-out {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.8rem;
    line-height: 1.75;
    white-space: pre-wrap;
    min-height: 48px;
  }
  .code-out.cnl-color {
    color: #6d28d9;
  }
  .code-out.asp-color {
    color: #047857;
  }
  .ph {
    color: #cbd5e1;
    font-style: italic;
    font-size: 0.75rem;
    font-family: "JetBrains Mono", monospace;
  }

  /* Buttons */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.33rem 0.8rem;
    border-radius: 6px;
    border: none;
    font-family: inherit;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.1s;
    white-space: nowrap;
  }
  .btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .btn-primary {
    background: #2563eb;
    color: #fff;
  }
  .btn-primary:hover:not(:disabled) {
    background: #1d4ed8;
  }
  .btn-secondary {
    background: #f8fafc;
    color: #64748b;
    border: 1px solid #e2e8f0;
  }
  .btn-secondary:hover:not(:disabled) {
    background: #f1f5f9;
    color: #475569;
    border-color: #cbd5e1;
  }
  .btn-success {
    background: rgba(5, 150, 105, 0.08);
    color: #059669;
    border: 1px solid rgba(5, 150, 105, 0.25);
  }
  .btn-success:hover:not(:disabled) {
    background: rgba(5, 150, 105, 0.14);
  }
  .btn-copy {
    background: none;
    border: 1px solid transparent;
    color: #94a3b8;
    padding: 0.28rem 0.6rem;
    border-radius: 5px;
    font-size: 0.68rem;
    transition: all 0.1s;
  }
  .btn-copy:hover {
    background: #f1f5f9;
    border-color: #e2e8f0;
    color: #64748b;
  }
  .btn-copy.copied {
    color: #059669;
    border-color: rgba(5, 150, 105, 0.3);
    background: rgba(5, 150, 105, 0.06);
  }
  .btn-danger {
    background: rgba(239, 68, 68, 0.06);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.2);
  }
  .btn-danger:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.1);
  }
  .row {
    display: flex;
    gap: 0.35rem;
    align-items: center;
    flex-wrap: wrap;
  }

  /* Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 20px;
  }
  .badge.valid {
    background: rgba(5, 150, 105, 0.08);
    color: #059669;
    border: 1px solid rgba(5, 150, 105, 0.25);
  }
  .badge.invalid {
    background: rgba(239, 68, 68, 0.07);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.2);
  }
  .badge.compiled {
    background: rgba(5, 150, 105, 0.08);
    color: #059669;
    border: 1px solid rgba(5, 150, 105, 0.25);
  }
  .badge.error {
    background: rgba(239, 68, 68, 0.07);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.2);
  }

  /* Error */
  .err-bar {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    background: rgba(239, 68, 68, 0.05);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 1rem;
    color: #dc2626;
    font-size: 0.72rem;
    font-family: "JetBrains Mono", monospace;
  }

  /* Spinner */
  .spin {
    display: inline-block;
    width: 10px;
    height: 10px;
    border: 2px solid rgba(255, 255, 255, 0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Ensemble info */
  .ensemble-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    background: rgba(124, 58, 237, 0.05);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 8px;
    padding: 0.5rem 0.85rem;
    margin-bottom: 1rem;
    font-size: 0.62rem;
    font-family: "JetBrains Mono", monospace;
    color: #7c3aed;
  }
  .fold-chip {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.58rem;
    color: #94a3b8;
  }
  .fold-chip.valid {
    border-color: rgba(5, 150, 105, 0.3);
    color: #059669;
    background: rgba(5, 150, 105, 0.05);
  }
  .fold-chip.invalid {
    border-color: rgba(239, 68, 68, 0.2);
    color: #dc2626;
    background: rgba(239, 68, 68, 0.04);
  }

  /* Progress */
  .prog-wrap {
    margin-bottom: 1rem;
  }
  .prog-bg {
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
    overflow: hidden;
  }
  .prog-fill {
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    transition: width 0.25s;
  }
  .prog-label {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: #94a3b8;
    margin-top: 4px;
  }

  /* Stats */
  .stats {
    display: flex;
    gap: 0.65rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }
  .stat {
    flex: 1;
    min-width: 90px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }
  .sv {
    font-size: 1.4rem;
    font-weight: 800;
    font-family: "JetBrains Mono", monospace;
  }
  .sv.b {
    color: #2563eb;
  }
  .sv.g {
    color: #059669;
  }
  .sv.p {
    color: #7c3aed;
  }
  .sl {
    font-size: 0.58rem;
    color: #94a3b8;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Drop zone */
  .drop {
    border: 2px dashed #e2e8f0;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    position: relative;
    transition: all 0.12s;
  }
  .drop:hover {
    border-color: #2563eb;
    background: rgba(37, 99, 235, 0.02);
  }
  .drop input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .drop-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
  }
  .drop-lbl {
    font-size: 0.78rem;
    color: #64748b;
  }
  .drop-sub {
    font-size: 0.62rem;
    color: #cbd5e1;
    margin-top: 3px;
    font-family: "JetBrains Mono", monospace;
  }

  /* Table */
  .tbl-wrap {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow-x: auto;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.72rem;
  }
  th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    background: #f8fafc;
    color: #94a3b8;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #e2e8f0;
    white-space: nowrap;
  }
  td {
    padding: 0.48rem 0.75rem;
    border-bottom: 1px solid #f8fafc;
    font-family: "JetBrains Mono", monospace;
    color: #64748b;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  tr:last-child td {
    border-bottom: none;
  }
  tr:hover td {
    background: #f8fafc;
  }
</style>
