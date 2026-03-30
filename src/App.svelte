<script>
  import { onMount } from "svelte";

  // ── State ──────────────────────────────────────────────────────────────────
  let selectedModel = "qwen3_8b";
  let pipelineMode = "direct";
  let activeTab = "single";

  let nlInput = "";
  let cnlOutput = "";
  let cnlEditMode = false;
  let aspOutput = "";
  let cnlStatus = "";
  let aspStatus = "";

  let batchFile = null;
  let batchFileName = "";
  let batchResults = [];
  let batchProgress = 0;
  let batchTotal = 0;

  // ── Dashboard (multi-model comparison) ───────────────────────────────────
  // NOTE: K-fold ensemble support temporarily disabled.
  // Currently supports full dataset comparison only.
  // K-fold functionality to be added per-model later.
  let dashModels = {};
  let dashInputMode = "single";
  let dashNLInput = "";
  let dashFile = null;
  let dashFileName = "";
  let dashRecords = [];
  let dashResults = [];
  let dashProgress = 0;
  let dashTotal = 0;
  let dashBatchLimit = 0;
  let dashRunning = false;
  let dashStopped = false;
  let dashDone = false;
  let dashResumeIndex = 0;
  let dashParsedRecords = [];
  let dashFilter = "all";
  let dashSearch = "";
  let dashExpandedRow = null;
  let dashErrorMsg = "";

  // ── Input history ───────────────────────────────────────────────────────────
  let history = [];
  let historyOpen = false;

  let loadingNL2CNL = false;
  let loadingCNL2ASP = false;
  let loadingDirect = false;
  let loadingBatch = false;
  let loadingModel = false;
  let cancelModelLoad = false; // flag to cancel model loading
  let serverStatus = "unknown";
  let availableModels = {};
  let loadedKeys = [];
  let gpuName = "";
  let errorMsg = "";
  let qwen35Worker = "unknown";

  // Copy feedback
  let copiedNL = false;
  let copiedCNL = false;
  let copiedASP = false;

  const API = "http://localhost:8000";

  // ── Model group display order / labels ────────────────────────────────
  const GROUP_META = {
    llm: { label: "Causal LLMs", color: "#2563eb" },
    t5small: { label: "T5-Small", color: "#7c3aed" },
    t5large: { label: "T5-Large", color: "#7c3aed" },
    t5_3b: { label: "T5-3B", color: "#7c3aed" },
  };
  const GROUP_ORDER = ["llm", "t5small", "t5large", "t5_3b"];

  // ── Health check ────────────────────────────────────────────────────────────
  onMount(async () => {
    try {
      history = JSON.parse(sessionStorage.getItem("nl2asp_history") || "[]");
    } catch {}
    await refreshHealth();
  });

  function handleOutsideClick(e) {
    if (
      !e.target.closest(".history-panel") &&
      !e.target.closest(".btn-history")
    )
      historyOpen = false;
  }

  // ── History ───────────────────────────────────────────────────────────────
  function saveToHistory(nl, cnl, asp, model, syntaxValid, compiled) {
    const entry = {
      id: Date.now(),
      ts: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
      nl,
      cnl,
      asp,
      model,
      syntaxValid,
      compiled,
    };
    history = [entry, ...history].slice(0, 10);
    try {
      sessionStorage.setItem("nl2asp_history", JSON.stringify(history));
    } catch {}
  }

  function loadFromHistory(entry) {
    nlInput = entry.nl;
    cnlOutput = entry.cnl;
    aspOutput = entry.asp;
    cnlStatus = entry.syntaxValid ? "valid" : "invalid";
    aspStatus = entry.compiled ? "compiled" : "error";
    historyOpen = false;
  }

  function clearHistory() {
    history = [];
    try {
      sessionStorage.removeItem("nl2asp_history");
    } catch {}
  }

  async function refreshHealth() {
    try {
      const r = await fetch(`${API}/api/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (r.ok) {
        const d = await r.json();
        serverStatus = "online";
        availableModels = d.available_models ?? {};
        loadedKeys = d.loaded_models ?? [];
        gpuName = d.gpu ?? "";
        qwen35Worker = d.qwen35_worker ?? "unknown";
        if (!availableModels[selectedModel]?.exists) {
          const first = Object.entries(availableModels).find(
            ([, m]) => m.exists,
          );
          if (first) selectedModel = first[0];
        }
      } else {
        serverStatus = "offline";
      }
    } catch {
      serverStatus = "offline";
    }
  }

  // ── Group models for sidebar display ──────────────────────────────────────
  $: groupedModels = GROUP_ORDER.map((gid) => ({
    gid,
    meta: GROUP_META[gid],
    models: Object.entries(availableModels)
      .filter(([, m]) => m.group === gid)
      .map(([key, m]) => ({ key, ...m })),
  })).filter((g) => g.models.length > 0);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function reset() {
    cnlOutput = "";
    aspOutput = "";
    cnlStatus = "";
    aspStatus = "";
    errorMsg = "";
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

  async function selectModel(key) {
    if (key === selectedModel) return;
    selectedModel = key;
    if (availableModels[key]?.group === "llm" && !loadedKeys.includes(key)) {
      errorMsg = `ℹ Model "${availableModels[key]?.label}" will be loaded on first use (may take 1-3 min for large models).`;
    } else {
      errorMsg = "";
    }
  }

  async function unloadModel(key) {
    try {
      await postJSON("/api/unload", { model: key });
      await refreshHealth();
    } catch (e) {
      errorMsg = e.message;
    }
  }

  async function unloadAll() {
    try {
      await postJSON("/api/unload", { model: null });
      await refreshHealth();
    } catch (e) {
      errorMsg = e.message;
    }
  }

  async function loadAll() {
    try {
      errorMsg =
        "⏳ Loading all models into VRAM... this may take several minutes.";
      const existingModels = Object.entries(availableModels)
        .filter(([, m]) => m.exists && m.group === "llm")
        .map(([key]) => key);

      if (existingModels.length === 0) {
        errorMsg = "No LLM models available to load.";
        return;
      }

      // Trigger loading for each model by doing a simple inference
      for (const modelKey of existingModels) {
        if (!loadedKeys.includes(modelKey)) {
          try {
            await postJSON("/api/nl2asp", {
              nl: "test",
              model: modelKey,
            });
          } catch (e) {
            console.error(`Error loading ${modelKey}:`, e);
          }
        }
      }

      await refreshHealth();
      errorMsg = "✓ All available models loaded successfully!";
      setTimeout(() => {
        errorMsg = "";
      }, 3000);
    } catch (e) {
      errorMsg = `Error loading models: ${e.message}`;
    }
  }

  function trackLoad(isLLM) {
    if (isLLM && !loadedKeys.includes(selectedModel)) {
      loadingModel = true;
      cancelModelLoad = false;
    }
  }
  function untrackLoad() {
    loadingModel = false;
    cancelModelLoad = false;
  }
  function cancelLoad() {
    cancelModelLoad = true;
    loadingModel = false;
  }

  async function runNL2CNL() {
    if (!nlInput.trim()) return;
    reset();
    loadingNL2CNL = true;
    trackLoad(availableModels[selectedModel]?.group === "llm");
    try {
      const d = await postJSON("/api/nl2cnl", {
        nl: nlInput,
        model: selectedModel,
      });
      cnlOutput = d.cnl;
      cnlStatus = d.syntax_valid ? "valid" : "invalid";
      await refreshHealth();
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loadingNL2CNL = false;
      untrackLoad();
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
    trackLoad(availableModels[selectedModel]?.group === "llm");
    try {
      const d = await postJSON("/api/nl2asp", {
        nl: nlInput,
        model: selectedModel,
      });
      cnlOutput = d.cnl;
      cnlStatus = d.syntax_valid ? "valid" : "invalid";
      aspOutput = d.asp;
      aspStatus = d.compiled ? "compiled" : "error";
      saveToHistory(
        nlInput,
        d.cnl,
        d.asp,
        selectedModel,
        d.syntax_valid,
        d.compiled,
      );
      await refreshHealth();
    } catch (e) {
      errorMsg = e.message;
    } finally {
      loadingDirect = false;
      untrackLoad();
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
    trackLoad(availableModels[selectedModel]?.group === "llm");
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
        if (i === 0) untrackLoad();
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
          const d = await postJSON("/api/nl2asp", { nl, model: selectedModel });
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
      await refreshHealth();
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

  // ── Dashboard ─────────────────────────────────────────────────────────────
  $: dashSelectedKeys = Object.entries(dashModels)
    .filter(([, v]) => v)
    .map(([k]) => k);

  function onDashFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    dashFile = f;
    dashFileName = f.name;
    dashRecords = [];
    dashResults = [];
    dashProgress = 0;
    dashTotal = 0;
    dashDone = false;
    dashBatchLimit = 0;
    dashErrorMsg = "";
    f.text().then((txt) => {
      try {
        const j = JSON.parse(txt);
        dashRecords = Array.isArray(j)
          ? j
          : Array.isArray(j.data_dict)
            ? j.data_dict
            : [];
        dashTotal = dashRecords.length;
        dashBatchLimit = dashRecords.length;
      } catch {
        dashErrorMsg = "Invalid JSON file";
      }
    });
  }

  async function runDashboard() {
    if (dashSelectedKeys.length === 0) {
      dashErrorMsg = "Select at least one model.";
      return;
    }
    dashErrorMsg = "";
    dashResults = [];
    dashProgress = 0;
    dashResumeIndex = 0;
    dashDone = false;
    dashRunning = true;
    dashStopped = false;

    if (dashInputMode === "single") {
      if (!dashNLInput.trim()) {
        dashErrorMsg = "Enter a NL sentence.";
        dashRunning = false;
        return;
      }
      dashParsedRecords = [{ NL_V2: dashNLInput, CNL_V2: "", ASP: "" }];
    } else {
      if (!dashRecords.length) {
        dashErrorMsg = "Load a JSON file first.";
        dashRunning = false;
        return;
      }
      const limit =
        dashBatchLimit > 0 && dashBatchLimit < dashRecords.length
          ? dashBatchLimit
          : dashRecords.length;
      dashParsedRecords = dashRecords.slice(0, limit);
    }
    dashTotal = dashParsedRecords.length;
    await _runDashFrom(0);
    dashRunning = false;
    dashDone = !dashStopped;
  }

  async function resumeDashboard() {
    if (!dashParsedRecords.length) return;
    dashRunning = true;
    dashStopped = false;
    dashDone = false;
    dashErrorMsg = "";
    await _runDashFrom(dashResumeIndex);
    dashRunning = false;
    dashDone = !dashStopped;
  }

  async function _runDashFrom(startIdx) {
    // Process each record through all selected models
    // Note: K-fold ensemble disabled for now. Only full model inference supported.
    for (let i = startIdx; i < dashParsedRecords.length; i++) {
      if (dashStopped) {
        dashResumeIndex = i;
        break;
      }
      const rec = dashParsedRecords[i];
      const nl = rec.NL_V2 ?? rec.nl ?? rec.NL ?? "";
      const row = {
        id: rec.id ?? rec.ID ?? "",
        category: rec.category ?? rec.Category ?? rec.cat ?? "",
        nl,
        gold_cnl: rec.CNL_V2 ?? rec.cnl ?? "",
        gold_asp: rec.ASP ?? rec.asp ?? "",
        models: {},
      };
      for (const key of dashSelectedKeys) {
        if (dashStopped) break;
        try {
          const d = await postJSON("/api/nl2asp", { nl, model: key });
          row.models[key] = {
            cnl: d.cnl,
            asp: d.asp,
            syntax_valid: d.syntax_valid,
            compiled: d.compiled,
          };
        } catch (e) {
          row.models[key] = {
            cnl: "ERROR",
            asp: "ERROR",
            syntax_valid: false,
            compiled: false,
          };
        }
      }
      dashResults = [...dashResults, row];
      dashProgress = i + 1;
    }
  }

  function stopDashboard() {
    dashStopped = true;
    dashRunning = false;
  }
  function resetDashboard() {
    dashResults = [];
    dashProgress = 0;
    dashTotal = 0;
    dashDone = false;
    dashStopped = false;
    dashResumeIndex = 0;
    dashParsedRecords = [];
    dashFile = null;
    dashFileName = "";
    dashRecords = [];
    dashBatchLimit = 0;
    dashErrorMsg = "";
  }

  function downloadDashCSV() {
    const modelCols = dashSelectedKeys.flatMap((k) => {
      const lbl = availableModels[k]?.label ?? k;
      return [
        `${lbl} CNL`,
        `${lbl} CNL Valid`,
        `${lbl} ASP`,
        `${lbl} Compiled`,
      ];
    });
    const header = [
      "ID",
      "Category",
      "NL",
      "Gold CNL",
      "Gold ASP",
      ...modelCols,
    ];
    const rows = dashResults.map((r) => {
      const base = [
        `"${String(r.id ?? "").replace(/"/g, '""')}"`,
        `"${String(r.category ?? "").replace(/"/g, '""')}"`,
        `"${r.nl.replace(/"/g, '""')}"`,
        `"${r.gold_cnl.replace(/"/g, '""')}"`,
        `"${r.gold_asp.replace(/"/g, '""')}"`,
      ];
      const modelData = dashSelectedKeys.flatMap((k) => {
        const m = r.models[k] ?? {
          cnl: "",
          asp: "",
          syntax_valid: false,
          compiled: false,
        };
        return [
          `"${m.cnl.replace(/"/g, '""')}"`,
          m.syntax_valid,
          `"${m.asp.replace(/"/g, '""')}"`,
          m.compiled,
        ];
      });
      return [...base, ...modelData];
    });
    const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `nl2asp_comparison_${Date.now()}.csv`;
    a.click();
  }

  function toggleDashModel(key) {
    dashModels = { ...dashModels, [key]: !dashModels[key] };
  }

  $: dashStats = dashSelectedKeys.map((k) => {
    const rows = dashResults.filter((r) => r.models[k]);
    const n = rows.length;
    return {
      key: k,
      label: availableModels[k]?.label ?? k,
      icon: availableModels[k]?.icon ?? "🤖",
      tag: availableModels[k]?.tag ?? "",
      total: n,
      syntaxOk: rows.filter((r) => r.models[k].syntax_valid).length,
      compiled: rows.filter((r) => r.models[k].compiled).length,
      syntaxPct: n
        ? (
            (rows.filter((r) => r.models[k].syntax_valid).length / n) *
            100
          ).toFixed(1)
        : "—",
      compilePct: n
        ? ((rows.filter((r) => r.models[k].compiled).length / n) * 100).toFixed(
            1,
          )
        : "—",
    };
  });

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
  $: selectedMeta = availableModels[selectedModel] ?? {};
</script>

<svelte:window on:click={handleOutsideClick} />

<!-- ═════════════════════════ MARKUP ═════════════════════════ -->
<div class="app">
  <!-- Header -->
  <header>
    <div class="brand">
      <img src="/logo.png" alt="NL2ASP" class="brand-logo" />
      <div>
        <div class="brand-name">NL2ASP</div>
        <div class="brand-sub">NL → CNL → ASP pipeline</div>
      </div>
    </div>

    <div class="header-mid">
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
      <button
        class="tab-btn"
        class:active={activeTab === "dashboard"}
        on:click={() => (activeTab = "dashboard")}>Dashboard</button
      >
    </div>

    <div class="header-right">
      {#if gpuName && gpuName !== "none"}
        <span class="gpu-tag">🎮 {gpuName}</span>
      {/if}
      <span class="status-pill">
        <span class="dot {serverStatus}"></span>
        Gateway {serverStatus === "online"
          ? "online"
          : serverStatus === "offline"
            ? "offline"
            : "..."}
      </span>
      <span
        class="status-pill"
        title="Qwen3.5 worker (server_qwen35.py under llm_peft_qwen35 env)"
      >
        <span
          class="dot {qwen35Worker === 'online'
            ? 'online'
            : qwen35Worker === 'offline'
              ? 'offline'
              : 'unknown'}"
        ></span>
        Qwen3.5 {qwen35Worker === "online"
          ? "worker ✓"
          : qwen35Worker === "offline"
            ? "worker ✗"
            : "worker …"}
      </span>
      <button
        class="btn-history"
        class:has-items={history.length}
        on:click={() => (historyOpen = !historyOpen)}
      >
        ⏱ History {#if history.length}<span class="hist-count"
            >{history.length}</span
          >{/if}
      </button>
    </div>
  </header>

  <div class="main">
    <!-- Sidebar -->
    <aside>
      <div class="sidebar-inner">
        <!-- Pipeline Mode -->
        <div class="group-section">
          <div class="s-label">Pipeline Mode</div>
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
            <span class="mode-label">Step-by-step</span>
            <span class="mode-sub">NL → CNL → ASP</span>
          </button>
        </div>

        <!-- Model Groups -->
        <div>
          <div class="s-label">Model Selection</div>
          {#each groupedModels as grp}
            <div class="group-section">
              <div class="group-header">
                <span class="group-dot" style="background:{grp.meta.color}"
                ></span>
                {grp.meta.label}
              </div>
              {#each grp.models as m}
                <button
                  class="model-btn"
                  class:active={selectedModel === m.key}
                  class:unavailable={!m.exists ||
                    (m.kind === "causal_qwen3_5" && !m.worker_online)}
                  on:click={() =>
                    m.exists &&
                    (m.kind !== "causal_qwen3_5" || m.worker_online) &&
                    selectModel(m.key)}
                  title={!m.exists
                    ? "Weights not found on server"
                    : m.kind === "causal_qwen3_5" && !m.worker_online
                      ? "Qwen3.5 worker offline — start server_qwen35.py"
                      : m.label}
                >
                  <span class="m-icon">{m.icon}</span>
                  <span class="m-label">{m.label}</span>
                  {#if m.kind === "causal_qwen3_5" && !m.worker_online && serverStatus === "online"}
                    <span
                      class="worker-badge"
                      title="Start: conda activate llm_peft_qwen35 && python server_qwen35.py"
                      >offline</span
                    >
                  {:else if loadedKeys.includes(m.key)}
                    <span class="loaded-dot" title="Loaded in VRAM"></span>
                  {/if}
                  <span class="m-tag">{m.tag}</span>
                  {#if loadedKeys.includes(m.key)}
                    <button
                      class="unload-btn"
                      on:click|stopPropagation={() => unloadModel(m.key)}
                      title="Unload from VRAM">✕</button
                    >
                  {/if}
                </button>
              {/each}
            </div>
          {/each}
        </div>
      </div>

      <!-- Footer -->
      <div class="sidebar-footer">
        <div class="server-info">
          {API}<br />
          {#if loadedKeys.length}
            In VRAM:<br />{loadedKeys.length} model{loadedKeys.length > 1
              ? "s"
              : ""}
          {:else}
            No models in VRAM
          {/if}
        </div>
        <div style="display:flex; gap:0.5rem">
          <button
            class="unload-all-btn"
            on:click={loadAll}
            disabled={loadedKeys.length ===
              Object.entries(availableModels).filter(([, m]) => m.exists)
                .length}
            style="background:#ecfdf5; color:#059669; border-color:#a7f3d0; flex:1"
          >
            ⬆ Load All
          </button>
          <button
            class="unload-all-btn"
            on:click={unloadAll}
            disabled={loadedKeys.length === 0}
            style="flex:1"
          >
            🗑 Unload All
          </button>
        </div>
      </div>
    </aside>

    <!-- Content -->
    <div class="content">
      <div class="panel">
        {#if activeTab === "single" || activeTab === "batch"}
          {#if loadingModel}
            <div
              style="display:flex; align-items:center; gap:0.6rem; background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:0.7rem 1rem; margin-bottom:1rem; color:#c2410c; font-size:0.72rem; font-family:'JetBrains Mono', monospace"
            >
              <span
                style="display:inline-block; width:14px; height:14px; border:2px solid rgba(245,158,11,0.2); border-top-color:#f59e0b; border-radius:50%; animation:spin 0.65s linear infinite; flex-shrink:0"
              ></span>
              <span style="flex:1">
                Loading {selectedMeta.label ?? selectedModel} into VRAM — this may
                take 1–3 minutes for 8B models…
              </span>
              <button
                class="btn btn-danger"
                on:click={cancelLoad}
                style="margin-left:auto; padding:0.25rem 0.65rem; font-size:0.65rem"
              >
                ✕ Cancel
              </button>
            </div>
          {/if}
          {#if errorMsg}
            <div class={errorMsg.startsWith("ℹ") ? "info-msg" : "err-bar"}>
              {errorMsg.startsWith("ℹ") ? "" : "⚠ "}{errorMsg}
            </div>
          {/if}
        {/if}

        <!-- ══ SINGLE TAB ══ -->
        {#if activeTab === "single"}
          <div class="pipe-bar">
            <span class="pnode nl">NL</span>
            <span class="parr">→</span>
            <span class="pnode cnl">CNL</span>
            <span class="parr">→</span>
            <span class="pnode asp">ASP</span>
            <span class="pipe-model">
              {selectedMeta.icon ?? ""}
              {selectedMeta.label ?? selectedModel}
            </span>
          </div>

          <!-- NL Input -->
          <div class="card">
            <div class="card-header">
              <span class="card-title"
                ><span class="pnode nl">NL</span> Natural Language</span
              >
              <div class="row">
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
                  {pipelineMode === "direct"
                    ? "⚡ NL → ASP"
                    : "⚡ Full Pipeline"}
                </button>
                {#if nlInput}
                  <button
                    class="btn-copy"
                    class:copied={copiedNL}
                    on:click={() => copyText(nlInput, "nl")}
                  >
                    {copiedNL ? "✓" : "📋"}
                    {copiedNL ? "Copied" : "Copy NL"}
                  </button>
                {/if}
                {#if nlInput || cnlOutput || aspOutput}
                  <button class="btn btn-secondary" on:click={reset}
                    >✕ Clear</button
                  >
                {/if}
              </div>
            </div>
            <div class="card-body">
              <textarea
                bind:value={nlInput}
                placeholder="Enter natural language specification…&#10;e.g. Every node must be reachable from the source vertex."
              ></textarea>
            </div>
          </div>

          <!-- CNL Output -->
          {#if cnlOutput || loadingNL2CNL || loadingDirect}
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
                  {#if pipelineMode === "stepwise" && cnlOutput && !cnlEditMode}
                    <button
                      class="btn btn-success"
                      on:click={runCNL2ASP}
                      disabled={loadingCNL2ASP}
                    >
                      {#if loadingCNL2ASP}<span class="spin"></span>{/if}
                      Compile → ASP
                    </button>
                  {/if}
                  {#if cnlOutput}
                    {#if cnlEditMode}
                      <button
                        class="btn btn-success"
                        on:click={() => (cnlEditMode = false)}
                      >
                        ✓ Save
                      </button>
                      <button
                        class="btn btn-secondary"
                        on:click={() => (cnlEditMode = false)}
                      >
                        ✕ Cancel
                      </button>
                    {:else}
                      <button
                        class="btn btn-secondary"
                        on:click={() => (cnlEditMode = true)}
                      >
                        ✏️ Edit
                      </button>
                      <button
                        class="btn-copy"
                        class:copied={copiedCNL}
                        on:click={() => copyText(cnlOutput, "cnl")}
                      >
                        {copiedCNL ? "✓" : "📋"}
                        {copiedCNL ? "Copied" : "Copy CNL"}
                      </button>
                    {/if}
                  {/if}
                </div>
              </div>
              <div class="card-body">
                {#if loadingNL2CNL || loadingDirect}
                  <p class="ph">Generating CNL…</p>
                {:else if cnlOutput}
                  {#if cnlEditMode}
                    <textarea bind:value={cnlOutput} class="cnl-textarea"
                    ></textarea>
                  {:else}
                    <div class="code-out cnl-color">{cnlOutput}</div>
                  {/if}
                {/if}
              </div>
            </div>
          {/if}

          <!-- ASP Output -->
          {#if aspOutput || loadingCNL2ASP || loadingDirect}
            <div class="card">
              <div class="card-header">
                <span class="card-title">
                  <span class="pnode asp">ASP</span> Answer Set Program
                  {#if aspStatus}
                    <span class="badge {aspStatus}"
                      >{aspStatus === "compiled"
                        ? "✓ Compiled"
                        : "✗ Error"}</span
                    >
                  {/if}
                </span>
                {#if aspOutput}
                  <button
                    class="btn-copy"
                    class:copied={copiedASP}
                    on:click={() => copyText(aspOutput, "asp")}
                  >
                    {copiedASP ? "✓" : "📋"}
                    {copiedASP ? "Copied" : "Copy"}
                  </button>
                {/if}
              </div>
              <div class="card-body">
                {#if loadingCNL2ASP || loadingDirect}
                  <p class="ph">Compiling ASP…</p>
                {:else if aspOutput}
                  <div class="code-out asp-color">{aspOutput}</div>
                {/if}
              </div>
            </div>
          {/if}

          <!-- ══ BATCH TAB ══ -->
        {:else if activeTab === "batch"}
          <div class="card">
            <div class="card-header">
              <span class="card-title">📁 Batch Processing</span>
              {#if batchFile}
                <button
                  class="btn btn-primary"
                  on:click={runBatch}
                  disabled={loadingBatch}
                >
                  {#if loadingBatch}<span class="spin"></span>{/if}
                  Run Batch
                </button>
              {/if}
            </div>
            <div class="card-body">
              <div class="drop">
                <input type="file" accept=".json" on:change={onFileChange} />
                <div class="drop-icon">📂</div>
                {#if batchFileName}
                  <div class="drop-lbl" style="color:#2563eb">
                    {batchFileName}
                  </div>
                  <div class="drop-sub">click to change</div>
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
              <div class="prog-header">
                <div class="prog-label">Processing Dataset</div>
                <div class="prog-pct">{batchProgress} / {batchTotal}</div>
              </div>
              <div class="prog-bg">
                <div
                  class="prog-fill"
                  style="width:{batchTotal
                    ? (batchProgress / batchTotal) * 100
                    : 0}%"
                ></div>
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
              <div class="stat" style="flex:0 0 auto">
                <button
                  class="btn btn-secondary"
                  on:click={downloadCSV}
                  style="margin-top:4px">⬇ CSV</button
                >
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
                      <td style="color:#94a3b8">{i + 1}</td>
                      <td title={r.nl}>{r.nl}</td>
                      <td title={r.predicted_cnl} style="color:#7c3aed"
                        >{r.predicted_cnl}</td
                      >
                      <td
                        ><span
                          class="badge {r.syntax_valid ? 'valid' : 'invalid'}"
                          >{r.syntax_valid ? "✓" : "✗"}</span
                        ></td
                      >
                      <td title={r.asp} style="color:#059669">{r.asp}</td>
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

          <!-- ══ DASHBOARD TAB ══ -->
        {:else if activeTab === "dashboard"}
          {#if dashErrorMsg}
            <div class="err-bar">⚠ {dashErrorMsg}</div>
          {/if}

          <!-- Model Selection -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">🔬 Compare Models</span>
              <span
                style="font-size:0.6rem; color:#94a3b8; font-family:'JetBrains Mono', monospace; margin-left:auto"
              >
                Full dataset mode
              </span>
            </div>
            <div class="card-body">
              <div style="margin-bottom:0.8rem">
                <div class="s-label" style="margin-bottom:0.5rem">
                  Select Models
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap">
                  {#each Object.entries(availableModels).filter(([, m]) => m.exists) as [key, m]}
                    <button
                      class="filter-btn"
                      class:active={dashModels[key]}
                      on:click={() => toggleDashModel(key)}
                    >
                      {m.icon}
                      {m.label}
                    </button>
                  {/each}
                </div>
              </div>
            </div>
          </div>

          <!-- Input Mode Selection -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">📝 Input Mode</span>
            </div>
            <div class="card-body">
              <div style="display:flex; gap:0.5rem; margin-bottom:0.8rem">
                <button
                  class="filter-btn"
                  class:active={dashInputMode === "single"}
                  on:click={() => (dashInputMode = "single")}
                >
                  Single Input
                </button>
                <button
                  class="filter-btn"
                  class:active={dashInputMode === "batch"}
                  on:click={() => (dashInputMode = "batch")}
                >
                  Batch File
                </button>
              </div>

              {#if dashInputMode === "single"}
                <textarea
                  bind:value={dashNLInput}
                  placeholder="Enter a single NL sentence…"
                  style="margin-bottom:0.5rem"
                ></textarea>
              {:else}
                <div class="drop" style="margin-bottom:0.5rem">
                  <input
                    type="file"
                    accept=".json"
                    on:change={onDashFileChange}
                  />
                  <div class="drop-icon">📂</div>
                  {#if dashFileName}
                    <div class="drop-lbl" style="color:#2563eb">
                      {dashFileName}
                    </div>
                    <div class="drop-sub">click to change</div>
                  {:else}
                    <div class="drop-lbl">
                      Drop JSON file or click to browse
                    </div>
                  {/if}
                </div>
                {#if dashRecords.length}
                  <div class="limit-control">
                    <label class="limit-label">
                      Limit to
                      <input
                        type="range"
                        bind:value={dashBatchLimit}
                        min="0"
                        max={dashRecords.length}
                        class="limit-slider"
                      />
                      <span class="limit-value">
                        {dashBatchLimit === 0 ? "all" : dashBatchLimit} / {dashRecords.length}
                      </span>
                    </label>
                  </div>
                {/if}
              {/if}

              <div style="display:flex; gap:0.5rem; margin-top:0.8rem">
                <button
                  class="btn btn-primary"
                  disabled={dashRunning ||
                    dashSelectedKeys.length === 0 ||
                    (dashInputMode === "single" && !dashNLInput.trim()) ||
                    (dashInputMode === "batch" && !dashRecords.length)}
                  on:click={runDashboard}
                >
                  {#if dashRunning}<span class="spin"></span>{/if}
                  Start Comparison
                </button>
                {#if dashRunning}
                  <button class="btn btn-danger" on:click={stopDashboard}
                    >Stop</button
                  >
                {:else if dashProgress > 0}
                  <button
                    class="btn btn-secondary"
                    on:click={resumeDashboard}
                    disabled={dashDone}>Resume</button
                  >
                  <button class="btn btn-secondary" on:click={resetDashboard}
                    >Reset</button
                  >
                {/if}
              </div>
            </div>
          </div>

          <!-- Progress -->
          {#if dashRunning || dashProgress > 0}
            <div class="prog-wrap">
              <div class="prog-header">
                <div class="prog-label">Comparing Models</div>
                <div class="prog-pct">{dashProgress} / {dashTotal}</div>
              </div>
              <div class="prog-bg">
                <div
                  class="prog-fill"
                  style="width:{dashTotal
                    ? (dashProgress / dashTotal) * 100
                    : 0}%"
                ></div>
              </div>
            </div>
          {/if}

          <!-- Results -->
          {#if dashResults.length}
            <!-- Stats -->
            <div class="stats">
              {#each dashStats as stat}
                <div class="stat">
                  <div class="sv b">{stat.syntaxPct}%</div>
                  <div class="sl">Syntax {stat.icon} {stat.label}</div>
                </div>
              {/each}
            </div>

            {#if dashSelectedKeys.length > 0}
              <div style="margin-bottom:1rem">
                <button class="btn btn-secondary" on:click={downloadDashCSV}
                  >⬇ Download CSV</button
                >
              </div>
            {/if}

            <!-- Search & Filter -->
            <div class="dash-toolbar">
              <input
                type="text"
                placeholder="Search…"
                class="dash-search"
                bind:value={dashSearch}
              />
              <button
                class="filter-btn"
                class:active={dashFilter === "all"}
                on:click={() => (dashFilter = "all")}>All</button
              >
            </div>

            <!-- Results Table -->
            <div class="tbl-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th><th>Natural Language</th>
                    {#each dashSelectedKeys as key}
                      <th
                        >{availableModels[key]?.icon ?? "🤖"}
                        {availableModels[key]?.label ?? key}</th
                      >
                    {/each}
                  </tr>
                </thead>
                <tbody>
                  {#each dashResults.filter((r) => !dashSearch || r.nl
                        .toLowerCase()
                        .includes(dashSearch.toLowerCase())) as r, i}
                    <tr>
                      <td style="color:#94a3b8">{i + 1}</td>
                      <td title={r.nl}>{r.nl}</td>
                      {#each dashSelectedKeys as key}
                        <td>
                          <div
                            style="display:flex; gap:0.3rem; align-items:center; flex-wrap:wrap; font-size:0.65rem"
                          >
                            <span
                              class="badge {r.models[key]?.syntax_valid
                                ? 'valid'
                                : 'invalid'}"
                              title="Syntax"
                              >{r.models[key]?.syntax_valid ? "✓" : "✗"}</span
                            >
                            <span
                              class="badge {r.models[key]?.compiled
                                ? 'compiled'
                                : 'error'}"
                              title="Compiled"
                              >{r.models[key]?.compiled ? "✓" : "✗"}</span
                            >
                          </div>
                        </td>
                      {/each}
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

  <!-- History Panel -->
  {#if historyOpen}
    <div class="history-panel">
      <div class="hp-header">
        <div class="hp-title">Recent Queries</div>
        <div style="display:flex; gap:0.3rem">
          {#if history.length}
            <button
              class="hp-close"
              title="Clear history"
              on:click={clearHistory}>🗑</button
            >
          {/if}
          <button class="hp-close" on:click={() => (historyOpen = false)}
            >✕</button
          >
        </div>
      </div>
      <div class="hp-list">
        {#if history.length}
          {#each history as entry}
            <div class="hp-item" on:click={() => loadFromHistory(entry)}>
              <div class="hp-nl" title={entry.nl}>{entry.nl}</div>
              <div class="hp-meta">
                <span class="hp-time">{entry.ts}</span>
                <span class="hp-model"
                  >{availableModels[entry.model]?.icon}
                  {availableModels[entry.model]?.label ?? entry.model}</span
                >
                <span
                  class="badge {entry.syntaxValid ? 'valid' : 'invalid'}"
                  style="margin-left:auto">{entry.syntaxValid ? "✓" : "✗"}</span
                >
              </div>
            </div>
          {/each}
        {:else}
          <div class="hp-empty">No history yet</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<!-- ═════════════════════════ STYLES ═════════════════════════ -->
<style>
  :global(*, *::before, *::after) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  :global(body) {
    font-family: "DM Sans", "Syne", system-ui, sans-serif;
    background: #f0f2f5;
    color: #1a1f2e;
    min-height: 100vh;
  }

  .app {
    display: grid;
    grid-template-rows: 54px 1fr;
    height: 100vh;
    overflow: hidden;
  }
  .main {
    display: grid;
    grid-template-columns: 268px 1fr;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    background: #ffffff;
    border-bottom: 1px solid #e8ecf0;
    display: flex;
    align-items: center;
    padding: 0 1.25rem;
    gap: 1rem;
    z-index: 20;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .brand-logo {
    width: 32px;
    height: 32px;
    border-radius: 7px;
    object-fit: contain;
    flex-shrink: 0;
  }
  .brand-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: #0f172a;
    letter-spacing: -0.02em;
  }
  .brand-sub {
    font-size: 0.58rem;
    color: #94a3b8;
    font-family: "JetBrains Mono", monospace;
  }

  .header-mid {
    display: flex;
    gap: 3px;
    background: #f0f2f5;
    padding: 3px;
    border-radius: 8px;
    margin-left: 1rem;
  }
  .tab-btn {
    padding: 0.28rem 0.85rem;
    border-radius: 6px;
    border: none;
    background: none;
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
    letter-spacing: 0.01em;
  }
  .tab-btn:hover {
    color: #1e293b;
  }
  .tab-btn.active {
    background: #ffffff;
    color: #1e293b;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .header-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .gpu-tag {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    font-weight: 500;
    color: #059669;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.65rem;
    color: #94a3b8;
    font-weight: 500;
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot.online {
    background: #22c55e;
    box-shadow: 0 0 0 2px #dcfce7;
  }
  .dot.offline {
    background: #ef4444;
    box-shadow: 0 0 0 2px #fee2e2;
  }
  .dot.unknown {
    background: #f59e0b;
    box-shadow: 0 0 0 2px #fef3c7;
  }

  /* ── Sidebar ── */
  aside {
    background: #ffffff;
    border-right: 1px solid #e8ecf0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-inner {
    flex: 1;
    overflow-y: auto;
    padding: 1.1rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 1.4rem;
  }
  .sidebar-inner::-webkit-scrollbar {
    width: 3px;
  }
  .sidebar-inner::-webkit-scrollbar-thumb {
    background: #e2e8f0;
    border-radius: 2px;
  }

  .s-label {
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin-bottom: 0.55rem;
  }
  .group-section {
    margin-bottom: 0.8rem;
  }
  .group-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.25rem 0.35rem;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 0.25rem;
  }
  .group-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .model-btn {
    width: 100%;
    padding: 0.45rem 0.65rem;
    border-radius: 7px;
    border: 1.5px solid #e8ecf0;
    background: #f8fafc;
    color: #64748b;
    font-family: "DM Sans", sans-serif;
    font-size: 0.74rem;
    text-align: left;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.12s;
    margin-bottom: 0.2rem;
    position: relative;
  }
  .model-btn:hover:not(.unavailable) {
    background: #f1f5f9;
    border-color: #cbd5e1;
  }
  .model-btn.active {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #2563eb;
    font-weight: 600;
  }
  .model-btn.unavailable {
    opacity: 0.3;
    cursor: not-allowed;
  }
  .m-icon {
    font-size: 0.9em;
    flex-shrink: 0;
  }
  .m-label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .m-tag {
    margin-left: auto;
    font-family: "JetBrains Mono", monospace;
    flex-shrink: 0;
    font-size: 0.56rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: #f1f5f9;
    color: #94a3b8;
  }
  .model-btn.active .m-tag {
    background: #dbeafe;
    color: #2563eb;
  }
  .loaded-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 4px #22c55e88;
    flex-shrink: 0;
  }
  .worker-badge {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.54rem;
    padding: 2px 6px;
    border-radius: 4px;
    flex-shrink: 0;
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }
  .unload-btn {
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid #e8ecf0;
    background: transparent;
    color: #ef4444;
    font-size: 0.58rem;
    cursor: pointer;
    flex-shrink: 0;
    display: none;
    font-family: "JetBrains Mono", monospace;
    transition: background 0.1s;
  }
  .model-btn:hover .unload-btn {
    display: inline-flex;
  }
  .unload-btn:hover {
    background: rgba(239, 68, 68, 0.08);
  }

  .sidebar-sep {
    height: 1px;
    background: #e8ecf0;
    margin: 0.6rem 0;
  }

  .mode-btn {
    width: 100%;
    padding: 0.55rem 0.3rem;
    border-radius: 9px;
    border: 1.5px solid #e8ecf0;
    background: #f8fafc;
    color: #94a3b8;
    font-family: inherit;
    font-size: 0.68rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.15s;
    line-height: 1.3;
  }
  .mode-btn:hover {
    border-color: #cbd5e1;
    color: #475569;
    background: #f1f5f9;
  }
  .mode-btn.active {
    background: #ffffff;
    border-color: #2563eb;
    color: #1e293b;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.12);
  }
  .mode-icon {
    font-size: 1rem;
    display: block;
    margin-bottom: 2px;
  }
  .mode-label {
    font-weight: 700;
    font-size: 0.68rem;
    display: block;
  }
  .mode-sub {
    font-size: 0.56rem;
    color: #94a3b8;
    display: block;
  }
  .mode-btn.active .mode-sub {
    color: #64748b;
  }

  .sidebar-footer {
    margin-top: auto;
    padding: 0.8rem;
    border-top: 1px solid #e8ecf0;
  }
  .server-info {
    font-size: 0.62rem;
    color: #94a3b8;
    line-height: 1.7;
    font-family: "JetBrains Mono", monospace;
  }
  .unload-all-btn {
    width: 100%;
    padding: 0.35rem;
    border-radius: 6px;
    border: 1.5px solid #fecaca;
    background: #fef2f2;
    color: #dc2626;
    font-size: 0.68rem;
    cursor: pointer;
    font-family: inherit;
    font-weight: 600;
    transition: background 0.12s;
    margin-top: 0.5rem;
  }
  .unload-all-btn:hover:not(:disabled) {
    background: #fee2e2;
  }
  .unload-all-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  /* ── Content ── */
  .content {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #f0f2f5;
  }
  .panel {
    flex: 1;
    overflow-y: auto;
    padding: 1.25rem 1.4rem;
  }
  .panel::-webkit-scrollbar {
    width: 5px;
  }
  .panel::-webkit-scrollbar-thumb {
    background: #e2e8f0;
    border-radius: 3px;
  }

  /* ── Pipeline bar ── */
  .pipe-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }
  .pnode {
    padding: 3px 11px;
    border-radius: 6px;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .pnode.nl {
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
  }
  .pnode.cnl {
    background: #f5f3ff;
    color: #7c3aed;
    border: 1px solid #ddd6fe;
  }
  .pnode.asp {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
  }
  .parr {
    color: #cbd5e1;
    font-size: 0.9rem;
  }
  .pipe-model {
    margin-left: auto;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.6rem;
    color: #94a3b8;
    font-weight: 500;
  }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border: 1.5px solid #e8ecf0;
    border-radius: 12px;
    margin-bottom: 0.9rem;
    overflow: hidden;
    transition:
      border-color 0.2s,
      box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
  }
  .card:focus-within {
    border-color: #93c5fd;
    box-shadow:
      0 0 0 3px rgba(37, 99, 235, 0.07),
      0 1px 4px rgba(0, 0, 0, 0.04);
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #f1f5f9;
    background: #fafbfc;
  }
  .card-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
  }
  .card-body {
    padding: 1rem;
  }

  textarea {
    width: 100%;
    background: none;
    border: none;
    outline: none;
    color: #1a1f2e;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    resize: vertical;
    min-height: 96px;
  }
  textarea::placeholder {
    color: #e2e8f0;
  }

  .code-out {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    white-space: pre-wrap;
    min-height: 48px;
  }
  .code-out.cnl-color {
    color: #5b21b6;
  }
  .code-out.asp-color {
    color: #065f46;
  }
  .ph {
    color: #d1d5db;
    font-style: italic;
    font-size: 0.75rem;
    font-family: "JetBrains Mono", monospace;
  }

  /* ── CNL Edit Mode ── */
  .cnl-textarea {
    width: 100%;
    background: none;
    border: none;
    outline: none;
    color: #5b21b6;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    resize: vertical;
    min-height: 120px;
  }

  /* ── Limit Slider ── */
  .limit-control {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.72rem;
    color: #64748b;
  }
  .limit-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    cursor: pointer;
  }
  .limit-slider {
    width: 140px;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(to right, #2563eb 0%, #7c3aed 100%);
    outline: none;
    -webkit-appearance: none;
    appearance: none;
  }
  .limit-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #2563eb;
    cursor: pointer;
    border: 2px solid #ffffff;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4);
    transition: all 0.2s;
  }
  .limit-slider::-webkit-slider-thumb:hover {
    background: #1d4ed8;
    box-shadow: 0 3px 8px rgba(37, 99, 235, 0.6);
    transform: scale(1.1);
  }
  .limit-slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #2563eb;
    cursor: pointer;
    border: 2px solid #ffffff;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4);
    transition: all 0.2s;
  }
  .limit-slider::-moz-range-thumb:hover {
    background: #1d4ed8;
    box-shadow: 0 3px 8px rgba(37, 99, 235, 0.6);
    transform: scale(1.1);
  }
  .limit-slider::-moz-range-track {
    background: none;
    border: none;
  }
  .limit-value {
    font-weight: 600;
    color: #2563eb;
    font-family: "JetBrains Mono", monospace;
    min-width: 50px;
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.85rem;
    border-radius: 7px;
    border: none;
    font-family: inherit;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .btn:disabled {
    opacity: 0.38;
    cursor: not-allowed;
  }
  .btn-primary {
    background: #2563eb;
    color: #fff;
    box-shadow: 0 1px 4px rgba(37, 99, 235, 0.25);
  }
  .btn-primary:hover:not(:disabled) {
    background: #1d4ed8;
  }
  .btn-secondary {
    background: #f8fafc;
    color: #64748b;
    border: 1.5px solid #e8ecf0;
  }
  .btn-secondary:hover:not(:disabled) {
    background: #f1f5f9;
    color: #475569;
    border-color: #cbd5e1;
  }
  .btn-success {
    background: #ecfdf5;
    color: #059669;
    border: 1.5px solid #a7f3d0;
    font-weight: 600;
  }
  .btn-success:hover:not(:disabled) {
    background: #d1fae5;
  }
  .btn-danger {
    background: #fef2f2;
    color: #dc2626;
    border: 1.5px solid #fecaca;
  }
  .btn-danger:hover:not(:disabled) {
    background: #fee2e2;
  }
  .btn-green {
    background: #ecfdf5;
    color: #059669;
    border: 1.5px solid #a7f3d0;
  }
  .btn-green:hover:not(:disabled) {
    background: #d1fae5;
  }
  .btn-ghost {
    background: #f8fafc;
    color: #64748b;
    border: 1.5px solid #e8ecf0;
  }
  .btn-ghost:hover:not(:disabled) {
    background: #f1f5f9;
    color: #475569;
    border-color: #cbd5e1;
  }
  .btn-copy {
    background: none;
    border: 1.5px solid transparent;
    color: #94a3b8;
    padding: 0.3rem 0.65rem;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 600;
    transition: all 0.12s;
  }
  .btn-copy:hover {
    background: #f1f5f9;
    border-color: #e2e8f0;
    color: #64748b;
  }
  .btn-copy.copied {
    color: #059669;
    border-color: #a7f3d0;
    background: #ecfdf5;
  }
  .btn-history {
    background: none;
    border: 1.5px solid #e8ecf0;
    color: #64748b;
    padding: 0.3rem 0.65rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.12s;
    position: relative;
    font-family: inherit;
  }
  .btn-history:hover {
    background: #f1f5f9;
  }
  .btn-history.has-items {
    border-color: #bfdbfe;
    color: #2563eb;
    background: #eff6ff;
  }
  .hist-count {
    position: absolute;
    top: -5px;
    right: -5px;
    background: #2563eb;
    color: #fff;
    font-size: 0.5rem;
    font-weight: 700;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .row {
    display: flex;
    gap: 0.35rem;
    align-items: center;
    flex-wrap: wrap;
  }

  /* ── Badges ── */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.02em;
  }
  .badge.valid {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
  }
  .badge.invalid {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
  }
  .badge.compiled {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
  }
  .badge.error {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
  }

  /* ── Error bar ── */
  .err-bar {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 1rem;
    color: #dc2626;
    font-size: 0.72rem;
    font-family: "JetBrains Mono", monospace;
  }
  .info-msg {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin-bottom: 1rem;
    color: #2563eb;
    font-size: 0.72rem;
    font-family: "JetBrains Mono", monospace;
  }

  /* ── Spinner ── */
  .spin {
    display: inline-block;
    width: 10px;
    height: 10px;
    border: 2px solid rgba(255, 255, 255, 0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.55s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* ── Progress ── */
  .prog-wrap {
    margin-bottom: 1rem;
  }
  .prog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
  }
  .prog-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #64748b;
  }
  .prog-pct {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.63rem;
    color: #94a3b8;
  }
  .prog-bg {
    height: 6px;
    background: #e8ecf0;
    border-radius: 3px;
    overflow: hidden;
  }
  .prog-fill {
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    transition: width 0.3s;
    border-radius: 3px;
  }
  .prog-lbl {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    margin-top: 3px;
  }

  /* ── Stats row ── */
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.65rem;
    margin-bottom: 1rem;
  }
  .stat {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }
  .sv {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: "JetBrains Mono", monospace;
    line-height: 1;
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
  .sv.r {
    color: #dc2626;
  }
  .sl {
    font-size: 0.57rem;
    color: #94a3b8;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
  }

  /* ── Drop zone ── */
  .drop {
    border: 2px dashed #e8ecf0;
    border-radius: 10px;
    padding: 2.2rem;
    text-align: center;
    cursor: pointer;
    position: relative;
    transition: all 0.15s;
    background: #fafbfc;
  }
  .drop:hover {
    border-color: #2563eb;
    background: #f0f7ff;
  }
  .drop input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .drop-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
  }
  .drop-lbl {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
  }
  .drop-sub {
    font-size: 0.62rem;
    color: #94a3b8;
    margin-top: 4px;
    font-family: "JetBrains Mono", monospace;
  }

  /* ── Table ── */
  .tbl-wrap {
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.72rem;
  }
  th {
    text-align: left;
    padding: 0.55rem 0.85rem;
    background: #fafbfc;
    color: #94a3b8;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.57rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    border-bottom: 1px solid #e8ecf0;
    white-space: nowrap;
  }
  td {
    padding: 0.5rem 0.85rem;
    border-bottom: 1px solid #f8fafc;
    font-family: "JetBrains Mono", monospace;
    color: #64748b;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: middle;
  }
  tr:last-child td {
    border-bottom: none;
  }
  tr:hover td {
    background: #f8fafc;
  }

  /* ── History panel ── */
  .history-panel {
    position: fixed;
    right: 1rem;
    top: 62px;
    width: 340px;
    z-index: 200;
    background: #ffffff;
    border: 1.5px solid #e8ecf0;
    border-radius: 12px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.12);
    overflow: hidden;
  }
  .hp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #f1f5f9;
    background: #fafbfc;
  }
  .hp-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #1e293b;
  }
  .hp-close {
    background: none;
    border: none;
    color: #94a3b8;
    cursor: pointer;
    font-size: 1rem;
    padding: 0;
  }
  .hp-close:hover {
    color: #1e293b;
  }
  .hp-list {
    max-height: 400px;
    overflow-y: auto;
  }
  .hp-list::-webkit-scrollbar {
    width: 3px;
  }
  .hp-list::-webkit-scrollbar-thumb {
    background: #e2e8f0;
  }
  .hp-item {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #f8fafc;
    cursor: pointer;
    transition: background 0.08s;
  }
  .hp-item:hover {
    background: #f8fafc;
  }
  .hp-item:last-child {
    border-bottom: none;
  }
  .hp-nl {
    font-size: 0.75rem;
    color: #1e293b;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 3px;
  }
  .hp-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .hp-time {
    font-size: 0.6rem;
    color: #94a3b8;
    font-family: "JetBrains Mono", monospace;
  }
  .hp-model {
    font-size: 0.58rem;
    color: #94a3b8;
    background: #f1f5f9;
    padding: 1px 6px;
    border-radius: 3px;
  }
  .hp-empty {
    padding: 1.5rem;
    text-align: center;
    font-size: 0.72rem;
    color: #94a3b8;
  }

  /* ── Dashboard toolbar ── */
  .dash-toolbar {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }
  .filter-btn {
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
    border: 1.5px solid #e8ecf0;
    background: #fff;
    color: #94a3b8;
    font-size: 0.68rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.12s;
    font-family: inherit;
  }
  .filter-btn:hover {
    border-color: #cbd5e1;
    color: #64748b;
  }
  .filter-btn.active {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #2563eb;
  }
  .dash-search {
    flex: 1;
    min-width: 150px;
    background: #fff;
    border: 1.5px solid #e8ecf0;
    border-radius: 7px;
    padding: 0.3rem 0.7rem 0.3rem 2rem;
    color: #1e293b;
    font-size: 0.72rem;
    outline: none;
    font-family: inherit;
    transition: border-color 0.12s;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: 0.55rem center;
  }
  .dash-search:focus {
    border-color: #2563eb;
  }
  .dash-search::placeholder {
    color: #cbd5e1;
  }
</style>
