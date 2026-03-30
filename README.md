# NL2ASP Pipeline — Setup & Run Guide

Web interface for the NL → CNL → ASP pipeline using fine-tuned LLaMA 3.1 8B, Qwen 3/3.5,
Ministral-8B, T5-Small, T5-Large, and T5-3B models. Runs on AMD ROCm GPU server with a
modern light-themed Svelte frontend, multi-model comparison dashboard, and query history.

---

## ✨ What's New (Latest Update)

- ✅ **T5 Models Now Live**: T5-Small (60M), T5-Large (770M), and T5-3B fully integrated and working with complete training datasets
- ✅ **Improved Dashboard UI**: Larger limit slider, copy buttons for NL/CNL, editable CNL output in both Direct and Step-by-Step modes
- ✅ **Dual-Server Architecture**: Separate Qwen3.5 worker for transformers 5.x compatibility
- ✅ **Modern UI**: Light theme redesign with improved UX
- ✅ **Dashboard Feature**: Multi-model comparison and batch analysis
- ✅ **Query History**: Browser-based sessionStorage for recent inputs
- ✅ **Model Controls**: Load All / Unload All buttons + Cancel button for long operations- ✅ **K-fold Preparation**: Commented out for future per-model re-enablement

---

## Project Structure

```
nl2asp_app/
├── server.py                           ← FastAPI gateway (processes all models except Qwen3.5-9B)
├── server_qwen35.py                    ← Worker server (handles Qwen3.5-9B inference)
├── src/
│   ├── main.js                         ← Svelte entry point
│   └── App.svelte                      ← Modern light-theme UI with dashboard
├── index.html                          ← Frontend entry point
├── vite.config.js
├── package.json
├── requirements.txt                    ← Python dependencies
├── public/
│   └── logo.png                        ← Application logo
├── t5-3b_Complete_Dataset/             ← T5-3B full dataset (newly trained)
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
├── t5large_Complete_Dataset/           ← T5-Large full dataset (newly trained)
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
├── t5small_Complete_Dataset/           ← T5-Small full dataset (newly trained)
│   ├── checkpoint-38000/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── tokenizer.json
│   └── ...
├── [LLM adapters]/                     ← Qwen, LLaMA, Ministral LoRA weights
├── backups/                            ← Source file backups
└── old_models/                         ← Legacy model storage
```

---

## System Requirements

- AMD ROCm GPU (tested on AMD Instinct MI210 with ROCm 7.0)
- Dual Conda environments (nl2asp + llm_peft_qwen35)
- Node.js 20+ (installed via conda)
- ~50GB VRAM for loading largest models simultaneously

---

## Step 1 — Create Conda Environments

### Main Environment (for T5 and other models):

```bash
conda create -n nl2asp python=3.10 -y
conda activate nl2asp
```

### Qwen3.5 Environment (transformers >= 5.0 required):

```bash
conda create -n llm_peft_qwen35 python=3.10 -y
conda activate llm_peft_qwen35
```

---

## Step 2 — Install PyTorch for ROCm 7.0

For **both** environments, install PyTorch from the nightly ROCm index:

```bash
pip install torch==2.11.0.dev20260206+rocm7.0 \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected output:
# True
# AMD Instinct MI210
```

---

## Step 3 — Install Python Dependencies

### For nl2asp environment:

```bash
conda activate nl2asp
pip install -r requirements.txt
```

### For llm_peft_qwen35 environment:

```bash
conda activate llm_peft_qwen35
pip install transformers==5.1.0 peft torch-rocm bitsandbytes pydantic fastapi uvicorn requests
```

---

## Step 4 — Install Node.js and Frontend Dependencies

```bash
# Install Node.js via conda if not already installed
conda install -c conda-forge nodejs=20 -y

# Install Svelte/Vite dependencies
cd nl2asp_app
npm install
```

---

## Step 5 — Run the Application

### **CRITICAL: Three Separate Terminals (all with project folder context)**

#### **Terminal 1 — Qwen3.5 Worker Server** (if using Qwen3.5-9B)

```bash
cd /home/manuel/nl2asp_app
conda activate llm_peft_qwen35
python server_qwen35.py
```

Expected output:
```
INFO: Qwen3.5-9B worker server starting...
INFO: Loaded models: qwen3_5_9b_r8, qwen3_5_9b_r16, qwen3_5_9b_r32
INFO: Uvicorn running on http://0.0.0.0:8005
```

> **Note:** The Qwen3.5 worker must be online before any Qwen3.5-9B models can be used.
> This runs on port 8005 (separate from main gateway).

---

#### **Terminal 2 — Main API Gateway Server**

```bash
cd /home/manuel/nl2asp_app
conda activate nl2asp
python server.py
```

Expected output:
```
INFO: Gateway server — transformers 4.57.3  (nl2asp env)
INFO: Qwen3.5 worker expected at http://localhost:8005
INFO: ✓ Qwen3.5 worker is reachable.
INFO: Uvicorn running on http://0.0.0.0:8000
```

> **Note:** This serves ALL models (Qwen, LLaMA, Ministral, T5) except Qwen3.5-9B,
> which is transparently proxied to the worker on port 8005.

**What the gateway does:**
- Loads and manages Qwen-3, LLaMA-3.1, Ministral-8B models
- Loads T5-Small, T5-Large, T5-3B from `old_models/`
- Manages VRAM (exclusive LLM group ensures only one Causal LLM in VRAM at a time)
- Proxies Qwen3.5-9B requests to the worker
- Exposes `/api/health`, `/api/nl2cnl`, `/api/nl2asp`, etc.

---

#### **Terminal 3 — Frontend Dev Server**

```bash
cd /home/manuel/nl2asp_app
conda activate nl2asp  # Same environment as gateway is fine
npm run dev
```

Expected output:
```
VITE v5.4.21  ready in 450 ms
➜  Local:   http://localhost:5173/
➜  Network: http://160.97.63.21:5173/
```

---

## Step 6 — Open in Browser

**Local access:**
```
http://localhost:5173
```

**Remote access (from another machine):**
```
http://SERVER_IP:5173
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Vite @ :5173)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Svelte UI (Light Theme)                      │   │
│  │  • Single/Batch/Dashboard tabs                       │   │
│  │  • Model selector (load/unload)                      │   │
│  │  • Query history panel                               │   │
│  │  • Real-time progress indicators                     │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (JSON)
                            ↓
        ┌───────────────────────────────────────┐
        │   Gateway Server (FastAPI @ :8000)    │
        │  ┌─────────────────────────────────┐  │
        │  │  Qwen-3 (LoRA, causal)          │  │
        │  │  LLaMA-3.1 (LoRA, causal)       │  │
        │  │  Ministral-8B (LoRA, causal)    │  │
        │  │  T5-Small/Large/3B (seq2seq)    │  │
        │  │  ↓                              │  │
        │  │  Exclusive LLM group (1 in RAM)│  │
        │  └─────────────────────────────────┘  │
        │         Proxy Layer                    │
        └────────────────┬─────────────────────┘
                         │
                         ├─→ Internal models (CNL/ASP compilation)
                         │
                         └─→ Qwen3.5-9B requests
                              │
                              ↓
        ┌───────────────────────────────────────┐
        │  Qwen3.5 Worker (FastAPI @ :8005)     │
        │  ┌─────────────────────────────────┐  │
        │  │  Qwen-3.5-9B (LoRA × 3 ranks)   │  │
        │  │  Exclusive group (1 in RAM)      │  │
        │  └─────────────────────────────────┘  │
        └───────────────────────────────────────┘
```

---

## Available Models

### Causal LLMs (One at a time in VRAM due to exclusive group)
- **Qwen-3 8B** (3 LoRA ranks: r=8, r=16, r=32)
- **Qwen-3.5-9B** (3 ranks, worker only, transformers 5.x required)
- **LLaMA-3.1 8B** (3 LoRA ranks)
- **Ministral-8B** (3 LoRA ranks)

### Seq2Seq Models (T5 family, can load independently)
- **T5-Small** (60M, float32) - complete dataset ✅ Fully working
- **T5-Large** (770M, float16) - complete dataset ✅ Fully working  
- **T5-3B** (3B, float16) - complete dataset ✅ Fully working

> **Status**: All T5 models are now fully integrated with trained weights from complete datasets.
> Located at `/home/manuel/nl2asp_app/t5-{small,large,3b}_Complete_Dataset/`
> K-fold ensemble variants are commented out and will be re-enabled per-model in future updates.

---

## API Endpoints

All endpoints available at `http://localhost:8000/api/`:

| Method | Endpoint      | Input                           | Output                    |
|--------|---------------|---------------------------------|---------------------------|
| GET    | `/health`     | —                               | Models, loaded, GPU info  |
| POST   | `/nl2cnl`     | `{"nl": "...", "model": "..."}` | `{"cnl": "...", "syntax_valid": bool}` |
| POST   | `/cnl2asp`    | `{"cnl": "..."}`                | `{"asp": "...", "compiled": bool}` |
| POST   | `/nl2asp`     | `{"nl": "...", "model": "..."}` | Full pipeline result      |
| POST   | `/unload`     | `{"model": "..."}` or `null`    | `{"unloaded": ["..."]}`   |

Example:
```bash
curl -X POST http://localhost:8000/api/nl2asp \
  -H "Content-Type: application/json" \
  -d '{"nl": "Every node must be reachable.", "model": "qwen3_8b_r8"}'
```

---

## UI Features

### **Single Mode**
- Type natural language, select model
- Choose pipeline: Step-by-step (NL→CNL→ASP) or Direct (NL→ASP)
- View CNL syntax validation and ASP compilation results
- **Copy NL button** for quick clipboard access
- **Edit CNL** - toggle edit mode to manually refine CNL output before compilation
- **Copy CNL** - copy refined CNL to clipboard after editing

### **Batch Mode**
- Upload JSON: `{"data_dict": [{"NL_V2": "...", "CNL_V2": "...", "ASP": "..."}]}`
- **Improved limit slider** - easily set batch size limit with visual feedback (`current / total` display)
- Processes all records with live progress bar
- Shows syntax accuracy % and compilation rate
- Download results as CSV

### **Dashboard (Multi-Model Comparison)**
- Select multiple models simultaneously
- Run same input through all models
- Compare outputs side-by-side
- Batch processing with comparison stats
- Download comparison results as CSV

### **History Panel**
- Automatically saves recent queries
- Quick access to previous inputs
- Filter by model/status
- Uses browser sessionStorage (cleared on browser close)

---

## Firewall Configuration

```bash
sudo ufw allow 5173   # Frontend (Vite)
sudo ufw allow 8000   # Gateway API
sudo ufw allow 8005   # Qwen3.5 worker (optional if running locally)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Ensure correct environment is active. Reinstall from nightly ROCm index. |
| `torch.cuda.is_available() = False` | Check `rocm-smi --version`. May need ROCm update. |
| Qwen3.5 models show "worker offline" | Terminal 1 (server_qwen35.py) not running. Start it in llm_peft_qwen35 environment. |
| T5 models fail to load | Verify tokenizer_config.json has been fixed (extra_special_tokens field removed). |
| `No module named 'transformers'` in worker (llm_peft_qwen35) | Install with `pip install transformers==5.1.0` |
| Port 8000/5173 already in use | Kill background process: `sudo lsof -i :8000` then `kill -9 <PID>` |
| `package.json not found` | Run npm commands inside `/home/manuel/nl2asp_app` directory |

---

## Performance Tips

1. **GPU Memory Management**
   - Load only needed models with sidebar buttons
   - Use "Load All" only if GPU has >40GB VRAM
   - Unload models when switching to different LoRA rank

2. **Query Optimization**
   - Shorter inputs (< 256 tokens) run faster
   - Batch mode can process 100+ records in parallel
   - Dashboard limits per-batch to prevent timeouts

3. **For Slow Operations**
   - Click "Cancel" button if model loading takes too long
   - Switch to smaller models (T5-Small instead of T5-3B)
   - Consider pre-loading on app startup

---

## Next Steps

- Run `python server.py --help` for additional gateway options
- Check `server_qwen35.py` for Qwen3.5 worker configuration
- Review `/api/health` response to verify all models loaded correctly
- Test dashboard feature with multiple models for comparison


