# NL2ASP Pipeline — Setup & Run Guide

Web interface for the NL → CNL → ASP pipeline using fine-tuned LLaMA 3.1 8B,
T5-Small, and BART-Base models. Runs on AMD ROCm GPU server with a Svelte frontend.

---

## Project Structure

```
nl2asp_app/
├── server.py              ← FastAPI backend (model inference)
├── src/
│   ├── main.js            ← Svelte entry point
│   └── App.svelte         ← Svelte UI
├── index.html
├── vite.config.js
├── package.json
├── requirements.txt       ← Python dependencies with exact versions
├── llamafinetune/         ← LLaMA LoRA adapter weights
├── t5small_weights/       ← T5-Small weights (optional)
└── bartbase_weights/      ← BART-Base weights (optional)
```

---

## System Requirements

- AMD ROCm GPU (tested on AMD Instinct MI210 with ROCm 7.0)
- Conda (Miniconda or Anaconda)
- Node.js 20+ (installed via conda)

---

## Step 1 — Create Conda Environment

```bash
conda create -n nl2asp python=3.10 -y
conda activate nl2asp
```

---

## Step 2 — Install PyTorch for ROCm 7.0

PyTorch for ROCm must be installed from the nightly index separately:

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

```bash
pip install -r requirements.txt
```

Verify all key packages:

```bash
python -c "import torch, transformers, peft, fastapi; print(torch.__version__, transformers.__version__, peft.__version__)"
# Expected output:
# 2.11.0.dev20260206+rocm7.0  4.57.3  0.18.0
```

---

## Step 4 — Install Node.js and Frontend Dependencies

```bash
# Install Node.js via conda if not already installed
conda install -c conda-forge nodejs -y

# Install Svelte/Vite dependencies
cd nl2asp_app
npm install
```

---

## Step 5 — Configure Weight Paths

Open server.py and update the paths to your model weights:

```python
LLAMA_ADAPTER_PATH = "./llamafinetune"      # LoRA adapter checkpoint
T5_SMALL_PATH      = "./t5small_weights"    # T5-Small (optional)
BART_BASE_PATH     = "./bartbase_weights"   # BART-Base (optional)
```

The server will skip any model whose folder does not exist — no crash.

---

## Step 6 — Run the Application

You need two terminals both inside the project folder with the conda env active.

### Terminal 1 — Backend

```bash
conda activate nl2asp
cd nl2asp_app
python server.py
```

Expected output:
```
INFO: GPU detected: AMD Instinct MI210
INFO: Device: cuda  |  dtype: torch.float16
INFO: Loading LLaMA adapter from ./llamafinetune ...
INFO: ✓  llama loaded
INFO: Ready — loaded: ['llama']
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 — Frontend

```bash
conda activate nl2asp
cd nl2asp_app
npm run dev
```

Expected output:
```
VITE v5.4.21  ready in 450 ms
➜  Local:   http://localhost:5173/
➜  Network: http://160.97.63.21:5173/
```

---

## Step 7 — Open in Browser

If browser is on the same server:
```
http://localhost:5173
```

If browser is on your laptop or another machine:
```
http://SERVER_IP:5173
```

Make sure ports are open:
```bash
sudo ufw allow 5173
sudo ufw allow 8000
```

---

## API Endpoints

The backend exposes these REST endpoints at http://localhost:8000:

| Method | Endpoint      | Description                    |
|--------|---------------|--------------------------------|
| GET    | /api/health   | Check server and loaded models |
| POST   | /api/nl2cnl   | Translate NL to CNL            |
| POST   | /api/cnl2asp  | Compile CNL to ASP             |
| POST   | /api/nl2asp   | Full pipeline NL to ASP        |

Example request:
```bash
curl -X POST http://localhost:8000/api/nl2asp \
  -H "Content-Type: application/json" \
  -d '{"nl": "Every node must be reachable from the source.", "model": "llama"}'
```

---

## UI Features

### Single Input Mode
- Direct (NL to ASP): paste natural language, click NL to ASP
- Step-by-step: translate NL to CNL first, inspect CNL output, then click Compile to ASP

### Batch Mode
- Upload a JSON dataset file in format: { "data_dict": [{"NL_V2": "...", "CNL_V2": "...", "ASP": "..."}] }
- Runs full pipeline on every record with live progress bar
- Shows syntax accuracy and compile rate statistics
- Download all results as CSV

### Model Selector
Switch between loaded models in the sidebar:
- LLaMA-3.1 8B (Fine-tuned LoRA)
- T5-Small
- BART-Base

Models not found on disk are automatically grayed out.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No module named torch | Reinstall torch with ROCm nightly index URL |
| torch.cuda.is_available() is False | ROCm version mismatch, check rocm-smi --version |
| modeling_layers not found | pip install transformers==4.57.3 |
| alora_invocation_tokens error | pip install peft==0.18.0 |
| operator torchvision nms error | pip uninstall torchvision torchaudio -y |
| package.json not found | Make sure you are inside nl2asp_app/ before running npm commands |

---

## Exact Package Versions (tested and working)

| Package      | Version                        |
|--------------|--------------------------------|
| torch        | 2.11.0.dev20260206+rocm7.0     |
| transformers | 4.57.3                         |
| peft         | 0.18.0                         |
| fastapi      | latest                         |
| uvicorn      | latest                         |
| node.js      | 20.x                           |
| vite         | 5.4.x                          |
| svelte       | 4.2.x                          |
