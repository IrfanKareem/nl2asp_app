# NL2ASP Pipeline UI — ROCm / AMD GPU Server

Run the full NL → CNL → ASP pipeline on your ROCm GPU server with a Svelte web UI.

---

## Project Structure

```
nl2asp-rocm/
├── server.py          ← FastAPI backend (loads models, GPU inference)
├── src/
│   ├── main.js
│   └── App.svelte     ← Svelte frontend
├── index.html
├── vite.config.js
├── package.json
└── README.md
```

---

## Step 1 — Fix weight paths in server.py

Open `server.py` and update the three paths to where your weights live:

```python
LLAMA_ADAPTER_PATH = "./tuning_results/checkpoint-13638"  # your best checkpoint
T5_SMALL_PATH      = "./t5small_weights"
BART_BASE_PATH     = "./bartbase_weights"
```

---

## Step 2 — Install Python dependencies

ROCm PyTorch must be installed separately (not from default pip index):

```bash
# Install ROCm version of PyTorch (adjust rocm version to match yours)
# Check your ROCm version first: rocminfo | grep "ROCm"

# For ROCm 6.0:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# For ROCm 5.7:
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# Then install the rest normally
pip install fastapi uvicorn transformers peft accelerate requests
```

---

## Step 3 — Install Node.js (for Svelte frontend)

```bash
# Check if Node is already installed
node --version

# If not installed:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install frontend dependencies
npm install
```

---

## Step 4 — Run everything

Open **two terminals** on the GPU server:

**Terminal 1 — Backend:**
```bash
python server.py
# Loads models and starts API at http://0.0.0.0:8000
```

**Terminal 2 — Frontend:**
```bash
npm run dev
# Starts Svelte UI at http://0.0.0.0:5173
```

---

## Step 5 — Open in browser

**If browser is on the same server:**
```
http://localhost:5173
```

**If browser is on your laptop/WSL and server has an IP (e.g. 192.168.1.50):**
```
http://192.168.1.50:5173
```

Make sure ports 5173 and 8000 are open in your firewall:
```bash
sudo ufw allow 5173
sudo ufw allow 8000
```

---

## Features

### Single Input
- **Direct mode:** paste NL, click ⚡ NL → ASP — get CNL and ASP in one shot
- **Step-by-step mode:** translate NL → CNL, inspect result, then compile → ASP separately

### Batch Mode
- Upload your JSON test file (`{ data_dict: [{NL_V2, CNL_V2, ASP}] }`)
- Runs full pipeline on every record with live progress bar
- Shows syntax accuracy and compile rate
- Download results as CSV

### Model Selector (sidebar)
- LLaMA-3.1 8B fine-tuned (LoRA adapter)
- T5-Small
- BART-Base
- Grayed out automatically if weights not found

### GPU indicator
The GPU name (e.g. AMD Radeon RX 6900 XT) is shown in the header when detected.

---

## Verify GPU is working

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

ROCm exposes itself as CUDA to PyTorch, so `torch.cuda.is_available()` should return `True`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` returns False | Reinstall PyTorch with correct ROCm index URL |
| Model loads but inference crashes | Try setting `PYTORCH_HIP_ALLREDUCE_TIMEOUT=1800` env var |
| Frontend can't reach backend | Check CORS in server.py, check firewall ports |
| LLaMA loads very slowly | Normal — 8B model takes 1-3 min to load into VRAM |
| `peft` not found | `pip install peft` |
