"""
NL2ASP — Qwen3.5-9B WORKER SERVER
════════════════════════════════════════════════════════════════════════════════
Run this under the  (llm_peft_qwen35)  conda environment [transformers 5.3.x].

    conda activate llm_peft_qwen35
    python server_qwen35.py

Listens on port 8001 (localhost only — NOT exposed to the browser directly).
The gateway server (server.py, port 8000) proxies Qwen3.5 requests here.

════════════════════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS
────────────────────────────────────────────────────────────────────────────────
Qwen3.5-9B requires apply_chat_template(..., enable_thinking=False), which is
only available in transformers >= 5.0.  This worker runs in the dedicated
conda environment that has that version.  All other models run under the
(nl2asp) env via server.py.
════════════════════════════════════════════════════════════════════════════════
"""

import gc
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

import requests as _http
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Environment guard ─────────────────────────────────────────────────────────
import transformers as _tf
_TF_MAJOR = int(_tf.__version__.split(".")[0])
if _TF_MAJOR < 5:
    import sys
    sys.exit(
        f"\nERROR: server_qwen35.py requires transformers >= 5.0.\n"
        f"       Detected: transformers {_tf.__version__}\n"
        f"       Activate the correct environment:\n"
        f"           conda activate llm_peft_qwen35\n"
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    log.info("GPU: %s", gpu_name)
else:
    DEVICE   = "cpu"
    gpu_name = "none"
    log.warning("No GPU — falling back to CPU.")

# ── External CNL2ASP API ──────────────────────────────────────────────────────
CNL2ASP_BASE    = "http://160.97.63.29:3003/api"
CNL2ASP_HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY":    "BwfZYaNwDUIHUQapxl0TB2B1veZ6AfW8",
}

SYSTEM_PROMPT = (
    "You are an expert in Translating the Natural language (NL) into Controlled Natural "
    "Language (CNL) translation. Always provide precise, syntactically correct translations "
    "of NL into CNL."
)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY  (Qwen3.5-9B variants — absolute paths)
# ════════════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY: dict = {
    "qwen3_5_9b_r8": {
        "label": "Qwen3.5-9B r=8",   "tag": "9B r8",  "icon": "🌟",
        "kind":  "causal_qwen3_5",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3.5-9B_r8",
        "group": "llm",
    },
    "qwen3_5_9b_r16": {
        "label": "Qwen3.5-9B r=16",  "tag": "9B r16", "icon": "🌟",
        "kind":  "causal_qwen3_5",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3.5-9B_r16",
        "group": "llm",
    },
    "qwen3_5_9b_r32": {
        "label": "Qwen3.5-9B r=32",  "tag": "9B r32", "icon": "🌟",
        "kind":  "causal_qwen3_5",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3.5-9B_r32",
        "group": "llm",
    },
}

_loaded: dict = {}
_load_lock = threading.Lock()


# ════════════════════════════════════════════════════════════════════════════════
# Think-token stripper  (belt-and-suspenders even with enable_thinking=False)
# ════════════════════════════════════════════════════════════════════════════════
def _strip_think(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    return cleaned.strip()


# ════════════════════════════════════════════════════════════════════════════════
# Model loader
# ════════════════════════════════════════════════════════════════════════════════
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

def _resolve_dtype(dtype_str: str):
    return _DTYPE_MAP.get(dtype_str, torch.float32)


def _model_input_device(model) -> torch.device:
    """Detect actual device for input_ids after accelerate device_map="auto"."""
    try:
        device_map = getattr(model, "hf_device_map", None)
        if device_map is None:
            device_map = getattr(model.base_model, "hf_device_map", None)
        if device_map:
            for key in ("model.embed_tokens", "transformer.wte"):
                if key in device_map:
                    dev = device_map[key]
                    return torch.device(dev if dev != "cpu" else "cpu")
            for dev in device_map.values():
                if dev != "cpu":
                    return torch.device(dev)
            return torch.device("cpu")
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def _load_qwen35(path: str, dtype) -> dict:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    log.info("Loading Qwen3.5 adapter: %s", path)
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            path, device_map="auto", dtype=dtype, trust_remote_code=True,
        )
    except TypeError:
        log.info("  dtype= not accepted, retrying with torch_dtype=")
        model = AutoPeftModelForCausalLM.from_pretrained(
            path, device_map="auto", torch_dtype=dtype, trust_remote_code=True,
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    input_device = _model_input_device(model)
    log.info("✓ Qwen3.5-9B loaded. embed_tokens device: %s", input_device)
    return {"model": model, "tokenizer": tok, "input_device": input_device}


# ════════════════════════════════════════════════════════════════════════════════
# Lazy loader
# ════════════════════════════════════════════════════════════════════════════════
def _ensure_loaded(model_key: str) -> dict:
    if model_key in _loaded:
        return _loaded[model_key]
    with _load_lock:
        if model_key in _loaded:
            return _loaded[model_key]
        if model_key not in MODEL_REGISTRY:
            raise HTTPException(400, f"Unknown model: '{model_key}'")
        meta = MODEL_REGISTRY[model_key]
        if not Path(meta["path"]).exists():
            raise HTTPException(503, f"Weights not found at: {meta['path']}\n"
                                     f"Update MODEL_REGISTRY in server_qwen35.py.")
        try:
            entry = _load_qwen35(meta["path"], _resolve_dtype(meta["dtype"]))
            entry["kind"] = meta["kind"]
            _loaded[model_key] = entry
            return entry
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Failed to load '{model_key}': {exc}") from exc


# ════════════════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════════════════
def _infer_qwen35(entry: dict, nl: str, max_new_tokens: int) -> str:
    """
    Qwen3.5-9B inference.
    enable_thinking=False is REQUIRED and only works in transformers >= 5.0.
    Without it the model emits <think>…</think> blocks that corrupt CNL output.
    """
    model, tok = entry["model"], entry["tokenizer"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Translate the following natural language to controlled natural language: {nl}"},
    ]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # ← requires transformers >= 5.0
    )
    input_device = entry["input_device"]
    inputs = tok(prompt, return_tensors="pt").to(input_device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    raw = tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return _strip_think(raw)   # safety net


# ════════════════════════════════════════════════════════════════════════════════
# CNL → ASP helpers  (duplicated from server.py — worker is self-contained)
# ════════════════════════════════════════════════════════════════════════════════
def _check_syntax(cnl: str) -> bool:
    try:
        r = _http.post(f"{CNL2ASP_BASE}/check_syntax", headers=CNL2ASP_HEADERS,
                       json={"cnls": cnl}, timeout=10)
        return r.json().get("cli_message") == "Input file fits the grammar."
    except Exception:
        return False


def _compile_cnl(cnl: str) -> Tuple[str, bool]:
    try:
        r = _http.post(f"{CNL2ASP_BASE}/compile", headers=CNL2ASP_HEADERS,
                       json={"cnls": cnl}, timeout=15)
        asp = r.json().get("asp", "").strip()
        return (asp, True) if asp else ("", False)
    except Exception:
        return ("", False)


# ════════════════════════════════════════════════════════════════════════════════
# FastAPI  —  binds to 127.0.0.1:8001 (localhost only, not exposed to browser)
# ════════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Qwen3.5 worker — transformers %s  (llm_peft_qwen35 env)", _tf.__version__)
    log.info("Listening on 127.0.0.1:8005 — gateway server will proxy requests here.")
    yield
    _loaded.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Qwen3.5 worker stopped.")


app = FastAPI(title="NL2ASP Qwen3.5 Worker", lifespan=lifespan)
# Allow only the gateway (localhost).  Change if you run processes on different hosts.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"],
                   allow_methods=["*"], allow_headers=["*"])


class NL2ASPRequest(BaseModel):
    nl: str;  model: str = "qwen3_5_9b";  max_new_tokens: int = 256

class UnloadRequest(BaseModel):
    model: Optional[str] = None


@app.get("/api/health")
def health():
    available = {}
    for key, meta in MODEL_REGISTRY.items():
        available[key] = {
            "label":  meta["label"], "tag": meta["tag"], "icon": meta["icon"],
            "kind":   meta["kind"],  "group": meta["group"],
            "exists": Path(meta["path"]).exists(),
            "loaded": key in _loaded,
        }
    return {
        "status":           "ok",
        "device":           DEVICE,
        "gpu":              gpu_name,
        "tf_version":       _tf.__version__,
        "loaded_models":    list(_loaded.keys()),
        "available_models": available,
    }


@app.post("/api/nl2asp")
def nl2asp(req: NL2ASPRequest):
    entry = _ensure_loaded(req.model)
    cnl           = _infer_qwen35(entry, req.nl, req.max_new_tokens)  # entry carries input_device
    valid         = _check_syntax(cnl)
    asp, compiled = _compile_cnl(cnl)
    return {"cnl": cnl, "syntax_valid": valid, "asp": asp, "compiled": compiled}


@app.post("/api/unload")
def unload(req: UnloadRequest):
    with _load_lock:
        keys = [req.model] if req.model else list(_loaded.keys())
        unloaded = []
        for k in keys:
            if k in _loaded:
                e = _loaded.pop(k)
                e.pop("model", None); e.pop("tokenizer", None); e.pop("input_device", None)
                unloaded.append(k)
            else:
                unloaded.append(k)   # already unloaded — still report success
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"unloaded": unloaded}


if __name__ == "__main__":
    import uvicorn
    # Bind to 127.0.0.1 only — this worker should never be directly browser-accessible
    uvicorn.run("server_qwen35:app", host="127.0.0.1", port=8005, reload=False)
