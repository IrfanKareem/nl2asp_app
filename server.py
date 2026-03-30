"""
NL2ASP Backend — GATEWAY SERVER
════════════════════════════════════════════════════════════════════════════════
Run this under the  (nl2asp)  conda environment  [transformers 4.57.3].

    conda activate nl2asp
    python server.py

This process handles ALL models except Qwen3.5-9B.
Requests for Qwen3.5-9B are transparently proxied to server_qwen35.py,
which must be running separately under the (llm_peft_qwen35) environment.

Startup order (two terminals on the same server):
    Terminal 1 — Qwen3.5 worker  (port 8001):
        conda activate llm_peft_qwen35
        python server_qwen35.py

    Terminal 2 — Gateway / main API  (port 8000):
        conda activate nl2asp
        python server.py

The Svelte frontend only ever talks to port 8000.  The two-process split is
completely transparent to it.

════════════════════════════════════════════════════════════════════════════════
WHY TWO PROCESSES?
────────────────────────────────────────────────────────────────────────────────
Qwen3.5-9B requires transformers >= 5.0 because it needs:
    tokenizer.apply_chat_template(..., enable_thinking=False)
This kwarg does not exist in transformers 4.x.  Without it the model emits
<think>…</think> reasoning blocks that corrupt CNL output.

All other models (Qwen3-8B, LLaMA3-8B, Ministral-8B×3, T5-Small×6,
T5-Large×6, T5-3B×6) work correctly with transformers 4.57.3.

A single Python process cannot switch conda environments, so two processes
are required.  The gateway makes this transparent.
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

# Rename the stdlib/third-party requests to avoid shadowing FastAPI's Request
import requests as _http

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Environment guard ─────────────────────────────────────────────────────────
import transformers as _tf
_TF_MAJOR = int(_tf.__version__.split(".")[0])
if _TF_MAJOR >= 5:
    import sys
    sys.exit(
        f"\nERROR: server.py must run under the (nl2asp) environment "
        f"[transformers 4.x].\n"
        f"       Detected: transformers {_tf.__version__}\n"
        f"       For Qwen3.5-9B use server_qwen35.py under (llm_peft_qwen35).\n"
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

# ── External APIs ─────────────────────────────────────────────────────────────
CNL2ASP_BASE    = "http://160.97.63.29:3003/api"
CNL2ASP_HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY":    "BwfZYaNwDUIHUQapxl0TB2B1veZ6AfW8",
}

# Address of the Qwen3.5 worker process (server_qwen35.py)
# Must match the port in server_qwen35.py  (default 8005)
QWEN35_WORKER_URL = "http://localhost:8005"

SYSTEM_PROMPT = (
    "You are an expert in Translating the Natural language (NL) into Controlled Natural "
    "Language (CNL) translation. Always provide precise, syntactically correct translations "
    "of NL into CNL."
)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY  —  all paths are absolute, matching /home/manuel/nl2asp_app/
# ════════════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY: dict = {

    # ── Qwen3-8B  (3 LoRA ranks, transformers 4.x, served locally) ───────────
    "qwen3_8b_r8": {
        "label": "Qwen3-8B r=8",   "tag": "8B r8",  "icon": "🔷",
        "kind":  "causal_qwen3",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3-8B_r8",
        "group": "llm",
    },
    "qwen3_8b_r16": {
        "label": "Qwen3-8B r=16",  "tag": "8B r16", "icon": "🔷",
        "kind":  "causal_qwen3",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3-8B_r16",
        "group": "llm",
    },
    "qwen3_8b_r32": {
        "label": "Qwen3-8B r=32",  "tag": "8B r32", "icon": "🔷",
        "kind":  "causal_qwen3",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3-8B_r32",
        "group": "llm",
    },

    # ── LLaMA-3.1 8B  (3 LoRA ranks) ─────────────────────────────────────────
    "llama3_8b_r8": {
        "label": "LLaMA-3.1 8B r=8",   "tag": "8B r8",  "icon": "🦙",
        "kind":  "causal_llama",        "dtype": "float16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_3.1-8B-Instruct_r8",
        "group": "llm",
    },
    "llama3_8b_r16": {
        "label": "LLaMA-3.1 8B r=16",  "tag": "8B r16", "icon": "🦙",
        "kind":  "causal_llama",        "dtype": "float16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_3.1-8B-Instruct_r16",
        "group": "llm",
    },
    "llama3_8b_r32": {
        "label": "LLaMA-3.1 8B r=32",  "tag": "8B r32", "icon": "🦙",
        "kind":  "causal_llama",        "dtype": "float16",
        "path":  "/home/manuel/nl2asp_app/final_adapter_3.1-8B-Instruct_r32",
        "group": "llm",
    },

    # ── Ministral-8B-2410  (3 LoRA ranks) ────────────────────────────────────
    "ministral_8b_r8": {
        "label": "ministral-8B r=8",   "tag": "8B r8",  "icon": "🎯",
        "kind":  "causal_ministral",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/ministral8B-2410_r8",
        "group": "llm",
    },
    "ministral_8b_r16": {
        "label": "ministral-8B r=16",  "tag": "8B r16", "icon": "🎯",
        "kind":  "causal_ministral",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/ministral8B-2410_r16",
        "group": "llm",
    },
    "ministral_8b_r32": {
        "label": "ministral-8B r=32",  "tag": "8B r32", "icon": "🎯",
        "kind":  "causal_ministral",   "dtype": "bfloat16",
        "path":  "/home/manuel/nl2asp_app/ministral8B-2410_r32",
        "group": "llm",
    },

    # ── T5-Small  (full dataset only) ──
    # TODO: Re-enable K-fold variants (fold_1 through fold_5) per-model later
    "t5small_full": {
        "label": "T5-Small (Full dataset)", "tag": "60M full", "icon": "🔬",
        "kind": "seq2seq", "dtype": "float32",
        "path": "/home/manuel/nl2asp_app/t5small_Complete_Dataset/checkpoint-38000",   "group": "t5small",
    },
    # DISABLED K-FOLD VARIANTS — To be re-enabled per-model in future updates
    # "t5small_k1":   {
    #     "label": "T5-Small K-Fold 1",  "tag": "60M k1", "icon": "🔬",
    #     "kind": "seq2seq", "dtype": "float32",
    #     "path": "/home/manuel/nl2asp_app/Kfold5_t5small/fold_1", "group": "t5small",
    # },
    # "t5small_k2":   {
    #     "label": "T5-Small K-Fold 2",  "tag": "60M k2", "icon": "🔬",
    #     "kind": "seq2seq", "dtype": "float32",
    #     "path": "/home/manuel/nl2asp_app/Kfold5_t5small/fold_2", "group": "t5small",
    # },
    # "t5small_k3":   {
    #     "label": "T5-Small K-Fold 3",  "tag": "60M k3", "icon": "🔬",
    #     "kind": "seq2seq", "dtype": "float32",
    #     "path": "/home/manuel/nl2asp_app/Kfold5_t5small/fold_3", "group": "t5small",
    # },
    # "t5small_k4":   {
    #     "label": "T5-Small K-Fold 4",  "tag": "60M k4", "icon": "🔬",
    #     "kind": "seq2seq", "dtype": "float32",
    #     "path": "/home/manuel/nl2asp_app/Kfold5_t5small/fold_4", "group": "t5small",
    # },
    # "t5small_k5":   {
    #     "label": "T5-Small K-Fold 5",  "tag": "60M k5", "icon": "🔬",
    #     "kind": "seq2seq", "dtype": "float32",
    #     "path": "/home/manuel/nl2asp_app/Kfold5_t5small/fold_5", "group": "t5small",
    # },

    # ── T5-Large  (full dataset only) ──
    # TODO: Re-enable K-fold variants (fold subdirs) per-model later if available
    "t5large_full": {
        "label": "T5-Large (Full dataset)", "tag": "770M full", "icon": "📊",
        "kind": "seq2seq", "dtype": "float16",
        "path": "/home/manuel/nl2asp_app/t5large_Complete_Dataset",   "group": "t5large",
    },

    # ── T5-3B  (full dataset only) ────────────────────────────
    # TODO: Re-enable K-fold variants (fold subdirs) per-model later if available
    "t5_3b_full":   {
        "label": "T5-3B (Full dataset)",    "tag": "3B full",   "icon": "🧬",
        "kind": "seq2seq", "dtype": "float16",
        "path": "/home/manuel/nl2asp_app/t5-3b_Complete_Dataset",      "group": "t5_3b",
    },
}

# ── Qwen3.5 stubs ─────────────────────────────────────────────────────────────
# Served by server_qwen35.py (worker).  Gateway only needs these for health
# reporting and proxy routing.  Keep in sync with server_qwen35.py MODEL_REGISTRY.
QWEN35_STUBS: dict = {
    "qwen3_5_9b_r8": {
        "label": "Qwen3.5-9B r=8",   "tag": "9B r8",  "icon": "🌟",
        "kind":  "causal_qwen3_5",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3.5-9B_r8",
        "group": "llm",
    },
    "qwen3_5_9b_r16": {
        "label": "Qwen3.5-9B r=16",  "tag": "9B r16", "icon": "🌟",
        "kind":  "causal_qwen3_5",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3.5-9B_r16",
        "group": "llm",
    },
    "qwen3_5_9b_r32": {
        "label": "Qwen3.5-9B r=32",  "tag": "9B r32", "icon": "🌟",
        "kind":  "causal_qwen3_5",
        "path":  "/home/manuel/nl2asp_app/final_adapter_Qwen3.5-9B_r32",
        "group": "llm",
    },
}

def _is_qwen35(model_key: str) -> bool:
    """True for any model key that must be routed to the Qwen3.5 worker."""
    return model_key in QWEN35_STUBS

# Only one model from these groups held in VRAM at a time
EXCLUSIVE_GROUPS = {"llm"}

# ── In-process model cache ─────────────────────────────────────────────────────
_loaded: dict = {}
_load_lock = threading.Lock()


# ════════════════════════════════════════════════════════════════════════════════
# Qwen3.5 worker helpers
# ════════════════════════════════════════════════════════════════════════════════
def _worker_health() -> dict:
    """Return the worker's health JSON, or {} if unreachable."""
    try:
        r = _http.get(f"{QWEN35_WORKER_URL}/api/health", timeout=2)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _worker_online() -> bool:
    return bool(_worker_health())


# ════════════════════════════════════════════════════════════════════════════════
# Think-token stripper  (belt-and-suspenders for Qwen3-8B on transformers 4.x)
# ════════════════════════════════════════════════════════════════════════════════
def _strip_think(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    return cleaned.strip()


# ════════════════════════════════════════════════════════════════════════════════
# Model loaders
# ════════════════════════════════════════════════════════════════════════════════
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

def _resolve_dtype(dtype_str: str):
    """Convert string dtype name to torch.dtype object."""
    return _DTYPE_MAP.get(dtype_str, torch.float32)


def _model_input_device(model) -> torch.device:
    """
    Return the device to place input_ids on.
    With device_map="auto", accelerate may split layers across GPU+CPU.
    The embedding layer (first consumer of input_ids) determines the
    correct device — NOT the global DEVICE constant.
    """
    try:
        # accelerate attaches hf_device_map; embedding is always "model.embed_tokens"
        # or "transformer.wte" etc. — just grab the first non-"cpu" device, or cpu.
        device_map = getattr(model, "hf_device_map", None)
        if device_map is None:
            device_map = getattr(model.base_model, "hf_device_map", None)
        if device_map:
            # Find where embed_tokens lives; fall back to first device in map
            for key in ("model.embed_tokens", "transformer.wte", "model.embed_tokens.weight"):
                if key in device_map:
                    dev = device_map[key]
                    return torch.device(dev if dev != "cpu" else "cpu")
            # fallback: first non-cpu device, else cpu
            for dev in device_map.values():
                if dev != "cpu":
                    return torch.device(dev)
            return torch.device("cpu")
    except Exception:
        pass
    # Final fallback: first parameter device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def _load_peft_causal(path: str, dtype) -> dict:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    log.info("Loading PEFT causal model: %s", path)
    # Try the new-style dtype= kwarg first; fall back to torch_dtype= for older
    # versions of transformers/PEFT that don't accept dtype=.
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
    # Detect where embed_tokens actually landed after accelerate device_map="auto".
    # With split GPU+CPU offloading, input_ids must go to the embed_tokens device,
    # NOT the global DEVICE constant — or you get the cross-device RuntimeError.
    input_device = _model_input_device(model)
    log.info("  embed_tokens device: %s", input_device)
    return {"model": model, "tokenizer": tok, "input_device": input_device}


# def _load_seq2seq(path: str, dtype) -> dict:
#     from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#     log.info("Loading seq2seq model: %s", path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=dtype).to(DEVICE)
#     model.eval()
#     tok = AutoTokenizer.from_pretrained(path)
#     return {"model": model, "tokenizer": tok}

def _load_seq2seq(path: str, dtype) -> dict:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import traceback
    log.info("Loading seq2seq model: %s", path)
    try:
        log.info("  Loading model weights from %s", path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=dtype).to(DEVICE)
        log.info("  Model loaded, evaluating...")
        model.eval()
        log.info("  Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(path)
        log.info("  Tokenizer loaded successfully")
        return {"model": model, "tokenizer": tok}
    except Exception as e:
        log.error("Failed to load seq2seq model from %s", path)
        log.error("Error type: %s", type(e).__name__)
        log.error("Error message: %s", str(e))
        log.error("Full traceback:\n%s", traceback.format_exc())
        raise


# ════════════════════════════════════════════════════════════════════════════════
# Lazy loader with VRAM eviction
# ════════════════════════════════════════════════════════════════════════════════
def _free_entry(e: dict) -> None:
    """Delete model/tokenizer/input_device from an entry and release GPU memory."""
    e.pop("model", None)
    e.pop("tokenizer", None)
    e.pop("input_device", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evict_group(group: str) -> None:
    keys = [k for k in list(_loaded) if MODEL_REGISTRY.get(k, {}).get("group") == group]
    for k in keys:
        log.info("Evicting '%s' from VRAM.", k)
        _free_entry(_loaded.pop(k))


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
                                     f"Update MODEL_REGISTRY in server.py.")
        if meta["group"] in EXCLUSIVE_GROUPS:
            _evict_group(meta["group"])
        try:
            kind  = meta["kind"]
            dtype = _resolve_dtype(meta["dtype"])   # string → torch.dtype
            if kind in ("causal_qwen3", "causal_llama", "causal_ministral"):
                entry = _load_peft_causal(meta["path"], dtype)
            elif kind == "seq2seq":
                entry = _load_seq2seq(meta["path"], dtype)
            else:
                raise HTTPException(500, f"Unknown kind: {kind}")
            entry["kind"] = kind
            _loaded[model_key] = entry
            log.info("✓ '%s' ready.", model_key)
            return entry
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Failed to load '{model_key}': {exc}") from exc


# ════════════════════════════════════════════════════════════════════════════════
# Inference functions
# ════════════════════════════════════════════════════════════════════════════════
def _gen(model, tok, prompt: str, max_new_tokens: int, input_device: torch.device) -> str:
    """
    Tokenize prompt, move input_ids to the correct device, generate, decode.
    input_device must come from entry["input_device"] — NOT the global DEVICE —
    because accelerate may have split layers across GPU+CPU.
    """
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
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def _infer_llama(entry: dict, nl: str, max_new_tokens: int) -> str:
    model, tok, dev = entry["model"], entry["tokenizer"], entry["input_device"]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Translate the following natural language to controlled natural language: {nl}"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return _gen(model, tok, prompt, max_new_tokens, dev)


def _infer_qwen3(entry: dict, nl: str, max_new_tokens: int) -> str:
    """
    Qwen3-8B under transformers 4.57.3.

    The  enable_thinking=False  kwarg only exists in transformers >= 5.0.
    The documented workaround for 4.x is to append a pre-filled empty
    <think></think> block to the prompt, which forces the model to skip its
    reasoning phase and emit the answer directly.

    Reference: https://huggingface.co/Qwen/Qwen3-8B#disable-thinking-mode
    """
    model, tok = entry["model"], entry["tokenizer"]  # entry["input_device"] used below
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Translate the following natural language to controlled natural language: {nl}"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Append empty think block — disables chain-of-thought on transformers 4.x
    prompt += "<think>\n\n</think>\n\n"
    raw = _gen(model, tok, prompt, max_new_tokens, entry["input_device"])
    return _strip_think(raw)   # safety net


def _infer_ministral(entry: dict, nl: str, max_new_tokens: int) -> str:
    """Ministral tokenizer silently drops the system role — prepend into user msg."""
    model, tok, dev = entry["model"], entry["tokenizer"], entry["input_device"]
    msgs = [{
        "role":    "user",
        "content": (f"{SYSTEM_PROMPT}\n\n"
                    f"Translate the following natural language to controlled natural language: {nl}"),
    }]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return _gen(model, tok, prompt, max_new_tokens, dev)


def _infer_seq2seq(entry: dict, nl: str, max_new_tokens: int) -> str:
    model, tok = entry["model"], entry["tokenizer"]
    inputs = tok("translate NL to CNL: " + nl,
                 return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _predict_local(model_key: str, nl: str, max_new_tokens: int) -> str:
    entry = _ensure_loaded(model_key)
    kind  = entry["kind"]
    if kind == "causal_llama":     return _infer_llama(entry, nl, max_new_tokens)
    if kind == "causal_qwen3":     return _infer_qwen3(entry, nl, max_new_tokens)
    if kind == "causal_ministral": return _infer_ministral(entry, nl, max_new_tokens)
    if kind == "seq2seq":          return _infer_seq2seq(entry, nl, max_new_tokens)
    raise HTTPException(500, f"Unhandled model kind: {kind}")


# ════════════════════════════════════════════════════════════════════════════════
# Qwen3.5 proxy  — forwards the full nl2asp pipeline to the worker
# ════════════════════════════════════════════════════════════════════════════════
def _proxy_qwen35(model_key: str, nl: str, max_new_tokens: int) -> dict:
    """Forward a request to the Qwen3.5 worker, passing the exact model_key."""
    if not _worker_online():
        raise HTTPException(
            503,
            "Qwen3.5-9B worker is not running.\n"
            "Start it:  conda activate llm_peft_qwen35 && python server_qwen35.py"
        )
    try:
        r = _http.post(
            f"{QWEN35_WORKER_URL}/api/nl2asp",
            # Pass the exact variant key (qwen3_5_9b / qwen3_5_9b_r8 / etc.)
            # so the worker loads the correct adapter checkpoint.
            json={"nl": nl, "model": model_key, "max_new_tokens": max_new_tokens},
            timeout=300,
        )
        if not r.ok:
            raise HTTPException(r.status_code, f"Qwen3.5 worker error: {r.text}")
        return r.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Qwen3.5 worker unreachable: {exc}") from exc


def _proxy_qwen35_cnl_only(model_key: str, nl: str, max_new_tokens: int) -> dict:
    result = _proxy_qwen35(model_key, nl, max_new_tokens)
    return {"cnl": result["cnl"], "syntax_valid": result.get("syntax_valid", False)}


# ════════════════════════════════════════════════════════════════════════════════
# CNL → ASP helpers
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
# FastAPI
# ════════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Gateway server — transformers %s  (nl2asp env)", _tf.__version__)
    log.info("Qwen3.5 worker expected at %s", QWEN35_WORKER_URL)
    if _worker_online():
        log.info("✓ Qwen3.5 worker is reachable.")
    else:
        log.warning("✗ Qwen3.5 worker offline — Qwen3.5-9B will be unavailable until started.")
    yield
    _loaded.clear()
    log.info("Gateway stopped.")


app = FastAPI(title="NL2ASP Gateway API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NL2CNLRequest(BaseModel):
    nl: str;  model: str = "qwen3_8b";  max_new_tokens: int = 256

class CNL2ASPRequest(BaseModel):
    cnl: str

class NL2ASPRequest(BaseModel):
    nl: str;  model: str = "qwen3_8b";  max_new_tokens: int = 256

class UnloadRequest(BaseModel):
    model: Optional[str] = None


@app.get("/api/health")
def health():
    wh = _worker_health()
    worker_online  = bool(wh)
    # worker_loaded_keys — live list from Qwen3.5 worker (may be empty if offline)

    available = {}
    # Qwen3.5 variants — served by worker, listed with live loaded/worker status
    worker_loaded_keys = wh.get("loaded_models", [])
    for key, meta in QWEN35_STUBS.items():
        available[key] = {
            "label":         meta["label"],
            "tag":           meta["tag"],
            "icon":          meta["icon"],
            "kind":          meta["kind"],
            "group":         meta["group"],
            "exists":        Path(meta["path"]).exists(),
            "loaded":        key in worker_loaded_keys,
            "worker_online": worker_online,
        }
    # All other models — served locally by this process
    for key, meta in MODEL_REGISTRY.items():
        available[key] = {
            "label":         meta["label"],
            "tag":           meta["tag"],
            "icon":          meta["icon"],
            "kind":          meta["kind"],
            "group":         meta["group"],
            "exists":        Path(meta["path"]).exists(),
            "loaded":        key in _loaded,
            "worker_online": True,
        }

    loaded_keys = list(_loaded.keys())
    # Prepend any Qwen3.5 variants that are currently loaded in the worker
    for k in worker_loaded_keys:
        if k not in loaded_keys:
            loaded_keys.insert(0, k)

    return {
        "status":           "ok",
        "device":           DEVICE,
        "gpu":              gpu_name,
        "tf_version":       _tf.__version__,
        "loaded_models":    loaded_keys,
        "available_models": available,
        "qwen35_worker":    "online" if worker_online else "offline",
    }


@app.post("/api/nl2cnl")
def nl2cnl(req: NL2CNLRequest):
    if _is_qwen35(req.model):
        return _proxy_qwen35_cnl_only(req.model, req.nl, req.max_new_tokens)
    cnl = _predict_local(req.model, req.nl, req.max_new_tokens)
    return {"cnl": cnl, "syntax_valid": _check_syntax(cnl)}


@app.post("/api/cnl2asp")
def cnl2asp(req: CNL2ASPRequest):
    asp, compiled = _compile_cnl(req.cnl)
    return {"asp": asp, "compiled": compiled}


@app.post("/api/nl2asp")
def nl2asp(req: NL2ASPRequest):
    if _is_qwen35(req.model):
        return _proxy_qwen35(req.model, req.nl, req.max_new_tokens)
    cnl           = _predict_local(req.model, req.nl, req.max_new_tokens)
    valid         = _check_syntax(cnl)
    asp, compiled = _compile_cnl(cnl)
    return {"cnl": cnl, "syntax_valid": valid, "asp": asp, "compiled": compiled}


@app.post("/api/unload")
def unload(req: UnloadRequest):
    with _load_lock:
        # ── Unload all local models ───────────────────────────────────────────
        if req.model is None:
            keys = list(_loaded.keys())
            for k in keys:
                _free_entry(_loaded.pop(k))
            # Also tell Qwen3.5 worker to unload everything
            if _worker_online():
                try:
                    _http.post(f"{QWEN35_WORKER_URL}/api/unload",
                               json={"model": None}, timeout=10)
                except Exception:
                    pass
            return {"unloaded": keys}

        # ── Qwen3.5 variants — forward to worker ──────────────────────────────
        if _is_qwen35(req.model):
            if _worker_online():
                try:
                    _http.post(f"{QWEN35_WORKER_URL}/api/unload",
                               json={"model": req.model}, timeout=10)
                except Exception:
                    pass
            # Always return success — worker state is authoritative
            return {"unloaded": [req.model]}

        # ── Single local model ────────────────────────────────────────────────
        if req.model not in _loaded:
            # Already evicted (e.g. auto-evicted when another LLM was loaded).
            # Return success instead of 404 — the goal (not in VRAM) is already met.
            log.info("Unload requested for '%s' — already not in memory.", req.model)
            return {"unloaded": [req.model], "note": "was already unloaded"}
        _free_entry(_loaded.pop(req.model))
        return {"unloaded": [req.model]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
