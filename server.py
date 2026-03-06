"""
NL2ASP Backend Server — ROCm / AMD GPU
Models supported:
  - llamafinetune        LLaMA-3.1 8B LoRA (original)
  - llamafinetune_HP     LLaMA-3.1 8B LoRA (HP-tuned)
  - Qwen3-8Bfinetune     Qwen3-8B LoRA (causal, needs <think> stripping)
  - t5small_weights      T5-Small seq2seq
  - t5large_weights      T5-Large seq2seq
  - t53b_weights         T5-3B   seq2seq
  - Kfold5_t5small/fold_N  KFold T5-Small ensemble
"""

import json
import logging
import re
import requests
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    log.info(f"GPU detected: {gpu_name}")
else:
    DEVICE = "cpu"
    log.warning("No GPU — falling back to CPU.")

log.info(f"Device: {DEVICE}")

# ── Weight paths ──────────────────────────────────────────────────────────────
# Causal LM (LoRA adapters)
LLAMA_PATH       = "./llamafinetune"
LLAMA_HP_PATH    = "./llamafinetune_HP"
QWEN3_PATH       = "./Qwen3-8Bfinetune"

# Seq2Seq
T5_SMALL_PATH    = "./t5small_weights"
T5_LARGE_PATH    = "./t5large_weights"
T5_3B_PATH       = "./t53b_weights"

# KFold T5-Small
KFOLD_T5_BASE    = "./Kfold5_t5small"
KFOLD_T5_FOLDS   = 5

# ── CNL2ASP API ───────────────────────────────────────────────────────────────
CNL2ASP_BASE    = "http://160.97.63.29:3003/api"
CNL2ASP_HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": "BwfZYaNwDUIHUQapxl0TB2B1veZ6AfW8",
}

SYSTEM_PROMPT = (
    "You are an expert in Translating the Natural language (NL) into Controlled Natural "
    "Language (CNL) translation. Always provide precise, syntactically correct translations "
    "of NL into CNL."
)

# ── Model registry ────────────────────────────────────────────────────────────
models: dict = {}


# ── Known fixes ───────────────────────────────────────────────────────────────

def _patch_tokenizer_config(path: str):
    """
    Fix two known transformers ≥4.47 regressions in tokenizer_config.json:
      1. extra_special_tokens saved as [] list instead of {} dict
         → AttributeError: 'list' object has no attribute 'keys'
      2. vocab_file saved as null
         → TypeError: not a string  (sentencepiece)
    Applied before every model load as a safety measure.
    """
    cfg_path = Path(path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text())
        changed = False

        # Fix 1: top-level extra_special_tokens
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg["extra_special_tokens"] = {}
            changed = True

        # Fix 1b: inside added_tokens_decoder entries
        for v in cfg.get("added_tokens_decoder", {}).values():
            if isinstance(v, dict) and isinstance(v.get("extra_special_tokens"), list):
                v["extra_special_tokens"] = {}
                changed = True

        # Fix 2: vocab_file is null — copy spiece.model from KFold fold_1 if available
        if cfg.get("vocab_file") is None:
            fallback = Path(KFOLD_T5_BASE) / "fold_1" / "spiece.model"
            local    = Path(path) / "spiece.model"
            if not local.exists() and fallback.exists():
                import shutil
                shutil.copy(fallback, local)
                log.info(f"Copied spiece.model from fold_1 → {path}")
            if local.exists():
                cfg["vocab_file"] = "spiece.model"
                changed = True

        if changed:
            cfg_path.write_text(json.dumps(cfg, indent=2))
            log.info(f"Patched tokenizer_config.json at {path}")
    except Exception as e:
        log.warning(f"Could not patch tokenizer config at {path}: {e}")


def _dtype_str() -> str:
    """Return dtype as plain string — newer transformers rejects torch.dtype objects."""
    return "float16" if DEVICE == "cuda" else "float32"


def _try_load(key: str, loader):
    try:
        models[key] = loader()
        log.info(f"✓  {key} loaded")
    except Exception as e:
        log.warning(f"✗  {key} skipped — {e}")
        log.warning(traceback.format_exc())


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_causal(path: str, is_qwen: bool = False) -> dict:
    """Load any LoRA causal-LM adapter (LLaMA or Qwen3)."""
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    log.info(f"Loading causal LM from {path} ...")
    _patch_tokenizer_config(path)
    dtype = _dtype_str()

    # Qwen3 was trained at bfloat16; use bf16 on GPU if available
    if is_qwen and DEVICE == "cuda":
        dtype = "bfloat16"

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            path, device_map="auto", dtype=dtype, trust_remote_code=True,
        )
    except TypeError:
        # Fallback for older transformers that don't accept dtype= as string
        torch_dtype = torch.bfloat16 if (is_qwen and DEVICE == "cuda") else (
            torch.float16 if DEVICE == "cuda" else torch.float32)
        model = AutoPeftModelForCausalLM.from_pretrained(
            path, device_map="auto", torch_dtype=torch_dtype, trust_remote_code=True,
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return {"model": model, "tokenizer": tokenizer, "type": "causal", "is_qwen": is_qwen}


def _load_seq2seq(path: str) -> dict:
    """Load any seq2seq model (T5-Small / T5-Large / T5-3B)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    log.info(f"Loading seq2seq from {path} ...")
    _patch_tokenizer_config(path)
    dtype = _dtype_str()
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(DEVICE)
    except TypeError:
        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch_dtype).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return {"model": model, "tokenizer": tokenizer, "type": "seq2seq", "is_qwen": False}


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── Seq2Seq models ─────────────────────────────────────────────────────────
    for key, path in [
        ("t5small",  T5_SMALL_PATH),
        ("t5large",  T5_LARGE_PATH),
        ("t53b",     T5_3B_PATH),
    ]:
        if Path(path).exists():
            _try_load(key, lambda p=path: _load_seq2seq(p))
        else:
            log.warning(f"{key} weights not found at: {path}")

    # ── KFold T5-Small ─────────────────────────────────────────────────────────
    kfold_loaded = []
    for n in range(1, KFOLD_T5_FOLDS + 1):
        fold_path = Path(KFOLD_T5_BASE) / f"fold_{n}"
        if fold_path.exists():
            key = f"t5small_fold{n}"
            _try_load(key, lambda p=str(fold_path): _load_seq2seq(p))
            if key in models:
                kfold_loaded.append(key)
        else:
            log.warning(f"KFold fold_{n} not found at: {fold_path}")
    if kfold_loaded:
        log.info(f"KFold folds loaded: {kfold_loaded}")

    # ── Causal LMs — load heaviest last ───────────────────────────────────────
    for key, path, is_qwen in [
        # ("llama",    LLAMA_PATH,    False),
        ("llama_HP", LLAMA_HP_PATH, False),
        ("qwen3",    QWEN3_PATH,    True),
    ]:
        if Path(path).exists():
            _try_load(key, lambda p=path, q=is_qwen: _load_causal(p, q))
        else:
            log.warning(f"{key} weights not found at: {path}")

    if not models:
        log.error("No models loaded! Check weight paths at top of server.py.")
    else:
        log.info(f"Ready — loaded: {list(models.keys())}")
    yield
    models.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="NL2ASP API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class NL2CNLRequest(BaseModel):
    nl: str
    model: str = "llama"
    max_new_tokens: int = 256

class CNL2ASPRequest(BaseModel):
    cnl: str

class NL2ASPRequest(BaseModel):
    nl: str
    model: str = "llama"
    max_new_tokens: int = 256

class KFoldEnsembleRequest(BaseModel):
    nl: str
    max_new_tokens: int = 256


# ── Qwen3 output cleaning ─────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> chain-of-thought blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer_causal(entry: dict, nl: str, max_new_tokens: int) -> str:
    model, tokenizer = entry["model"], entry["tokenizer"]
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Translate the following natural language to controlled natural language: {nl}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # Always strip think tokens — harmless no-op for LLaMA, required for Qwen3
    return _strip_think(result)


def _infer_seq2seq(entry: dict, nl: str, max_new_tokens: int) -> str:
    model, tokenizer = entry["model"], entry["tokenizer"]
    inputs = tokenizer(
        "translate NL to CNL: " + nl,
        return_tensors="pt", truncation=True, max_length=512,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _predict(model_key: str, nl: str, max_new_tokens: int) -> str:
    if model_key not in models:
        raise HTTPException(400, f"Model '{model_key}' not loaded. Available: {list(models.keys())}")
    entry = models[model_key]
    if entry["type"] == "causal":
        return _infer_causal(entry, nl, max_new_tokens)
    return _infer_seq2seq(entry, nl, max_new_tokens)


def _check_syntax(cnl: str) -> bool:
    try:
        r = requests.post(f"{CNL2ASP_BASE}/check_syntax", headers=CNL2ASP_HEADERS,
                          json={"cnls": cnl}, timeout=10)
        return r.json().get("cli_message") == "Input file fits the grammar."
    except Exception:
        return False


def _compile_cnl(cnl: str) -> tuple:
    try:
        r = requests.post(f"{CNL2ASP_BASE}/compile", headers=CNL2ASP_HEADERS,
                          json={"cnls": cnl}, timeout=15)
        asp = r.json().get("asp", "").strip()
        return (asp, True) if asp else ("", False)
    except Exception:
        return ("", False)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":        "ok",
        "device":        DEVICE,
        "gpu":           torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none",
        "loaded_models": list(models.keys()),
        "kfold_folds":   [k for k in models if k.startswith("t5small_fold")],
    }


@app.post("/api/nl2cnl")
def nl2cnl(req: NL2CNLRequest):
    cnl = _predict(req.model, req.nl, req.max_new_tokens)
    return {"cnl": cnl, "syntax_valid": _check_syntax(cnl)}


@app.post("/api/cnl2asp")
def cnl2asp(req: CNL2ASPRequest):
    asp, compiled = _compile_cnl(req.cnl)
    return {"asp": asp, "compiled": compiled}


@app.post("/api/nl2asp")
def nl2asp(req: NL2ASPRequest):
    cnl           = _predict(req.model, req.nl, req.max_new_tokens)
    valid         = _check_syntax(cnl)
    asp, compiled = _compile_cnl(cnl)
    return {"cnl": cnl, "syntax_valid": valid, "asp": asp, "compiled": compiled}


@app.post("/api/nl2asp_kfold_ensemble")
def nl2asp_kfold_ensemble(req: KFoldEnsembleRequest):
    fold_keys = sorted(k for k in models if k.startswith("t5small_fold"))
    if not fold_keys:
        raise HTTPException(400, "No KFold models loaded.")

    fold_results = []
    for key in fold_keys:
        try:
            cnl   = _predict(key, req.nl, req.max_new_tokens)
            valid = _check_syntax(cnl)
        except Exception as e:
            cnl, valid = f"ERROR: {e}", False
        fold_results.append({"fold": key, "cnl": cnl, "syntax_valid": valid})

    # Prefer valid predictions for majority vote
    valid_cnls = [r["cnl"] for r in fold_results if r["syntax_valid"]]
    if valid_cnls:
        winner, cnl_source = Counter(valid_cnls).most_common(1)[0][0], "majority_vote_valid"
    else:
        winner, cnl_source = Counter(r["cnl"] for r in fold_results).most_common(1)[0][0], "majority_vote_all"

    asp, compiled = _compile_cnl(winner)
    return {
        "cnl": winner, "syntax_valid": _check_syntax(winner),
        "asp": asp, "compiled": compiled,
        "cnl_source": cnl_source, "fold_details": fold_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
