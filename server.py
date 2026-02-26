"""
NL2ASP Backend Server — ROCm / AMD GPU
Serves inference endpoints for the Svelte frontend.

Setup on ROCm server:
    pip install fastapi uvicorn requests
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
    pip install transformers peft accelerate

Run:
    python server.py

API at: http://0.0.0.0:8000
"""

import json
import logging
import requests
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Device detection — ROCm exposes itself as CUDA to PyTorch ─────────────────
if torch.cuda.is_available():
    DEVICE   = "cuda"
    DTYPE    = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    log.info(f"GPU detected: {gpu_name}")
else:
    DEVICE   = "cpu"
    DTYPE    = torch.float32
    log.warning("No GPU found — falling back to CPU.")

log.info(f"Device: {DEVICE}  |  dtype: {DTYPE}")

# ── Weight paths ───────────────────────────────────────────────────────────────
LLAMA_ADAPTER_PATH = "./llamafinetune"
T5_SMALL_PATH      = "./t5small_weights"
BART_BASE_PATH     = "./bartbase_weights"

# KFold T5-Small — contains fold_1 … fold_5 subdirectories
KFOLD_T5_BASE_PATH = "./Kfold5_t5small"
KFOLD_T5_FOLDS     = 5

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


def _try_load(key: str, loader):
    import traceback
    try:
        models[key] = loader()
        log.info(f"✓  {key} loaded")
    except Exception as e:
        log.warning(f"✗  {key} skipped — {e}")
        log.debug(traceback.format_exc())


def _patch_tokenizer_config(path: str):
    """
    Transformers ≥ 4.47 regression: extra_special_tokens saved as a list []
    causes AttributeError: 'list' object has no attribute 'keys' on load.
    Patch tokenizer_config.json in-place to convert it to a dict {}.
    Safe to call multiple times — only writes if the bug is present.
    """
    cfg_path = Path(path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if isinstance(cfg.get("extra_special_tokens"), list):
        log.info(f"Patching extra_special_tokens list→dict in {cfg_path}")
        cfg["extra_special_tokens"] = {}
        cfg_path.write_text(json.dumps(cfg, indent=2))


def _load_llama():
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    log.info(f"Loading LLaMA adapter from {LLAMA_ADAPTER_PATH} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        LLAMA_ADAPTER_PATH,
        device_map="auto",
        torch_dtype=DTYPE,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_ADAPTER_PATH)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return {"model": model, "tokenizer": tokenizer, "type": "causal"}


def _load_seq2seq(path: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    log.info(f"Loading seq2seq from {path} ...")
    # Fix transformers >=4.47 tokenizer_config regression before loading
    _patch_tokenizer_config(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return {"model": model, "tokenizer": tokenizer, "type": "seq2seq"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Original single-weight models ─────────────────────────────────────────
    if Path(T5_SMALL_PATH).exists():
        _try_load("t5small", lambda: _load_seq2seq(T5_SMALL_PATH))
    else:
        log.warning(f"T5-Small weights not found at: {T5_SMALL_PATH}")

    if Path(BART_BASE_PATH).exists():
        _try_load("bartbase", lambda: _load_seq2seq(BART_BASE_PATH))
    else:
        log.warning(f"BART-Base weights not found at: {BART_BASE_PATH}")

    # ── KFold T5-Small — load each fold that exists ───────────────────────────
    kfold_base = Path(KFOLD_T5_BASE_PATH)
    if kfold_base.exists():
        loaded_folds = []
        for fold_n in range(1, KFOLD_T5_FOLDS + 1):
            fold_path = kfold_base / f"fold_{fold_n}"
            if fold_path.exists():
                key = f"t5small_fold{fold_n}"
                _try_load(key, lambda p=str(fold_path): _load_seq2seq(p))
                if key in models:
                    loaded_folds.append(key)
            else:
                log.warning(f"KFold fold not found: {fold_path}")
        if loaded_folds:
            log.info(f"KFold T5-Small folds loaded: {loaded_folds}")
        else:
            log.warning(f"No KFold folds loaded from {KFOLD_T5_BASE_PATH}")
    else:
        log.warning(f"KFold T5-Small base path not found: {KFOLD_T5_BASE_PATH}")

    # ── LLaMA last — largest model ────────────────────────────────────────────
    if Path(LLAMA_ADAPTER_PATH).exists():
        _try_load("llama", _load_llama)
    else:
        log.warning(f"LLaMA adapter not found at: {LLAMA_ADAPTER_PATH}")

    if not models:
        log.error("No models loaded! Check weight paths at the top of server.py.")
    else:
        log.info(f"Ready — loaded: {list(models.keys())}")
    yield
    models.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="NL2ASP API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class EnsembleRequest(BaseModel):
    nl: str
    max_new_tokens: int = 256


# ── Inference helpers ─────────────────────────────────────────────────────────
def _infer_causal(entry: dict, nl: str, max_new_tokens: int) -> str:
    model     = entry["model"]
    tokenizer = entry["tokenizer"]
    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Translate the following natural language to controlled natural language: {nl}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()


def _infer_seq2seq(entry: dict, nl: str, max_new_tokens: int) -> str:
    model     = entry["model"]
    tokenizer = entry["tokenizer"]
    inputs    = tokenizer(
        "translate NL to CNL: " + nl,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _predict(model_key: str, nl: str, max_new_tokens: int) -> str:
    if model_key not in models:
        raise HTTPException(
            400,
            f"Model '{model_key}' not loaded. Available: {list(models.keys())}"
        )
    entry = models[model_key]
    if entry["type"] == "causal":
        return _infer_causal(entry, nl, max_new_tokens)
    return _infer_seq2seq(entry, nl, max_new_tokens)


def _check_syntax(cnl: str) -> bool:
    try:
        r = requests.post(
            f"{CNL2ASP_BASE}/check_syntax",
            headers=CNL2ASP_HEADERS,
            json={"cnls": cnl},
            timeout=10,
        )
        return r.json().get("cli_message") == "Input file fits the grammar."
    except Exception:
        return False


def _compile_cnl(cnl: str) -> tuple:
    try:
        r = requests.post(
            f"{CNL2ASP_BASE}/compile",
            headers=CNL2ASP_HEADERS,
            json={"cnls": cnl},
            timeout=15,
        )
        asp = r.json().get("asp", "").strip()
        return (asp, True) if asp else ("", False)
    except Exception:
        return ("", False)


def _get_kfold_keys() -> list:
    """Return all currently loaded KFold fold keys, sorted."""
    return sorted(k for k in models if k.startswith("t5small_fold"))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":        "ok",
        "device":        DEVICE,
        "gpu":           torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none",
        "loaded_models": list(models.keys()),
        "kfold_folds":   _get_kfold_keys(),
    }


@app.post("/api/nl2cnl")
def nl2cnl(req: NL2CNLRequest):
    cnl   = _predict(req.model, req.nl, req.max_new_tokens)
    valid = _check_syntax(cnl)
    return {"cnl": cnl, "syntax_valid": valid}


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
def nl2asp_kfold_ensemble(req: EnsembleRequest):
    """
    Run all loaded KFold T5-Small folds and pick the best CNL via majority vote.

    Strategy:
      1. Collect CNL predictions from every fold.
      2. Check syntax for each.
      3. Prefer the most common syntax-valid prediction (majority vote).
      4. Fall back to most common prediction overall if none are valid.
      5. Compile the chosen CNL to ASP.
    """
    fold_keys = _get_kfold_keys()
    if not fold_keys:
        raise HTTPException(400, "No KFold T5-Small folds are loaded.")

    fold_results = []
    for key in fold_keys:
        try:
            cnl   = _predict(key, req.nl, req.max_new_tokens)
            valid = _check_syntax(cnl)
        except Exception as e:
            log.warning(f"Fold {key} inference failed: {e}")
            cnl, valid = "", False
        fold_results.append({"fold": key, "cnl": cnl, "syntax_valid": valid})

    valid_preds = [r["cnl"] for r in fold_results if r["syntax_valid"] and r["cnl"]]
    all_preds   = [r["cnl"] for r in fold_results if r["cnl"]]

    if valid_preds:
        best_cnl   = Counter(valid_preds).most_common(1)[0][0]
        cnl_source = "majority_vote_valid"
    elif all_preds:
        best_cnl   = Counter(all_preds).most_common(1)[0][0]
        cnl_source = "majority_vote_all"
    else:
        raise HTTPException(500, "All folds returned empty predictions.")

    syntax_valid  = _check_syntax(best_cnl)
    asp, compiled = _compile_cnl(best_cnl)

    return {
        "cnl":          best_cnl,
        "syntax_valid": syntax_valid,
        "asp":          asp,
        "compiled":     compiled,
        "cnl_source":   cnl_source,
        "folds_used":   len(fold_keys),
        "fold_details": fold_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
