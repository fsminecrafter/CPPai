#!/usr/bin/env python3
"""
Pure-Python + numpy Decoder-only Transformer LM  (GPU-capable via CuPy)
=========================================================================
Architecture  (GPT-style)
  - Token embedding    wte : [vocab_size x embed_dim]
  - Position embedding wpe : [block_size x embed_dim]   (learned, absolute)
  - N transformer blocks, each:
        LN1 -> causal multi-head self-attention -> residual add
        LN2 -> FFN (Linear -> GELU -> Linear)   -> residual add
  - Final LayerNorm
  - Linear head -> logits over vocab
  - Loss: cross-entropy at every position (teacher forcing over the
    whole block_size window, not just a single next token)

This replaces the old fixed-context "concat last N embeddings -> one
hidden layer -> softmax" model, which had no attention and could not
represent dependencies beyond a tiny, un-ordered window — hence the
repetition regardless of how much training data you fed it.

GPU support
  - Requires CuPy (pip install cupy-cuda11x / cupy-cuda12x).
  - Set  use_gpu=True  and  gpu_device=<id>  in settings.
  - Every array op below goes through the module-level `_xp` backend
    (numpy or cupy), same convention as the original file.

Memory-safe training
  - Files are processed in chunks; only one chunk's tokens live in RAM
    at a time. Each chunk is sliced into non-overlapping [block_size]
    sequences for training.

Book downloader
  - Unchanged from the original: multi-threaded concurrent downloads
    with per-book progress bars and ETA.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# GPU / Array-backend  (xp = numpy or cupy)
# ─────────────────────────────────────────────────────────────────────────────

_cupy_available   = False
_curand_available = False
_cupy_devices: List[Tuple[int, str]] = []

_cupy_import_error: str = ""

try:
    import cupy as cp          # type: ignore
    _cupy_available = True
except BaseException as _e:
    _cupy_available    = False
    _curand_available  = False
    _cupy_import_error = str(_e)

if _cupy_available:
    try:
        _ = cp.random.permutation(4)
        _curand_available = True
    except BaseException:
        _curand_available = False
    try:
        n_dev = cp.cuda.runtime.getDeviceCount()
        for _di in range(n_dev):
            try:
                cp.cuda.Device(_di).use()
                _props = cp.cuda.runtime.getDeviceProperties(_di)
                _name  = (_props["name"].decode()
                          if isinstance(_props["name"], bytes)
                          else str(_props["name"]))
                _cupy_devices.append((_di, _name))
            except BaseException:
                _cupy_devices.append((_di, f"CUDA device {_di}"))
    except BaseException:
        try:
            cp.cuda.Device(0).use()
            _cupy_devices.append((0, "CUDA device 0"))
        except BaseException:
            pass

_xp        = np
_device_id: int = -1


def _set_backend(use_gpu: bool, gpu_device: int = 0) -> None:
    global _xp, _device_id
    if use_gpu and _cupy_available:
        _device_id = gpu_device
        cp.cuda.Device(gpu_device).use()
        _xp = cp
    else:
        _xp = np
        _device_id = -1


def _to_numpy(arr) -> np.ndarray:
    if _xp is np:
        return arr
    return cp.asnumpy(arr)


def _to_xp(arr: np.ndarray):
    if _xp is np:
        return arr
    return cp.asarray(arr)


def gpu_info_string() -> str:
    if not _cupy_available:
        if _cupy_import_error:
            return (f"CuPy failed to import — GPU unavailable.\n"
                    f"  Reason: {_cupy_import_error}")
        return "CuPy not installed — GPU unavailable."
    lines = []
    if not _curand_available:
        lines.append(
            "  WARNING: libcurand.so not found — random shuffling will use "
            "numpy (all other GPU ops are fine)."
        )
    if not _cupy_devices:
        lines.append("CuPy imported OK but no CUDA devices found/accessible.")
        return "\n".join(lines)
    lines.insert(0, "Available CUDA devices:")
    for did, name in _cupy_devices:
        lines.append(f"  [{did}] {name}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GUTENBERG_URL       = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_ID_MIN    = 1
GUTENBERG_ID_MAX    = 75000
AD_MIN_BYTES        = 40 * 1024
AD_PROBE_WORKERS    = 6
AD_PROBE_TIMEOUT    = 10
AD_DOWNLOAD_TIMEOUT = 60
DISCOVERED_FILE     = "discovered.json"
REJECTED_FILE       = "rejected.json"

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "»", "\u201d", "'"}
NO_SPACE_AFTER  = {"(", "[", "{", "«", "\u201c", "'", "$", "£", "€"}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

CHUNK_RAM_FRACTION = 0.40   # fraction of available RAM for the raw-token buffer

INIT_STD = 0.02
LN_EPS   = 1e-5

DEFAULT_SETTINGS = {
    "input_folder":        "input_files",
    "model_file":          "model.npz",
    "settings_file":       "settings.json",
    # ── Vocabulary ───────────────────────────────────────────────────────
    "vocab_size":          20000,
    "lowercase":           True,
    # ── Architecture (Transformer) ───────────────────────────────────────
    "block_size":          128,     # context length (sequence length)
    "embed_dim":           128,     # d_model
    "n_layers":             4,
    "n_heads":               4,
    "ffn_dim":              512,    # inner dim of the per-block MLP
    # ── Training ─────────────────────────────────────────────────────────
    "epochs":              3,
    "batch_size":           32,     # sequences per batch (not tokens)
    "learning_rate":        0.0003,
    "workers":             max(1, (os.cpu_count() or 2) // 2),
    "single_thread":       False,
    "show_progress":       True,
    # "normal"  — end-to-end backprop through every layer, every step (default)
    # "staged"  — greedy layer-by-layer warm-up first (see train_staged_block
    #             below), THEN a normal end-to-end fine-tune for `epochs`.
    #             Lower peak memory during warm-up (only one block's
    #             activations/grads live at a time); useful for very deep
    #             models or constrained RAM.
    "train_mode":          "normal",
    "stage_epochs":        1,       # epochs per layer during staged warm-up
    # ── Generation ───────────────────────────────────────────────────────
    "max_generate_tokens": 80,
    "temperature":         0.8,
    "top_k":               20,
    # ── GPU ──────────────────────────────────────────────────────────────
    "use_gpu":             False,
    "gpu_device":          0,
    # ── Auto-Data Downloader ─────────────────────────────────────────────
    "auto_download":       False,
    "ad_max_books":        10,
    "ad_max_bytes":        104857600,
    "ad_refill_below":     5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Settings I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_settings(path: str) -> dict:
    s = dict(DEFAULT_SETTINGS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                s.update(loaded)
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    with open(s["settings_file"], "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)


def ensure_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return WORD_RE.findall(text)


def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    out = tokens[0]
    for tok in tokens[1:]:
        if tok in NO_SPACE_BEFORE:
            out += tok
        elif out and out[-1] in NO_SPACE_AFTER:
            out += tok
        else:
            out += " " + tok
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, max_size: int = 20000) -> None:
        self.max_size  = max_size
        self.tok2id: Dict[str, int] = {}
        self.id2tok: List[str]      = []

    def build(self, token_lists: List[List[str]]) -> None:
        counts: Counter = Counter()
        for toks in token_lists:
            counts.update(toks)
        special     = [PAD_TOKEN, UNK_TOKEN]
        most_common = [t for t, _ in counts.most_common(self.max_size - len(special))]
        self.id2tok = special + most_common
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def encode(self, token: str) -> int:
        return self.tok2id.get(token, self.tok2id[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id2tok):
            return self.id2tok[idx]
        return UNK_TOKEN

    @property
    def size(self) -> int:
        return len(self.id2tok)

    def to_dict(self) -> dict:
        return {"max_size": self.max_size, "id2tok": self.id2tok}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        v = cls(max_size=d["max_size"])
        v.id2tok = d["id2tok"]
        v.tok2id = {t: i for i, t in enumerate(v.id2tok)}
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimiser  (generic over any {name: array} param/grad dict)
# ─────────────────────────────────────────────────────────────────────────────

class AdamState:
    def __init__(self, lr: float = 0.0003,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m: Dict[str, object] = {}
        self.v: Dict[str, object] = {}

    def step(self, params: Dict[str, object],
             grads:  Dict[str, object]) -> None:
        self.t += 1
        xp = _xp
        for name, g in grads.items():
            if name not in self.m:
                self.m[name] = xp.zeros_like(params[name])
                self.v[name] = xp.zeros_like(params[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Functional layers: LayerNorm, GELU, causal multi-head self-attention
# All operate through the module-level `_xp` backend (numpy or CuPy).
# Each *_forward returns (out, cache); each *_backward takes (dout, cache)
# and returns gradients.
# ─────────────────────────────────────────────────────────────────────────────

_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_forward(x):
    xp = _xp
    inner = _GELU_C * (x + 0.044715 * x ** 3)
    t = xp.tanh(inner)
    out = 0.5 * x * (1.0 + t)
    return out, (x, t)


def gelu_backward(dout, cache):
    x, t = cache
    dinner_dx = _GELU_C * (1.0 + 3.0 * 0.044715 * x ** 2)
    dgelu_dx  = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner_dx
    return dout * dgelu_dx


def ln_forward(x, g, b, eps: float = LN_EPS):
    xp = _xp
    mu   = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    rstd = 1.0 / xp.sqrt(var + eps)
    xhat = (x - mu) * rstd
    out  = g * xhat + b
    return out, (xhat, rstd, g)


def ln_backward(dout, cache):
    xhat, rstd, g = cache
    D = dout.shape[-1]
    flat_dout = dout.reshape(-1, D)
    flat_xhat = xhat.reshape(-1, D)
    dg = (flat_dout * flat_xhat).sum(axis=0)
    db = flat_dout.sum(axis=0)

    dxhat        = dout * g
    sum_dxhat    = dxhat.sum(axis=-1, keepdims=True)
    sum_dxhat_xh = (dxhat * xhat).sum(axis=-1, keepdims=True)
    dx = rstd / D * (D * dxhat - sum_dxhat - xhat * sum_dxhat_xh)
    return dx, dg, db


def _causal_mask(T):
    xp = _xp
    i = xp.arange(T).reshape(T, 1)
    j = xp.arange(T).reshape(1, T)
    return j > i   # True where attention must be blocked (future tokens)


def attn_forward(x, Wq, Wk, Wv, Wo, bo, n_heads: int):
    """x: [B,T,D] (already LayerNorm'd). Returns out [B,T,D] + cache."""
    xp = _xp
    B, T, D = x.shape
    hd = D // n_heads

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    Qh = Q.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)  # [B,nh,T,hd]
    Kh = K.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    Vh = V.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)

    scores = xp.einsum('bhid,bhjd->bhij', Qh, Kh) / math.sqrt(hd)  # [B,nh,T,T]
    mask = _causal_mask(T)
    neg_inf = xp.asarray(-1e9, dtype=scores.dtype)
    scores = xp.where(mask, neg_inf, scores)

    m = scores.max(axis=-1, keepdims=True)
    e = xp.exp(scores - m)
    attn = e / e.sum(axis=-1, keepdims=True)

    ctxv = xp.einsum('bhij,bhjd->bhid', attn, Vh)               # [B,nh,T,hd]
    ctxv_flat = ctxv.transpose(0, 2, 1, 3).reshape(B, T, D)
    out = ctxv_flat @ Wo + bo

    cache = dict(x=x, Qh=Qh, Kh=Kh, Vh=Vh, attn=attn, ctxv_flat=ctxv_flat,
                 Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, n_heads=n_heads, hd=hd, B=B, T=T, D=D)
    return out, cache


def attn_backward(dout, cache):
    xp = _xp
    x, Qh, Kh, Vh, attn = cache['x'], cache['Qh'], cache['Kh'], cache['Vh'], cache['attn']
    ctxv_flat = cache['ctxv_flat']
    Wq, Wk, Wv, Wo = cache['Wq'], cache['Wk'], cache['Wv'], cache['Wo']
    n_heads, hd, B, T, D = (cache['n_heads'], cache['hd'],
                             cache['B'], cache['T'], cache['D'])

    dWo = ctxv_flat.reshape(-1, D).T @ dout.reshape(-1, D)
    dbo = dout.reshape(-1, D).sum(axis=0)

    dctxv_flat = dout @ Wo.T
    dctxv = dctxv_flat.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)  # [B,nh,T,hd]

    dattn = xp.einsum('bhid,bhjd->bhij', dctxv, Vh)             # [B,nh,T,T]
    dVh   = xp.einsum('bhij,bhid->bhjd', attn, dctxv)           # [B,nh,T,hd]

    dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
    dscores = dscores / math.sqrt(hd)

    dQh = xp.einsum('bhij,bhjd->bhid', dscores, Kh)
    dKh = xp.einsum('bhij,bhid->bhjd', dscores, Qh)

    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

    x_flat = x.reshape(-1, D)
    dWq = x_flat.T @ dQ.reshape(-1, D)
    dWk = x_flat.T @ dK.reshape(-1, D)
    dWv = x_flat.T @ dV.reshape(-1, D)

    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    grads = dict(attn_Wq=dWq, attn_Wk=dWk, attn_Wv=dWv, attn_Wo=dWo, attn_bo=dbo)
    return dx, grads


# ── Transformer block ───────────────────────────────────────────────────────

def block_forward(x, p: Dict[str, object], li: int, n_heads: int):
    pre = f"h{li}."
    ln1_out, ln1_cache = ln_forward(x, p[pre + "ln1_g"], p[pre + "ln1_b"])
    attn_out, attn_cache = attn_forward(
        ln1_out, p[pre + "attn_Wq"], p[pre + "attn_Wk"], p[pre + "attn_Wv"],
        p[pre + "attn_Wo"], p[pre + "attn_bo"], n_heads)
    x2 = x + attn_out

    ln2_out, ln2_cache = ln_forward(x2, p[pre + "ln2_g"], p[pre + "ln2_b"])
    h1 = ln2_out @ p[pre + "mlp_W1"] + p[pre + "mlp_b1"]
    a1, gelu_cache = gelu_forward(h1)
    mlp_out = a1 @ p[pre + "mlp_W2"] + p[pre + "mlp_b2"]
    x3 = x2 + mlp_out

    cache = dict(
        ln1_cache=ln1_cache, attn_cache=attn_cache,
        ln2_cache=ln2_cache, ln2_out=ln2_out,
        a1=a1, gelu_cache=gelu_cache,
        mlp_W1=p[pre + "mlp_W1"], mlp_W2=p[pre + "mlp_W2"],
    )
    return x3, cache


def block_backward(dx3, cache, li: int):
    pre = f"h{li}."
    grads: Dict[str, object] = {}

    # ── MLP branch (x2 -> x3 = x2 + mlp(ln2(x2))) ───────────────────────
    dmlp_out = dx3            # gradient into the MLP branch
    a1     = cache['a1']
    mlp_W1 = cache['mlp_W1']
    mlp_W2 = cache['mlp_W2']

    grads[pre + "mlp_W2"] = a1.reshape(-1, a1.shape[-1]).T @ dmlp_out.reshape(-1, dmlp_out.shape[-1])
    grads[pre + "mlp_b2"] = dmlp_out.reshape(-1, dmlp_out.shape[-1]).sum(axis=0)

    da1 = dmlp_out @ mlp_W2.T
    dh1 = gelu_backward(da1, cache['gelu_cache'])

    ln2_out = cache['ln2_out']
    grads[pre + "mlp_W1"] = ln2_out.reshape(-1, ln2_out.shape[-1]).T @ dh1.reshape(-1, dh1.shape[-1])
    grads[pre + "mlp_b1"] = dh1.reshape(-1, dh1.shape[-1]).sum(axis=0)

    dln2_out = dh1 @ mlp_W1.T
    dx2_from_ln2, dln2_g, dln2_b = ln_backward(dln2_out, cache['ln2_cache'])
    grads[pre + "ln2_g"] = dln2_g
    grads[pre + "ln2_b"] = dln2_b

    dx2 = dx3 + dx2_from_ln2   # residual path (dx3) + branch path through LN2

    # ── Attention branch (x -> x2 = x + attn(ln1(x))) ────────────────────
    dattn_out = dx2
    dln1_out, attn_grads = attn_backward(dattn_out, cache['attn_cache'])
    for k, v in attn_grads.items():
        grads[pre + k] = v

    dx_from_ln1, dln1_g, dln1_b = ln_backward(dln1_out, cache['ln1_cache'])
    grads[pre + "ln1_g"] = dln1_g
    grads[pre + "ln1_b"] = dln1_b

    dx = dx2 + dx_from_ln1     # residual path (dx2) + branch path through LN1
    return dx, grads


# ─────────────────────────────────────────────────────────────────────────────
# GPT model
# ─────────────────────────────────────────────────────────────────────────────

def gpt_forward(p: Dict[str, object], ctx_ids, n_layers: int, n_heads: int):
    """ctx_ids: [B,T] int array on the active backend. Returns logits [B,T,V] + cache."""
    B, T = ctx_ids.shape
    tok_emb = p['wte'][ctx_ids]                 # [B,T,D]
    pos_emb = p['wpe'][:T][None, :, :]           # [1,T,D]
    x = tok_emb + pos_emb

    block_caches = []
    for li in range(n_layers):
        x, c = block_forward(x, p, li, n_heads)
        block_caches.append(c)

    lnf_out, lnf_cache = ln_forward(x, p['lnf_g'], p['lnf_b'])
    logits = lnf_out @ p['head_W'] + p['head_b']

    cache = dict(ctx_ids=ctx_ids, T=T, block_caches=block_caches,
                 lnf_cache=lnf_cache, lnf_out=lnf_out)
    return logits, cache


def gpt_backward(p: Dict[str, object], dlogits, cache, n_layers: int, n_heads: int):
    xp = _xp
    lnf_out = cache['lnf_out']
    B, T, D = lnf_out.shape
    V = dlogits.shape[-1]

    grads: Dict[str, object] = {}
    grads['head_W'] = lnf_out.reshape(-1, D).T @ dlogits.reshape(-1, V)
    grads['head_b'] = dlogits.reshape(-1, V).sum(axis=0)

    dlnf_out = dlogits @ p['head_W'].T
    dx, dlnf_g, dlnf_b = ln_backward(dlnf_out, cache['lnf_cache'])
    grads['lnf_g'] = dlnf_g
    grads['lnf_b'] = dlnf_b

    for li in reversed(range(n_layers)):
        dx, block_grads = block_backward(dx, cache['block_caches'][li], li)
        grads.update(block_grads)

    ctx_ids = cache['ctx_ids']
    grads['wte'] = xp.zeros_like(p['wte'])
    xp.add.at(grads['wte'], ctx_ids, dx)

    grads['wpe'] = xp.zeros_like(p['wpe'])
    grads['wpe'][:T] = dx.sum(axis=0)

    return grads


def compute_loss_and_dlogits(logits, targets):
    """logits: [B,T,V], targets: [B,T] int. Returns (loss: float, dlogits: [B,T,V])."""
    xp = _xp
    B, T, V = logits.shape
    logits_s = logits - logits.max(axis=-1, keepdims=True)
    exp_l = xp.exp(logits_s)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    bi = xp.arange(B).reshape(B, 1)
    ti = xp.arange(T).reshape(1, T)
    target_probs = probs[bi, ti, targets]
    loss = -xp.log(target_probs + 1e-12).mean()

    dlogits = probs.copy()
    dlogits[bi, ti, targets] -= 1.0
    dlogits = dlogits / (B * T)
    return float(_to_numpy(loss)), dlogits


# ─────────────────────────────────────────────────────────────────────────────
# Staged (layer-by-layer) training
#
# Normal mode backprops through the whole stack every step. Staged mode
# instead trains one block at a time: blocks before it run forward-only
# (their caches are thrown away immediately, so nothing about them is kept
# for backprop) and a small throwaway "stage head" (LayerNorm + Linear ->
# vocab) sits on top of the block being trained so it has something to fit
# against. Only that one block's activations/gradients are ever live at
# once. Once every block has had its warm-up pass, the stage head is
# discarded and a normal end-to-end fine-tune trains the whole stack
# (plus the real head) together.
# ─────────────────────────────────────────────────────────────────────────────

def _init_stage_head(embed_dim: int, vocab_size: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        'stage_ln_g':   np.ones(embed_dim, dtype=np.float32),
        'stage_ln_b':   np.zeros(embed_dim, dtype=np.float32),
        'stage_head_W': (rng.standard_normal((embed_dim, vocab_size)) * INIT_STD).astype(np.float32),
        'stage_head_b': np.zeros(vocab_size, dtype=np.float32),
    }


def _forward_frozen_prefix(p: Dict[str, object], ctx_ids, upto_li: int, n_heads: int):
    """Forward through blocks [0, upto_li) with no cache kept — nothing here
    is ever backpropagated through, so activations aren't retained."""
    T = ctx_ids.shape[1]
    x = p['wte'][ctx_ids] + p['wpe'][:T][None, :, :]
    for li in range(upto_li):
        x, _ = block_forward(x, p, li, n_heads)   # cache discarded on purpose
    return x


def _stage_forward(p: Dict[str, object], stage_p: Dict[str, object],
                    ctx_ids, li: int, n_heads: int):
    x_in = _forward_frozen_prefix(p, ctx_ids, li, n_heads)     # frozen blocks 0..li-1
    x_out, block_cache = block_forward(x_in, p, li, n_heads)   # the block being trained
    ln_out, ln_cache = ln_forward(x_out, stage_p['stage_ln_g'], stage_p['stage_ln_b'])
    logits = ln_out @ stage_p['stage_head_W'] + stage_p['stage_head_b']
    cache = dict(block_cache=block_cache, ln_cache=ln_cache, ln_out=ln_out)
    return logits, cache


def _stage_backward(stage_p: Dict[str, object], dlogits, cache, li: int):
    ln_out = cache['ln_out']
    D = ln_out.shape[-1]
    V = dlogits.shape[-1]

    stage_grads: Dict[str, object] = {}
    stage_grads['stage_head_W'] = ln_out.reshape(-1, D).T @ dlogits.reshape(-1, V)
    stage_grads['stage_head_b'] = dlogits.reshape(-1, V).sum(axis=0)

    dln_out = dlogits @ stage_p['stage_head_W'].T
    dx_out, dln_g, dln_b = ln_backward(dln_out, cache['ln_cache'])
    stage_grads['stage_ln_g'] = dln_g
    stage_grads['stage_ln_b'] = dln_b

    # dx into the frozen prefix is thrown away — nothing before block li
    # is being trained during its stage.
    _dx_in, block_grads = block_backward(dx_out, cache['block_cache'], li)
    return block_grads, stage_grads


def train_staged_block(model: "GPT", vocab: Vocabulary, files: List[str],
                        chunk_tokens: int, li: int, n_heads: int, block_size: int,
                        batch_size: int, lr: float, lowercase: bool,
                        stage_epochs: int, show_prog: bool) -> None:
    """Run the staged warm-up for a single block `li`, mutating model.params
    in place. Everything else in model.params is left untouched (frozen)."""
    stage_p = _init_stage_head(model.hp['embed_dim'], vocab.size, seed=1234 + li)
    stage_p = {k: _to_xp(v) for k, v in stage_p.items()}
    adam_block = AdamState(lr=lr)
    adam_stage = AdamState(lr=lr)

    for ep in range(1, stage_epochs + 1):
        ep_loss = 0.0
        ep_batches = 0
        for chunk_toks in _iter_file_chunks(files, lowercase, chunk_tokens):
            X_cpu, Y_cpu = build_dataset_from_tokens(chunk_toks, vocab, block_size)
            N = X_cpu.shape[0]
            if N == 0:
                continue
            perm = np.random.permutation(N)
            X_cpu, Y_cpu = X_cpu[perm], Y_cpu[perm]
            if _xp is not np:
                X_dev, Y_dev = _to_xp(X_cpu), _to_xp(Y_cpu)
            else:
                X_dev, Y_dev = X_cpu, Y_cpu

            for b_start in range(0, N, batch_size):
                Xb = X_dev[b_start: b_start + batch_size]
                Yb = Y_dev[b_start: b_start + batch_size]

                logits, cache = _stage_forward(model.params, stage_p, Xb, li, n_heads)
                loss, dlogits = compute_loss_and_dlogits(logits, Yb)
                block_grads, stage_grads = _stage_backward(stage_p, dlogits, cache, li)
                adam_block.step(model.params, block_grads)
                adam_stage.step(stage_p, stage_grads)

                ep_loss += loss
                ep_batches += 1

        if show_prog:
            print(f"    epoch {ep}/{stage_epochs}  "
                  f"avg loss={ep_loss / max(1, ep_batches):.4f}")
    # stage_p (the throwaway head) simply goes out of scope here — only
    # block li's now-trained weights persist in model.params.


class GPT:
    """Decoder-only transformer. `params` is a flat {name: array} dict so the
    existing generic AdamState (and the save/load code) need no changes."""

    def __init__(self, hp: dict) -> None:
        assert hp['embed_dim'] % hp['n_heads'] == 0, \
            "embed_dim must be divisible by n_heads"
        self.hp = dict(hp)
        self.params = self._init_weights()

    def _init_weights(self) -> Dict[str, np.ndarray]:
        hp = self.hp
        V, D, L, H, F, T = (hp['vocab_size'], hp['embed_dim'], hp['n_layers'],
                             hp['n_heads'], hp['ffn_dim'], hp['block_size'])
        rng = np.random.default_rng(42)
        proj_std = INIT_STD / math.sqrt(2 * max(1, L))

        def randn(shape, std):
            return (rng.standard_normal(shape) * std).astype(np.float32)

        p: Dict[str, np.ndarray] = {}
        p['wte'] = randn((V, D), INIT_STD)
        p['wpe'] = randn((T, D), INIT_STD)

        for li in range(L):
            pre = f"h{li}."
            p[pre + "ln1_g"] = np.ones(D, dtype=np.float32)
            p[pre + "ln1_b"] = np.zeros(D, dtype=np.float32)
            p[pre + "attn_Wq"] = randn((D, D), INIT_STD)
            p[pre + "attn_Wk"] = randn((D, D), INIT_STD)
            p[pre + "attn_Wv"] = randn((D, D), INIT_STD)
            p[pre + "attn_Wo"] = randn((D, D), proj_std)
            p[pre + "attn_bo"] = np.zeros(D, dtype=np.float32)
            p[pre + "ln2_g"] = np.ones(D, dtype=np.float32)
            p[pre + "ln2_b"] = np.zeros(D, dtype=np.float32)
            p[pre + "mlp_W1"] = randn((D, F), INIT_STD)
            p[pre + "mlp_b1"] = np.zeros(F, dtype=np.float32)
            p[pre + "mlp_W2"] = randn((F, D), proj_std)
            p[pre + "mlp_b2"] = np.zeros(D, dtype=np.float32)

        p['lnf_g'] = np.ones(D, dtype=np.float32)
        p['lnf_b'] = np.zeros(D, dtype=np.float32)
        p['head_W'] = randn((D, V), INIT_STD)
        p['head_b'] = np.zeros(V, dtype=np.float32)
        return p

    def to_device(self) -> None:
        self.params = {k: _to_xp(v) for k, v in self.params.items()}

    def to_cpu(self) -> None:
        self.params = {k: _to_numpy(v) for k, v in self.params.items()}

    def forward(self, ctx_ids):
        return gpt_forward(self.params, ctx_ids, self.hp['n_layers'], self.hp['n_heads'])

    def backward(self, dlogits, cache):
        return gpt_backward(self.params, dlogits, cache, self.hp['n_layers'], self.hp['n_heads'])

    def num_params(self) -> int:
        return sum(int(np.prod(_to_numpy(v).shape)) for v in self.params.values())

    def to_npz(self) -> io.BytesIO:
        buf = io.BytesIO()
        cpu_params = {k: _to_numpy(v) for k, v in self.params.items()}
        np.savez_compressed(buf, **cpu_params)
        buf.seek(0)
        return buf

    @classmethod
    def from_npz(cls, buf: io.BytesIO, hp: dict) -> "GPT":
        m = cls(hp)
        data = np.load(buf)
        for k in m.params:
            m.params[k] = data[k]
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Model save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: GPT, vocab: Vocabulary, model_file: str) -> None:
    header       = {"vocab": vocab.to_dict(), "hparams": model.hp}
    header_bytes = json.dumps(header).encode("utf-8")
    weights_buf  = model.to_npz()

    with gzip.open(model_file, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(weights_buf.read())


def load_model(model_file: str) -> Optional[Tuple[GPT, Vocabulary, dict]]:
    if not os.path.exists(model_file):
        return None
    with gzip.open(model_file, "rb") as f:
        raw = f.read()

    hlen   = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8: 8 + hlen])
    weights_buf = io.BytesIO(raw[8 + hlen:])

    vocab = Vocabulary.from_dict(header["vocab"])
    hp    = header["hparams"]
    model = GPT.from_npz(weights_buf, hp)
    return model, vocab, hp


# ─────────────────────────────────────────────────────────────────────────────
# Training data builder  (streaming / chunked)
# ─────────────────────────────────────────────────────────────────────────────

def load_tokens_from_file(path: str, lowercase: bool) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return tokenize(text, lowercase=lowercase)


def _estimate_chunk_size(block_size: int, sample_tokens: int = 500_000) -> int:
    """How many raw tokens to hold in RAM at once (CHUNK_RAM_FRACTION of
    available RAM), independent of how the chunk is later sliced into
    training sequences."""
    if _PSUTIL:
        avail = _psutil.virtual_memory().available
    else:
        avail = 2 * 1024 ** 3

    bytes_per_token = 4  # one int32 id per token, worst case
    budget_bytes    = int(avail * CHUNK_RAM_FRACTION)
    max_tokens      = budget_bytes // bytes_per_token
    return max(block_size * 64, min(max_tokens, 10_000_000))


def build_dataset_from_tokens(tokens: List[str], vocab: Vocabulary,
                               block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build non-overlapping [N, block_size] sequences (X) and their
    shifted-by-one targets (Y) from a flat token list."""
    ids = [vocab.encode(t) for t in tokens]
    n = len(ids)
    n_seq = (n - 1) // block_size
    if n_seq <= 0:
        return (np.empty((0, block_size), dtype=np.int32),
                np.empty((0, block_size), dtype=np.int32))

    usable = n_seq * block_size
    ids_arr = np.asarray(ids[: usable + 1], dtype=np.int32)
    X = ids_arr[:usable].reshape(n_seq, block_size)
    Y = ids_arr[1: usable + 1].reshape(n_seq, block_size)
    return X, Y


def _iter_file_chunks(files: List[str], lowercase: bool,
                      chunk_tokens: int) -> Iterator[List[str]]:
    """Yield flat token lists of <= chunk_tokens tokens, reading files one
    by one. Never holds more than ~2 files' worth of tokens in memory."""
    buf: List[str] = []
    for path in files:
        toks = load_tokens_from_file(path, lowercase)
        buf.extend(toks)
        while len(buf) >= chunk_tokens:
            yield buf[:chunk_tokens]
            buf = buf[chunk_tokens:]
    if buf:
        yield buf


# ─────────────────────────────────────────────────────────────────────────────
# Progress / display helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "=" * width + "]"
    ratio  = max(0.0, min(1.0, done / total))
    filled = int(ratio * width)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def human_num(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.1f}T"


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _sys_usage_str() -> str:
    if not _PSUTIL:
        return ""
    cpu  = _psutil.cpu_percent(interval=None)
    mem  = _psutil.virtual_memory()
    used = mem.used  // (1024 ** 2)
    tot  = mem.total // (1024 ** 2)
    s    = f"CPU {cpu:4.1f}%  RAM {used}/{tot} MB"

    if _xp is not np:
        try:
            pool  = cp.get_default_memory_pool()
            g_used = pool.used_bytes()  // (1024 ** 2)
            g_tot  = pool.total_bytes() // (1024 ** 2)
            s     += f"  GPU-mem {g_used}/{g_tot} MB"
        except Exception:
            pass
    return s


def _print_file_loading_progress(done: int, total: int, token_count: int) -> None:
    bar = format_bar(done, total, width=28)
    sys.stdout.write(
        f"\r  Loading {bar} {done}/{total} files  "
        f"({human_num(token_count)} tokens)" + " " * 4
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Two-pass vocab builder (streaming, low memory)
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_streaming(files: List[str], lowercase: bool,
                           vocab_size: int,
                           workers: int, single_thread: bool,
                           show_prog: bool) -> Tuple[Vocabulary, int]:
    counts: Counter = Counter()
    total_tokens = 0
    n_files = len(files)

    print("  Pass 1/2 — counting tokens for vocabulary…")

    if single_thread:
        for i, path in enumerate(files):
            toks = load_tokens_from_file(path, lowercase)
            counts.update(toks)
            total_tokens += len(toks)
            if show_prog:
                _print_file_loading_progress(i + 1, n_files, total_tokens)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {
                ex.submit(load_tokens_from_file, p, lowercase): idx
                for idx, p in enumerate(files)
            }
            done = 0
            for fut in as_completed(future_map):
                toks = fut.result()
                counts.update(toks)
                total_tokens += len(toks)
                done += 1
                if show_prog:
                    _print_file_loading_progress(done, n_files, total_tokens)

    if show_prog:
        sys.stdout.write("\n")

    special     = [PAD_TOKEN, UNK_TOKEN]
    most_common = [t for t, _ in counts.most_common(vocab_size - len(special))]
    vocab       = Vocabulary(max_size=vocab_size)
    vocab.id2tok = special + most_common
    vocab.tok2id = {t: i for i, t in enumerate(vocab.id2tok)}
    return vocab, total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(settings: dict) -> None:
    folder        = settings["input_folder"]
    lowercase     = bool(settings["lowercase"])
    single_thread = bool(settings.get("single_thread", False))
    show_prog     = bool(settings.get("show_progress", True))
    auto_dl       = bool(settings.get("auto_download", False))

    vocab_size = int(settings["vocab_size"])
    block_size = int(settings["block_size"])
    embed_dim  = int(settings["embed_dim"])
    n_layers   = int(settings["n_layers"])
    n_heads    = int(settings["n_heads"])
    ffn_dim    = int(settings["ffn_dim"])
    epochs     = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    lr         = float(settings["learning_rate"])
    workers    = max(1, int(settings["workers"]))

    if embed_dim % n_heads != 0:
        print(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}).")
        return

    use_gpu    = bool(settings.get("use_gpu", False))
    gpu_device = int(settings.get("gpu_device", 0))
    _set_backend(use_gpu, gpu_device)
    if _xp is not np:
        dev_name = _cupy_devices[gpu_device][1] if gpu_device < len(_cupy_devices) else "?"
        print(f"[GPU] Using CUDA device {gpu_device}: {dev_name}")
        if not _curand_available:
            print("[GPU] Note: libcurand not found — shuffling via numpy (no impact on training).")
    else:
        if use_gpu and not _cupy_available:
            print("[GPU] CuPy not available — falling back to CPU.")
        elif use_gpu:
            print("[GPU] No valid CUDA device found — falling back to CPU.")
        else:
            print("[CPU] Running on CPU.")

    ensure_folder(folder)

    dl_thread: Optional["AutoDownloadThread"] = None
    if auto_dl:
        if single_thread:
            auto_download_blocking(settings)
        else:
            dl_thread = AutoDownloadThread(settings)
            dl_thread.start()

    files = find_text_files(folder)
    if not files:
        if dl_thread:
            print("Waiting for downloader…")
            for _ in range(30):
                time.sleep(2)
                files = find_text_files(folder)
                if files:
                    break
    if not files:
        print(f"No .txt files found in '{folder}'.")
        if dl_thread:
            dl_thread.stop()
        return

    backend_label = f"GPU:{gpu_device}" if _xp is not np else "CPU"
    print(f"\n=== Transformer LM Training  [{backend_label}] ===")
    print(f"Files: {len(files)}  |  Vocab: {vocab_size}  |  "
          f"block_size={block_size}  d_model={embed_dim}  "
          f"layers={n_layers}  heads={n_heads}  ffn={ffn_dim}")
    print(f"Epochs: {epochs}  |  Batch: {batch_size} sequences  |  LR: {lr}")
    if _PSUTIL:
        mem = _psutil.virtual_memory()
        print(f"System RAM: {mem.available // (1024**2)} MB available of "
              f"{mem.total // (1024**2)} MB total")
    print()

    vocab, total_tokens = build_vocab_streaming(
        files, lowercase, vocab_size, workers, single_thread, show_prog
    )
    print(f"  Vocab size: {vocab.size}  |  Total tokens: {human_num(total_tokens)}")

    chunk_tokens = _estimate_chunk_size(block_size)
    n_chunks_approx = max(1, total_tokens // chunk_tokens)
    print(f"  Chunk size: ~{human_num(chunk_tokens)} tokens  "
          f"(≈{n_chunks_approx} chunk(s) per epoch)")

    print("\n  Pass 2/2 — initialising model…")
    hp = dict(vocab_size=vocab.size, block_size=block_size, embed_dim=embed_dim,
              n_layers=n_layers, n_heads=n_heads, ffn_dim=ffn_dim)
    model = GPT(hp)
    model.to_device()
    adam = AdamState(lr=lr)

    print(f"  Parameters: {human_num(model.num_params())}")

    train_mode   = str(settings.get("train_mode", "normal")).lower()
    stage_epochs = int(settings.get("stage_epochs", 1))

    if train_mode == "staged":
        print(f"\n=== Staged warm-up: {n_layers} block(s), "
              f"{stage_epochs} epoch(s) each ===")
        for li in range(n_layers):
            print(f"  -- Stage {li+1}/{n_layers}: training block h{li} "
                  f"(blocks 0..{li-1} frozen) --" if li > 0 else
                  f"  -- Stage {li+1}/{n_layers}: training block h{li} --")
            train_staged_block(model, vocab, files, chunk_tokens, li, n_heads,
                                block_size, batch_size, lr, lowercase,
                                stage_epochs, show_prog)
            model.to_cpu()
            save_model(model, vocab, settings["model_file"])
            model.to_device()
        print(f"\n=== Joint fine-tune: all {n_layers} layers together "
              f"(train_mode=normal from here) ===")

    global_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start   = time.perf_counter()
        epoch_loss    = 0.0
        epoch_batches = 0
        epoch_seqs    = 0

        print(f"\n── Epoch {epoch}/{epochs} ──────────────────────────────────────")

        chunk_idx   = 0
        chunk_start = time.perf_counter()

        for chunk_toks in _iter_file_chunks(files, lowercase, chunk_tokens):
            chunk_idx += 1

            X_cpu, Y_cpu = build_dataset_from_tokens(chunk_toks, vocab, block_size)
            N = X_cpu.shape[0]
            del chunk_toks
            if N == 0:
                continue

            perm  = np.random.permutation(N)
            X_cpu = X_cpu[perm]
            Y_cpu = Y_cpu[perm]

            if _xp is not np:
                X_dev = _to_xp(X_cpu)
                Y_dev = _to_xp(Y_cpu)
                del X_cpu, Y_cpu
            else:
                X_dev = X_cpu
                Y_dev = Y_cpu

            batches_in_chunk = (N + batch_size - 1) // batch_size
            chunk_loss    = 0.0
            chunk_batches = 0

            for b_start in range(0, N, batch_size):
                Xb = X_dev[b_start: b_start + batch_size]
                Yb = Y_dev[b_start: b_start + batch_size]

                logits, cache = model.forward(Xb)
                loss, dlogits = compute_loss_and_dlogits(logits, Yb)
                grads = model.backward(dlogits, cache)
                adam.step(model.params, grads)

                chunk_loss    += loss
                chunk_batches += 1
                epoch_loss    += loss
                epoch_batches += 1
                epoch_seqs    += Xb.shape[0]

                if show_prog and (chunk_batches % 5 == 0
                                  or chunk_batches == batches_in_chunk):
                    elapsed   = max(0.001, time.perf_counter() - chunk_start)
                    avg_loss  = chunk_loss / chunk_batches
                    pct       = chunk_batches / batches_in_chunk
                    bar       = format_bar(chunk_batches, batches_in_chunk, 24)
                    seq_sec   = (chunk_batches * batch_size) / elapsed
                    usage     = _sys_usage_str()
                    sys.stdout.write(
                        f"\r  Chunk {chunk_idx} {bar} {pct*100:5.1f}%  "
                        f"loss={avg_loss:.4f}  {human_num(seq_sec)} seq/s"
                        + (f"  |  {usage}" if usage else "")
                        + " " * 4
                    )
                    sys.stdout.flush()

            if show_prog:
                sys.stdout.write("\n")

            del X_dev, Y_dev

        epoch_elapsed  = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        print(
            f"  Epoch {epoch} done — "
            f"{chunk_idx} chunk(s)  "
            f"avg loss={avg_epoch_loss:.4f}  "
            f"sequences={human_num(epoch_seqs)}  "
            f"time={epoch_elapsed:.1f}s"
        )

        model.to_cpu()
        save_model(model, vocab, settings["model_file"])
        model.to_device()
        print(f"  Checkpoint saved → {settings['model_file']}")

    if dl_thread:
        dl_thread.stop()

    total_elapsed = time.perf_counter() - global_start
    print(f"\nTraining complete in {total_elapsed:.1f}s")
    print(f"Model saved to: {settings['model_file']}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(model: GPT, vocab: Vocabulary,
                  prompt: str, max_tokens: int,
                  temperature: float, top_k: int,
                  lowercase: bool) -> str:
    block_size = model.hp['block_size']

    prompt_toks = tokenize(prompt, lowercase=lowercase)
    ids = [vocab.encode(t) for t in prompt_toks]
    if not ids:
        ids = [vocab.tok2id.get(UNK_TOKEN, 1)]
    ids = ids[-block_size:]

    generated = list(prompt_toks)

    for _ in range(max_tokens):
        ctx_np = np.asarray(ids[-block_size:], dtype=np.int32)[None, :]
        ctx_dev = _to_xp(ctx_np)
        logits, _ = model.forward(ctx_dev)
        last_logits = _to_numpy(logits[0, -1])   # [V]

        temp = max(0.05, float(temperature))
        z = last_logits / temp
        z -= z.max()
        probs = np.exp(z)
        probs /= probs.sum()

        if top_k and top_k > 0:
            k = min(top_k, len(probs))
            top_idx = np.argpartition(probs, -k)[-k:]
            mask = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            probs = mask / mask.sum()

        next_id  = int(np.random.choice(len(probs), p=probs))
        next_tok = vocab.decode(next_id)

        if next_tok in (PAD_TOKEN, UNK_TOKEN):
            probs[vocab.tok2id[PAD_TOKEN]] = 0
            probs[vocab.tok2id[UNK_TOKEN]] = 0
            s = probs.sum()
            if s > 0:
                probs   /= s
                next_id  = int(np.random.choice(len(probs), p=probs))
                next_tok = vocab.decode(next_id)
            else:
                break

        generated.append(next_tok)
        ids.append(next_id)

    return detokenize(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

def chat(settings: dict) -> None:
    use_gpu    = bool(settings.get("use_gpu", False))
    gpu_device = int(settings.get("gpu_device", 0))
    _set_backend(use_gpu, gpu_device)

    result = load_model(settings["model_file"])
    if result is None:
        print("No model found. Train first.")
        return

    model, vocab, hp = result
    model.to_device()

    backend_label = f"GPU:{gpu_device}" if _xp is not np else "CPU"
    print(f"Model loaded [{backend_label}] — vocab={vocab.size}  "
          f"block_size={hp['block_size']}  d_model={hp['embed_dim']}  "
          f"layers={hp['n_layers']}  heads={hp['n_heads']}")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break

        out = generate_text(
            model, vocab,
            prompt      = prompt,
            max_tokens  = int(settings["max_generate_tokens"]),
            temperature = float(settings["temperature"]),
            top_k       = int(settings["top_k"]),
            lowercase   = bool(settings["lowercase"]),
        )
        print(f"\nAI> {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_text_files(folder: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".txt"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Data Downloader  (unchanged from the original — multi-threaded,
# multi-source Gutenberg fetcher with live progress bars)
# ─────────────────────────────────────────────────────────────────────────────

def _load_id_cache(path: str) -> Set[int]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {int(x) for x in data}
        except Exception:
            pass
    return set()


def _save_id_cache(path: str, ids: Set[int]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(ids), f)
    except Exception:
        pass


def _folder_state(folder: str) -> Tuple[int, int]:
    total_bytes = 0
    count = 0
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".txt"):
                count += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return count, total_bytes


def _already_downloaded(folder: str) -> Set[int]:
    present: Set[int] = set()
    for root, _, files in os.walk(folder):
        for name in files:
            m = re.match(r"pg(\d+)\.txt$", name, re.IGNORECASE)
            if m:
                present.add(int(m.group(1)))
    return present


def _probe_id(book_id: int) -> bool:
    url     = GUTENBERG_URL.format(id=book_id)
    headers = {"User-Agent": "neural-lm-probe/1.0"}
    try:
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=AD_PROBE_TIMEOUT) as resp:
            if resp.status == 404:
                return False
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                return int(cl) >= AD_MIN_BYTES
    except Exception:
        return False
    try:
        req = urllib.request.Request(
            url, headers={**headers, "Range": f"bytes=0-{AD_MIN_BYTES}"}
        )
        with urllib.request.urlopen(req, timeout=AD_PROBE_TIMEOUT) as resp:
            chunk = resp.read(AD_MIN_BYTES + 1)
        return len(chunk) >= AD_MIN_BYTES
    except Exception:
        return False


def _download_one_book(book_id: int, folder: str) -> Optional[str]:
    dest = os.path.join(folder, f"pg{book_id}.txt")
    if os.path.exists(dest):
        return None
    url = GUTENBERG_URL.format(id=book_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neural-lm/1.0"})
        with urllib.request.urlopen(req, timeout=AD_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
        if len(data) < AD_MIN_BYTES:
            return None
        with open(dest, "wb") as f:
            f.write(data)
        return dest
    except Exception:
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        return None


def _download_one_book_progress(book_id: int, folder: str,
                                 status_dict: dict,
                                 lock: threading.Lock) -> Optional[str]:
    dest = os.path.join(folder, f"pg{book_id}.txt")
    if os.path.exists(dest):
        with lock:
            status_dict[book_id] = {'done': 0, 'total': 0, 'state': 'exists'}
        return None

    url = GUTENBERG_URL.format(id=book_id)
    with lock:
        status_dict[book_id] = {'done': 0, 'total': 0, 'state': 'connecting'}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neural-lm/1.0"})
        with urllib.request.urlopen(req, timeout=AD_DOWNLOAD_TIMEOUT) as resp:
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl else 0
            with lock:
                status_dict[book_id]['total'] = total
                status_dict[book_id]['state'] = 'downloading'

            chunks = []
            done   = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                done += len(chunk)
                with lock:
                    status_dict[book_id]['done'] = done

        data = b"".join(chunks)
        if len(data) < AD_MIN_BYTES:
            with lock:
                status_dict[book_id]['state'] = 'too_small'
            return None

        with open(dest, "wb") as f:
            f.write(data)
        with lock:
            status_dict[book_id] = {'done': len(data), 'total': len(data), 'state': 'done'}
        return dest

    except Exception:
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        with lock:
            status_dict[book_id] = {'done': 0, 'total': 0, 'state': 'failed'}
        return None


def _render_download_status(status_dict: dict, lock: threading.Lock,
                              n_done: int, n_total: int,
                              started_at: float) -> None:
    with lock:
        snap = dict(status_dict)

    lines = []
    bar_w = 20
    for bid, info in sorted(snap.items()):
        state = info['state']
        done  = info['done']
        total = info['total']

        if state == 'done':
            size = human_bytes(done)
            line = f"  pg{bid:<6}  [{'='*bar_w}] ✓  {size}"
        elif state == 'failed':
            line = f"  pg{bid:<6}  [{'✗':^{bar_w}}] FAILED"
        elif state == 'too_small':
            line = f"  pg{bid:<6}  [{'~':^{bar_w}}] too small, skipped"
        elif state == 'exists':
            line = f"  pg{bid:<6}  [already on disk]"
        elif state == 'connecting':
            line = f"  pg{bid:<6}  [{'…':^{bar_w}}] connecting…"
        else:
            if total > 0:
                ratio  = done / total
                filled = int(ratio * bar_w)
                bar    = "=" * filled + "-" * (bar_w - filled)
                pct    = f"{ratio*100:4.0f}%  {human_bytes(done)}/{human_bytes(total)}"
            else:
                filled = int((done / (AD_MIN_BYTES * 4)) * bar_w)
                filled = min(filled, bar_w - 1)
                bar    = "=" * filled + ">" + "-" * (bar_w - filled - 1)
                pct    = f"~{human_bytes(done)}"
            line = f"  pg{bid:<6}  [{bar}] {pct}"
        lines.append(line)

    elapsed = max(0.001, time.perf_counter() - started_at)
    if n_done > 0 and n_total > 0:
        eta_s = (elapsed / n_done) * (n_total - n_done)
        eta   = f"ETA {eta_s:.0f}s" if eta_s < 3600 else f"ETA {eta_s/60:.1f}m"
    else:
        eta = "ETA …"
    overall = format_bar(n_done, n_total, 30)
    lines.append(f"  Overall {overall} {n_done}/{n_total} books  {eta}  "
                 f"elapsed {elapsed:.0f}s")

    output = "\n".join(lines)
    sys.stdout.write(output)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _discover_ids(n_needed: int, already_have: Set[int],
                  discovered: Set[int], rejected: Set[int],
                  probe_workers: int = AD_PROBE_WORKERS,
                  verbose: bool = False) -> List[int]:
    skip      = already_have | rejected | discovered
    all_ids   = list(range(GUTENBERG_ID_MIN, GUTENBERG_ID_MAX + 1))
    random.shuffle(all_ids)
    unchecked = [i for i in all_ids if i not in skip]

    good: List[int] = []
    batch_size = probe_workers * 4
    idx = 0

    while len(good) < n_needed and idx < len(unchecked):
        batch = unchecked[idx: idx + batch_size]
        idx  += batch_size
        if verbose:
            sys.stdout.write(
                f"\r  Probing…  found {len(good)}/{n_needed} usable IDs   "
            )
            sys.stdout.flush()
        with ThreadPoolExecutor(max_workers=probe_workers) as ex:
            futures = {ex.submit(_probe_id, bid): bid for bid in batch}
            for fut in as_completed(futures):
                bid = futures[fut]
                ok  = False
                try:
                    ok = fut.result()
                except Exception:
                    pass
                (discovered if ok else rejected).add(bid)
                if ok:
                    good.append(bid)

    if verbose:
        sys.stdout.write("\n")
    return good


def _ready_ids(folder: str, n_needed: int,
               verbose: bool = False) -> List[int]:
    discovered = _load_id_cache(DISCOVERED_FILE)
    rejected   = _load_id_cache(REJECTED_FILE)
    have       = _already_downloaded(folder)

    queued = [i for i in discovered if i not in have]
    random.shuffle(queued)

    if len(queued) < n_needed:
        still_needed = n_needed - len(queued)
        if verbose:
            print(f"  Discovery: probing for {still_needed} more usable IDs…")
        new_good = _discover_ids(still_needed, have, discovered, rejected,
                                 verbose=verbose)
        _save_id_cache(DISCOVERED_FILE, discovered)
        _save_id_cache(REJECTED_FILE,   rejected)
        queued.extend(new_good)

    random.shuffle(queued)
    return queued[:n_needed]


def auto_download_blocking(settings: dict,
                            label_prefix: str = "",
                            multithreaded: Optional[bool] = None) -> None:
    folder    = settings["input_folder"]
    max_books = int(settings["ad_max_books"])
    max_bytes = int(settings["ad_max_bytes"])
    workers   = max(1, int(settings.get("workers", 4)))
    if multithreaded is None:
        multithreaded = not bool(settings.get("single_thread", False))

    ensure_folder(folder)
    count, used = _folder_state(folder)
    if count >= max_books or used >= max_bytes:
        print(f"{label_prefix}Cap already reached ({count}/{max_books} books).")
        return

    slots = max_books - count
    print(f"{label_prefix}Auto-download: {count}/{max_books} books — "
          f"fetching up to {slots} more…")

    ids = _ready_ids(folder, n_needed=slots * 2, verbose=True)
    if not ids:
        print(f"{label_prefix}No usable IDs found.")
        return

    ids = ids[:slots]

    if multithreaded and len(ids) > 1:
        _download_books_parallel(ids, folder, max_books, max_bytes, workers)
    else:
        _download_books_serial(ids, folder, max_books, max_bytes)

    count, used = _folder_state(folder)
    print(f"\n{label_prefix}Done: {count} books, {used//1048576} MB.")


def _download_books_serial(ids: List[int], folder: str,
                            max_books: int, max_bytes: int) -> None:
    rejected_local: Set[int] = set()
    n_total = len(ids)
    for i, book_id in enumerate(ids):
        count, used = _folder_state(folder)
        if count >= max_books or used >= max_bytes:
            break
        bar = format_bar(i, n_total, 28)
        sys.stdout.write(f"\r  {bar}  Downloading pg{book_id}.txt …" + " " * 10)
        sys.stdout.flush()
        path = _download_one_book(book_id, folder)
        if path:
            size = os.path.getsize(path)
            sys.stdout.write(f"\r  ✓ pg{book_id}.txt  ({size//1024} KB)" + " " * 20 + "\n")
        else:
            sys.stdout.write(f"\r  ✗ pg{book_id}.txt  (failed)" + " " * 20 + "\n")
            rejected_local.add(book_id)
        sys.stdout.flush()

    _flush_rejected(rejected_local)


def _download_books_parallel(ids: List[int], folder: str,
                              max_books: int, max_bytes: int,
                              workers: int) -> None:
    n_total       = len(ids)
    status_dict: dict = {}
    lock          = threading.Lock()
    n_done        = 0
    rejected_local: Set[int] = set()
    started_at    = time.perf_counter()

    _prev_lines   = [0]

    def _refresh() -> None:
        if _prev_lines[0] > 0:
            sys.stdout.write(f"\033[{_prev_lines[0]}A\033[J")
        _render_download_status(status_dict, lock,
                                 n_done, n_total, started_at)
        with lock:
            _prev_lines[0] = len(status_dict) + 1

    print(f"  Downloading {n_total} books with {min(workers, n_total)} threads…\n")

    with ThreadPoolExecutor(max_workers=min(workers, n_total)) as ex:
        future_map = {}
        submitted = 0
        for bid in ids:
            count, used = _folder_state(folder)
            if count + submitted >= max_books or used >= max_bytes:
                break
            fut = ex.submit(
                _download_one_book_progress,
                bid, folder, status_dict, lock
            )
            future_map[fut] = bid
            submitted += 1

        display_thread_stop = threading.Event()

        def _display_loop():
            while not display_thread_stop.is_set():
                _refresh()
                time.sleep(0.25)

        disp = threading.Thread(target=_display_loop, daemon=True)
        disp.start()

        for fut in as_completed(future_map):
            bid  = future_map[fut]
            try:
                path = fut.result()
            except Exception:
                path = None
            if path is None:
                with lock:
                    st = status_dict.get(bid, {}).get('state', '')
                if st not in ('exists', 'done'):
                    rejected_local.add(bid)
            n_done += 1

        display_thread_stop.set()
        disp.join()

    if _prev_lines[0] > 0:
        sys.stdout.write(f"\033[{_prev_lines[0]}A\033[J")
    _render_download_status(status_dict, lock, n_done, n_total, started_at)

    _flush_rejected(rejected_local)


def _flush_rejected(rejected_local: Set[int]) -> None:
    if rejected_local:
        rej  = _load_id_cache(REJECTED_FILE)
        disc = _load_id_cache(DISCOVERED_FILE)
        rej  |= rejected_local
        disc -= rejected_local
        _save_id_cache(REJECTED_FILE,   rej)
        _save_id_cache(DISCOVERED_FILE, disc)


class AutoDownloadThread(threading.Thread):
    POLL_INTERVAL = 5

    def __init__(self, settings: dict) -> None:
        super().__init__(daemon=True, name="AutoDownloader")
        self._s        = settings
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        folder       = self._s["input_folder"]
        max_books    = int(self._s["ad_max_books"])
        max_bytes    = int(self._s["ad_max_bytes"])
        refill_below = int(self._s["ad_refill_below"])

        ensure_folder(folder)
        queue: List[int] = _ready_ids(folder, n_needed=max_books * 2, verbose=False)
        rejected_local: Set[int] = set()

        while not self._stop_evt.is_set():
            count, used = _folder_state(folder)
            if (count < refill_below) or (used < max_bytes and count < max_books):
                if not queue:
                    queue = _ready_ids(folder, n_needed=max_books * 2, verbose=False)
                while queue and not self._stop_evt.is_set():
                    count, used = _folder_state(folder)
                    if count >= max_books or used >= max_bytes:
                        break
                    bid  = queue.pop(0)
                    path = _download_one_book(bid, folder)
                    if path is None:
                        rejected_local.add(bid)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

        _flush_rejected(rejected_local)


# ─────────────────────────────────────────────────────────────────────────────
# Settings menu
# ─────────────────────────────────────────────────────────────────────────────

def show_settings(s: dict) -> None:
    count, used = _folder_state(s["input_folder"])
    disc = _load_id_cache(DISCOVERED_FILE)
    rej  = _load_id_cache(REJECTED_FILE)
    print("\nCurrent settings:")
    groups = [
        ("Files",        ["input_folder", "model_file", "lowercase"]),
        ("Vocabulary",   ["vocab_size"]),
        ("Architecture", ["block_size", "embed_dim", "n_layers", "n_heads", "ffn_dim"]),
        ("Training",     ["epochs", "batch_size", "learning_rate",
                          "workers", "single_thread", "show_progress",
                          "train_mode", "stage_epochs"]),
        ("Generation",   ["max_generate_tokens", "temperature", "top_k"]),
    ]
    for label, keys in groups:
        print(f"  ── {label}")
        for k in keys:
            print(f"    {k}: {s[k]}")

    use_gpu    = bool(s.get("use_gpu", False))
    gpu_device = int(s.get("gpu_device", 0))
    print(f"  ── GPU")
    print(f"    use_gpu:    {'ON' if use_gpu else 'OFF'}")
    if _cupy_available:
        if not _curand_available:
            print("    WARNING: libcurand.so missing — shuffle uses numpy fallback.")
        if _cupy_devices:
            for did, name in _cupy_devices:
                marker = " ◀ selected" if (use_gpu and did == gpu_device) else ""
                print(f"    [{did}] {name}{marker}")
        else:
            print("    (CuPy installed but no CUDA devices found)")
    else:
        if _cupy_import_error:
            print(f"    (CuPy import failed: {_cupy_import_error[:80]})")
        else:
            print("    (CuPy not installed — install cupy-cuda11x or cupy-cuda12x)")

    ad_on = s.get("auto_download", False)
    mt    = not bool(s.get("single_thread", False))
    print(f"  ── Auto-Data Downloader")
    print(f"    auto_download:   {'ON' if ad_on else 'OFF'}")
    print(f"    multi-thread DL: {'ON' if mt else 'OFF'} (workers: {s['workers']})")
    print(f"    ad_max_books:    {s['ad_max_books']}")
    print(f"    ad_max_bytes:    {s['ad_max_bytes']//1048576} MB")
    print(f"    ad_refill_below: {s['ad_refill_below']}")
    print(f"    folder now:      {count} books, {used//1048576} MB")
    print(f"    discovered IDs:  {len(disc)}   rejected: {len(rej)}")


def _gpu_submenu(s: dict) -> None:
    while True:
        use_gpu    = bool(s.get("use_gpu", False))
        gpu_device = int(s.get("gpu_device", 0))

        print(f"\n── GPU Settings ────────────────────────────────────────")
        print(f"  Status: {'ON' if use_gpu else 'OFF'}  |  "
              f"Selected device: {gpu_device}")
        print()
        print(gpu_info_string())
        print()
        print("1) Toggle GPU ON/OFF")
        print("2) Select GPU device")
        print("3) Run GPU diagnostics")
        print("0) Back")

        c = input("> ").strip()

        if c == "3":
            print(gpu_info_string())
        elif c == "1":
            if not _cupy_available:
                if _cupy_import_error:
                    print(f"CuPy failed to import: {_cupy_import_error[:120]}")
                else:
                    print("CuPy is not installed.  "
                          "Install with:  pip install cupy-cuda11x  "
                          "(or cupy-cuda12x for CUDA 12)")
            elif not _cupy_devices:
                print("No CUDA devices detected.")
            else:
                s["use_gpu"] = not use_gpu
                print("GPU:", "ON" if s["use_gpu"] else "OFF")
        elif c == "2":
            if not _cupy_available or not _cupy_devices:
                print("No CUDA devices available.")
            else:
                print("Device IDs:")
                for did, name in _cupy_devices:
                    print(f"  {did}  {name}")
                v = input("Enter device ID: ").strip()
                try:
                    n = int(v)
                    if any(d == n for d, _ in _cupy_devices):
                        s["gpu_device"] = n
                        print(f"GPU device set to {n}.")
                    else:
                        print("Invalid device ID.")
                except ValueError:
                    print("Invalid input.")
        elif c == "0":
            save_settings(s)
            return
        else:
            print("Invalid.")

        save_settings(s)


def _auto_dl_submenu(s: dict) -> None:
    while True:
        count, used = _folder_state(s["input_folder"])
        disc = _load_id_cache(DISCOVERED_FILE)
        rej  = _load_id_cache(REJECTED_FILE)
        ad_on = s.get("auto_download", False)
        mt    = not bool(s.get("single_thread", False))
        print(f"\n── Auto-Data Downloader ────────────────────────────────")
        print(f"  Status: {'ON' if ad_on else 'OFF'}  |  "
              f"{count}/{s['ad_max_books']} books  "
              f"{used//1048576}/{s['ad_max_bytes']//1048576} MB")
        print(f"  Multi-thread DL: {'ON' if mt else 'OFF'}  |  "
              f"Workers: {s['workers']}")
        print(f"  Discovered: {len(disc)}   Rejected: {len(rej)}   "
              f"Probe range: {GUTENBERG_ID_MIN}–{GUTENBERG_ID_MAX}   "
              f"Min size: {AD_MIN_BYTES//1024} KB")
        print()
        print("1) Toggle ON/OFF")
        print("2) Max books")
        print("3) Max total MB")
        print("4) Refill-below threshold")
        print("5) Download now (multi-thread if enabled)")
        print("6) Download now (single-thread)")
        print("7) Probe for new IDs now")
        print("8) Clear caches")
        print("0) Back")

        c = input("> ").strip()

        if c == "1":
            s["auto_download"] = not bool(s.get("auto_download", False))
            print("Auto-DL:", "ON" if s["auto_download"] else "OFF")
        elif c == "2":
            v = input("Max books (1-500): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 500:
                    s["ad_max_books"] = n
            except ValueError:
                pass
        elif c == "3":
            v = input("Max MB (1-2000): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 2000:
                    s["ad_max_bytes"] = n * 1048576
            except ValueError:
                pass
        elif c == "4":
            v = input("Refill below N books: ").strip()
            try:
                n = int(v)
                if n >= 0:
                    s["ad_refill_below"] = n
            except ValueError:
                pass
        elif c == "5":
            save_settings(s)
            auto_download_blocking(s, multithreaded=mt)
        elif c == "6":
            save_settings(s)
            auto_download_blocking(s, multithreaded=False)
        elif c == "7":
            folder = s["input_folder"]
            have   = _already_downloaded(folder)
            disc2  = _load_id_cache(DISCOVERED_FILE)
            rej2   = _load_id_cache(REJECTED_FILE)
            print("Probing… (Ctrl-C to stop)")
            try:
                _discover_ids(20, have, disc2, rej2, verbose=True)
                _save_id_cache(DISCOVERED_FILE, disc2)
                _save_id_cache(REJECTED_FILE,   rej2)
                print(f"Done. {len(disc2)} total discovered IDs.")
            except KeyboardInterrupt:
                _save_id_cache(DISCOVERED_FILE, disc2)
                _save_id_cache(REJECTED_FILE,   rej2)
                print("\nInterrupted — partial results saved.")
        elif c == "8":
            if input("Clear both caches? (y/N): ").strip().lower() == "y":
                _save_id_cache(DISCOVERED_FILE, set())
                _save_id_cache(REJECTED_FILE,   set())
                print("Cleared.")
        elif c == "0":
            save_settings(s)
            return
        else:
            print("Invalid.")

        save_settings(s)


def settings_menu(s: dict) -> None:
    while True:
        show_settings(s)
        print("\nSettings menu:")
        print(" Files / vocab")
        print("  1) Input folder")
        print("  2) Model file")
        print("  3) Toggle lowercase")
        print("  4) Vocab size")
        print(" Architecture")
        print("  5) Block size (context length)")
        print("  6) Embedding dim (d_model)")
        print("  7) FFN inner dim")
        print("  n) Number of layers")
        print("  h) Number of attention heads")
        print(" Training")
        print("  8) Epochs")
        print("  9) Batch size (sequences)")
        print("  l) Learning rate")
        print("  w) Worker count")
        print("  t) Toggle single-thread")
        print("  p) Toggle progress display")
        print("  m) Toggle training mode (normal/staged)")
        print("  s) Stage epochs (staged mode)")
        print(" Generation")
        print("  g) Max generated tokens")
        print("  e) Temperature")
        print("  k) Top-k")
        print(" Other")
        print("  u) GPU settings")
        print("  a) Auto-Data Downloader")
        print("  0) Back")

        c = input("> ").strip().lower()

        if c == "1":
            v = input("Input folder: ").strip()
            if v:
                s["input_folder"] = v
                ensure_folder(v)
        elif c == "2":
            v = input("Model file: ").strip()
            if v:
                s["model_file"] = v
        elif c == "3":
            s["lowercase"] = not bool(s["lowercase"])
            print("Lowercase:", s["lowercase"])
        elif c == "4":
            v = input("Vocab size (1000-100000): ").strip()
            try:
                n = int(v)
                if 1000 <= n <= 100000:
                    s["vocab_size"] = n
            except ValueError:
                pass
        elif c == "5":
            v = input("Block size / context length (8-4096): ").strip()
            try:
                n = int(v)
                if 8 <= n <= 4096:
                    s["block_size"] = n
            except ValueError:
                pass
        elif c == "6":
            v = input("embed_dim / d_model (16-4096): ").strip()
            try:
                n = int(v)
                if 16 <= n <= 4096:
                    s["embed_dim"] = n
            except ValueError:
                pass
        elif c == "7":
            v = input("FFN inner dim (16-16384): ").strip()
            try:
                n = int(v)
                if 16 <= n <= 16384:
                    s["ffn_dim"] = n
            except ValueError:
                pass
        elif c == "n":
            v = input("Number of layers (1-96): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 96:
                    s["n_layers"] = n
            except ValueError:
                pass
        elif c == "h":
            v = input("Number of attention heads (1-64): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 64:
                    if s["embed_dim"] % n == 0:
                        s["n_heads"] = n
                    else:
                        print(f"embed_dim ({s['embed_dim']}) must be divisible by n_heads.")
            except ValueError:
                pass
        elif c == "8":
            v = input("Epochs (1-100): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 100:
                    s["epochs"] = n
            except ValueError:
                pass
        elif c == "9":
            v = input("Batch size, sequences (1-1024): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 1024:
                    s["batch_size"] = n
            except ValueError:
                pass
        elif c == "l":
            v = input("Learning rate (e.g. 0.0003): ").strip()
            try:
                x = float(v)
                if 0 < x < 1:
                    s["learning_rate"] = x
            except ValueError:
                pass
        elif c == "w":
            v = input("Workers: ").strip()
            try:
                n = int(v)
                if n >= 1:
                    s["workers"] = n
            except ValueError:
                pass
        elif c == "t":
            s["single_thread"] = not bool(s.get("single_thread", False))
            print("Single-thread:", s["single_thread"])
        elif c == "p":
            s["show_progress"] = not bool(s["show_progress"])
            print("Show progress:", s["show_progress"])
        elif c == "m":
            cur = str(s.get("train_mode", "normal")).lower()
            s["train_mode"] = "staged" if cur != "staged" else "normal"
            print("Training mode:", s["train_mode"])
        elif c == "s":
            v = input("Stage epochs (1-50): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 50:
                    s["stage_epochs"] = n
            except ValueError:
                pass
        elif c == "g":
            v = input("Max generated tokens: ").strip()
            try:
                n = int(v)
                if n >= 1:
                    s["max_generate_tokens"] = n
            except ValueError:
                pass
        elif c == "e":
            v = input("Temperature (0.1-3.0): ").strip()
            try:
                x = float(v)
                if 0.1 <= x <= 3.0:
                    s["temperature"] = x
            except ValueError:
                pass
        elif c == "k":
            v = input("Top-k (0=off): ").strip()
            try:
                n = int(v)
                if n >= 0:
                    s["top_k"] = n
            except ValueError:
                pass
        elif c == "u":
            _gpu_submenu(s)
        elif c == "a":
            _auto_dl_submenu(s)
        elif c == "0":
            save_settings(s)
            print("Settings saved.")
            return
        else:
            print("Invalid choice.")

        save_settings(s)
        print("Saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    s = load_settings(DEFAULT_SETTINGS["settings_file"])
    ensure_folder(s["input_folder"])

    parser = argparse.ArgumentParser(
        description="Pure Python + numpy Decoder-only Transformer LM (GPU-capable via CuPy)"
    )
    parser.add_argument(
        "mode", nargs="?",
        choices=["train", "chat", "settings", "download"],
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(s);         return
    if args.mode == "chat":
        chat(s);          return
    if args.mode == "settings":
        settings_menu(s); return
    if args.mode == "download":
        auto_download_blocking(s); return

    while True:
        count, used = _folder_state(s["input_folder"])
        ad_ind = (f" [Auto-DL ON | {count} books, {used//1048576}MB]"
                  if s.get("auto_download") else "")
        use_gpu    = bool(s.get("use_gpu", False))
        gpu_device = int(s.get("gpu_device", 0))
        if use_gpu and _cupy_available and _cupy_devices:
            dev_name = _cupy_devices[gpu_device][1] if gpu_device < len(_cupy_devices) else "?"
            gpu_ind  = f" [GPU:{gpu_device} {dev_name}]"
            if not _curand_available:
                gpu_ind += " (curand missing—shuffle via numpy)"
        elif use_gpu:
            if not _cupy_available:
                reason = f": {_cupy_import_error[:60]}" if _cupy_import_error else " (not installed)"
                gpu_ind = f" [GPU: import failed{reason} — using CPU]"
            elif not _cupy_devices:
                gpu_ind = " [GPU: no devices found — using CPU]"
            else:
                gpu_ind = " [GPU: unavailable — using CPU]"
        else:
            gpu_ind = " [CPU]"

        model_exists = os.path.exists(s["model_file"])
        print(f"\n=== Transformer LM ==={ad_ind}{gpu_ind}")
        print(f"  model: {s['model_file']}"
              + (" ✓" if model_exists else " (not trained yet)"))
        print("1) Train")
        print("2) Chat / Generate")
        print("3) Settings")
        print("4) Download books now")
        print("5) Exit")

        c = input("> ").strip()
        if c == "1":
            train(s)
        elif c == "2":
            chat(s)
        elif c == "3":
            settings_menu(s)
        elif c == "4":
            auto_download_blocking(s)
        elif c == "5":
            save_settings(s)
            break
        else:
            print("Invalid.")


if __name__ == "__main__":
    main()