"""
adapters/frameworks/encoder.py

EmbeddingEncoder — local semantic embeddings for CASPIAN channel feature vectors.

Model: sentence-transformers/all-MiniLM-L6-v2
Backend: FastEmbed (ONNX Runtime, quantized) — no remote API calls, no PyTorch.
Cache: SHA-256(text) -> projected embedding, avoids re-encoding repeated content.
Projection: fixed Gaussian random matrix (seeded) from 384-d -> target d_c.

On first use the model downloads once to ~/.cache/fastembed/ (~24 MB).
All subsequent encodes are local ONNX inference, typically <5ms per unique text.

Channel feature vector layout (paper §3, d_c values):
  comm (16-d): [semantic_proj x12, token_count_norm, text_len_norm, role_flag, handoff_flag]
  mem  (12-d): [semantic_proj x10, write_flag, content_len_norm]
  tool (12-d): [semantic_proj x8,  tool_name_proj x2, arg_len_norm, output_len_norm]
  exec  (8-d): [prompt_tok_norm, completion_tok_norm, latency_norm, token_ratio, zeros x4]

exec does not use semantic embeddings — it encodes execution metadata only.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

import numpy as np

try:
    import os as _os
    import warnings as _warnings
    # Suppress ONNX Runtime thread affinity warnings (cosmetic on HPC clusters)
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", "1")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except ImportError:
    _FASTEMBED_AVAILABLE = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM  = 384   # output dimension of all-MiniLM-L6-v2

# Channel output dimensions — must match tensor.py CHANNEL_DIM
CHANNEL_DIM: dict[str, int] = {
    "comm": 16,
    "mem":  12,
    "tool": 12,
    "exec": 8,
}

# Semantic projection dimensions per channel (remaining dims are structured features)
_SEMANTIC_DIM: dict[str, int] = {
    "comm": 12,
    "mem":  10,
    "tool": 8,    # 8 for joint name+args+output, 2 for tool name alone
}


class EmbeddingEncoder:
    """
    Local semantic embedding encoder with projection and caching.

    One instance shared across all adapter instances within a process.
    Thread-safe for read operations (cache lookup), not for concurrent writes —
    acceptable since CASPIAN processes one turn at a time.
    """

    # Class-level model instance — loaded once per process
    _model: ClassVar["TextEmbedding | None"] = None
    # Fixed random projection matrices per (source_dim, target_dim) — seeded, stable
    _projections: ClassVar[dict[tuple[int, int], np.ndarray]] = {}

    def __init__(self) -> None:
        # Per-instance embedding cache: sha256(text) -> projected vector per dim
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    @classmethod
    def _get_model(cls) -> "TextEmbedding":
        if not _FASTEMBED_AVAILABLE:
            raise RuntimeError(
                "fastembed is not installed. Run: pip install fastembed"
            )
        if cls._model is None:
            cls._model = TextEmbedding(MODEL_NAME)
        return cls._model

    @classmethod
    def _get_projection(cls, src_dim: int, tgt_dim: int) -> np.ndarray:
        """
        Fixed Gaussian random projection matrix, shape (src_dim, tgt_dim).
        Seeded by (src_dim, tgt_dim) for reproducibility.
        Normalized so projected vectors preserve approximate norms.
        """
        key = (src_dim, tgt_dim)
        if key not in cls._projections:
            rng = np.random.default_rng(seed=src_dim * 10000 + tgt_dim)
            P   = rng.standard_normal((src_dim, tgt_dim))
            P  /= np.sqrt(tgt_dim)   # Johnson-Lindenstrauss normalization
            cls._projections[key] = P
        return cls._projections[key]

    def _embed_and_project(self, text: str, out_dim: int) -> np.ndarray:
        """
        Embed text with MiniLM, project to out_dim, normalize to unit sphere.
        Cached by (sha256(text), out_dim).
        """
        key = (hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(), out_dim)
        if key in self._cache:
            return self._cache[key]

        model   = self._get_model()
        raw     = np.array(next(iter(model.embed([text]))), dtype=np.float64)
        raw    /= (np.linalg.norm(raw) + 1e-10)

        P       = self._get_projection(EMBED_DIM, out_dim)
        proj    = raw @ P
        proj   /= (np.linalg.norm(proj) + 1e-10)
        proj    = np.clip((proj + 1.0) / 2.0, 0.0, 1.0)   # map [-1,1] -> [0,1]

        self._cache[key] = proj
        return proj

    # ------------------------------------------------------------------
    # Public channel vector builders
    # ------------------------------------------------------------------

    def comm_vector(self, content: str, token_count: int, is_handoff: bool = False) -> np.ndarray:
        """
        Communication channel feature vector, shape (16,).

        [0:12]  semantic projection of message content
        [12]    token_count / 2048, clipped to [0, 1]
        [13]    len(content) / 4096, clipped to [0, 1]
        [14]    1.0 if content contains explicit instruction/role keywords
        [15]    1.0 if this is a handoff message
        """
        d     = CHANNEL_DIM["comm"]
        sd    = _SEMANTIC_DIM["comm"]
        vec   = np.zeros(d, dtype=np.float64)

        vec[:sd] = self._embed_and_project(content, sd)
        vec[12]  = min(token_count / 2048.0, 1.0)
        vec[13]  = min(len(content) / 4096.0, 1.0)
        vec[14]  = float(any(kw in content.lower() for kw in
                             ("you must", "your task", "ignore", "system:", "instruction")))
        vec[15]  = float(is_handoff)
        return vec

    def mem_vector(self, content: str, op: str) -> np.ndarray:
        """
        Memory channel feature vector, shape (12,).

        [0:10]  semantic projection of memory content
        [10]    1.0 if write, 0.0 if read
        [11]    len(content) / 2048, clipped to [0, 1]
        """
        d     = CHANNEL_DIM["mem"]
        sd    = _SEMANTIC_DIM["mem"]
        vec   = np.zeros(d, dtype=np.float64)

        vec[:sd] = self._embed_and_project(content, sd)
        vec[10]  = 1.0 if op == "write" else 0.0
        vec[11]  = min(len(content) / 2048.0, 1.0)
        return vec

    def tool_vector(self, tool_name: str, arguments: str, output: str) -> np.ndarray:
        """
        Tool channel feature vector, shape (12,).

        [0:8]   semantic projection of joint (tool_name + arguments + output)
        [8:10]  semantic projection of tool_name alone (2-d)
        [10]    len(arguments) / 1024, clipped to [0, 1]
        [11]    len(output) / 2048, clipped to [0, 1]
        """
        d      = CHANNEL_DIM["tool"]
        sd     = _SEMANTIC_DIM["tool"]
        vec    = np.zeros(d, dtype=np.float64)

        joint       = tool_name + " " + arguments + " " + output
        vec[:sd]    = self._embed_and_project(joint, sd)
        vec[8:10]   = self._embed_and_project(tool_name, 2)
        vec[10]     = min(len(arguments) / 1024.0, 1.0)
        vec[11]     = min(len(output) / 2048.0, 1.0)
        return vec

    @staticmethod
    def exec_vector(prompt_tokens: int, completion_tokens: int, latency_ms: float) -> np.ndarray:
        """
        Execution channel feature vector, shape (8,).
        No semantic content — execution metadata only.

        [0]  prompt_tokens / 4096
        [1]  completion_tokens / 2048
        [2]  latency_ms / 10000
        [3]  completion / (prompt + completion + 1)
        [4:8] zeros (reserved)
        """
        d     = CHANNEL_DIM["exec"]
        vec   = np.zeros(d, dtype=np.float64)
        total = prompt_tokens + completion_tokens
        vec[0] = min(prompt_tokens / 4096.0, 1.0)
        vec[1] = min(completion_tokens / 2048.0, 1.0)
        vec[2] = min(latency_ms / 10_000.0, 1.0)
        vec[3] = completion_tokens / (total + 1.0)
        return vec