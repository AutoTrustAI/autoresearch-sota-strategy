#!/usr/bin/env python3
# =============================================================================
# train.py -- B200 leaderboard submission for the karpathy/autoresearch harness.
# Single-GPU, fixed 300s budget, validation bits-per-byte metric.
#
# Full-capacity strict-CRN candidate: construct and initialize the exact 64x
# baseline first, expand all eight per-layer bigram tables to 512x, then expand
# the shared-trigram pair to 2048x/2048x. Both tables preserve the exact
# 1024x/1024x strict-CRN prefixes and then receive private-RNG tails in original
# table order; the ambient RNG is restored. To fit the 183.36-GiB target while
# preserving FP32 gradient accumulation, both trigram tables and only the two
# layer-1 512x bigram tables use independent occurrence-compact DirectScratch
# buffers and independent dense int32 owner maps. Each FP32 scratch needs only
# B*T rows rather than all hash-table rows. At B=72,T=2048 every scratch is
# exactly 216 MiB; each trigram owner is 64 MiB and each L1 bigram owner is
# 16 MiB. Weight, moment, hash, and optimizer arithmetic remain unchanged.
# Combined candidate: layers 5/6/7 additionally form Q/K/V and attention gates
# from the saved post-layer-4 activation while their residual/MLP streams stay
# current. This attention-source reuse adds no parameters or random draws.
#
#   run it:   python train.py            (defaults to SEED=42)
#   or:       SEED=43 python train.py
#   microcheck: TRITON_DIRECT_COMPACT_SCRATCH_MICROCHECK=1 python train.py
#
# Clean reference on 1x NVIDIA B200 with full cached climbmix data + 8192 BPE
# tokenizer (depth 8 / width 768, SEED=42, 300s budget, val_bpb metric):
#   champion baseline val_bpb ~= 0.9002 on this box (measured 0.900182 / 0.900216 across
#   runs; run-to-run seed noise is ~0.004, so anything within that band is a tie).
#
# Rejected: batch-64 candidate measured val_bpb=0.902112 (regression).
# Rejected: layer-5 bigram deletion measured val_bpb=0.901529 (regression).
# Rejected: exact-semantics n-gram hash rewrite measured val_bpb=0.900386
#   (neutral -- no quality change from int64-modulo to int32-bitmask hashing).
# Rejected: global NGRAM_MULT 64->32 (halving both bigram and trigram tables)
#   measured val_bpb=0.901361 at 501.1M tokens (regression).
# Rejected: trigram-only halving (TRIGRAM_MULT=32) measured val_bpb=0.900416
#   (neutral -- no quality change from halving trigram capacity alone).
# Rejected: K=1 full-width n-gram tables (one kv_dim-wide table per site,
#   removing the second hash/lookup/factorization) measured val_bpb=0.900721
#   (regression -- factored K=2 multi-hash is better than single full-width).
#
# This candidate keeps the measured shared-trigram model unchanged and replaces
# only the n-gram RMSProp implementation.  The default path deliberately keeps
# dense embedding gradients (the most reliable torch.compile/fullgraph path),
# then uses one custom Triton kernel per table to atomically deduplicate hashes
# and update claimed rows in place, avoiding index_select/index_copy materialization.
# For the
# benchmark's BF16 moment state and beta2 >= 0.999, a dense zero-gradient lerp
# rounds back to the same positive normal BF16 value, so skipped rows require
# no materialization.  NGRAM_SPARSE_GRAD=1 also
# enables real sparse embedding gradients and coalesces duplicate COO rows,
# but that opt-in requires a PyTorch build whose compiled embedding backward
# supports sparse output; the leaderboard runtime must validate it first.
#
# This preserves the dense BF16 recurrence for normal moment values, including
# the dynamic beta2 warmdown and its original current_beta**step bias
# correction. Zero and every positive finite BF16 value (including
# subnormals) is exact under the skipped-row identity; NaN/Inf states remain
# outside this argument.
#
# Base model change (MEASURED, kept as current best): shared trigram value embedding.
# A single original K=2 factored trigram table pair (half-width, 524288 rows)
# is shared across all three trigram layers {1,5,7}. One pair of trigram
# hashes/lookups is computed once per forward pass and the resulting 768-wide
# value is reused at all three layers, while each layer retains its own
# independent trigram_gate. The shared table uses the champion's layer-1
# trigram hash pair (FNV+Murmur family); the layer-5 and layer-7 trigram table
# pairs, hash families, and lookup paths are removed. This reduces trigram
# embedding parameters to 1/3 of the champion's. Every bigram path, trigram
# placement, table size, K=2 half-width factorization, initialization,
# surviving sparse-optimizer behavior, model geometry, batch, schedules, seed,
# FA4, native BF16, and COMPILE_MODE are unchanged.
#
# Result on this box: framework-measured val_bpb = 0.900171 (fresh re-measure 0.899885),
# vs the champion baseline ~=0.9002 -- within the ~0.004 seed-noise band. The gain is
# architectural, not a score jump: trigram value-embedding parameters are cut to 1/3
# (three per-layer tables -> one shared table). Published as vora's board entry.
#
# This is SELF-CONTAINED: the full model + training code below is recursive's
# optimized_from_karpathy.py (env-parametrized), and our tuned config is baked
# in via the os.environ defaults right here. Current B200 defaults:
#   MODEL_DIM=768, NGRAM_MULT=64, MATRIX_LR=0.035, EMBEDDING_LR=0.6
#   WARMDOWN_RATIO=0.90, COMPILE_MODE=max-autotune, TINY_DIV=8
# Needs Blackwell/SM100 with flash-attn-4. Attention fallback is disabled.
# =============================================================================
import os as _os
for _k, _v in {
    "DEPTH": "8",
    "MODEL_DIM": "768",
    "NGRAM_MULT": "64",
    "BIGRAM_CRN_MULT": "512",
    "TRIGRAM_CRN_MULT_0": "2048",
    "TRIGRAM_CRN_MULT_1": "2048",
    "MATRIX_LR": "0.035",
    "EMBEDDING_LR": "0.6",
    "WARMDOWN_RATIO": "0.90",
    "COMPILE_MODE": "max-autotune",
    "NS_STEPS": "5",
    "SEED": "42",
    "WINDOW_PATTERN": "TTTL",
    "DEVICE_BATCH_SIZE": "72",
    "TINY_DIV": "8",
    "NGRAM_DIRECT_SCRATCH_GRAD": "1",
    "DIRECT_SCRATCH_FUSED_CLEAR": "1",
    "TRITON_RMS_BLOCK_R": "4",
    "TRITON_RMS_NUM_WARPS": "4",
    "DIRECT_SCRATCH_BLOCK_R": "4",
    "DIRECT_SCRATCH_NUM_WARPS": "4",
    "MUON_PEAK_MOMENTUM": "0.94",
    "MUON_WARMDOWN_MOMENTUM": "0.80",
    "MLP_DEPTH_PROFILE": "middle",
    "TRIGRAM_LR_SCALE": "1.0",
    "COMPACT_FP32_TRIGRAM_SCRATCH_INDICES": "0,1",
    "COMPACT_FP32_BIGRAM_LAYERS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    _os.environ.setdefault(_k, _v)   # env still wins if the user sets it


def _parse_linux_cpu_list(spec):
    cpus = set()
    for part in spec.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        cpus.update(range(start, stop + 1))
    return cpus


def _pin_to_single_nvidia_gpu_numa_node():
    """Keep the tokenizer/packer and trainer on the GPU-local CPU node."""
    if not hasattr(_os, "sched_getaffinity"):
        return
    try:
        # Containers often expose every PCI record in sysfs but only one
        # /dev/nvidia<N> device. Map that visible device minor back to its bus.
        visible_minors = {
            int(name.removeprefix("nvidia"))
            for name in _os.listdir("/dev")
            if name.startswith("nvidia") and name.removeprefix("nvidia").isdigit()
        }
        if len(visible_minors) != 1:
            return
        visible_minor = next(iter(visible_minors))
        visible_bus = None
        gpu_info_root = "/proc/driver/nvidia/gpus"
        for bus in _os.listdir(gpu_info_root):
            with open(
                _os.path.join(gpu_info_root, bus, "information"), encoding="ascii"
            ) as f:
                fields = {
                    key.strip(): value.strip()
                    for line in f
                    if ":" in line
                    for key, value in [line.split(":", 1)]
                }
            if int(fields.get("Device Minor", "-1")) == visible_minor:
                visible_bus = fields.get("Bus Location", bus)
                break
        if visible_bus is None:
            return
        with open(
            f"/sys/bus/pci/devices/{visible_bus}/numa_node", encoding="ascii"
        ) as f:
            node = int(f.read().strip())
        if node < 0:
            return
        with open(
            f"/sys/devices/system/node/node{node}/cpulist", encoding="ascii"
        ) as f:
            local_cpus = _parse_linux_cpu_list(f.read())
        local_cpus &= set(_os.sched_getaffinity(0))
        if local_cpus:
            _os.sched_setaffinity(0, local_cpus)
    except (OSError, ValueError):
        # Affinity is a locality optimization, never a portability requirement.
        return


_pin_to_single_nvidia_gpu_numa_node()
# ----- below: full champion model + training code (env-parametrized). vora's ONLY change
# ----- from the champion is the shared trigram VE described above; the remaining inline
# ----- comments below are the champion base's own rationale, not new claims. -----------
# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""
Nanochat pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: /venv/main/bin/python train.py
"""

import os
import sys


_FAST_LOADER_WORKER_FLAG = "--fast-loader-worker"
_FAST_LOADER_MAGIC = b"ARF1"


def _write_all(fd, data):
    """Write a complete binary frame, handling short POSIX pipe writes."""
    view = memoryview(data).cast("B")
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BrokenPipeError("fast-loader pipe closed during frame write")
        view = view[written:]


def _fast_loader_worker_main():
    """CPU-only ordered tokenizer/packer subprocess.

    This is intentionally dispatched before the training process imports torch
    or initializes CUDA. stdout is a framed binary protocol; diagnostics and
    tracebacks go only to stderr.
    """
    import heapq
    import struct
    import time as worker_time
    import traceback
    from collections import defaultdict, deque
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from prepare import Tokenizer, _document_batches

    header = struct.Struct("<4sQqI")
    batch_size = int(os.environ["FAST_LOADER_BATCH_SIZE"])
    sequence_len = int(os.environ["FAST_LOADER_SEQUENCE_LEN"])
    buffer_size = 1000
    if os.environ.get("FAST_LOADER_BUFFER_SIZE", "1000") != "1000":
        raise ValueError(
            "FAST_LOADER_BUFFER_SIZE is fixed at 1000 to preserve official packing semantics"
        )
    token_threads = int(os.environ.get("FAST_LOADER_TOKEN_THREADS", "8"))
    token_prefetch = int(os.environ.get("FAST_LOADER_TOKEN_PREFETCH", "8"))
    verify_selections = os.environ.get("FAST_LOADER_VERIFY_SELECTIONS", "0") == "1"
    stats_every = int(os.environ.get("FAST_LOADER_STATS_EVERY", "0"))

    if sys.byteorder != "little":
        raise RuntimeError("fast-loader binary protocol currently requires little-endian host")
    if batch_size <= 0 or sequence_len <= 0 or buffer_size <= 0:
        raise ValueError("fast-loader dimensions and buffer size must be positive")
    if token_threads <= 0 or token_prefetch <= 0:
        raise ValueError("fast-loader token thread/prefetch settings must be positive")

    worker_nice = int(os.environ.get("FAST_LOADER_NICE", "5"))
    if worker_nice:
        try:
            os.nice(worker_nice)
        except OSError:
            pass

    tokenizer = Tokenizer.from_directory()
    bos_token = tokenizer.get_bos_token_id()
    document_batches = _document_batches("train", tokenizer_batch_size=128)

    # Frame 0 uses the exact synchronous source behavior. Only after the parent
    # releases the first-frame barrier do we enable bounded token lookahead, so
    # no work for batch 1 is moved ahead of the measured training loop.
    token_executor = None
    pending_token_batches = deque()

    def submit_token_batch():
        if token_executor is None:
            raise RuntimeError("token prefetch executor has not been started")
        texts, batch_epoch = next(document_batches)
        future = token_executor.submit(
            tokenizer.encode,
            texts,
            prepend=bos_token,
            num_threads=token_threads,
        )
        pending_token_batches.append((future, batch_epoch))

    def start_token_prefetch():
        nonlocal token_executor
        if token_executor is not None:
            return
        # A single executor worker lets Rust/tiktoken run ahead of Python
        # packing without concurrent Encoding calls. Logical 128-doc groups
        # and their epoch tags remain unchanged.
        token_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenize")
        for _ in range(token_prefetch):
            submit_token_batch()

    def next_token_batch():
        if token_executor is None:
            texts, batch_epoch = next(document_batches)
            return (
                tokenizer.encode(
                    texts,
                    prepend=bos_token,
                    num_threads=token_threads,
                ),
                batch_epoch,
            )
        future, batch_epoch = pending_token_batches.popleft()
        token_lists = future.result()
        submit_token_batch()
        return token_lists, batch_epoch

    row_capacity = sequence_len + 1
    buckets = defaultdict(deque)  # token length -> documents in source insertion order
    fit_length_mask = 0           # active lengths <= row_capacity
    min_length_heap = []          # one lazy entry per possibly-active length
    heap_lengths = set()
    active_count = 0
    epoch = 1
    shadow_buffer = [] if verify_selections else None

    def add_document(doc):
        nonlocal fit_length_mask, active_count
        doc_len = len(doc)
        bucket = buckets[doc_len]
        if not bucket:
            if doc_len <= row_capacity:
                fit_length_mask |= 1 << doc_len
            if doc_len not in heap_lengths:
                heapq.heappush(min_length_heap, doc_len)
                heap_lengths.add(doc_len)
        bucket.append(doc)
        active_count += 1
        if shadow_buffer is not None:
            shadow_buffer.append(doc)

    def refill_documents():
        nonlocal epoch
        token_lists, epoch = next_token_batch()
        for token_list in token_lists:
            add_document(token_list)

    def pop_document(remaining):
        """Exact equivalent of prepare.make_dataloader's two linear scans."""
        nonlocal fit_length_mask, active_count
        eligible = fit_length_mask & ((1 << (remaining + 1)) - 1)
        if eligible:
            chosen_len = eligible.bit_length() - 1
            chosen = buckets[chosen_len].popleft()
        else:
            while min_length_heap and not buckets[min_length_heap[0]]:
                stale_len = heapq.heappop(min_length_heap)
                heap_lengths.remove(stale_len)
            if not min_length_heap:
                raise RuntimeError("fast-loader document index unexpectedly empty")
            chosen_len = min_length_heap[0]
            chosen = buckets[chosen_len].popleft()

        active_count -= 1
        if not buckets[chosen_len] and chosen_len <= row_capacity:
            fit_length_mask &= ~(1 << chosen_len)

        if shadow_buffer is not None:
            best_idx = -1
            best_len = 0
            for doc_idx, doc in enumerate(shadow_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = doc_idx
                    best_len = doc_len
            if best_idx < 0:
                best_idx = min(range(len(shadow_buffer)), key=lambda i: len(shadow_buffer[i]))
            reference = shadow_buffer.pop(best_idx)
            if reference is not chosen:
                raise AssertionError(
                    "optimized best-fit selection diverged from the official linear scan"
                )
        return chosen

    rows = np.empty((batch_size, row_capacity), dtype="<i8")
    payload = np.empty((2, batch_size, sequence_len), dtype="<i8")
    payload_nbytes = payload.nbytes
    frame_sequence = 0
    produced_time = 0.0

    try:
        while True:
            batch_started = worker_time.perf_counter()
            for row_idx in range(batch_size):
                pos = 0
                while pos < row_capacity:
                    while active_count < buffer_size:
                        refill_documents()
                    remaining = row_capacity - pos
                    doc = pop_document(remaining)
                    take = min(len(doc), remaining)
                    # NumPy converts the Python integer list directly into the
                    # destination slice, avoiding one torch allocation per doc.
                    source = doc if take == len(doc) else doc[:take]
                    rows[row_idx, pos:pos + take] = source
                    pos += take

            np.copyto(payload[0], rows[:, :-1])
            np.copyto(payload[1], rows[:, 1:])
            frame_header = header.pack(
                _FAST_LOADER_MAGIC,
                frame_sequence,
                epoch,
                payload_nbytes,
            )
            _write_all(sys.stdout.fileno(), frame_header)
            _write_all(sys.stdout.fileno(), payload)

            if frame_sequence == 0:
                gate = os.read(sys.stdin.fileno(), 1)
                if gate != b"\x01":
                    return
                start_token_prefetch()

            produced_time += worker_time.perf_counter() - batch_started
            frame_sequence += 1
            if stats_every and frame_sequence % stats_every == 0:
                mean_ms = 1000.0 * produced_time / frame_sequence
                print(
                    f"fast-loader produced={frame_sequence} mean_ms={mean_ms:.2f} "
                    f"active_docs={active_count}",
                    file=sys.stderr,
                    flush=True,
                )
    except BrokenPipeError:
        # Normal parent shutdown after the training window.
        return
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if token_executor is not None:
            token_executor.shutdown(wait=False, cancel_futures=True)


if _FAST_LOADER_WORKER_FLAG in sys.argv:
    try:
        _fast_loader_worker_main()
    except BrokenPipeError:
        pass
    sys.exit(0)

if hasattr(_os, "sched_getaffinity"):
    _affinity = sorted(_os.sched_getaffinity(0))
    if _affinity:
        print(
            f"CPU affinity: {_affinity[0]}-{_affinity[-1]} "
            f"({len(_affinity)} CPUs)"
        )

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import atexit
import gc
import math
import queue
import struct
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import torch
import torch._inductor.config as inductor_config

# The full-capacity layout leaves less than the physical 128 MiB L2 size free
# during first-kernel autotuning.  Inductor's cache flush is benchmarking
# scratch, not model state; a smaller flush keeps config timing representative
# while avoiding a transient-only allocation failure.
from torch._inductor.runtime.benchmarking import benchmarker as _inductor_benchmarker

_inductor_benchmarker.__dict__["L2_cache_size"] = 32 * 1024 * 1024

# Keep default inductor settings for this compile-capture ablation.
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.language.extra import libdevice

_USE_TRITON_LAZY_RMS = os.environ.get("USE_TRITON_LAZY_RMS", "1") == "1"
_NGRAM_UNCOALESCED_SPARSE = (
    os.environ.get("NGRAM_UNCOALESCED_SPARSE", "0") == "1"
)
_DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK = (
    os.environ.get("TRITON_DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK", "0") == "1"
)
_NGRAM_DIRECT_SCRATCH_GRAD = (
    os.environ.get("NGRAM_DIRECT_SCRATCH_GRAD", "0") == "1"
    or _DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK
)
_DIRECT_SCRATCH_FUSED_CLEAR = (
    os.environ.get("DIRECT_SCRATCH_FUSED_CLEAR", "0") == "1"
    or _DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK
)
_DIRECT_SCRATCH_OWNER_MIN = (
    os.environ.get("DIRECT_SCRATCH_OWNER_MIN", "0") == "1"
)
_DIRECT_SCRATCH_PAIR_FUSION = (
    os.environ.get("DIRECT_SCRATCH_PAIR_FUSION", "0") == "1"
)
if _DIRECT_SCRATCH_OWNER_MIN and not _NGRAM_DIRECT_SCRATCH_GRAD:
    raise ValueError(
        "DIRECT_SCRATCH_OWNER_MIN=1 requires NGRAM_DIRECT_SCRATCH_GRAD=1"
    )
if _DIRECT_SCRATCH_PAIR_FUSION and not _NGRAM_DIRECT_SCRATCH_GRAD:
    raise ValueError(
        "DIRECT_SCRATCH_PAIR_FUSION=1 requires NGRAM_DIRECT_SCRATCH_GRAD=1"
    )
if _DIRECT_SCRATCH_FUSED_CLEAR and not _NGRAM_DIRECT_SCRATCH_GRAD:
    raise ValueError(
        "DIRECT_SCRATCH_FUSED_CLEAR=1 requires NGRAM_DIRECT_SCRATCH_GRAD=1"
    )
if _DIRECT_SCRATCH_PAIR_FUSION and _DIRECT_SCRATCH_OWNER_MIN:
    raise ValueError(
        "DIRECT_SCRATCH_PAIR_FUSION and DIRECT_SCRATCH_OWNER_MIN are "
        "mutually exclusive"
    )
if _NGRAM_DIRECT_SCRATCH_GRAD and _NGRAM_UNCOALESCED_SPARSE:
    raise ValueError(
        "NGRAM_DIRECT_SCRATCH_GRAD and NGRAM_UNCOALESCED_SPARSE are "
        "mutually exclusive"
    )
if (
    _NGRAM_DIRECT_SCRATCH_GRAD
    and os.environ.get("NGRAM_SPARSE_GRAD", "0") == "1"
):
    raise ValueError(
        "NGRAM_DIRECT_SCRATCH_GRAD uses a normal dense embedding weight and "
        "cannot be combined with NGRAM_SPARSE_GRAD=1"
    )
# The fused uncoalesced path consumes real sparse embedding gradients.  Keep
# the published dense-gradient path as the default, but make the single opt-in
# switch sufficient unless the caller explicitly requested an incompatible
# layout (validated after model construction below).
if _NGRAM_UNCOALESCED_SPARSE:
    os.environ.setdefault("NGRAM_SPARSE_GRAD", "1")
_TRITON_RMS_BLOCK_R = int(os.environ.get("TRITON_RMS_BLOCK_R", "8"))
_TRITON_RMS_NUM_WARPS = int(os.environ.get("TRITON_RMS_NUM_WARPS", "8"))
_DIRECT_SCRATCH_BLOCK_R = int(
    os.environ.get("DIRECT_SCRATCH_BLOCK_R", str(_TRITON_RMS_BLOCK_R))
)
_DIRECT_SCRATCH_NUM_WARPS = int(
    os.environ.get("DIRECT_SCRATCH_NUM_WARPS", str(_TRITON_RMS_NUM_WARPS))
)
_TRITON_RMS_ROUND_PROFILES = {
    # Matches the inspected Inductor dense oracle: one FP32 fused expression,
    # with BF16 conversion only at the two final stores.
    "inductor": 0,
    # Materialize the new BF16 moment, then use that rounded value downstream.
    "moment": 1,
    # Logical eager tensor boundaries: square/moment/div/sqrt/add/div round.
    "logical": 2,
    # Intermediate diagnostic: square and moment round, denominator stays FP32.
    "grad_moment": 3,
}
_TRITON_RMS_ROUND_PROFILE_NAME = os.environ.get(
    "TRITON_RMS_ROUND_PROFILE", "inductor"
)
if _TRITON_RMS_ROUND_PROFILE_NAME not in _TRITON_RMS_ROUND_PROFILES:
    raise ValueError(
        "TRITON_RMS_ROUND_PROFILE must be one of "
        + ", ".join(_TRITON_RMS_ROUND_PROFILES)
    )
_TRITON_RMS_ROUND_PROFILE = _TRITON_RMS_ROUND_PROFILES[
    _TRITON_RMS_ROUND_PROFILE_NAME
]
_NGRAM_UNIQUE_ROWS = os.environ.get("NGRAM_UNIQUE_ROWS", "1") == "1"
if _TRITON_RMS_BLOCK_R not in (1, 2, 4, 8, 16):
    raise ValueError("TRITON_RMS_BLOCK_R must be one of 1, 2, 4, 8, 16")
if _TRITON_RMS_NUM_WARPS not in (4, 8):
    raise ValueError("TRITON_RMS_NUM_WARPS must be 4 or 8")
if _DIRECT_SCRATCH_BLOCK_R not in (1, 2, 4, 8, 16):
    raise ValueError("DIRECT_SCRATCH_BLOCK_R must be one of 1, 2, 4, 8, 16")
if _DIRECT_SCRATCH_NUM_WARPS not in (4, 8):
    raise ValueError("DIRECT_SCRATCH_NUM_WARPS must be 4 or 8")

_USE_QUACK_CE = os.environ.get("USE_QUACK_CE", "0") == "1"
_REUSE_NGRAM_ROWS = (
    os.environ.get("REUSE_NGRAM_ROWS", "1") == "1"
    and not _NGRAM_UNCOALESCED_SPARSE
)
_DIRECT_SCRATCH_DTYPES = (torch.float32, torch.bfloat16)
_COMPACT_FP32_TRIGRAM_SCRATCH_INDICES = frozenset(
    int(part.strip())
    for part in os.environ.get(
        "COMPACT_FP32_TRIGRAM_SCRATCH_INDICES", "0,1"
    ).split(",")
    if part.strip()
)
if _COMPACT_FP32_TRIGRAM_SCRATCH_INDICES != frozenset((0, 1)):
    raise ValueError(
        "this dual-compact candidate requires "
        "COMPACT_FP32_TRIGRAM_SCRATCH_INDICES=0,1"
    )
_COMPACT_FP32_BIGRAM_LAYERS = frozenset(
    int(part.strip())
    for part in os.environ.get("COMPACT_FP32_BIGRAM_LAYERS", "1").split(",")
    if part.strip()
)
if _COMPACT_FP32_BIGRAM_LAYERS != frozenset((1,)):
    raise ValueError(
        "this candidate requires COMPACT_FP32_BIGRAM_LAYERS=1 so only "
        "the two layer-1 bigram tables use compact FP32 scratch"
    )


@torch.library.custom_op(
    "ngram_direct::scratch_scatter",
    mutates_args={"grad_scratch"},
    device_types="cuda",
)
def _ngram_direct_scratch_scatter_op(
    indices: torch.Tensor,
    grad_output: torch.Tensor,
    grad_scratch: torch.Tensor,
) -> None:
    """Clear touched rows and atomically accumulate occurrence grads in FP32."""
    _triton_direct_scratch_scatter(indices, grad_output, grad_scratch)


@_ngram_direct_scratch_scatter_op.register_fake
def _ngram_direct_scratch_scatter_fake(indices, grad_output, grad_scratch):
    return None


@torch.library.custom_op(
    "ngram_direct::scratch_scatter_compact",
    mutates_args={"grad_scratch", "owner_key"},
    device_types="cuda",
)
def _ngram_direct_scratch_scatter_compact_op(
    indices: torch.Tensor,
    grad_output: torch.Tensor,
    grad_scratch: torch.Tensor,
    owner_key: torch.Tensor,
) -> None:
    """Accumulate FP32 row sums in first-occurrence compact slots."""
    _triton_direct_compact_scratch_scatter(
        indices,
        grad_output,
        grad_scratch,
        owner_key,
    )


@_ngram_direct_scratch_scatter_compact_op.register_fake
def _ngram_direct_scratch_scatter_compact_fake(
    indices,
    grad_output,
    grad_scratch,
    owner_key,
):
    return None


@torch.library.custom_op(
    "ngram_direct::scratch_scatter_pair",
    mutates_args={"grad_scratch0", "grad_scratch1"},
    device_types="cuda",
)
def _ngram_direct_scratch_scatter_pair_op(
    indices0: torch.Tensor,
    indices1: torch.Tensor,
    grad_output: torch.Tensor,
    grad_scratch0: torch.Tensor,
    grad_scratch1: torch.Tensor,
) -> None:
    """Accumulate both halves of one K=2 lookup with two total launches."""
    _triton_direct_scratch_scatter_pair(
        indices0,
        indices1,
        grad_output,
        grad_scratch0,
        grad_scratch1,
    )


@_ngram_direct_scratch_scatter_pair_op.register_fake
def _ngram_direct_scratch_scatter_pair_fake(
    indices0,
    indices1,
    grad_output,
    grad_scratch0,
    grad_scratch1,
):
    return None


@torch.library.custom_op(
    "ngram_direct::scratch_scatter_owner_min",
    mutates_args={"grad_scratch", "owner_key"},
    device_types="cuda",
)
def _ngram_direct_scratch_scatter_owner_min_op(
    indices: torch.Tensor,
    grad_output: torch.Tensor,
    grad_scratch: torch.Tensor,
    owner_key: torch.Tensor,
) -> None:
    """Initialize from the first occurrence, then atomically add duplicates."""
    _triton_direct_scratch_scatter_owner_min(
        indices,
        grad_output,
        grad_scratch,
        owner_key,
    )


@_ngram_direct_scratch_scatter_owner_min_op.register_fake
def _ngram_direct_scratch_scatter_owner_min_fake(
    indices,
    grad_output,
    grad_scratch,
    owner_key,
):
    return None


@torch.library.custom_op(
    "ngram_direct::embedding",
    mutates_args=(),
    device_types="cuda",
)
def _ngram_direct_embedding_op(
    weight: torch.Tensor,
    indices: torch.Tensor,
    grad_scratch: torch.Tensor,
) -> torch.Tensor:
    # Forward is an ordinary gather.  Only its registered backward differs
    # from nn.Embedding: it writes a persistent scratch table, not weight.grad.
    return F.embedding(indices, weight)


@_ngram_direct_embedding_op.register_fake
def _ngram_direct_embedding_fake(weight, indices, grad_scratch):
    return weight.new_empty((*indices.shape, weight.shape[1]))


def _ngram_direct_embedding_setup_context(ctx, inputs, output):
    _weight, indices, grad_scratch = inputs
    ctx.save_for_backward(indices, grad_scratch)


def _ngram_direct_embedding_backward(ctx, grad_output):
    indices, grad_scratch = ctx.saved_tensors
    # grad_output may be a non-contiguous slice produced by cat backward.  The
    # Triton kernel consumes its runtime strides directly, avoiding a values()
    # tensor, COO construction, and a contiguous materialization here.
    torch.ops.ngram_direct.scratch_scatter(indices, grad_output, grad_scratch)
    return None, None, None


_ngram_direct_embedding_op.register_autograd(
    _ngram_direct_embedding_backward,
    setup_context=_ngram_direct_embedding_setup_context,
)


@torch.library.custom_op(
    "ngram_direct::embedding_compact",
    mutates_args=(),
    device_types="cuda",
)
def _ngram_direct_embedding_compact_op(
    weight: torch.Tensor,
    indices: torch.Tensor,
    grad_scratch: torch.Tensor,
    owner_key: torch.Tensor,
) -> torch.Tensor:
    return F.embedding(indices, weight)


@_ngram_direct_embedding_compact_op.register_fake
def _ngram_direct_embedding_compact_fake(
    weight,
    indices,
    grad_scratch,
    owner_key,
):
    return weight.new_empty((*indices.shape, weight.shape[1]))


def _ngram_direct_embedding_compact_setup_context(ctx, inputs, output):
    _weight, indices, grad_scratch, owner_key = inputs
    ctx.save_for_backward(indices, grad_scratch, owner_key)


def _ngram_direct_embedding_compact_backward(ctx, grad_output):
    indices, grad_scratch, owner_key = ctx.saved_tensors
    torch.ops.ngram_direct.scratch_scatter_compact(
        indices,
        grad_output,
        grad_scratch,
        owner_key,
    )
    return None, None, None, None


_ngram_direct_embedding_compact_op.register_autograd(
    _ngram_direct_embedding_compact_backward,
    setup_context=_ngram_direct_embedding_compact_setup_context,
)


@torch.library.custom_op(
    "ngram_direct::embedding_pair",
    mutates_args=(),
    device_types="cuda",
)
def _ngram_direct_embedding_pair_op(
    weight0: torch.Tensor,
    weight1: torch.Tensor,
    indices0: torch.Tensor,
    indices1: torch.Tensor,
    grad_scratch0: torch.Tensor,
    grad_scratch1: torch.Tensor,
) -> torch.Tensor:
    """Gather two half-width tables directly into one contiguous K=2 value."""
    return _triton_direct_embedding_pair(
        weight0,
        weight1,
        indices0,
        indices1,
        grad_scratch0,
        grad_scratch1,
    )


@_ngram_direct_embedding_pair_op.register_fake
def _ngram_direct_embedding_pair_fake(
    weight0,
    weight1,
    indices0,
    indices1,
    grad_scratch0,
    grad_scratch1,
):
    return weight0.new_empty((*indices0.shape, weight0.shape[1] * 2))


def _ngram_direct_embedding_pair_setup_context(ctx, inputs, output):
    (
        _weight0,
        _weight1,
        indices0,
        indices1,
        grad_scratch0,
        grad_scratch1,
    ) = inputs
    ctx.save_for_backward(indices0, indices1, grad_scratch0, grad_scratch1)


def _ngram_direct_embedding_pair_backward(ctx, grad_output):
    indices0, indices1, grad_scratch0, grad_scratch1 = ctx.saved_tensors
    torch.ops.ngram_direct.scratch_scatter_pair(
        indices0,
        indices1,
        grad_output,
        grad_scratch0,
        grad_scratch1,
    )
    return None, None, None, None, None, None


_ngram_direct_embedding_pair_op.register_autograd(
    _ngram_direct_embedding_pair_backward,
    setup_context=_ngram_direct_embedding_pair_setup_context,
)


@torch.library.custom_op(
    "ngram_direct::embedding_owner_min",
    mutates_args=(),
    device_types="cuda",
)
def _ngram_direct_embedding_owner_min_op(
    weight: torch.Tensor,
    indices: torch.Tensor,
    grad_scratch: torch.Tensor,
    owner_key: torch.Tensor,
) -> torch.Tensor:
    return F.embedding(indices, weight)


@_ngram_direct_embedding_owner_min_op.register_fake
def _ngram_direct_embedding_owner_min_fake(
    weight,
    indices,
    grad_scratch,
    owner_key,
):
    return weight.new_empty((*indices.shape, weight.shape[1]))


def _ngram_direct_embedding_owner_min_setup_context(ctx, inputs, output):
    _weight, indices, grad_scratch, owner_key = inputs
    ctx.save_for_backward(indices, grad_scratch, owner_key)


def _ngram_direct_embedding_owner_min_backward(ctx, grad_output):
    indices, grad_scratch, owner_key = ctx.saved_tensors
    torch.ops.ngram_direct.scratch_scatter_owner_min(
        indices,
        grad_output,
        grad_scratch,
        owner_key,
    )
    return None, None, None, None


_ngram_direct_embedding_owner_min_op.register_autograd(
    _ngram_direct_embedding_owner_min_backward,
    setup_context=_ngram_direct_embedding_owner_min_setup_context,
)


class DirectScratchEmbedding(nn.Embedding):
    """Dense-weight embedding whose backward writes to selected scratch precision."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim, sparse=False)
        # A None buffer survives meta materialization and the wholesale BF16
        # cast without allocation/conversion. Installing the real table
        # afterwards makes it visible as mutable module state to torch.compile
        # while keeping it out of checkpoints.
        self.register_buffer("grad_scratch", None, persistent=False)
        self.register_buffer("compact_owner_key", None, persistent=False)
        self.compact_scratch = False
        if _DIRECT_SCRATCH_OWNER_MIN:
            # Only the opt-in path registers this extra mutable AOT input.  The
            # default direct-scatter graph and buffer set therefore stay
            # unchanged when DIRECT_SCRATCH_OWNER_MIN=0.
            self.register_buffer("owner_key", None, persistent=False)

    def initialize_grad_scratch(
        self,
        dtype=torch.float32,
        compact_rows=None,
    ):
        if self.weight.is_meta:
            raise RuntimeError("direct n-gram scratch cannot be initialized on meta")
        if dtype not in _DIRECT_SCRATCH_DTYPES:
            raise TypeError(f"unsupported direct scratch dtype: {dtype}")
        if compact_rows is not None:
            compact_rows = int(compact_rows)
            if compact_rows <= 0:
                raise ValueError("compact direct scratch requires positive rows")
            if dtype != torch.float32:
                raise TypeError("compact direct scratch preserves FP32 accumulation")
            if _DIRECT_SCRATCH_OWNER_MIN:
                raise ValueError(
                    "compact direct scratch and owner-min mode are mutually exclusive"
                )
            if _DIRECT_SCRATCH_PAIR_FUSION:
                raise ValueError(
                    "compact direct scratch is incompatible with pair fusion"
                )
            if not _DIRECT_SCRATCH_FUSED_CLEAR:
                raise ValueError(
                    "compact direct scratch requires fused post-consume clearing"
                )
            self.compact_scratch = True
            scratch_shape = (compact_rows, self.weight.shape[1])
            self.compact_owner_key = torch.empty(
                self.weight.shape[0],
                dtype=torch.int32,
                device=self.weight.device,
            )
        else:
            self.compact_scratch = False
            scratch_shape = self.weight.shape
        scratch_factory = torch.zeros if _DIRECT_SCRATCH_FUSED_CLEAR else torch.empty
        self.grad_scratch = scratch_factory(
            scratch_shape,
            dtype=dtype,
            device=self.weight.device,
        )
        if _DIRECT_SCRATCH_OWNER_MIN:
            self.owner_key = torch.empty(
                self.weight.shape[0],
                dtype=torch.int32,
                device=self.weight.device,
            )

    def forward(self, indices):
        if self.grad_scratch is None:
            raise RuntimeError("direct n-gram scratch has not been initialized")
        if self.compact_scratch:
            if self.compact_owner_key is None:
                raise RuntimeError("compact direct scratch has no owner table")
            if self.compact_owner_key.numel() != self.weight.shape[0]:
                raise RuntimeError(
                    "compact direct owner map must have one entry per weight row"
                )
            compact_tensors = (
                self.weight,
                indices,
                self.grad_scratch,
                self.compact_owner_key,
            )
            if len({tensor.device for tensor in compact_tensors}) != 1:
                raise RuntimeError("compact direct tensors must share one device")
            if not self.grad_scratch.is_contiguous():
                raise RuntimeError("compact direct scratch must be contiguous")
            if not self.compact_owner_key.is_contiguous():
                raise RuntimeError("compact direct owner map must be contiguous")
            if indices.numel() > self.grad_scratch.shape[0]:
                raise RuntimeError(
                    "compact direct scratch has fewer rows than occurrences"
                )
            return torch.ops.ngram_direct.embedding_compact(
                self.weight,
                indices,
                self.grad_scratch,
                self.compact_owner_key,
            )
        if _DIRECT_SCRATCH_OWNER_MIN:
            if self.owner_key is None:
                raise RuntimeError("direct n-gram owner table has not been initialized")
            return torch.ops.ngram_direct.embedding_owner_min(
                self.weight,
                indices,
                self.grad_scratch,
                self.owner_key,
            )
        return torch.ops.ngram_direct.embedding(
            self.weight,
            indices,
            self.grad_scratch,
        )

if _USE_QUACK_CE:
    from quack.cross_entropy import cross_entropy_bwd, cross_entropy_fwd

    class _QuackCrossEntropyFunction(torch.autograd.Function):
        """Quack CE with a contiguous loss-gradient compatibility shim.

        Reduction backward expands a scalar gradient with stride zero, while
        Quack 0.4.1's Cutlass kernel requires the row-gradient stride to be
        one.  Materializing this tiny [B*T] vector fixes that interface without
        touching the logits or changing the CE math.
        """

        @staticmethod
        def forward(ctx, x, target, ignore_index):
            loss, lse = cross_entropy_fwd(
                x, target, ignore_index=ignore_index, return_lse=True
            )
            ctx.save_for_backward(x, target, lse)
            ctx.ignore_index = ignore_index
            return loss

        @staticmethod
        def backward(ctx, dloss):
            x, target, lse = ctx.saved_tensors
            dx = cross_entropy_bwd(
                x,
                target,
                dloss.contiguous(),
                lse,
                ctx.ignore_index,
            )
            return dx, None, None

    def quack_cross_entropy(x, target, *, ignore_index=-100, reduction="mean"):
        loss = _QuackCrossEntropyFunction.apply(x, target, ignore_index)
        if reduction == "mean":
            return loss.sum() / (target != ignore_index).sum().float()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"unsupported CE reduction: {reduction}")

cap = torch.cuda.get_device_capability()

if cap[0] >= 10:
    # Blackwell (B200, SM100): wrap flash-attn-4 as a custom op so torch.compile
    # treats it as opaque (no tracing into cutlass DSL, no recompile-cache thrash,
    # no per-call Python kernel build). Always load the pinned kernel through
    # the existing allowed dependency chain; never depend on an ambient
    # flash-attn package whose version is outside pyproject.toml.
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    from huggingface_hub import snapshot_download

    _fa4_root = Path(snapshot_download(
        "kernels-community/flash-attn4",
        repo_type="model",
        revision="7f952e7e7ec1787ad1f7d209d0bdefdb34747af2",
        allow_patterns=["build/torch-cuda/*"],
    ))
    _fa4_variant = _fa4_root / "build" / "torch-cuda"
    _fa4_module_name = "_autoresearch_flash_attn4"
    _fa4_spec = importlib.util.spec_from_file_location(
        _fa4_module_name,
        _fa4_variant / "__init__.py",
        submodule_search_locations=[str(_fa4_variant)],
    )
    if _fa4_spec is None or _fa4_spec.loader is None:
        raise ImportError(f"Cannot load pinned FA4 kernel from {_fa4_variant}")
    _fa4_module = importlib.util.module_from_spec(_fa4_spec)
    sys.modules[_fa4_module_name] = _fa4_module
    _fa4_spec.loader.exec_module(_fa4_module)
    _fa4_interface = importlib.import_module(f"{_fa4_module_name}.interface")
    _fa4_raw = _fa4_module.flash_attn_func
    _fa4_bwd_raw = _fa4_interface._flash_attn_bwd

    @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
    def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (window_left, 0) if window_left > 0 else (None, None)
        out_lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
        out, lse = out_lse if isinstance(out_lse, tuple) else (out_lse, None)
        if lse is None:
            raise RuntimeError("FA4 did not return LSE; cannot run custom backward")
        return out, lse

    @_fa4_causal_op.register_fake
    def _fa4_causal_fake(q, k, v, window_left):
        B, T, H, D = q.shape
        return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    def _fa4_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl = window_left if window_left > 0 else None
        dq, dk, dv = _fa4_bwd_raw(
            q, k, v, out, grad_output, lse,
            causal=True, window_size_left=wl, window_size_right=0,
        )
        return dq, dk, dv

    @_fa4_bwd_op.register_fake
    def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_backward(ctx, grad_output, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
        return dq, dk, dv, None

    _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
        return out

    print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
else:
    raise RuntimeError("FA4 is required on this B200 run; attention fallback is disabled")

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb  # noqa: E402
TRAIN_SEQ_LEN = MAX_SEQ_LEN

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


QK_NORM_SCALE = float(os.environ.get("QK_NORM_SCALE", 1.0))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Separate gate for bigram VE on ALL VE layers reading decorrelated channels (32:64)
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Trigram gate placement; default remains published layers {1,5,7}.
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        trigram_placement = os.environ.get("TRIGRAM_PLACEMENT", "late")
        if trigram_placement not in {"late", "early"}:
            raise ValueError("TRIGRAM_PLACEMENT must be late or early")
        trigram_middle = ve_layers[1] if trigram_placement == "early" else ve_layers[-2]
        trigram_layers = (
            {ve_layers[0], trigram_middle, ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in trigram_layers
            else None
        )
        # Head-level MoE gate on ALL layers for attention output routing
        self.head_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Bigram VE with its own independent gate reading from decorrelated channels (32:64)
        if bigram_ve is not None:
            bigram_ve = bigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bigram_gate(x[..., self.ve_gate_channels:2*self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bigram_ve

        # Trigram VE with its own gate reading from channels 64:96
        if trigram_ve is not None:
            trigram_ve = trigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            tg_gate = 2 * torch.sigmoid(self.trigram_gate(x[..., 2*self.ve_gate_channels:3*self.ve_gate_channels]))
            v = v + tg_gate.unsqueeze(-1) * trigram_ve

        cos, sin = cos_sin
        # QK-norm refinement: normalize BEFORE rotary instead of after
        q, k = norm(q) * QK_NORM_SCALE, norm(k) * QK_NORM_SCALE
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Per-head RMSNorm on attention output (DiffTransformer-inspired sub-layer normalization)
        y = norm(y)

        # Head-level MoE: per-head routing gate on all layers
        head_gates = 2.0 * torch.sigmoid(self.head_gate(x[..., :self.ve_gate_channels]))
        y = y * head_gates.unsqueeze(-1)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        depth_profile = os.environ.get("MLP_DEPTH_PROFILE", "")
        if depth_profile not in {"", "middle", "long3"}:
            raise ValueError(
                "MLP_DEPTH_PROFILE must be empty, middle, or long3"
            )
        named_depth_profile = depth_profile in {"middle", "long3"}
        redistribute_depth = (
            named_depth_profile
            or os.environ.get("MLP_DEPTH_REDISTRIBUTE", "0") == "1"
        )
        if redistribute_depth:
            if config.n_layer != 8:
                raise ValueError("MLP depth redistribution requires DEPTH=8")
            if config.n_embd % 128 != 0:
                raise ValueError(
                    "MLP depth redistribution requires MODEL_DIM to be a multiple of 128"
                )
            if named_depth_profile:
                expansion_profiles = {
                    "middle": (3, 3, 3, 4, 4, 5, 5, 5),
                    # Preserve the proven 3x/5x shape histogram while moving
                    # one wide MLP from layer 4 to the intermediate
                    # long-attention layer 3.
                    "long3": (3, 3, 3, 5, 3, 5, 5, 5),
                }
                expansion_profile = expansion_profiles[depth_profile]
                self.hidden_dim = expansion_profile[layer_idx] * config.n_embd
            else:
                shift_units = int(os.environ.get("MLP_DEPTH_SHIFT_UNITS", "6"))
                if shift_units <= 0:
                    raise ValueError("MLP_DEPTH_SHIFT_UNITS must be positive")
                baseline_units = (4 * config.n_embd) // 128
                hidden_units = (
                    baseline_units - shift_units
                    if layer_idx < 4
                    else baseline_units + shift_units
                )
                if hidden_units <= 0:
                    raise ValueError(
                        "MLP_DEPTH_SHIFT_UNITS makes an MLP hidden width non-positive"
                    )
                self.hidden_dim = hidden_units * 128
        else:
            self.hidden_dim = 4 * config.n_embd
        if redistribute_depth and self.hidden_dim <= 0:
            raise ValueError("redistributed MLP hidden dimensions must be positive")
        if redistribute_depth and self.hidden_dim % 128 != 0:
            raise ValueError("redistributed MLP hidden dimensions must be multiples of 128")
        self.c_fc = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
        self.c_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        # Uniform tau=0.5: confirmed optimal threshold
        self.tau = float(os.environ.get("MLP_TAU", 0.5))

    def forward(self, x):
        h = self.c_fc(x)
        h = F.relu(h - self.tau).square()
        h = self.c_proj(h)
        return h


ATTN_SOURCE_LAYERS = frozenset((5, 6, 7))
ATTN_SOURCE_AFTER_LAYER = 4


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        bigram_ve=None,
        trigram_ve=None,
        attn_source_norm=None,
    ):
        attn_x = norm(x) if attn_source_norm is None else attn_source_norm
        # Simplified attention residual: per-head norm + head gate inside CSA already sufficient
        x = x + self.attn(
            attn_x,
            ve,
            cos_sin,
            window_size,
            bigram_ve=bigram_ve,
            trigram_ve=trigram_ve,
        )
        x = x + norm(self.mlp(norm(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        if (
            os.environ.get("MLP_DEPTH_PROFILE", "") in {"middle", "long3"}
            or os.environ.get("MLP_DEPTH_REDISTRIBUTE", "0") == "1"
        ):
            # Named profiles and shift mode preserve the 32x total hidden width
            # of eight uniform 4x MLPs. At d=768, the default shift is 24 +/- 6 units.
            total_mlp_hidden = sum(block.mlp.hidden_dim for block in self.transformer.h)
            if total_mlp_hidden != 32 * config.n_embd:
                raise RuntimeError("redistributed MLPs must preserve total hidden width")
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # JEPA MTP removed: multi-token prediction hurts step count in 5-min budget
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Opt-in reproduction of the public learned skip-2 claim.  Keeping the
        # parameter unregistered when disabled preserves the baseline model and
        # optimizer state exactly.
        if os.environ.get("LEARNED_SKIP2", "0") == "1":
            self.skip2_lambdas = nn.Parameter(torch.empty(config.n_layer))
        else:
            self.register_parameter("skip2_lambdas", None)
        # Input-dependent x0 gating: per-layer scale for sigmoid gate on x0 skip (all layers)
        # gate = 2*sigmoid(scale * x.mean(-1)) modulates x0_lambdas contribution
        # Zero-init so gate starts at 1.0 (neutral = same as current scalar behavior)
        self.x0_gate_scales = nn.Parameter(torch.zeros(config.n_layer))
        # Multi-layer output pooling: aggregate last-K intermediate layers as additive correction
        self.n_pool_layers = min(4, config.n_layer)  # layers [n-4, n-3, n-2] contribute (3 weights)
        self.layer_pool_weights = nn.Parameter(torch.zeros(self.n_pool_layers - 1))
        # Value embeddings (unigram)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # Factored multi-hash bigram VE: K=2 half-dim tables concatenated per layer
        # Crossover: K=2 simplification recovers throughput
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.bigram_ve_layers = set(ve_layers)
        self.bigram_table_size = config.vocab_size * int(
            os.environ.get("BIGRAM_MULT", os.environ.get("NGRAM_MULT", 64))
        )  # CROSSOVER B: 64x bigram tables
        if self.bigram_table_size & (self.bigram_table_size - 1):
            raise ValueError("bigram table size must be a power of two")
        self.bigram_table_mask = self.bigram_table_size - 1
        self.bigram_K = 2
        # Dense gradients are the safe default because the target runtime's
        # fullgraph AOTAutograd support for sparse embedding backward must be
        # checked empirically.  The optimizer below handles either layout.
        self.ngram_direct_scratch_grad = _NGRAM_DIRECT_SCRATCH_GRAD
        self.ngram_sparse_grad = (
            os.environ.get("NGRAM_SPARSE_GRAD", "0") == "1"
            and not self.ngram_direct_scratch_grad
        )
        ngram_embedding_cls = (
            DirectScratchEmbedding
            if self.ngram_direct_scratch_grad
            else nn.Embedding
        )
        ngram_embedding_kwargs = (
            {} if self.ngram_direct_scratch_grad else {"sparse": self.ngram_sparse_grad}
        )
        half_kv_dim = kv_dim // 2
        # PER-LAYER DECORRELATED: completely disjoint hash prime pairs per bigram VE layer
        # Each layer uses entirely distinct multipliers -- zero prime reuse within bigram type
        # Constants from Murmur/FNV/golden-ratio family for good avalanche behavior
        _decorr_bigram_primes = [
            [(2654435761, 2246822519), (1013904223, 6291469)],   # layer 1: golden-ratio family
            [(374761393, 668265263), (3266489917, 104729)],      # layer 3: prime family
            [(1640531527, 97531), (48271, 40503)],               # layer 5: LCG/Knuth family
            [(16777619, 2166136261), (3432918353, 461845907)],   # layer 7: MurmurHash3 family
        ]
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(ve_layers):
            self.bigram_ves[str(layer_i)] = nn.ModuleList([
                ngram_embedding_cls(
                    self.bigram_table_size, half_kv_dim, **ngram_embedding_kwargs
                ),
                ngram_embedding_cls(
                    self.bigram_table_size, half_kv_dim, **ngram_embedding_kwargs
                ),
            ])
            self.bigram_hash_primes_per_layer[layer_i] = _decorr_bigram_primes[j % len(_decorr_bigram_primes)]
        # Shared factored trigram VE: ONE K=2 half-dim table pair reused across
        # all trigram layers {1,5,7}. Uses the champion's layer-1 trigram hash
        # pair; the other two per-layer hash families and table pairs are removed.
        # Each layer retains its own independent trigram_gate (defined per-block).
        trigram_placement = os.environ.get("TRIGRAM_PLACEMENT", "late")
        if trigram_placement not in {"late", "early"}:
            raise ValueError("TRIGRAM_PLACEMENT must be late or early")
        trigram_middle = ve_layers[1] if trigram_placement == "early" else ve_layers[-2]
        self.trigram_ve_layers = (
            {ve_layers[0], trigram_middle, ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        baseline_trigram_table_size = config.vocab_size * int(
            os.environ.get("TRIGRAM_MULT", os.environ.get("NGRAM_MULT", 64))
        )  # independently tunable; defaults to the published 64x table
        if baseline_trigram_table_size & (baseline_trigram_table_size - 1):
            raise ValueError("trigram table size must be a power of two")
        self.trigram_table_sizes = (
            baseline_trigram_table_size,
            baseline_trigram_table_size,
        )
        self.trigram_table_masks = tuple(
            table_size - 1 for table_size in self.trigram_table_sizes
        )
        # Champion layer-1 trigram hash pair (FNV+Murmur family) -- shared across all trigram layers.
        self.trigram_hash_primes = (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477)
        self.trigram_ves = nn.ModuleList([
            ngram_embedding_cls(
                self.trigram_table_sizes[0], half_kv_dim, **ngram_embedding_kwargs
            ),
            ngram_embedding_cls(
                self.trigram_table_sizes[1], half_kv_dim, **ngram_embedding_kwargs
            ),
        ])
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        transformer_init_scale = float(
            os.environ.get("TRANSFORMER_INIT_SCALE", "1.0")
        )
        if transformer_init_scale <= 0.0:
            raise ValueError("TRANSFORMER_INIT_SCALE must be positive")
        transformer_s = s * transformer_init_scale
        qkv_init_scale = float(
            os.environ.get("QKV_INIT_SCALE", str(transformer_init_scale))
        )
        if qkv_init_scale <= 0.0:
            raise ValueError("QKV_INIT_SCALE must be positive")
        qkv_s = s * qkv_init_scale
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -qkv_s, qkv_s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -qkv_s, qkv_s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -qkv_s, qkv_s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -transformer_s, transformer_s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(float(os.environ.get("X0_LAMBDA_INIT", 0.1)))
        if self.skip2_lambdas is not None:
            self.skip2_lambdas.fill_(0.05)
        self.x0_gate_scales.fill_(0.0)  # Zero-init: sigmoid(0)=0.5, 2*0.5=1.0 = neutral gate
        self.layer_pool_weights.fill_(0.0)
        # Value embeddings
        unigram_ve_init = os.environ.get("UNIGRAM_VE_INIT", "uniform")
        if unigram_ve_init not in ("uniform", "normal"):
            raise ValueError("UNIGRAM_VE_INIT must be uniform or normal")
        for ve in self.value_embeds.values():
            if unigram_ve_init == "normal":
                torch.nn.init.normal_(ve.weight, mean=0.0, std=s)
            else:
                torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.bigram_gate is not None:
                torch.nn.init.zeros_(block.attn.bigram_gate.weight)
            if block.attn.trigram_gate is not None:
                torch.nn.init.zeros_(block.attn.trigram_gate.weight)
            torch.nn.init.zeros_(block.attn.head_gate.weight)
        # Bigram VE: same init as regular VE (factored: two half-dim tables per layer)
        for layer_ves in self.bigram_ves.values():
            for bve in layer_ves:
                torch.nn.init.uniform_(bve.weight, -s, s)
                bve.to(dtype=torch.bfloat16)
        # Shared trigram VE init (factored: two half-dim tables, shared across layers {1,5,7})
        for tve in self.trigram_ves:
            torch.nn.init.uniform_(tve.weight, -s, s)
            tve.to(dtype=torch.bfloat16)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def initialize_direct_ngram_scratch(self):
        """Allocate dense/compact FP32 scratch after the model BF16 cast."""
        if not self.ngram_direct_scratch_grad:
            return
        bigram_inventory = tuple(
            (int(layer_key), table_i, embedding)
            for layer_key, layer_ves in self.bigram_ves.items()
            for table_i, embedding in enumerate(layer_ves)
        )
        bigram_embeddings = tuple(
            embedding for _layer_i, _table_i, embedding in bigram_inventory
        )
        trigram_embeddings = tuple(self.trigram_ves)
        if len(trigram_embeddings) != 2:
            raise RuntimeError("expected exactly two shared trigram tables")
        for embedding in bigram_embeddings + trigram_embeddings:
            if not isinstance(embedding, DirectScratchEmbedding):
                raise RuntimeError("direct scratch mode has a non-direct embedding")
        compact_rows = DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN
        if compact_rows >= 0x7FFFFFFF:
            raise ValueError("compact direct scratch requires B*T < 2**31-1")
        for layer_i, _table_i, embedding in bigram_inventory:
            embedding.initialize_grad_scratch(
                dtype=torch.float32,
                compact_rows=(
                    compact_rows
                    if layer_i in _COMPACT_FP32_BIGRAM_LAYERS
                    else None
                ),
            )
        for table_i, embedding in enumerate(trigram_embeddings):
            embedding.initialize_grad_scratch(
                dtype=torch.float32,
                compact_rows=(
                    compact_rows
                    if table_i in _COMPACT_FP32_TRIGRAM_SCRATCH_INDICES
                    else None
                ),
            )
        compact_bigram_records = [
            (layer_i, table_i, embedding)
            for layer_i, table_i, embedding in bigram_inventory
            if layer_i in _COMPACT_FP32_BIGRAM_LAYERS
        ]
        if [(layer_i, table_i) for layer_i, table_i, _embedding in compact_bigram_records] != [
            (1, 0),
            (1, 1),
        ]:
            raise RuntimeError(
                "compact bigram inventory must contain exactly L1 table0/table1"
            )
        if any(
            embedding.compact_scratch or embedding.compact_owner_key is not None
            for layer_i, _table_i, embedding in bigram_inventory
            if layer_i not in _COMPACT_FP32_BIGRAM_LAYERS
        ):
            raise RuntimeError("only layer-1 bigram tables may use compact scratch")
        compact_buffers = [
            trigram_embeddings[table_i].grad_scratch
            for table_i in sorted(_COMPACT_FP32_TRIGRAM_SCRATCH_INDICES)
        ]
        compact_owners = [
            trigram_embeddings[table_i].compact_owner_key
            for table_i in sorted(_COMPACT_FP32_TRIGRAM_SCRATCH_INDICES)
        ]
        if len({buffer.data_ptr() for buffer in compact_buffers}) != 2:
            raise RuntimeError("trigram compact scratch buffers must be independent")
        if len({owner.data_ptr() for owner in compact_owners}) != 2:
            raise RuntimeError("trigram compact owner maps must be independent")
        all_compact_embeddings = [
            embedding for _layer_i, _table_i, embedding in compact_bigram_records
        ] + [
            trigram_embeddings[table_i]
            for table_i in sorted(_COMPACT_FP32_TRIGRAM_SCRATCH_INDICES)
        ]
        if len({embedding.grad_scratch.data_ptr() for embedding in all_compact_embeddings}) != 4:
            raise RuntimeError("all four compact scratch buffers must be independent")
        if len({embedding.compact_owner_key.data_ptr() for embedding in all_compact_embeddings}) != 4:
            raise RuntimeError("all four compact owner maps must be independent")
        bigram_dense_bytes = 0
        bigram_compact_bytes = 0
        for layer_i, table_i, compact in compact_bigram_records:
            dense_bytes = compact.weight.numel() * 4
            compact_bytes = (
                compact.grad_scratch.numel() * compact.grad_scratch.element_size()
                + compact.compact_owner_key.numel()
                * compact.compact_owner_key.element_size()
            )
            bigram_dense_bytes += dense_bytes
            bigram_compact_bytes += compact_bytes
            print(
                "Compact FP32 bigram scratch: "
                f"layer={layer_i} table={table_i} "
                f"dense={dense_bytes / 2**30:.2f} GiB "
                f"compact+owner={compact_bytes / 2**20:.1f} MiB "
                f"saved={(dense_bytes - compact_bytes) / 2**30:.2f} GiB"
            )
        print(
            "L1-pair compact FP32 bigram scratch: "
            f"compact+owners={bigram_compact_bytes / 2**20:.1f} MiB "
            f"saved={(bigram_dense_bytes - bigram_compact_bytes) / 2**30:.3f} GiB"
        )
        total_dense_bytes = 0
        total_compact_bytes = 0
        for table_i in sorted(_COMPACT_FP32_TRIGRAM_SCRATCH_INDICES):
            compact = trigram_embeddings[table_i]
            dense_bytes = compact.weight.numel() * 4  # FP32 bytes/element
            compact_bytes = (
                compact.grad_scratch.numel() * compact.grad_scratch.element_size()
                + compact.compact_owner_key.numel()
                * compact.compact_owner_key.element_size()
            )
            total_dense_bytes += dense_bytes
            total_compact_bytes += compact_bytes
            print(
                "Compact FP32 trigram scratch: "
                f"table={table_i} "
                f"dense={dense_bytes / 2**30:.2f} GiB "
                f"compact+owner={compact_bytes / 2**20:.1f} MiB "
                f"saved={(dense_bytes - compact_bytes) / 2**30:.2f} GiB"
            )
        print(
            "Dual-compact FP32 trigram scratch: "
            f"compact+owners={total_compact_bytes / 2**20:.1f} MiB "
            f"saved={(total_dense_bytes - total_compact_bytes) / 2**30:.3f} GiB"
        )

    @torch.no_grad()
    def expand_bigram_tables_crn(self, target_mult, seed):
        """Expand 64x bigram tables without perturbing baseline state.

        The caller invokes this only after ``init_weights`` has initialized the
        complete baseline, including trigram tables. Prefix rows are copied
        exactly. Candidate-only tail rows use a private generator, and the
        ambient device RNG is restored in ``finally`` as an explicit CRN
        invariant even though no candidate draw is taken from it.
        """
        target_mult = int(target_mult)
        if target_mult <= 0:
            raise ValueError("BIGRAM_CRN_MULT must be positive")
        baseline_mult = 64
        expected_baseline_size = self.config.vocab_size * baseline_mult
        if self.bigram_table_size != expected_baseline_size:
            raise ValueError(
                "strict-CRN bigram expansion requires the model to be "
                f"constructed at {baseline_mult}x; got "
                f"{self.bigram_table_size / self.config.vocab_size:g}x"
            )
        target_size = self.config.vocab_size * target_mult
        if target_size < self.bigram_table_size:
            raise ValueError("BIGRAM_CRN_MULT cannot shrink the baseline tables")
        if target_size == self.bigram_table_size:
            return
        if target_size & (target_size - 1):
            raise ValueError("expanded bigram table size must be a power of two")

        embeddings = tuple(
            embedding
            for layer_i in sorted(self.bigram_ve_layers)
            for embedding in self.bigram_ves[str(layer_i)]
        )
        if len(embeddings) != len(self.bigram_ve_layers) * self.bigram_K:
            raise RuntimeError("unexpected bigram table inventory")
        device = embeddings[0].weight.device
        if any(embedding.weight.device != device for embedding in embeddings):
            raise RuntimeError("all bigram tables must be on one device")
        if device.type == "cuda":
            baseline_rng_state = torch.cuda.get_rng_state(device)
            restore_rng_state = lambda state: torch.cuda.set_rng_state(
                state, device
            )
        elif device.type == "cpu":
            baseline_rng_state = torch.get_rng_state()
            restore_rng_state = torch.set_rng_state
        else:
            raise RuntimeError(
                f"unsupported bigram expansion device type: {device.type}"
            )

        private_generator = torch.Generator(device=device)
        private_generator.manual_seed(
            (int(seed) ^ 0x42494752414D3132) & 0x7FFF_FFFF_FFFF_FFFF
        )
        init_bound = 3**0.5 * self.config.n_embd**-0.5
        original_size = self.bigram_table_size
        try:
            for embedding in embeddings:
                old_weight = embedding.weight
                if old_weight.is_meta:
                    raise RuntimeError("expand bigram tables only after materialization")
                if old_weight.shape[0] != original_size:
                    raise RuntimeError("bigram table row count differs from baseline")
                tail_fp32 = torch.empty(
                    (target_size - original_size, old_weight.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                torch.nn.init.uniform_(
                    tail_fp32,
                    -init_bound,
                    init_bound,
                    generator=private_generator,
                )
                expanded_weight = torch.empty(
                    (target_size, old_weight.shape[1]),
                    dtype=old_weight.dtype,
                    device=device,
                )
                expanded_weight[:original_size].copy_(old_weight)
                expanded_weight[original_size:].copy_(tail_fp32)
                embedding.weight = nn.Parameter(
                    expanded_weight,
                    requires_grad=old_weight.requires_grad,
                )
                embedding.num_embeddings = target_size
                # Drop transient references before allocating the next table;
                # the module now owns the expanded parameter.
                del old_weight, tail_fp32, expanded_weight
        finally:
            restore_rng_state(baseline_rng_state)

        self.bigram_table_size = target_size
        self.bigram_table_mask = target_size - 1
        print(
            "Strict-CRN bigram capacity: "
            f"{baseline_mult}x -> {target_mult}x "
            f"({len(embeddings)} tables; original rows preserved)"
        )

    @torch.no_grad()
    def expand_trigram_tables_crn(self, target_mults, seed):
        """Expand the shared trigram pair to independent strict-CRN sizes.

        Both tables first consume the exact private-RNG draws used by the
        symmetric 1024x/1024x candidate, then extend in original table order
        to the common 2048x/2048x target. The ambient device RNG is restored
        unconditionally.
        """
        target_mults = tuple(int(target_mult) for target_mult in target_mults)
        if len(target_mults) != 2:
            raise ValueError("trigram expansion requires exactly two multipliers")
        if any(target_mult <= 0 for target_mult in target_mults):
            raise ValueError("TRIGRAM_CRN_MULT_0/1 must be positive")

        baseline_mult = 64
        expected_baseline_size = self.config.vocab_size * baseline_mult
        expected_baseline_sizes = (expected_baseline_size, expected_baseline_size)
        if tuple(self.trigram_table_sizes) != expected_baseline_sizes:
            got_mults = tuple(
                table_size / self.config.vocab_size
                for table_size in self.trigram_table_sizes
            )
            raise ValueError(
                "strict-CRN trigram expansion requires both tables to be "
                f"constructed at {baseline_mult}x; got {got_mults}"
            )

        target_sizes = tuple(
            self.config.vocab_size * target_mult for target_mult in target_mults
        )
        if any(
            target_size < original_size
            for target_size, original_size in zip(
                target_sizes, self.trigram_table_sizes
            )
        ):
            raise ValueError("TRIGRAM_CRN_MULT_0/1 cannot shrink baseline tables")
        if any(target_size & (target_size - 1) for target_size in target_sizes):
            raise ValueError("expanded trigram table sizes must be powers of two")
        if target_sizes == expected_baseline_sizes:
            return

        embeddings = tuple(self.trigram_ves)
        if len(embeddings) != 2:
            raise RuntimeError("unexpected shared trigram table inventory")
        device = embeddings[0].weight.device
        if any(embedding.weight.device != device for embedding in embeddings):
            raise RuntimeError("all trigram tables must be on one device")
        if device.type == "cuda":
            baseline_rng_state = torch.cuda.get_rng_state(device)
            restore_rng_state = lambda state: torch.cuda.set_rng_state(state, device)
        elif device.type == "cpu":
            baseline_rng_state = torch.get_rng_state()
            restore_rng_state = torch.set_rng_state
        else:
            raise RuntimeError(
                f"unsupported trigram expansion device type: {device.type}"
            )

        private_generator = torch.Generator(device=device)
        private_generator.manual_seed(
            (int(seed) ^ 0x5452494752414D32) & 0x7FFF_FFFF_FFFF_FFFF
        )
        init_bound = 3**0.5 * self.config.n_embd**-0.5
        original_sizes = tuple(self.trigram_table_sizes)
        # Preserve the exact already-tested 1024x/1024x checkpoint before
        # drawing either table's new 1024x tail. Using min(target_sizes) here
        # would consume table0's 1024x->2048x tail before table1's old prefix
        # and would therefore silently break strict CRN for table1.
        crn_prefix_size = self.config.vocab_size * 1024
        if any(target_size < crn_prefix_size for target_size in target_sizes):
            raise ValueError("2048x/2048x candidate requires a 1024x CRN prefix")
        try:
            # Phase one deliberately keeps the tested 1024x/1024x
            # table0->table1 draw order. Allocate each final-size tensor once;
            # all post-prefix tails are filled in phase two.
            for table_i, (embedding, original_size, target_size) in enumerate(
                zip(embeddings, original_sizes, target_sizes)
            ):
                old_weight = embedding.weight
                if old_weight.is_meta:
                    raise RuntimeError("expand trigram tables only after materialization")
                if old_weight.shape[0] != original_size:
                    raise RuntimeError(
                        f"trigram table{table_i} row count differs from baseline"
                    )
                expanded_weight = torch.empty(
                    (target_size, old_weight.shape[1]),
                    dtype=old_weight.dtype,
                    device=device,
                )
                expanded_weight[:original_size].copy_(old_weight)
                if crn_prefix_size > original_size:
                    common_tail_fp32 = torch.empty(
                        (crn_prefix_size - original_size, old_weight.shape[1]),
                        dtype=torch.float32,
                        device=device,
                    )
                    torch.nn.init.uniform_(
                        common_tail_fp32,
                        -init_bound,
                        init_bound,
                        generator=private_generator,
                    )
                    expanded_weight[original_size:crn_prefix_size].copy_(
                        common_tail_fp32
                    )
                    del common_tail_fp32
                embedding.weight = nn.Parameter(
                    expanded_weight,
                    requires_grad=old_weight.requires_grad,
                )
                embedding.num_embeddings = target_size
                del old_weight, expanded_weight

            # Only after both prefixes match the tested symmetric 1024x/1024x
            # candidate do the two 2048x tails consume private-RNG draws in
            # original table order.
            for embedding, target_size in zip(embeddings, target_sizes):
                if target_size == crn_prefix_size:
                    continue
                asymmetric_tail_fp32 = torch.empty(
                    (
                        target_size - crn_prefix_size,
                        embedding.weight.shape[1],
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                torch.nn.init.uniform_(
                    asymmetric_tail_fp32,
                    -init_bound,
                    init_bound,
                    generator=private_generator,
                )
                embedding.weight[crn_prefix_size:].copy_(asymmetric_tail_fp32)
                del asymmetric_tail_fp32
        finally:
            restore_rng_state(baseline_rng_state)

        self.trigram_table_sizes = target_sizes
        self.trigram_table_masks = tuple(
            table_size - 1 for table_size in self.trigram_table_sizes
        )
        print(
            "Strict-CRN trigram capacity: "
            f"{baseline_mult}x/{baseline_mult}x -> "
            f"{target_mults[0]}x/{target_mults[1]}x "
            "(common prefix and ambient RNG preserved)"
        )

    def _lookup_ngram_pair(self, embeddings, indices):
        """Lookup one K=2 table pair, optionally without half-output tensors."""
        if len(embeddings) != 2 or len(indices) != 2:
            raise RuntimeError("K=2 n-gram lookup requires exactly two tables/indices")
        if _DIRECT_SCRATCH_PAIR_FUSION:
            embedding0, embedding1 = embeddings
            if not isinstance(embedding0, DirectScratchEmbedding) or not isinstance(
                embedding1,
                DirectScratchEmbedding,
            ):
                raise RuntimeError("pair fusion requires direct-scratch embeddings")
            return torch.ops.ngram_direct.embedding_pair(
                embedding0.weight,
                embedding1.weight,
                indices[0],
                indices[1],
                embedding0.grad_scratch,
                embedding1.grad_scratch,
            )
        return torch.cat(
            [embeddings[0](indices[0]), embeddings[1](indices[1])],
            dim=-1,
        )

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=int(float(os.environ.get("ROTARY_BASE", 1000000))), device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SLT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // int(os.environ.get("SHORT_DIV", 2))
        tiny_window = long_window // int(os.environ.get("TINY_DIV", 4))
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0), "T": (tiny_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + (self.skip2_lambdas.numel() if self.skip2_lambdas is not None else 0)
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.layer_pool_weights.numel()
            + (self.skip2_lambdas.numel() if self.skip2_lambdas is not None else 0)
        )
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        ngram_ve_betas=None,  # if None, uses adam_betas
        ngram_ve_lr_scale=1.0,  # discriminative LR scale for n-gram VE (ULMFiT-inspired)
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas, self.x0_gate_scales]  # gate scales grouped with x0 lambdas
        skip2_params = [] if self.skip2_lambdas is None else [self.skip2_lambdas]
        # Build parameters and hash specifications in the exact same order.
        # The dense-grad lazy optimizer recomputes these rows from the current
        # token batch; this avoids a full-table nonzero scan and does not need
        # torch.unique because the dense embedding backward has already summed
        # every repeated/colliding row.
        bigram_ve_params = []
        bigram_direct_grad_scratch = []
        bigram_direct_compact_owner_key = []
        bigram_active_index_specs = []
        for layer_i in sorted(self.bigram_ve_layers):
            layer_primes = self.bigram_hash_primes_per_layer[layer_i]
            for table_i, embedding in enumerate(self.bigram_ves[str(layer_i)]):
                parameter = embedding.weight
                p1, p2 = layer_primes[table_i]
                bigram_ve_params.append(parameter)
                bigram_direct_grad_scratch.append(
                    getattr(embedding, "grad_scratch", None)
                )
                bigram_direct_compact_owner_key.append(
                    getattr(embedding, "compact_owner_key", None)
                )
                bigram_active_index_specs.append(
                    ("bigram", p1, p2, self.bigram_table_size)
                )

        trigram_ve_params = list(self.trigram_ves.parameters())
        trigram_direct_grad_scratch = [
            getattr(embedding, "grad_scratch", None)
            for embedding in self.trigram_ves
        ]
        trigram_direct_compact_owner_key = [
            getattr(embedding, "compact_owner_key", None)
            for embedding in self.trigram_ves
        ]
        lp = self.trigram_hash_primes
        trigram_active_index_specs = [
            ("trigram", lp[0], lp[1], lp[2], self.trigram_table_sizes[0]),
            ("trigram", lp[3], lp[4], lp[5], self.trigram_table_sizes[1]),
        ]
        assert len(bigram_ve_params) == len(bigram_active_index_specs)
        assert len(trigram_ve_params) == len(trigram_active_index_specs)
        pool_params = [self.layer_pool_weights]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
            + len(bigram_ve_params)
            + len(trigram_ve_params)
            + len(pool_params)
            + len(skip2_params)
        )
        # Scale LR ~ 1/sqrt(dmodel) (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if ngram_ve_betas is None:
            ngram_ve_betas = adam_betas
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": float(os.environ.get("LM_HEAD_WEIGHT_DECAY", "0.0")),
                "demon_beta1": True,  # Apply Demon beta1 scheduling
            },
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": float(os.environ.get("TOKEN_EMBED_WEIGHT_DECAY", "0.0")),
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": value_embeds_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": float(os.environ.get("VALUE_EMBED_WEIGHT_DECAY", "0.0")),
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                # No demon_beta1: scalar params keep fixed beta1
            },
            {
                "kind": "adamw",
                "params": x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.002,  # x0WD=0.002 (proven optimal)
                "is_x0_muon_warmdown": True,  # x0 Muon warmdown
            },
            {
                "kind": "rmsprop",
                "params": bigram_ve_params,
                "lr": (
                    embedding_lr
                    * dmodel_lr_scale
                    * ngram_ve_lr_scale
                    * float(os.environ.get("BIGRAM_LR_SCALE", "1.0"))
                ),
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
                "active_index_specs": tuple(bigram_active_index_specs),
                "sparse_grad": self.ngram_sparse_grad,
                "direct_grad_scratch": (
                    tuple(bigram_direct_grad_scratch)
                    if self.ngram_direct_scratch_grad
                    else None
                ),
                "direct_compact_owner_key": (
                    tuple(bigram_direct_compact_owner_key)
                    if self.ngram_direct_scratch_grad
                    else None
                ),
            },
            {
                "kind": "rmsprop",
                "params": trigram_ve_params,
                "lr": (
                    embedding_lr
                    * dmodel_lr_scale
                    * ngram_ve_lr_scale
                    * float(os.environ.get("TRIGRAM_LR_SCALE", "1.0"))
                ),
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
                "active_index_specs": tuple(trigram_active_index_specs),
                "sparse_grad": self.ngram_sparse_grad,
                "direct_grad_scratch": (
                    tuple(trigram_direct_grad_scratch)
                    if self.ngram_direct_scratch_grad
                    else None
                ),
                "direct_compact_owner_key": (
                    tuple(trigram_direct_compact_owner_key)
                    if self.ngram_direct_scratch_grad
                    else None
                ),
            },
            {
                "kind": "adamw",
                "params": pool_params,
                "lr": scalar_lr * 0.15,  # revert to formula (0.75*0.15=0.1125)
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ]
        if skip2_params:
            # Match the public claim: a separate fixed-beta AdamW scalar group
            # at SCALAR_LR * 0.01, with no weight decay or Demon scheduling.
            param_groups.insert(
                5,
                {
                    "kind": "adamw",
                    "params": skip2_params,
                    "lr": scalar_lr * 0.01,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                },
            )
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": "muon",
                    "params": group_params,
                    "lr": matrix_lr,
                    "momentum": float(os.environ.get("MUON_MOMENTUM", 0.95)),
                    "ns_steps": int(os.environ.get("NS_STEPS", 5)),
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        if self.skip2_lambdas is not None:
            # These buffers follow the public implementation's pre-block
            # residual states; at layer i, x_prev2 is x[i-2].
            x_prev2 = x
            x_prev1 = x
        # Precompute n-gram hash indices before the layer loop: shifted token indices +
        # per-layer bigram hashes (decorrelated primes), plus ONE shared trigram hash pair
        # (this run's change) reused at all trigram layers {1,5,7}.
        raw_prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        raw_prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        at_bos = idx == BOS_TOKEN_ID
        after_bos = raw_prev_idx == BOS_TOKEN_ID
        prev_idx = torch.where(at_bos, idx, raw_prev_idx)
        prev2_idx = torch.where(
            at_bos | after_bos,
            torch.full_like(idx, BOS_TOKEN_ID),
            raw_prev2_idx,
        )
        # Precompute per-layer bigram hash indices (different primes per layer for collision decorrelation)
        bigram_indices_per_layer = {}
        for layer_i in self.bigram_ve_layers:
            layer_bg_primes = self.bigram_hash_primes_per_layer[layer_i]
            bigram_indices_per_layer[layer_i] = [
                ((prev_idx * p1) ^ (idx * p2)) & self.bigram_table_mask
                for p1, p2 in layer_bg_primes
            ]
        # Shared trigram hash indices: ONE pair computed once (champion layer-1 primes),
        # reused at all trigram layers {1,5,7}.
        lp = self.trigram_hash_primes
        trigram_indices = (
            ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2]))
            & self.trigram_table_masks[0],
            ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5]))
            & self.trigram_table_masks[1],
        )
        # Shared trigram VE lookup: computed once, reused at every trigram layer.
        shared_tgve = self._lookup_ngram_pair(self.trigram_ves, trigram_indices)
        n_layer = len(self.transformer.h)
        pool_start = n_layer - self.n_pool_layers
        pool_residual = None
        saved_attn_source_norm = None
        for i, block in enumerate(self.transformer.h):
            # Input-dependent x0 gate on ALL 8 layers: 2*sigmoid(scale*mean(x)) modulates x0 contribution
            # Starts at 1.0 (gate_scales=0 -> sigmoid(0)=0.5 -> 2*0.5=1.0)
            x0_gate = 2.0 * torch.sigmoid(self.x0_gate_scales[i] * x.float().mean(-1, keepdim=True)).to(x.dtype)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0_gate * x0
            if self.skip2_lambdas is not None:
                if i >= 2:
                    x = x + self.skip2_lambdas[i] * x_prev2
                x_prev2 = x_prev1
                x_prev1 = x
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                ve = None
            # Factored multi-hash bigram VE: concat K=2 half-dim lookups from independent hashes (per-layer primes)
            if i in self.bigram_ve_layers:
                layer_ves = self.bigram_ves[str(i)]
                layer_indices = bigram_indices_per_layer[i]
                bgve = self._lookup_ngram_pair(layer_ves, layer_indices)
            else:
                bgve = None
            # Shared factored trigram VE: reuse the single shared 768-wide lookup at every trigram layer.
            # Each layer applies its own independent trigram_gate inside block(...).
            if i in self.trigram_ve_layers:
                tgve = shared_tgve
            else:
                tgve = None
            attn_source_norm = (
                saved_attn_source_norm if i in ATTN_SOURCE_LAYERS else None
            )
            x = block(
                x,
                ve,
                cos_sin,
                self.window_sizes[i],
                bigram_ve=bgve,
                trigram_ve=tgve,
                attn_source_norm=attn_source_norm,
            )
            if i == ATTN_SOURCE_AFTER_LAYER:
                saved_attn_source_norm = norm(x)
            if i == pool_start:
                pool_residual = self.layer_pool_weights[0] * x
            elif i == pool_start + 1:
                pool_residual = pool_residual + self.layer_pool_weights[1] * x
            elif i == pool_start + 2:
                pool_residual = pool_residual + self.layer_pool_weights[2] * x
        if pool_residual is not None:
            x = x + pool_residual
        x = norm(x)

        # Decoupled softcap in BF16: skip float() cast, halve logit tensor memory
        # Since model is natively BF16, softcap in BF16 should be numerically adequate
        logits = self.lm_head(x)
        logits = 16.5 * torch.tanh(logits / 15.0)

        if targets is not None:
            if _USE_QUACK_CE:
                # Quack accumulates the softmax/loss in FP32 while consuming
                # BF16 logits directly.  This avoids materializing the 4.8 GB
                # float32 logits copy made by the reference PyTorch CE path.
                loss = quack_cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1,
                    reduction=reduction,
                )
            else:
                # Reference path used by the published SOTA.
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction=reduction,
                )
            if _REUSE_NGRAM_ROWS and reduction == "mean":
                # These exact index tensors are already live for embedding
                # backward.  Returning views lets lazy RMSProp reuse them
                # instead of hashing all ten tables a second time.
                active_rows = tuple(
                    table_rows.flatten()
                    for layer_i in sorted(self.bigram_ve_layers)
                    for table_rows in bigram_indices_per_layer[layer_i]
                ) + tuple(table_rows.flatten() for table_rows in trigram_indices)
                return loss, active_rows
            return loss
        # Eval path: need float32 logits
        return logits.float()


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

_OPTIMIZER_COMPILE_MODE = os.environ.get("OPTIMIZER_COMPILE_MODE", "").strip()
if _OPTIMIZER_COMPILE_MODE not in {
    "",
    "default",
    "reduce-overhead",
    "max-autotune",
}:
    raise ValueError(
        "OPTIMIZER_COMPILE_MODE must be one of: "
        "default, reduce-overhead, max-autotune"
    )
_OPTIMIZER_COMPILE_KWARGS = (
    {} if not _OPTIMIZER_COMPILE_MODE else {"mode": _OPTIMIZER_COMPILE_MODE}
)

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(
    dynamic=False,
    fullgraph=True,
    **_OPTIMIZER_COMPILE_KWARGS,
)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


@torch.compile(
    dynamic=False,
    fullgraph=True,
    **_OPTIMIZER_COMPILE_KWARGS,
)
def rmsprop_step_fused(p, grad, exp_avg_sq, step_t, lr_t, beta2_t, eps_t, wd_t):
    """Dense published recurrence, retained as the opt-in GPU A/B oracle."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(grad / denom, alpha=-lr_t)


@triton.jit
def _triton_lazy_rmsprop_dense_kernel(
    parameter_ptr,
    dense_grad_ptr,
    moment_ptr,
    row_index_ptr,
    row_marker_ptr,
    n_active,
    n_rows,
    step_marker,
    lr,
    beta2,
    eps,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_PROFILE: tl.constexpr,
):
    """Claim duplicate row ids once, then update the complete row in place.

    One Triton program owns ``BLOCK_R`` candidate rows and all ``D`` columns.
    ``atomic_xchg`` makes exactly one occurrence of a duplicated hash the
    owner for this step.  The dense embedding backward has already summed all
    token and hash collisions, so every duplicate would otherwise perform the
    same update.
    """
    row_slot = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    slot_mask = row_slot < n_active
    row = tl.load(row_index_ptr + row_slot, mask=slot_mask, other=0).to(tl.int64)
    row_mask = slot_mask & (row >= 0) & (row < n_rows)
    previous_marker = tl.atomic_xchg(
        row_marker_ptr + row,
        step_marker,
        mask=row_mask,
        sem="relaxed",
        scope="gpu",
    )
    owns_row = row_mask & (previous_marker != step_marker)

    col = tl.arange(0, BLOCK_D)
    element_mask = owns_row[:, None] & (col[None, :] < D)
    offset = row[:, None] * D + col[None, :]

    grad = tl.load(dense_grad_ptr + offset, mask=element_mask, other=0.0).to(tl.float32)
    moment = tl.load(moment_ptr + offset, mask=element_mask, other=0.0).to(tl.float32)
    parameter = tl.load(parameter_ptr + offset, mask=element_mask, other=0.0).to(tl.float32)

    # Profile 0 mirrors the actual Inductor oracle generated for the dense
    # recurrence: all operations remain FP32 inside one fused kernel, including
    # CUDA libdevice powf for bias correction, and only the two stores round.
    # Other profiles isolate plausible materialization boundaries for A/B.
    grad_sq = grad * grad
    if ROUND_PROFILE == 2 or ROUND_PROFILE == 3:
        grad_sq = grad_sq.to(tl.bfloat16).to(tl.float32)
    moment_new = moment + (1.0 - beta2) * (grad_sq - moment)
    if ROUND_PROFILE == 1 or ROUND_PROFILE == 2 or ROUND_PROFILE == 3:
        moment_new = moment_new.to(tl.bfloat16).to(tl.float32)

    # Multiplication works for both Triton's specialized step=1 Python value
    # and the regular runtime int32 signature used by subsequent steps.
    step_float32 = step_marker * 1.0
    bias2 = 1.0 - libdevice.pow(beta2, step_float32)
    scaled_moment = moment_new / bias2
    if ROUND_PROFILE == 2:
        scaled_moment = scaled_moment.to(tl.bfloat16).to(tl.float32)
    denom = libdevice.sqrt(scaled_moment)
    if ROUND_PROFILE == 2:
        denom = denom.to(tl.bfloat16).to(tl.float32)
    denom = denom + eps
    if ROUND_PROFILE == 2:
        denom = denom.to(tl.bfloat16).to(tl.float32)

    normalized_grad = grad / denom
    if ROUND_PROFILE == 2:
        normalized_grad = normalized_grad.to(tl.bfloat16).to(tl.float32)
    parameter_new = parameter + normalized_grad * (-lr)

    tl.store(moment_ptr + offset, moment_new, mask=element_mask)
    tl.store(parameter_ptr + offset, parameter_new, mask=element_mask)


def triton_lazy_rmsprop_dense_rows_step(
    parameter,
    row_indices,
    dense_grad,
    exp_avg_sq,
    row_marker,
    step,
    lr,
    beta2,
    eps,
    round_profile=None,
):
    """Launch the single-pass, duplicate-safe active-row RMSProp kernel."""
    if parameter.dtype != torch.bfloat16:
        raise TypeError(f"Triton lazy RMSProp requires BF16 parameters, got {parameter.dtype}")
    if dense_grad.dtype != torch.bfloat16 or exp_avg_sq.dtype != torch.bfloat16:
        raise TypeError("Triton lazy RMSProp requires BF16 gradient and moment tensors")
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"row indices must be int32/int64, got {row_indices.dtype}")
    if row_marker.dtype != torch.int32:
        raise TypeError(f"row marker must be int32, got {row_marker.dtype}")
    if parameter.ndim != 2 or parameter.shape != dense_grad.shape or parameter.shape != exp_avg_sq.shape:
        raise ValueError("parameter, dense gradient, and moment must have one common [rows, dim] shape")
    if row_marker.ndim != 1 or row_marker.numel() != parameter.shape[0]:
        raise ValueError("row marker length must equal the embedding table row count")
    if not all(t.is_cuda for t in (parameter, dense_grad, exp_avg_sq, row_indices, row_marker)):
        raise ValueError("Triton lazy RMSProp tensors must all be CUDA tensors")
    if not all(t.is_contiguous() for t in (parameter, dense_grad, exp_avg_sq, row_indices, row_marker)):
        raise ValueError("Triton lazy RMSProp currently requires contiguous tensors")
    if not 0 < step < 2**31:
        raise ValueError(f"step marker must fit a positive int32, got {step}")

    n_active = row_indices.numel()
    if n_active == 0:
        return
    n_rows, width = parameter.shape
    if width > 512:
        raise ValueError(f"embedding width {width} exceeds the BLOCK_D=512 kernel limit")
    # The published implementation transports changing hyperparameters in CPU
    # float32 tensors. Round those scalar inputs before handing them to Triton;
    # bias correction itself uses CUDA powf inside the kernel, like Inductor.
    def as_float32(value):
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    lr_float32 = as_float32(lr)
    beta2_float32 = as_float32(beta2)
    eps_float32 = as_float32(eps)
    if round_profile is None:
        round_profile = _TRITON_RMS_ROUND_PROFILE
    if round_profile not in _TRITON_RMS_ROUND_PROFILES.values():
        raise ValueError(f"unknown Triton RMSProp round profile code: {round_profile}")
    grid = (triton.cdiv(n_active, _TRITON_RMS_BLOCK_R),)
    _triton_lazy_rmsprop_dense_kernel[grid](
        parameter,
        dense_grad,
        exp_avg_sq,
        row_indices,
        row_marker,
        n_active,
        n_rows,
        int(step),
        lr_float32,
        beta2_float32,
        eps_float32,
        D=width,
        BLOCK_R=_TRITON_RMS_BLOCK_R,
        BLOCK_D=512,
        ROUND_PROFILE=round_profile,
        num_warps=_TRITON_RMS_NUM_WARPS,
        num_stages=1,
    )


@triton.jit
def _triton_zero_uncoalesced_sparse_rows_kernel(
    grad_accum_ptr,
    row_index_ptr,
    row_marker_ptr,
    n_active,
    n_rows,
    clear_marker,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Clear only rows touched by this sparse batch, once per distinct row.

    The scratch table is deliberately allocated with ``torch.empty``.  A
    negative optimizer-step generation claims each row before it is cleared;
    the following scatter kernel therefore never observes stale or
    uninitialized data.  This is an active-row clear, not a full-table clear.
    """
    row_slot = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    slot_mask = row_slot < n_active
    row = tl.load(row_index_ptr + row_slot, mask=slot_mask, other=0).to(tl.int64)
    row_mask = slot_mask & (row >= 0) & (row < n_rows)
    previous_marker = tl.atomic_xchg(
        row_marker_ptr + row,
        clear_marker,
        mask=row_mask,
        sem="relaxed",
        scope="gpu",
    )
    owns_row = row_mask & (previous_marker != clear_marker)

    col = tl.arange(0, BLOCK_D)
    element_mask = owns_row[:, None] & (col[None, :] < D)
    offset = row[:, None] * D + col[None, :]
    tl.store(grad_accum_ptr + offset, 0.0, mask=element_mask)


@triton.jit
def _triton_scatter_uncoalesced_sparse_rows_kernel(
    grad_accum_ptr,
    sparse_grad_ptr,
    row_index_ptr,
    n_active,
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate duplicate sparse COO occurrences in FP32 by table row."""
    row_slot = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    slot_mask = row_slot < n_active
    row = tl.load(row_index_ptr + row_slot, mask=slot_mask, other=0).to(tl.int64)
    row_mask = slot_mask & (row >= 0) & (row < n_rows)

    col = tl.arange(0, BLOCK_D)
    element_mask = row_mask[:, None] & (col[None, :] < D)
    table_offset = row[:, None] * D + col[None, :]
    occurrence_offset = row_slot[:, None] * D + col[None, :]
    occurrence_grad = tl.load(
        sparse_grad_ptr + occurrence_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        grad_accum_ptr + table_offset,
        occurrence_grad,
        mask=element_mask,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _triton_zero_direct_scratch_rows_kernel(
    grad_scratch_ptr,
    index_ptr,
    index_stride_b,
    index_stride_t,
    B,
    T,
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Clear scratch only for this lookup's rows; duplicate zero stores agree."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row = tl.load(
        index_ptr + batch * index_stride_b + token * index_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)

    col = tl.arange(0, BLOCK_D)
    element_mask = row_mask[:, None] & (col[None, :] < D)
    tl.store(
        grad_scratch_ptr + row[:, None] * D + col[None, :],
        0.0,
        mask=element_mask,
    )


@triton.jit
def _triton_direct_embedding_pair_kernel(
    weight0_ptr,
    weight1_ptr,
    index0_ptr,
    index1_ptr,
    output_ptr,
    index0_stride_b,
    index0_stride_t,
    index1_stride_b,
    index1_stride_t,
    output_stride_b,
    output_stride_t,
    output_stride_d,
    B,
    T,
    n_rows0,
    n_rows1,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather two D-wide tables straight into adjacent output halves."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row0 = tl.load(
        index0_ptr + batch * index0_stride_b + token * index0_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row1 = tl.load(
        index1_ptr + batch * index1_stride_b + token * index1_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)

    col = tl.arange(0, BLOCK_D)
    col_mask = col < D
    mask0 = (
        occurrence_mask[:, None]
        & (row0[:, None] >= 0)
        & (row0[:, None] < n_rows0)
        & col_mask[None, :]
    )
    mask1 = (
        occurrence_mask[:, None]
        & (row1[:, None] >= 0)
        & (row1[:, None] < n_rows1)
        & col_mask[None, :]
    )
    value0 = tl.load(
        weight0_ptr + row0[:, None] * D + col[None, :],
        mask=mask0,
        other=0.0,
    )
    value1 = tl.load(
        weight1_ptr + row1[:, None] * D + col[None, :],
        mask=mask1,
        other=0.0,
    )
    output_base = (
        batch[:, None] * output_stride_b
        + token[:, None] * output_stride_t
    )
    tl.store(
        output_ptr + output_base + col[None, :] * output_stride_d,
        value0,
        mask=mask0,
    )
    tl.store(
        output_ptr + output_base + (D + col[None, :]) * output_stride_d,
        value1,
        mask=mask1,
    )


@triton.jit
def _triton_zero_direct_scratch_pair_kernel(
    grad_scratch0_ptr,
    grad_scratch1_ptr,
    index0_ptr,
    index1_ptr,
    index0_stride_b,
    index0_stride_t,
    index1_stride_b,
    index1_stride_t,
    B,
    T,
    n_rows0,
    n_rows1,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Clear both K=2 scratch rows in one launch."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row0 = tl.load(
        index0_ptr + batch * index0_stride_b + token * index0_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row1 = tl.load(
        index1_ptr + batch * index1_stride_b + token * index1_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    col = tl.arange(0, BLOCK_D)
    col_mask = col < D
    mask0 = (
        occurrence_mask[:, None]
        & (row0[:, None] >= 0)
        & (row0[:, None] < n_rows0)
        & col_mask[None, :]
    )
    mask1 = (
        occurrence_mask[:, None]
        & (row1[:, None] >= 0)
        & (row1[:, None] < n_rows1)
        & col_mask[None, :]
    )
    tl.store(
        grad_scratch0_ptr + row0[:, None] * D + col[None, :],
        0.0,
        mask=mask0,
    )
    tl.store(
        grad_scratch1_ptr + row1[:, None] * D + col[None, :],
        0.0,
        mask=mask1,
    )


@triton.jit
def _triton_scatter_direct_scratch_pair_kernel(
    grad_scratch0_ptr,
    grad_scratch1_ptr,
    grad_output_ptr,
    index0_ptr,
    index1_ptr,
    index0_stride_b,
    index0_stride_t,
    index1_stride_b,
    index1_stride_t,
    grad_stride_b,
    grad_stride_t,
    grad_stride_d,
    B,
    T,
    n_rows0,
    n_rows1,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCRATCH0_BF16: tl.constexpr,
    SCRATCH1_BF16: tl.constexpr,
):
    """Scatter both output halves into independently typed scratch tables."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row0 = tl.load(
        index0_ptr + batch * index0_stride_b + token * index0_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row1 = tl.load(
        index1_ptr + batch * index1_stride_b + token * index1_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    col = tl.arange(0, BLOCK_D)
    col_mask = col < D
    mask0 = (
        occurrence_mask[:, None]
        & (row0[:, None] >= 0)
        & (row0[:, None] < n_rows0)
        & col_mask[None, :]
    )
    mask1 = (
        occurrence_mask[:, None]
        & (row1[:, None] >= 0)
        & (row1[:, None] < n_rows1)
        & col_mask[None, :]
    )
    grad_base = (
        batch[:, None] * grad_stride_b
        + token[:, None] * grad_stride_t
    )
    grad0 = tl.load(
        grad_output_ptr + grad_base + col[None, :] * grad_stride_d,
        mask=mask0,
        other=0.0,
    ).to(tl.float32)
    grad1 = tl.load(
        grad_output_ptr + grad_base + (D + col[None, :]) * grad_stride_d,
        mask=mask1,
        other=0.0,
    ).to(tl.float32)
    if SCRATCH0_BF16:
        grad0 = grad0.to(tl.bfloat16)
    if SCRATCH1_BF16:
        grad1 = grad1.to(tl.bfloat16)
    tl.atomic_add(
        grad_scratch0_ptr + row0[:, None] * D + col[None, :],
        grad0,
        mask=mask0,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        grad_scratch1_ptr + row1[:, None] * D + col[None, :],
        grad1,
        mask=mask1,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _triton_scatter_direct_scratch_rows_kernel(
    grad_scratch_ptr,
    grad_output_ptr,
    index_ptr,
    index_stride_b,
    index_stride_t,
    grad_stride_b,
    grad_stride_t,
    grad_stride_d,
    B,
    T,
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCRATCH_BF16: tl.constexpr,
):
    """Scatter a possibly strided [B,T,D] gradient into FP32 or BF16."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row = tl.load(
        index_ptr + batch * index_stride_b + token * index_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)

    col = tl.arange(0, BLOCK_D)
    element_mask = row_mask[:, None] & (col[None, :] < D)
    grad_offset = (
        batch[:, None] * grad_stride_b
        + token[:, None] * grad_stride_t
        + col[None, :] * grad_stride_d
    )
    occurrence_grad = tl.load(
        grad_output_ptr + grad_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    if SCRATCH_BF16:
        occurrence_grad = occurrence_grad.to(tl.bfloat16)
    tl.atomic_add(
        grad_scratch_ptr + row[:, None] * D + col[None, :],
        occurrence_grad,
        mask=element_mask,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _triton_zero_direct_owner_key_kernel(
    owner_key_ptr,
    n_rows,
    BLOCK: tl.constexpr,
):
    """Reset the compact per-row owner table, not the D-wide scratch."""
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(owner_key_ptr + row, 0, mask=row < n_rows)


@triton.jit
def _triton_select_direct_owner_min_kernel(
    owner_key_ptr,
    index_ptr,
    index_stride_b,
    index_stride_t,
    B,
    T,
    n_rows,
    BLOCK: tl.constexpr,
):
    """Select the first flattened occurrence of every active table row."""
    occurrence = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row = tl.load(
        index_ptr + batch * index_stride_b + token * index_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)
    # owner_key starts at zero.  Larger inverted keys represent earlier flat
    # occurrences, so atomic_max deterministically chooses min(occurrence).
    inverted_occurrence = (0x7FFFFFFF - occurrence).to(tl.int32)
    tl.atomic_max(
        owner_key_ptr + row,
        inverted_occurrence,
        mask=row_mask,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _triton_scatter_direct_compact_owner_kernel(
    compact_scratch_ptr,
    owner_key_ptr,
    grad_output_ptr,
    index_ptr,
    index_stride_b,
    index_stride_t,
    grad_stride_b,
    grad_stride_t,
    grad_stride_d,
    B,
    T,
    n_rows,
    compact_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Sum each table row into its deterministic first-occurrence slot."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row = tl.load(
        index_ptr + batch * index_stride_b + token * index_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)
    owner_key = tl.load(owner_key_ptr + row, mask=row_mask, other=0)
    owner_occurrence = (0x7FFFFFFF - owner_key).to(tl.int64)
    owner_mask = (
        row_mask
        & (owner_key > 0)
        & (owner_occurrence >= 0)
        & (owner_occurrence < compact_rows)
    )

    col = tl.arange(0, BLOCK_D)
    element_mask = owner_mask[:, None] & (col[None, :] < D)
    grad_offset = (
        batch[:, None] * grad_stride_b
        + token[:, None] * grad_stride_t
        + col[None, :] * grad_stride_d
    )
    occurrence_grad = tl.load(
        grad_output_ptr + grad_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        compact_scratch_ptr + owner_occurrence[:, None] * D + col[None, :],
        occurrence_grad,
        mask=element_mask,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _triton_initialize_direct_owner_rows_kernel(
    grad_scratch_ptr,
    owner_key_ptr,
    grad_output_ptr,
    index_ptr,
    index_stride_b,
    index_stride_t,
    grad_stride_b,
    grad_stride_t,
    grad_stride_d,
    B,
    T,
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCRATCH_BF16: tl.constexpr,
):
    """Plain-store the first occurrence gradient for each active row."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row = tl.load(
        index_ptr + batch * index_stride_b + token * index_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)
    selected_key = tl.load(
        owner_key_ptr + row,
        mask=row_mask,
        other=0,
    )
    occurrence_key = (0x7FFFFFFF - occurrence).to(tl.int32)
    owns_row = row_mask & (selected_key == occurrence_key)

    col = tl.arange(0, BLOCK_D)
    element_mask = owns_row[:, None] & (col[None, :] < D)
    grad_offset = (
        batch[:, None] * grad_stride_b
        + token[:, None] * grad_stride_t
        + col[None, :] * grad_stride_d
    )
    occurrence_grad = tl.load(
        grad_output_ptr + grad_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    if SCRATCH_BF16:
        occurrence_grad = occurrence_grad.to(tl.bfloat16)
    tl.store(
        grad_scratch_ptr + row[:, None] * D + col[None, :],
        occurrence_grad,
        mask=element_mask,
    )


@triton.jit
def _triton_scatter_direct_duplicate_rows_kernel(
    grad_scratch_ptr,
    owner_key_ptr,
    grad_output_ptr,
    index_ptr,
    index_stride_b,
    index_stride_t,
    grad_stride_b,
    grad_stride_t,
    grad_stride_d,
    B,
    T,
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCRATCH_BF16: tl.constexpr,
):
    """Atomically add only non-owner occurrences after owner rows exist."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n_occurrences = B * T
    occurrence_mask = occurrence < n_occurrences
    batch = occurrence // T
    token = occurrence - batch * T
    row = tl.load(
        index_ptr + batch * index_stride_b + token * index_stride_t,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)
    selected_key = tl.load(
        owner_key_ptr + row,
        mask=row_mask,
        other=0,
    )
    occurrence_key = (0x7FFFFFFF - occurrence).to(tl.int32)
    is_duplicate = row_mask & (selected_key != occurrence_key)

    col = tl.arange(0, BLOCK_D)
    element_mask = is_duplicate[:, None] & (col[None, :] < D)
    grad_offset = (
        batch[:, None] * grad_stride_b
        + token[:, None] * grad_stride_t
        + col[None, :] * grad_stride_d
    )
    occurrence_grad = tl.load(
        grad_output_ptr + grad_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    if SCRATCH_BF16:
        occurrence_grad = occurrence_grad.to(tl.bfloat16)
    tl.atomic_add(
        grad_scratch_ptr + row[:, None] * D + col[None, :],
        occurrence_grad,
        mask=element_mask,
        sem="relaxed",
        scope="gpu",
    )


def _triton_direct_scratch_scatter(indices, grad_output, grad_scratch):
    """Launch active-row clear then direct typed scatter on the current stream."""
    if indices.ndim != 2:
        raise ValueError(f"direct n-gram indices must be [B,T], got {indices.shape}")
    if grad_output.ndim != 3:
        raise ValueError(
            f"direct n-gram grad_output must be [B,T,D], got {grad_output.shape}"
        )
    if tuple(grad_output.shape[:2]) != tuple(indices.shape):
        raise ValueError("direct n-gram indices/grad_output batch shapes differ")
    if grad_scratch.ndim != 2 or grad_scratch.shape[1] != grad_output.shape[2]:
        raise ValueError("direct n-gram scratch must have shape [rows,D]")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"direct n-gram indices must be int32/int64, got {indices.dtype}")
    if grad_output.dtype != torch.bfloat16:
        raise TypeError(
            f"direct n-gram occurrence gradients must be BF16, got {grad_output.dtype}"
        )
    if grad_scratch.dtype not in _DIRECT_SCRATCH_DTYPES:
        raise TypeError(
            "direct n-gram scratch must be FP32 or BF16, "
            f"got {grad_scratch.dtype}"
        )
    if not all(t.is_cuda for t in (indices, grad_output, grad_scratch)):
        raise ValueError("direct n-gram scatter tensors must all be CUDA tensors")

    B, T = indices.shape
    width = grad_output.shape[2]
    if width > 512:
        raise ValueError(f"embedding width {width} exceeds the BLOCK_D=512 limit")
    n_occurrences = B * T
    if n_occurrences == 0:
        return
    grid = (triton.cdiv(n_occurrences, _DIRECT_SCRATCH_BLOCK_R),)
    common_args = (
        indices.stride(0),
        indices.stride(1),
        B,
        T,
        grad_scratch.shape[0],
    )
    common_meta = {
        "D": width,
        "BLOCK_R": _DIRECT_SCRATCH_BLOCK_R,
        "BLOCK_D": 512,
        "num_warps": _DIRECT_SCRATCH_NUM_WARPS,
        "num_stages": 1,
    }
    # In the default path, both launches use the custom op's current CUDA
    # stream, so launch order clears every active row before scatter.  The
    # fused-clear path instead starts from an all-zero table and has the RMS
    # owner clear each consumed row before the next backward can begin.
    if not _DIRECT_SCRATCH_FUSED_CLEAR:
        _triton_zero_direct_scratch_rows_kernel[grid](
            grad_scratch,
            indices,
            *common_args,
            **common_meta,
        )
    _triton_scatter_direct_scratch_rows_kernel[grid](
        grad_scratch,
        grad_output,
        indices,
        indices.stride(0),
        indices.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        B,
        T,
        grad_scratch.shape[0],
        SCRATCH_BF16=grad_scratch.dtype == torch.bfloat16,
        **common_meta,
    )


def _triton_direct_embedding_pair(
    weight0,
    weight1,
    indices0,
    indices1,
    grad_scratch0,
    grad_scratch1,
):
    """Return the exact concatenation of two direct embedding gathers."""
    if indices0.ndim != 2 or indices1.shape != indices0.shape:
        raise ValueError("paired n-gram indices must share one [B,T] shape")
    if indices0.dtype not in (torch.int32, torch.int64) or indices1.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("paired n-gram indices must be int32 or int64")
    if weight0.ndim != 2 or weight1.ndim != 2:
        raise ValueError("paired n-gram weights must be two-dimensional")
    if weight0.shape[1] != weight1.shape[1]:
        raise ValueError("paired n-gram weights must have one common width")
    if weight0.dtype != torch.bfloat16 or weight1.dtype != torch.bfloat16:
        raise TypeError("paired n-gram weights must be BF16")
    if grad_scratch0.shape != weight0.shape or grad_scratch1.shape != weight1.shape:
        raise ValueError("paired direct scratch tables must match their weights")
    if (
        grad_scratch0.dtype not in _DIRECT_SCRATCH_DTYPES
        or grad_scratch1.dtype not in _DIRECT_SCRATCH_DTYPES
    ):
        raise TypeError("paired direct scratch tables must be FP32 or BF16")
    tensors = (
        weight0,
        weight1,
        indices0,
        indices1,
        grad_scratch0,
        grad_scratch1,
    )
    if not all(t.is_cuda for t in tensors):
        raise ValueError("paired direct embedding tensors must all be CUDA tensors")
    if not all(t.is_contiguous() for t in (weight0, weight1, grad_scratch0, grad_scratch1)):
        raise ValueError("paired direct embedding tables must be contiguous")

    B, T = indices0.shape
    width = weight0.shape[1]
    if width > 512:
        raise ValueError(f"paired embedding half-width {width} exceeds 512")
    output = torch.empty(
        (B, T, width * 2),
        dtype=weight0.dtype,
        device=weight0.device,
    )
    n_occurrences = B * T
    if n_occurrences == 0:
        return output
    pair_block_r = max(1, _DIRECT_SCRATCH_BLOCK_R // 2)
    _triton_direct_embedding_pair_kernel[
        (triton.cdiv(n_occurrences, pair_block_r),)
    ](
        weight0,
        weight1,
        indices0,
        indices1,
        output,
        indices0.stride(0),
        indices0.stride(1),
        indices1.stride(0),
        indices1.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        B,
        T,
        weight0.shape[0],
        weight1.shape[0],
        D=width,
        BLOCK_R=pair_block_r,
        BLOCK_D=512,
        num_warps=_DIRECT_SCRATCH_NUM_WARPS,
        num_stages=1,
    )
    return output


def _triton_direct_scratch_scatter_pair(
    indices0,
    indices1,
    grad_output,
    grad_scratch0,
    grad_scratch1,
):
    """Clear and scatter two K=2 halves into independently typed sums."""
    if indices0.ndim != 2 or indices1.shape != indices0.shape:
        raise ValueError("paired direct indices must share one [B,T] shape")
    if indices0.dtype not in (torch.int32, torch.int64) or indices1.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("paired direct indices must be int32 or int64")
    if grad_output.ndim != 3 or tuple(grad_output.shape[:2]) != tuple(indices0.shape):
        raise ValueError("paired direct grad_output must have shape [B,T,2D]")
    if grad_output.dtype != torch.bfloat16:
        raise TypeError("paired direct occurrence gradients must be BF16")
    if grad_scratch0.ndim != 2 or grad_scratch1.ndim != 2:
        raise ValueError("paired direct scratch tables must be two-dimensional")
    if grad_scratch0.shape[1] != grad_scratch1.shape[1]:
        raise ValueError("paired direct scratch tables must have one common width")
    width = grad_scratch0.shape[1]
    if grad_output.shape[2] != width * 2:
        raise ValueError("paired direct grad_output width must be twice the table width")
    if (
        grad_scratch0.dtype not in _DIRECT_SCRATCH_DTYPES
        or grad_scratch1.dtype not in _DIRECT_SCRATCH_DTYPES
    ):
        raise TypeError("paired direct scratch tables must be FP32 or BF16")
    tensors = (indices0, indices1, grad_output, grad_scratch0, grad_scratch1)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("paired direct scatter tensors must all be CUDA tensors")
    if not grad_scratch0.is_contiguous() or not grad_scratch1.is_contiguous():
        raise ValueError("paired direct scratch tables must be contiguous")
    if width > 512:
        raise ValueError(f"paired embedding half-width {width} exceeds 512")

    B, T = indices0.shape
    n_occurrences = B * T
    if n_occurrences == 0:
        return
    pair_block_r = max(1, _DIRECT_SCRATCH_BLOCK_R // 2)
    grid = (triton.cdiv(n_occurrences, pair_block_r),)
    common_indices = (
        indices0.stride(0),
        indices0.stride(1),
        indices1.stride(0),
        indices1.stride(1),
        B,
        T,
        grad_scratch0.shape[0],
        grad_scratch1.shape[0],
    )
    common_meta = {
        "D": width,
        "BLOCK_R": pair_block_r,
        "BLOCK_D": 512,
        "num_warps": _DIRECT_SCRATCH_NUM_WARPS,
        "num_stages": 1,
    }
    if not _DIRECT_SCRATCH_FUSED_CLEAR:
        _triton_zero_direct_scratch_pair_kernel[grid](
            grad_scratch0,
            grad_scratch1,
            indices0,
            indices1,
            *common_indices,
            **common_meta,
        )
    _triton_scatter_direct_scratch_pair_kernel[grid](
        grad_scratch0,
        grad_scratch1,
        grad_output,
        indices0,
        indices1,
        indices0.stride(0),
        indices0.stride(1),
        indices1.stride(0),
        indices1.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        B,
        T,
        grad_scratch0.shape[0],
        grad_scratch1.shape[0],
        SCRATCH0_BF16=grad_scratch0.dtype == torch.bfloat16,
        SCRATCH1_BF16=grad_scratch1.dtype == torch.bfloat16,
        **common_meta,
    )


def _triton_direct_compact_scratch_scatter(
    indices,
    grad_output,
    grad_scratch,
    owner_key,
):
    """Build first-occurrence owners, then scatter into compact FP32 rows."""
    if indices.ndim != 2:
        raise ValueError(f"compact direct indices must be [B,T], got {indices.shape}")
    if grad_output.ndim != 3 or tuple(grad_output.shape[:2]) != tuple(indices.shape):
        raise ValueError("compact direct grad_output must have shape [B,T,D]")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("compact direct indices must be int32 or int64")
    if grad_output.dtype != torch.bfloat16:
        raise TypeError("compact direct occurrence gradients must be BF16")
    if grad_scratch.dtype != torch.float32 or grad_scratch.ndim != 2:
        raise TypeError("compact direct scratch must be a 2D FP32 tensor")
    if grad_scratch.shape[1] != grad_output.shape[2]:
        raise ValueError("compact direct scratch width differs from grad_output")
    if owner_key.dtype != torch.int32 or owner_key.ndim != 1:
        raise TypeError("compact direct owner table must be one-dimensional int32")
    if not _DIRECT_SCRATCH_FUSED_CLEAR:
        raise RuntimeError(
            "compact direct scratch requires fused post-consume clearing"
        )
    tensors = (indices, grad_output, grad_scratch, owner_key)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("compact direct tensors must all be CUDA tensors")
    if not grad_scratch.is_contiguous() or not owner_key.is_contiguous():
        raise ValueError("compact direct scratch and owners must be contiguous")

    B, T = indices.shape
    n_occurrences = B * T
    if n_occurrences == 0:
        return
    if n_occurrences >= 0x7FFFFFFF:
        raise ValueError("compact direct scatter requires B*T < 2**31-1")
    if grad_scratch.shape[0] < n_occurrences:
        raise ValueError(
            f"compact scratch has {grad_scratch.shape[0]} rows for "
            f"{n_occurrences} occurrences"
        )
    width = grad_output.shape[2]
    if width > 512:
        raise ValueError(f"compact embedding width {width} exceeds 512")
    n_rows = owner_key.numel()
    owner_block = 256
    _triton_zero_direct_owner_key_kernel[
        (triton.cdiv(n_rows, owner_block),)
    ](
        owner_key,
        n_rows,
        BLOCK=owner_block,
        num_warps=4,
        num_stages=1,
    )
    _triton_select_direct_owner_min_kernel[
        (triton.cdiv(n_occurrences, owner_block),)
    ](
        owner_key,
        indices,
        indices.stride(0),
        indices.stride(1),
        B,
        T,
        n_rows,
        BLOCK=owner_block,
        num_warps=4,
        num_stages=1,
    )
    _triton_scatter_direct_compact_owner_kernel[
        (triton.cdiv(n_occurrences, _DIRECT_SCRATCH_BLOCK_R),)
    ](
        grad_scratch,
        owner_key,
        grad_output,
        indices,
        indices.stride(0),
        indices.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        B,
        T,
        n_rows,
        grad_scratch.shape[0],
        D=width,
        BLOCK_R=_DIRECT_SCRATCH_BLOCK_R,
        BLOCK_D=512,
        num_warps=_DIRECT_SCRATCH_NUM_WARPS,
        num_stages=1,
    )


def _triton_direct_scratch_scatter_owner_min(
    indices,
    grad_output,
    grad_scratch,
    owner_key,
):
    """Four ordered launches: zero keys, choose owners, store, add repeats."""
    if indices.ndim != 2:
        raise ValueError(f"direct n-gram indices must be [B,T], got {indices.shape}")
    if grad_output.ndim != 3:
        raise ValueError(
            f"direct n-gram grad_output must be [B,T,D], got {grad_output.shape}"
        )
    if tuple(grad_output.shape[:2]) != tuple(indices.shape):
        raise ValueError("direct n-gram indices/grad_output batch shapes differ")
    if grad_scratch.ndim != 2 or grad_scratch.shape[1] != grad_output.shape[2]:
        raise ValueError("direct n-gram scratch must have shape [rows,D]")
    if owner_key.ndim != 1 or owner_key.numel() != grad_scratch.shape[0]:
        raise ValueError("direct n-gram owner table must have one entry per row")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"direct n-gram indices must be int32/int64, got {indices.dtype}")
    if grad_output.dtype != torch.bfloat16:
        raise TypeError(
            f"direct n-gram occurrence gradients must be BF16, got {grad_output.dtype}"
        )
    if grad_scratch.dtype not in _DIRECT_SCRATCH_DTYPES:
        raise TypeError(
            "direct n-gram scratch must be FP32 or BF16, "
            f"got {grad_scratch.dtype}"
        )
    if owner_key.dtype != torch.int32:
        raise TypeError(
            f"direct n-gram owner table must be int32, got {owner_key.dtype}"
        )
    tensors = (indices, grad_output, grad_scratch, owner_key)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("direct n-gram owner-min tensors must all be CUDA tensors")
    if not grad_scratch.is_contiguous() or not owner_key.is_contiguous():
        raise ValueError("direct n-gram scratch and owner table must be contiguous")

    B, T = indices.shape
    width = grad_output.shape[2]
    if width > 512:
        raise ValueError(f"embedding width {width} exceeds the BLOCK_D=512 limit")
    n_occurrences = B * T
    if n_occurrences == 0:
        return
    if n_occurrences >= 0x7FFFFFFF:
        raise ValueError("owner-min direct scatter requires B*T < 2**31-1")

    n_rows = grad_scratch.shape[0]
    owner_block = 256
    _triton_zero_direct_owner_key_kernel[
        (triton.cdiv(n_rows, owner_block),)
    ](
        owner_key,
        n_rows,
        BLOCK=owner_block,
        num_warps=4,
        num_stages=1,
    )
    _triton_select_direct_owner_min_kernel[
        (triton.cdiv(n_occurrences, owner_block),)
    ](
        owner_key,
        indices,
        indices.stride(0),
        indices.stride(1),
        B,
        T,
        n_rows,
        BLOCK=owner_block,
        num_warps=4,
        num_stages=1,
    )

    grid = (triton.cdiv(n_occurrences, _DIRECT_SCRATCH_BLOCK_R),)
    common_args = (
        grad_scratch,
        owner_key,
        grad_output,
        indices,
        indices.stride(0),
        indices.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        B,
        T,
        n_rows,
    )
    common_meta = {
        "D": width,
        "BLOCK_R": _DIRECT_SCRATCH_BLOCK_R,
        "BLOCK_D": 512,
        "SCRATCH_BF16": grad_scratch.dtype == torch.bfloat16,
        "num_warps": _DIRECT_SCRATCH_NUM_WARPS,
        "num_stages": 1,
    }
    # Kernel boundaries on the custom op's current stream guarantee that all
    # owner stores are visible before any duplicate atomic accumulation.
    _triton_initialize_direct_owner_rows_kernel[grid](
        *common_args,
        **common_meta,
    )
    _triton_scatter_direct_duplicate_rows_kernel[grid](
        *common_args,
        **common_meta,
    )


@triton.jit
def _triton_lazy_rmsprop_uncoalesced_sparse_kernel(
    parameter_ptr,
    grad_accum_ptr,
    moment_ptr,
    row_index_ptr,
    row_marker_ptr,
    n_active,
    n_rows,
    step_marker,
    lr,
    beta2,
    eps,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_PROFILE: tl.constexpr,
    CLEAR_GRAD_ACCUM: tl.constexpr,
):
    """Round each FP32 row sum to BF16, then apply RMSProp exactly once."""
    row_slot = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    slot_mask = row_slot < n_active
    row = tl.load(row_index_ptr + row_slot, mask=slot_mask, other=0).to(tl.int64)
    row_mask = slot_mask & (row >= 0) & (row < n_rows)
    previous_marker = tl.atomic_xchg(
        row_marker_ptr + row,
        step_marker,
        mask=row_mask,
        sem="relaxed",
        scope="gpu",
    )
    owns_row = row_mask & (previous_marker != step_marker)

    col = tl.arange(0, BLOCK_D)
    element_mask = owns_row[:, None] & (col[None, :] < D)
    offset = row[:, None] * D + col[None, :]

    # FP32 scratch paths sum collisions before one BF16 materialization. The
    # selected BF16 scratch already contains a BF16 atomic sum. In both cases,
    # canonicalize the optimizer input to BF16 before the published recurrence.
    grad = tl.load(
        grad_accum_ptr + offset, mask=element_mask, other=0.0
    ).to(tl.bfloat16).to(tl.float32)
    moment = tl.load(moment_ptr + offset, mask=element_mask, other=0.0).to(
        tl.float32
    )
    parameter = tl.load(
        parameter_ptr + offset, mask=element_mask, other=0.0
    ).to(tl.float32)

    grad_sq = grad * grad
    if ROUND_PROFILE == 2 or ROUND_PROFILE == 3:
        grad_sq = grad_sq.to(tl.bfloat16).to(tl.float32)
    moment_new = moment + (1.0 - beta2) * (grad_sq - moment)
    if ROUND_PROFILE == 1 or ROUND_PROFILE == 2 or ROUND_PROFILE == 3:
        moment_new = moment_new.to(tl.bfloat16).to(tl.float32)

    step_float32 = step_marker * 1.0
    bias2 = 1.0 - libdevice.pow(beta2, step_float32)
    scaled_moment = moment_new / bias2
    if ROUND_PROFILE == 2:
        scaled_moment = scaled_moment.to(tl.bfloat16).to(tl.float32)
    denom = libdevice.sqrt(scaled_moment)
    if ROUND_PROFILE == 2:
        denom = denom.to(tl.bfloat16).to(tl.float32)
    denom = denom + eps
    if ROUND_PROFILE == 2:
        denom = denom.to(tl.bfloat16).to(tl.float32)

    normalized_grad = grad / denom
    if ROUND_PROFILE == 2:
        normalized_grad = normalized_grad.to(tl.bfloat16).to(tl.float32)
    parameter_new = parameter + normalized_grad * (-lr)

    tl.store(moment_ptr + offset, moment_new, mask=element_mask)
    tl.store(parameter_ptr + offset, parameter_new, mask=element_mask)
    if CLEAR_GRAD_ACCUM:
        # Keep the clear after every computation and result store that consumes
        # ``grad``.  Besides making the lifecycle explicit, this prevents a
        # compiler from scheduling the destructive store ahead of any use of
        # the scratch load.  Only the row-marker owner reaches this store.
        tl.store(grad_accum_ptr + offset, 0.0, mask=element_mask)


@triton.jit
def _triton_lazy_rmsprop_compact_owner_kernel(
    parameter_ptr,
    compact_scratch_ptr,
    moment_ptr,
    row_index_ptr,
    owner_key_ptr,
    n_occurrences,
    n_rows,
    step_marker,
    lr,
    beta2,
    eps,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_PROFILE: tl.constexpr,
):
    """Consume one FP32 compact slot per unique global table row."""
    occurrence = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    occurrence_mask = occurrence < n_occurrences
    row = tl.load(
        row_index_ptr + occurrence,
        mask=occurrence_mask,
        other=0,
    ).to(tl.int64)
    row_mask = occurrence_mask & (row >= 0) & (row < n_rows)
    owner_key = tl.load(owner_key_ptr + row, mask=row_mask, other=0)
    occurrence_key = (0x7FFFFFFF - occurrence).to(tl.int32)
    owns_row = row_mask & (owner_key == occurrence_key)

    col = tl.arange(0, BLOCK_D)
    element_mask = owns_row[:, None] & (col[None, :] < D)
    table_offset = row[:, None] * D + col[None, :]
    compact_offset = occurrence[:, None] * D + col[None, :]

    # Match the dense DirectScratch boundary exactly: all occurrence gradients
    # were accumulated in FP32, then the complete row sum rounds once to BF16
    # before entering the published RMSProp recurrence.
    grad = tl.load(
        compact_scratch_ptr + compact_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.bfloat16).to(tl.float32)
    moment = tl.load(
        moment_ptr + table_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    parameter = tl.load(
        parameter_ptr + table_offset,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)

    grad_sq = grad * grad
    if ROUND_PROFILE == 2 or ROUND_PROFILE == 3:
        grad_sq = grad_sq.to(tl.bfloat16).to(tl.float32)
    moment_new = moment + (1.0 - beta2) * (grad_sq - moment)
    if ROUND_PROFILE == 1 or ROUND_PROFILE == 2 or ROUND_PROFILE == 3:
        moment_new = moment_new.to(tl.bfloat16).to(tl.float32)

    step_float32 = step_marker * 1.0
    bias2 = 1.0 - libdevice.pow(beta2, step_float32)
    scaled_moment = moment_new / bias2
    if ROUND_PROFILE == 2:
        scaled_moment = scaled_moment.to(tl.bfloat16).to(tl.float32)
    denom = libdevice.sqrt(scaled_moment)
    if ROUND_PROFILE == 2:
        denom = denom.to(tl.bfloat16).to(tl.float32)
    denom = denom + eps
    if ROUND_PROFILE == 2:
        denom = denom.to(tl.bfloat16).to(tl.float32)

    normalized_grad = grad / denom
    if ROUND_PROFILE == 2:
        normalized_grad = normalized_grad.to(tl.bfloat16).to(tl.float32)
    parameter_new = parameter + normalized_grad * (-lr)

    tl.store(moment_ptr + table_offset, moment_new, mask=element_mask)
    tl.store(parameter_ptr + table_offset, parameter_new, mask=element_mask)
    # Restore the all-zero compact-scratch invariant for the next backward.
    tl.store(compact_scratch_ptr + compact_offset, 0.0, mask=element_mask)


def triton_lazy_rmsprop_uncoalesced_sparse_rows_step(
    parameter,
    row_indices,
    row_grad,
    grad_accum_fp32,
    exp_avg_sq,
    row_marker,
    step,
    lr,
    beta2,
    eps,
    round_profile=None,
):
    """Sum uncoalesced COO values before squaring, without a full-table clear."""
    if parameter.dtype != torch.bfloat16 or row_grad.dtype != torch.bfloat16:
        raise TypeError("uncoalesced sparse RMSProp requires BF16 parameter/values")
    if exp_avg_sq.dtype != torch.bfloat16:
        raise TypeError("uncoalesced sparse RMSProp requires a BF16 moment table")
    if grad_accum_fp32.dtype != torch.float32:
        raise TypeError("uncoalesced sparse RMSProp requires an FP32 scratch table")
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"row indices must be int32/int64, got {row_indices.dtype}")
    if row_marker.dtype != torch.int32:
        raise TypeError(f"row marker must be int32, got {row_marker.dtype}")
    if row_indices.ndim != 1:
        raise ValueError("uncoalesced sparse row indices must be one-dimensional")
    if row_grad.ndim != 2 or row_grad.shape[0] != row_indices.numel():
        raise ValueError("uncoalesced sparse values must have shape [nnz, dim]")
    if not (
        parameter.shape
        == exp_avg_sq.shape
        == grad_accum_fp32.shape
    ):
        raise ValueError("parameter, moment, and FP32 scratch must share one shape")
    if row_grad.shape[1] != parameter.shape[1]:
        raise ValueError("uncoalesced sparse value width must match the table")
    tensors = (
        parameter,
        row_indices,
        row_grad,
        grad_accum_fp32,
        exp_avg_sq,
        row_marker,
    )
    if not all(t.is_cuda for t in tensors):
        raise ValueError("uncoalesced sparse RMSProp tensors must all be CUDA tensors")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("uncoalesced sparse RMSProp tensors must be contiguous")
    if not 0 < step < 2**31:
        raise ValueError(f"step marker must fit a positive int32, got {step}")

    n_active = row_indices.numel()
    if n_active == 0:
        return
    n_rows, width = parameter.shape
    if width > 512:
        raise ValueError(f"embedding width {width} exceeds the BLOCK_D=512 limit")

    def as_float32(value):
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    if round_profile is None:
        round_profile = _TRITON_RMS_ROUND_PROFILE
    if round_profile not in _TRITON_RMS_ROUND_PROFILES.values():
        raise ValueError(f"unknown Triton RMSProp round profile: {round_profile}")
    grid = (triton.cdiv(n_active, _TRITON_RMS_BLOCK_R),)
    common_meta = {
        "D": width,
        "BLOCK_R": _TRITON_RMS_BLOCK_R,
        "BLOCK_D": 512,
        "num_warps": _TRITON_RMS_NUM_WARPS,
        "num_stages": 1,
    }
    _triton_zero_uncoalesced_sparse_rows_kernel[grid](
        grad_accum_fp32,
        row_indices,
        row_marker,
        n_active,
        n_rows,
        -int(step),
        **common_meta,
    )
    _triton_scatter_uncoalesced_sparse_rows_kernel[grid](
        grad_accum_fp32,
        row_grad,
        row_indices,
        n_active,
        n_rows,
        **common_meta,
    )
    _triton_lazy_rmsprop_uncoalesced_sparse_kernel[grid](
        parameter,
        grad_accum_fp32,
        exp_avg_sq,
        row_indices,
        row_marker,
        n_active,
        n_rows,
        int(step),
        as_float32(lr),
        as_float32(beta2),
        as_float32(eps),
        ROUND_PROFILE=round_profile,
        CLEAR_GRAD_ACCUM=False,
        **common_meta,
    )


def triton_lazy_rmsprop_compact_direct_scratch_rows_step(
    parameter,
    row_indices,
    grad_scratch,
    owner_key,
    exp_avg_sq,
    step,
    lr,
    beta2,
    eps,
    round_profile=None,
):
    """Consume first-occurrence compact FP32 sums and clear owner slots."""
    if parameter.dtype != torch.bfloat16 or exp_avg_sq.dtype != torch.bfloat16:
        raise TypeError("compact RMSProp requires BF16 parameter and moment")
    if grad_scratch.dtype != torch.float32 or grad_scratch.ndim != 2:
        raise TypeError("compact RMSProp requires a 2D FP32 scratch")
    if owner_key.dtype != torch.int32 or owner_key.ndim != 1:
        raise TypeError("compact RMSProp requires a 1D int32 owner table")
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("compact RMSProp row indices must be int32 or int64")
    if row_indices.ndim != 1:
        raise ValueError("compact RMSProp row indices must be one-dimensional")
    if parameter.shape != exp_avg_sq.shape:
        raise ValueError("compact RMSProp parameter/moment shapes differ")
    if grad_scratch.shape[1] != parameter.shape[1]:
        raise ValueError("compact RMSProp scratch width differs from parameter")
    if owner_key.numel() != parameter.shape[0]:
        raise ValueError("compact RMSProp owner count differs from table rows")
    if grad_scratch.shape[0] < row_indices.numel():
        raise ValueError("compact RMSProp has fewer scratch rows than occurrences")
    tensors = (parameter, row_indices, grad_scratch, owner_key, exp_avg_sq)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("compact RMSProp tensors must all be CUDA tensors")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("compact RMSProp tensors must be contiguous")
    if not 0 < step < 2**31:
        raise ValueError(f"step marker must fit a positive int32, got {step}")

    n_occurrences = row_indices.numel()
    if n_occurrences == 0:
        return
    if n_occurrences >= 0x7FFFFFFF:
        raise ValueError("compact RMSProp requires occurrences < 2**31-1")
    width = parameter.shape[1]
    if width > 512:
        raise ValueError(f"compact RMSProp width {width} exceeds 512")

    def as_float32(value):
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    if round_profile is None:
        round_profile = _TRITON_RMS_ROUND_PROFILE
    if round_profile not in _TRITON_RMS_ROUND_PROFILES.values():
        raise ValueError(f"unknown Triton RMSProp round profile: {round_profile}")
    _triton_lazy_rmsprop_compact_owner_kernel[
        (triton.cdiv(n_occurrences, _TRITON_RMS_BLOCK_R),)
    ](
        parameter,
        grad_scratch,
        exp_avg_sq,
        row_indices,
        owner_key,
        n_occurrences,
        parameter.shape[0],
        int(step),
        as_float32(lr),
        as_float32(beta2),
        as_float32(eps),
        D=width,
        BLOCK_R=_TRITON_RMS_BLOCK_R,
        BLOCK_D=512,
        ROUND_PROFILE=round_profile,
        num_warps=_TRITON_RMS_NUM_WARPS,
        num_stages=1,
    )


def triton_lazy_rmsprop_direct_scratch_rows_step(
    parameter,
    row_indices,
    grad_scratch,
    exp_avg_sq,
    row_marker,
    step,
    lr,
    beta2,
    eps,
    round_profile=None,
    clear_grad_accum=None,
):
    """Consume scratch populated by embedding backward; never materialize p.grad."""
    if parameter.dtype != torch.bfloat16:
        raise TypeError("direct-scratch RMSProp requires a BF16 parameter")
    if exp_avg_sq.dtype != torch.bfloat16:
        raise TypeError("direct-scratch RMSProp requires a BF16 moment table")
    if grad_scratch.dtype not in _DIRECT_SCRATCH_DTYPES:
        raise TypeError(
            "direct-scratch RMSProp requires an FP32 or BF16 scratch table"
        )
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"row indices must be int32/int64, got {row_indices.dtype}")
    if row_marker.dtype != torch.int32:
        raise TypeError(f"row marker must be int32, got {row_marker.dtype}")
    if row_indices.ndim != 1:
        raise ValueError("direct-scratch active rows must be one-dimensional")
    if not (
        parameter.shape == exp_avg_sq.shape == grad_scratch.shape
    ):
        raise ValueError("parameter, moment, and direct scratch must share one shape")
    tensors = (
        parameter,
        row_indices,
        grad_scratch,
        exp_avg_sq,
        row_marker,
    )
    if not all(t.is_cuda for t in tensors):
        raise ValueError("direct-scratch RMSProp tensors must all be CUDA tensors")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("direct-scratch RMSProp tensors must be contiguous")
    if not 0 < step < 2**31:
        raise ValueError(f"step marker must fit a positive int32, got {step}")

    n_active = row_indices.numel()
    if n_active == 0:
        return
    n_rows, width = parameter.shape
    if width > 512:
        raise ValueError(f"embedding width {width} exceeds the BLOCK_D=512 limit")

    def as_float32(value):
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    if round_profile is None:
        round_profile = _TRITON_RMS_ROUND_PROFILE
    if round_profile not in _TRITON_RMS_ROUND_PROFILES.values():
        raise ValueError(f"unknown Triton RMSProp round profile: {round_profile}")
    if clear_grad_accum is None:
        clear_grad_accum = _DIRECT_SCRATCH_FUSED_CLEAR
    if not isinstance(clear_grad_accum, bool):
        raise TypeError("direct-scratch clear_grad_accum must be a bool")
    grid = (triton.cdiv(n_active, _TRITON_RMS_BLOCK_R),)
    _triton_lazy_rmsprop_uncoalesced_sparse_kernel[grid](
        parameter,
        grad_scratch,
        exp_avg_sq,
        row_indices,
        row_marker,
        n_active,
        n_rows,
        int(step),
        as_float32(lr),
        as_float32(beta2),
        as_float32(eps),
        D=width,
        BLOCK_R=_TRITON_RMS_BLOCK_R,
        BLOCK_D=512,
        ROUND_PROFILE=round_profile,
        CLEAR_GRAD_ACCUM=clear_grad_accum,
        num_warps=_TRITON_RMS_NUM_WARPS,
        num_stages=1,
    )


def _lazy_rmsprop_rows_step_impl(
    p,
    row_indices,
    row_grad,
    exp_avg_sq,
    step_t,
    lr_t,
    beta2_t,
    eps_t,
):
    """Apply the baseline RMSProp update to the rows touched this step.

    The dense baseline also visits untouched rows with zero gradients.  Its
    BF16 state and beta2 >= 0.999 make that operation an identity for every
    positive normal value after round-to-nearest-even, so there is deliberately
    no delayed decay here.
    """
    moment_rows = exp_avg_sq.index_select(0, row_indices)
    moment_rows.lerp_(row_grad.square(), 1 - beta2_t)

    # Preserve the dense baseline's bias correction exactly.  In particular,
    # that code uses current_beta**global_step even while beta changes; it does
    # not use the cumulative beta product used solely for lazy state decay.
    bias2 = 1 - beta2_t**step_t
    denom = (moment_rows / bias2).sqrt() + eps_t
    parameter_rows = p.index_select(0, row_indices)
    parameter_rows.add_(row_grad / denom, alpha=-lr_t)

    exp_avg_sq.index_copy_(0, row_indices, moment_rows)
    p.index_copy_(0, row_indices, parameter_rows)


# Keep the dense-grad gather inside the compiled region so Inductor can fold it
# into the row update instead of materializing another B*T-by-D tensor in eager
# mode.  Dense gradients always produce B*T row entries.  Real sparse
# embedding gradients are coalesced first and need a dynamic nnz.
def _lazy_rmsprop_dense_rows_step_impl(
    p,
    row_indices,
    dense_grad,
    exp_avg_sq,
    step_t,
    lr_t,
    beta2_t,
    eps_t,
):
    return _lazy_rmsprop_rows_step_impl(
        p,
        row_indices,
        dense_grad.index_select(0, row_indices),
        exp_avg_sq,
        step_t,
        lr_t,
        beta2_t,
        eps_t,
    )


lazy_rmsprop_dense_rows_step_dynamic = torch.compile(
    _lazy_rmsprop_dense_rows_step_impl,
    # torch.unique returns a different active-row count for every hash table.
    # Make nnz symbolic so Dynamo does not compile once per table.
    dynamic=True,
    fullgraph=True,
    **_OPTIMIZER_COMPILE_KWARGS,
)
lazy_rmsprop_sparse_rows_step_dynamic = torch.compile(
    _lazy_rmsprop_rows_step_impl,
    dynamic=True,
    fullgraph=True,
    **_OPTIMIZER_COMPILE_KWARGS,
)


def _ngram_active_rows_impl(
    idx,
    bigram_primes,
    trigram_primes,
    bigram_table_size,
    trigram_table_sizes,
):
    """Recompute all table rows in two fused, vectorized hash families."""
    raw_prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
    raw_prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
    at_bos = idx == BOS_TOKEN_ID
    after_bos = raw_prev_idx == BOS_TOKEN_ID
    prev_idx = torch.where(at_bos, idx, raw_prev_idx)
    prev2_idx = torch.where(
        at_bos | after_bos,
        torch.full_like(idx, BOS_TOKEN_ID),
        raw_prev2_idx,
    )
    bigram_rows = torch.bitwise_and(
        (prev_idx.unsqueeze(0) * bigram_primes[:, 0, None, None])
        ^ (idx.unsqueeze(0) * bigram_primes[:, 1, None, None]),
        bigram_table_size - 1,
    )
    trigram_rows = torch.bitwise_and(
        (prev2_idx.unsqueeze(0) * trigram_primes[:, 0, None, None])
        ^ (prev_idx.unsqueeze(0) * trigram_primes[:, 1, None, None])
        ^ (idx.unsqueeze(0) * trigram_primes[:, 2, None, None]),
        (trigram_table_sizes - 1)[:, None, None],
    )
    return bigram_rows.flatten(1), trigram_rows.flatten(1)


ngram_active_rows_fused = torch.compile(
    _ngram_active_rows_impl,
    dynamic=False,
    fullgraph=True,
    **_OPTIMIZER_COMPILE_KWARGS,
)


def _run_triton_lazy_rms_gpu_microcheck():
    """Compare the candidate against the published dense compiled recurrence.

    This opt-in check uses a tiny synthetic table, intentionally repeats row
    ids, and carries state across several beta2 values.  It exits before model
    construction or data loading.  By default any bit mismatch is a failure;
    setting TRITON_RMS_MICROCHECK_ALLOW_MISMATCH=1 leaves the detailed report
    visible for one-ULP investigations.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("TRITON_LAZY_RMS_MICROCHECK requires CUDA")
    profile_selector = os.environ.get(
        "TRITON_RMS_MICROCHECK_PROFILES", _TRITON_RMS_ROUND_PROFILE_NAME
    )
    if profile_selector == "all":
        profile_names = tuple(_TRITON_RMS_ROUND_PROFILES)
    else:
        profile_names = tuple(name.strip() for name in profile_selector.split(","))
    unknown_profiles = set(profile_names) - set(_TRITON_RMS_ROUND_PROFILES)
    if unknown_profiles:
        raise ValueError(f"unknown microcheck round profiles: {sorted(unknown_profiles)}")

    device = torch.device("cuda")
    n_rows, width = 257, 384
    exact_profiles = []
    for profile_name in profile_names:
        # Re-seeding makes every profile consume identical synthetic cases.
        torch.manual_seed(0xB200)
        torch.cuda.manual_seed(0xB200)
        parameter_reference = (
            torch.rand(n_rows, width, device=device) - 0.5
        ).bfloat16()
        parameter_candidate = parameter_reference.clone()
        moment_reference = (
            torch.rand(n_rows, width, device=device) * 1e-2
        ).bfloat16()
        moment_candidate = moment_reference.clone()
        dense_grad = torch.zeros_like(parameter_reference)
        row_marker = torch.zeros(n_rows, dtype=torch.int32, device=device)

        step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        eps_t = torch.tensor(1e-10, dtype=torch.float32, device="cpu")
        wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        profile_mismatch = False

        for step, beta2 in enumerate((0.999, 0.999, 0.9995, 0.9999), start=1):
            # 1024 candidates for only 257 table rows guarantees substantial
            # duplication. Gradients are table-shaped and already aggregated.
            row_indices = torch.randint(
                0, n_rows, (1024,), dtype=torch.int64, device=device
            )
            unique_rows = torch.unique(row_indices)
            dense_grad.zero_()
            dense_grad[unique_rows] = (
                torch.rand(unique_rows.numel(), width, device=device) - 0.5
            ).bfloat16() * 0.08

            step_t.fill_(step)
            lr_t.fill_(0.6)
            beta2_t.fill_(beta2)
            rmsprop_step_fused(
                parameter_reference,
                dense_grad,
                moment_reference,
                step_t,
                lr_t,
                beta2_t,
                eps_t,
                wd_t,
            )
            triton_lazy_rmsprop_dense_rows_step(
                parameter_candidate,
                row_indices,
                dense_grad,
                moment_candidate,
                row_marker,
                step,
                0.6,
                beta2,
                1e-10,
                round_profile=_TRITON_RMS_ROUND_PROFILES[profile_name],
            )
            torch.cuda.synchronize()

            parameter_mismatches = int(
                (parameter_candidate != parameter_reference).sum().item()
            )
            moment_mismatches = int(
                (moment_candidate != moment_reference).sum().item()
            )
            parameter_max_abs = float(
                (parameter_candidate.float() - parameter_reference.float())
                .abs()
                .max()
                .item()
            )
            moment_max_abs = float(
                (moment_candidate.float() - moment_reference.float())
                .abs()
                .max()
                .item()
            )
            claimed_rows = int((row_marker == step).sum().item())
            expected_rows = unique_rows.numel()
            print(
                "TRITON_RMS_MICROCHECK "
                f"profile={profile_name} step={step} beta2={beta2} "
                f"candidates={row_indices.numel()} unique={expected_rows} "
                f"claimed={claimed_rows} p_mismatch={parameter_mismatches} "
                f"v_mismatch={moment_mismatches} p_max_abs={parameter_max_abs:.9g} "
                f"v_max_abs={moment_max_abs:.9g}"
            )
            if claimed_rows != expected_rows:
                raise AssertionError("atomic row-marker ownership count is incorrect")
            profile_mismatch |= bool(parameter_mismatches or moment_mismatches)

        if not profile_mismatch:
            exact_profiles.append(profile_name)

    if not exact_profiles and os.environ.get(
        "TRITON_RMS_MICROCHECK_ALLOW_MISMATCH", "0"
    ) != "1":
        raise AssertionError(
            "no Triton RMSProp profile matched the dense compiled BF16 oracle"
        )
    print(f"TRITON_RMS_MICROCHECK_PASS exact_profiles={exact_profiles}")


def _run_triton_uncoalesced_sparse_gpu_microcheck():
    """Bit-check sum-before-round and RMS state across duplicate COO rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("uncoalesced sparse microcheck requires CUDA")
    device = torch.device("cuda")
    n_rows, width = 257, 384
    torch.manual_seed(0x5A17)
    torch.cuda.manual_seed(0x5A17)
    parameter_reference = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16()
    parameter_candidate = parameter_reference.clone()
    moment_reference = (
        torch.rand(n_rows, width, device=device) * 1e-2
    ).bfloat16()
    moment_candidate = moment_reference.clone()
    reference_marker = torch.zeros(n_rows, dtype=torch.int32, device=device)
    candidate_marker = torch.zeros_like(reference_marker)
    # Empty is intentional: the candidate must initialize only touched rows.
    candidate_scratch = torch.empty(
        n_rows, width, dtype=torch.float32, device=device
    )

    for step, beta2 in enumerate((0.999, 0.999, 0.9995, 0.9999), start=1):
        row_indices = torch.randint(
            0, n_rows, (1024,), dtype=torch.int64, device=device
        )
        # Power-of-two-scaled integers make FP32 duplicate sums independent of
        # atomic arrival order, so any mismatch isolates the implementation.
        occurrence_grad = (
            torch.randint(
                -8,
                9,
                (row_indices.numel(), width),
                dtype=torch.int32,
                device=device,
            ).float()
            * (2.0 ** -9)
        ).bfloat16()
        dense_accum = torch.zeros(
            n_rows, width, dtype=torch.float32, device=device
        )
        dense_accum.index_add_(0, row_indices, occurrence_grad.float())
        dense_grad = dense_accum.bfloat16()

        triton_lazy_rmsprop_dense_rows_step(
            parameter_reference,
            row_indices,
            dense_grad,
            moment_reference,
            reference_marker,
            step,
            0.6,
            beta2,
            1e-10,
        )
        triton_lazy_rmsprop_uncoalesced_sparse_rows_step(
            parameter_candidate,
            row_indices,
            occurrence_grad,
            candidate_scratch,
            moment_candidate,
            candidate_marker,
            step,
            0.6,
            beta2,
            1e-10,
        )
        torch.cuda.synchronize()

        unique_rows = torch.unique(row_indices)
        scratch_mismatches = int(
            (
                candidate_scratch.index_select(0, unique_rows)
                != dense_accum.index_select(0, unique_rows)
            ).sum().item()
        )
        parameter_mismatches = int(
            (parameter_candidate != parameter_reference).sum().item()
        )
        moment_mismatches = int(
            (moment_candidate != moment_reference).sum().item()
        )
        claimed_rows = int((candidate_marker == step).sum().item())
        expected_rows = unique_rows.numel()
        print(
            "TRITON_UNCOALESCED_SPARSE_MICROCHECK "
            f"step={step} beta2={beta2} candidates={row_indices.numel()} "
            f"unique={expected_rows} claimed={claimed_rows} "
            f"scratch_mismatch={scratch_mismatches} "
            f"p_mismatch={parameter_mismatches} "
            f"v_mismatch={moment_mismatches}"
        )
        if claimed_rows != expected_rows:
            raise AssertionError("uncoalesced sparse marker ownership is incorrect")
        if scratch_mismatches or parameter_mismatches or moment_mismatches:
            raise AssertionError("uncoalesced sparse path differs from dense oracle")
    print("TRITON_UNCOALESCED_SPARSE_MICROCHECK_PASS")


def _run_triton_direct_scratch_grad_gpu_microcheck():
    """Compile-check custom autograd, shared reuse, scatter, and RMS update."""
    if not torch.cuda.is_available():
        raise RuntimeError("direct-scratch gradient microcheck requires CUDA")
    device = torch.device("cuda")
    n_rows, width = 97, 32
    torch.manual_seed(0xD1EC7)
    torch.cuda.manual_seed(0xD1EC7)

    weight = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16().requires_grad_()
    scratch = torch.empty(n_rows, width, dtype=torch.float32, device=device)
    indices = torch.randint(0, n_rows, (4, 31), device=device)
    # Force repeated rows/hash collisions, including duplicates in one warp.
    indices[:, 4:12] = indices[:, :1]
    coefficient = torch.full(
        (*indices.shape, width),
        2.0 ** -5,
        dtype=torch.bfloat16,
        device=device,
    )

    def direct_loss_fn(table, lookup_rows, grad_table, coeff):
        lookup = torch.ops.ngram_direct.embedding(table, lookup_rows, grad_table)
        # The benchmark's shared trigram lookup feeds three layers.  Reusing
        # one custom-op output here checks that autograd sums all three users
        # before the embedding node performs its single direct scatter.
        return (
            (lookup * coeff).sum()
            + (lookup * (coeff * 2)).sum()
            - (lookup * (coeff / 2)).sum()
        )

    compiled_direct_loss = torch.compile(
        direct_loss_fn,
        dynamic=False,
        fullgraph=True,
    )
    loss = compiled_direct_loss(weight, indices, scratch, coefficient)
    loss.backward()
    torch.cuda.synchronize()
    if weight.grad is not None:
        raise AssertionError("direct embedding backward materialized weight.grad")

    # 1 + 2 - 1/2 users, exactly representable in BF16, so FP32 atomic order
    # cannot hide a discrepancy in the occurrence-to-row accumulation.
    occurrence_grad = (coefficient * 2.5).float().reshape(-1, width)
    dense_accum = torch.zeros(n_rows, width, dtype=torch.float32, device=device)
    flat_rows = indices.flatten()
    dense_accum.index_add_(0, flat_rows, occurrence_grad)
    unique_rows = torch.unique(flat_rows)
    scratch_mismatches = int(
        (
            scratch.index_select(0, unique_rows)
            != dense_accum.index_select(0, unique_rows)
        ).sum().item()
    )
    if scratch_mismatches:
        raise AssertionError(
            f"direct scratch differs from FP32 oracle: {scratch_mismatches} values"
        )

    # Exercise the cat-backward layout explicitly: width is every other lane
    # of a larger allocation, so no hidden contiguous() is permitted.
    strided_storage = torch.empty(
        *indices.shape,
        width * 2,
        dtype=torch.bfloat16,
        device=device,
    )
    strided_occurrence_grad = strided_storage[..., ::2]
    strided_occurrence_grad.copy_((coefficient * 2.5))
    if strided_occurrence_grad.is_contiguous():
        raise AssertionError("direct-scratch stride test unexpectedly became contiguous")
    torch.ops.ngram_direct.scratch_scatter(
        indices,
        strided_occurrence_grad,
        scratch,
    )
    torch.cuda.synchronize()
    strided_scratch_mismatches = int(
        (
            scratch.index_select(0, unique_rows)
            != dense_accum.index_select(0, unique_rows)
        ).sum().item()
    )
    if strided_scratch_mismatches:
        raise AssertionError(
            "strided direct scatter differs from FP32 oracle: "
            f"{strided_scratch_mismatches} values"
        )

    parameter_reference = weight.detach().clone()
    parameter_candidate = parameter_reference.clone()
    moment_reference = (
        torch.rand(n_rows, width, device=device) * 1e-2
    ).bfloat16()
    moment_candidate = moment_reference.clone()
    reference_marker = torch.zeros(n_rows, dtype=torch.int32, device=device)
    candidate_marker = torch.zeros_like(reference_marker)
    dense_grad = dense_accum.bfloat16()
    triton_lazy_rmsprop_dense_rows_step(
        parameter_reference,
        flat_rows,
        dense_grad,
        moment_reference,
        reference_marker,
        1,
        0.6,
        0.999,
        1e-10,
    )
    triton_lazy_rmsprop_direct_scratch_rows_step(
        parameter_candidate,
        flat_rows,
        scratch,
        moment_candidate,
        candidate_marker,
        1,
        0.6,
        0.999,
        1e-10,
    )
    torch.cuda.synchronize()
    parameter_mismatches = int(
        (parameter_candidate != parameter_reference).sum().item()
    )
    moment_mismatches = int((moment_candidate != moment_reference).sum().item())
    claimed_rows = int((candidate_marker == 1).sum().item())
    print(
        "TRITON_DIRECT_SCRATCH_GRAD_MICROCHECK "
        f"occurrences={flat_rows.numel()} unique={unique_rows.numel()} "
        f"claimed={claimed_rows} scratch_mismatch={scratch_mismatches} "
        f"strided_scratch_mismatch={strided_scratch_mismatches} "
        f"p_mismatch={parameter_mismatches} v_mismatch={moment_mismatches}"
    )
    if claimed_rows != unique_rows.numel():
        raise AssertionError("direct-scratch marker ownership is incorrect")
    if parameter_mismatches or moment_mismatches:
        raise AssertionError("direct-scratch RMSProp differs from dense oracle")
    print("TRITON_DIRECT_SCRATCH_GRAD_MICROCHECK_PASS")


def _run_triton_direct_scratch_fused_clear_gpu_microcheck():
    """Check two complete scatter/update/clear lifecycles across CUDA streams."""
    if not torch.cuda.is_available():
        raise RuntimeError("direct-scratch fused-clear microcheck requires CUDA")
    if not _NGRAM_DIRECT_SCRATCH_GRAD or not _DIRECT_SCRATCH_FUSED_CLEAR:
        raise RuntimeError(
            "fused-clear microcheck must enable direct scratch and fused clear"
        )
    if not _USE_TRITON_LAZY_RMS:
        raise RuntimeError("direct-scratch fused-clear microcheck requires lazy RMSProp")
    if _DIRECT_SCRATCH_PAIR_FUSION or _DIRECT_SCRATCH_OWNER_MIN:
        raise RuntimeError(
            "fused-clear lifecycle microcheck requires pair fusion and owner-min off"
        )

    device = torch.device("cuda")
    n_rows, width, batch, sequence = 97, 32, 4, 16
    torch.manual_seed(0xF05ED)
    torch.cuda.manual_seed(0xF05ED)

    parameter_candidate = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16().requires_grad_()
    parameter_reference = parameter_candidate.detach().clone()
    parameter_nonclear = parameter_reference.clone()
    moment_candidate = (
        torch.rand(n_rows, width, device=device) * 1e-2
    ).bfloat16()
    moment_reference = moment_candidate.clone()
    moment_nonclear = moment_candidate.clone()
    candidate_marker = torch.zeros(n_rows, dtype=torch.int32, device=device)
    reference_marker = torch.zeros_like(candidate_marker)
    nonclear_marker = torch.zeros_like(candidate_marker)
    # This is the production fused-clear invariant before the first backward:
    # every row starts at zero and subsequent RMS updates restore that state.
    scratch = torch.zeros(n_rows, width, dtype=torch.float32, device=device)
    scratch_before_rms = torch.empty_like(scratch)
    scratch_nonclear = torch.empty_like(scratch)

    flat_position = torch.arange(
        batch * sequence,
        dtype=torch.int64,
        device=device,
    )
    flat_rows0 = ((flat_position * 7 + 3) % 41).clone()
    flat_rows0[4:16] = 7
    flat_rows0[32::3] = 11
    flat_rows1 = ((flat_position * 13 + 5) % 47).clone()
    flat_rows1[:10] = 7       # overlap with step 1, with repeated occurrences
    flat_rows1[20:28] = 53    # rows absent from step 1
    indices0 = flat_rows0.view(batch, sequence)
    indices1 = flat_rows1.view(batch, sequence)

    lane = torch.arange(width, dtype=torch.int64, device=device)
    raw_coefficient0 = (
        (flat_position[:, None] * 3 + lane[None, :]) % 15
    ) - 7
    raw_coefficient1 = (
        (flat_position[:, None] * 5 + lane[None, :] * 2) % 17
    ) - 8
    # Integer multiples of a power of two make all duplicate-row FP32 sums
    # exact regardless of relaxed atomic arrival order.
    coefficient0 = (
        raw_coefficient0.view(batch, sequence, width).float() * (2.0 ** -8)
    ).bfloat16()
    coefficient1 = (
        raw_coefficient1.view(batch, sequence, width).float() * (2.0 ** -8)
    ).bfloat16()

    unique0 = torch.unique(flat_rows0)
    unique1 = torch.unique(flat_rows1)
    overlap = int(torch.isin(unique1, unique0).sum().item())
    new_rows = int((~torch.isin(unique1, unique0)).sum().item())
    repeated0 = flat_rows0.numel() - unique0.numel()
    repeated1 = flat_rows1.numel() - unique1.numel()
    if not overlap or not new_rows or not repeated0 or not repeated1:
        raise AssertionError("fused-clear two-step row coverage is inconsistent")

    def direct_loss_fn(table, lookup_rows, grad_table, coefficient):
        lookup = torch.ops.ngram_direct.embedding(
            table,
            lookup_rows,
            grad_table,
        )
        return (lookup * coefficient).sum()

    compiled_direct_loss = torch.compile(
        direct_loss_fn,
        dynamic=False,
        fullgraph=True,
    )
    current_stream = torch.cuda.current_stream()
    rms_stream = torch.cuda.Stream()

    batches = (
        (indices0, coefficient0, 0.999),
        (indices1, coefficient1, 0.9999),
    )
    for step, (indices, coefficient, beta2) in enumerate(batches, start=1):
        flat_rows = indices.flatten()
        unique_rows = torch.unique(flat_rows)
        dense_accum = torch.zeros_like(scratch)
        dense_accum.index_add_(
            0,
            flat_rows,
            coefficient.float().reshape(-1, width),
        )
        if not int((dense_accum != 0).sum().item()):
            raise AssertionError("fused-clear oracle unexpectedly has no gradients")
        dense_grad = dense_accum.bfloat16()

        loss = compiled_direct_loss(
            parameter_candidate,
            indices,
            scratch,
            coefficient,
        )
        loss.backward()
        if parameter_candidate.grad is not None:
            raise AssertionError("direct fused-clear backward materialized weight.grad")
        # Snapshot the value consumed by RMS on the same stream as backward.
        # This distinguishes scatter/oracle disagreement from update-codegen
        # disagreement even though the production kernel subsequently clears.
        scratch_before_rms.copy_(scratch)
        scratch_nonclear.copy_(scratch)

        # Match MuonAdamW.step(): RMS waits for the compiled backward, then the
        # model stream waits for RMS before another backward can touch scratch.
        rms_stream.wait_stream(current_stream)
        with torch.cuda.stream(rms_stream):
            triton_lazy_rmsprop_direct_scratch_rows_step(
                parameter_candidate,
                flat_rows,
                scratch,
                moment_candidate,
                candidate_marker,
                step,
                0.6,
                beta2,
                1e-10,
            )
            # An otherwise identical kernel specialization without the final
            # clear isolates any arithmetic-codegen effect of CLEAR_GRAD_ACCUM.
            triton_lazy_rmsprop_direct_scratch_rows_step(
                parameter_nonclear,
                flat_rows,
                scratch_nonclear,
                moment_nonclear,
                nonclear_marker,
                step,
                0.6,
                beta2,
                1e-10,
                clear_grad_accum=False,
            )
            triton_lazy_rmsprop_dense_rows_step(
                parameter_reference,
                flat_rows,
                dense_grad,
                moment_reference,
                reference_marker,
                step,
                0.6,
                beta2,
                1e-10,
            )
        current_stream.wait_stream(rms_stream)
        torch.cuda.synchronize()

        scratch_pre_mismatches = int(
            (scratch_before_rms != dense_accum).sum().item()
        )
        scratch_pre_max_abs = float(
            (scratch_before_rms - dense_accum).abs().max().item()
        )
        parameter_mismatches = int(
            (parameter_candidate != parameter_reference).sum().item()
        )
        moment_mismatches = int(
            (moment_candidate != moment_reference).sum().item()
        )
        parameter_max_abs = float(
            (parameter_candidate.float() - parameter_reference.float())
            .abs()
            .max()
            .item()
        )
        moment_max_abs = float(
            (moment_candidate.float() - moment_reference.float())
            .abs()
            .max()
            .item()
        )
        clear_nonclear_p_mismatches = int(
            (parameter_candidate != parameter_nonclear).sum().item()
        )
        clear_nonclear_v_mismatches = int(
            (moment_candidate != moment_nonclear).sum().item()
        )
        nonclear_dense_p_mismatches = int(
            (parameter_nonclear != parameter_reference).sum().item()
        )
        nonclear_dense_v_mismatches = int(
            (moment_nonclear != moment_reference).sum().item()
        )
        active_scratch_nonzero = int(
            (scratch.index_select(0, unique_rows) != 0).sum().item()
        )
        full_scratch_nonzero = int((scratch != 0).sum().item())
        candidate_claimed = int((candidate_marker == step).sum().item())
        nonclear_claimed = int((nonclear_marker == step).sum().item())
        reference_claimed = int((reference_marker == step).sum().item())
        expected_claimed = unique_rows.numel()
        print(
            "TRITON_DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK "
            f"step={step} beta2={beta2} occurrences={flat_rows.numel()} "
            f"unique={expected_claimed} candidate_claimed={candidate_claimed} "
            f"nonclear_claimed={nonclear_claimed} "
            f"reference_claimed={reference_claimed} "
            f"scratch_pre_mismatch={scratch_pre_mismatches} "
            f"scratch_pre_max_abs={scratch_pre_max_abs:.3e} "
            f"active_scratch_nonzero={active_scratch_nonzero} "
            f"full_scratch_nonzero={full_scratch_nonzero} "
            f"p_mismatch={parameter_mismatches} p_max_abs={parameter_max_abs:.3e} "
            f"v_mismatch={moment_mismatches} v_max_abs={moment_max_abs:.3e} "
            f"clear_vs_nonclear_p={clear_nonclear_p_mismatches} "
            f"clear_vs_nonclear_v={clear_nonclear_v_mismatches} "
            f"nonclear_vs_dense_p={nonclear_dense_p_mismatches} "
            f"nonclear_vs_dense_v={nonclear_dense_v_mismatches}"
        )
        if candidate_claimed != expected_claimed:
            raise AssertionError("fused-clear candidate marker ownership is incorrect")
        if nonclear_claimed != expected_claimed:
            raise AssertionError("nonclear control marker ownership is incorrect")
        if reference_claimed != expected_claimed:
            raise AssertionError("fused-clear reference marker ownership is incorrect")
        if scratch_pre_mismatches:
            raise AssertionError("direct scatter differs from the dense FP32 oracle")
        if active_scratch_nonzero or full_scratch_nonzero:
            raise AssertionError("fused-clear RMS update left stale scratch values")
        if clear_nonclear_p_mismatches or clear_nonclear_v_mismatches:
            raise AssertionError("fused clear changes the direct RMS arithmetic")
        if parameter_mismatches or moment_mismatches:
            raise AssertionError("fused-clear RMS lifecycle differs from dense oracle")

    print(
        "TRITON_DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK_COVERAGE "
        f"step1_repeated={repeated0} step2_repeated={repeated1} "
        f"step2_overlap={overlap} step2_new={new_rows}"
    )
    print("TRITON_DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK_PASS")


def _run_triton_direct_scratch_owner_min_gpu_microcheck():
    """Check owner selection and dense-like sums across row multiplicities."""
    if not torch.cuda.is_available():
        raise RuntimeError("direct-scratch owner-min microcheck requires CUDA")
    device = torch.device("cuda")
    n_rows, width = 47, 32
    row_counts = ((3, 1), (7, 2), (11, 3), (19, 10), (29, 11), (41, 17))
    flat_rows = [
        row
        for occurrence_rank in range(max(count for _row, count in row_counts))
        for row, count in row_counts
        if occurrence_rank < count
    ]
    if len(flat_rows) != 44:
        raise AssertionError("owner-min microcheck row construction is inconsistent")
    indices = torch.tensor(
        flat_rows,
        dtype=torch.int64,
        device=device,
    ).view(4, 11)

    torch.manual_seed(0x0A11CE)
    torch.cuda.manual_seed(0x0A11CE)
    weight = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16().requires_grad_()
    scratch = torch.empty(n_rows, width, dtype=torch.float32, device=device)
    owner_key = torch.empty(n_rows, dtype=torch.int32, device=device)
    coefficient = (
        torch.randn(*indices.shape, width, device=device) * 0.0713
    ).bfloat16()

    def owner_min_loss_fn(table, lookup_rows, grad_table, owners, coeff):
        lookup = torch.ops.ngram_direct.embedding_owner_min(
            table,
            lookup_rows,
            grad_table,
            owners,
        )
        # Reuse one output as the shared trigram does.  This first pass checks
        # that AOTAutograd retains the side-effecting registered backward; the
        # explicit strided scatter below supplies the numeric oracle input.
        return (
            (lookup * coeff).sum()
            + (lookup * (coeff * 2)).sum()
            - (lookup * (coeff / 2)).sum()
        )

    compiled_owner_min_loss = torch.compile(
        owner_min_loss_fn,
        dynamic=False,
        fullgraph=True,
    )
    loss = compiled_owner_min_loss(
        weight,
        indices,
        scratch,
        owner_key,
        coefficient,
    )
    loss.backward()
    torch.cuda.synchronize()
    if weight.grad is not None:
        raise AssertionError("owner-min embedding materialized weight.grad")

    # Exercise the cat-backward-style strided D dimension with arbitrary BF16
    # values.  This overwrites every touched scratch row from the AOT check.
    strided_storage = torch.empty(
        *indices.shape,
        width * 2,
        dtype=torch.bfloat16,
        device=device,
    )
    occurrence_grad = strided_storage[..., ::2]
    occurrence_grad.copy_(coefficient)
    if occurrence_grad.is_contiguous():
        raise AssertionError("owner-min stride test unexpectedly became contiguous")
    torch.ops.ngram_direct.scratch_scatter_owner_min(
        indices,
        occurrence_grad,
        scratch,
        owner_key,
    )
    torch.cuda.synchronize()

    # Reproduce PyTorch 2.9's large dense-embedding reduction structure:
    # stable row/flat-occurrence order, ten FP32 values per partial, then an
    # FP32 sum of partials and one final BF16 materialization boundary.
    flat_rows_cpu = indices.flatten().cpu()
    occurrence_grad_cpu = occurrence_grad.flatten(0, 1).float().cpu()
    dense_oracle = torch.zeros(n_rows, width, dtype=torch.float32)
    expected_owner_keys = {}
    for row, _count in row_counts:
        positions = (flat_rows_cpu == row).nonzero(as_tuple=False).flatten()
        expected_owner_keys[row] = 0x7FFFFFFF - int(positions[0])
        partials = []
        for chunk_start in range(0, positions.numel(), 10):
            partial = torch.zeros(width, dtype=torch.float32)
            for position in positions[chunk_start : chunk_start + 10]:
                partial.add_(occurrence_grad_cpu[int(position)])
            partials.append(partial)
        total = torch.zeros(width, dtype=torch.float32)
        for partial in partials:
            total.add_(partial)
        dense_oracle[row].copy_(total)

    touched_rows = torch.tensor(
        [row for row, _count in row_counts],
        dtype=torch.int64,
        device=device,
    )
    candidate_fp32 = scratch.index_select(0, touched_rows).cpu()
    oracle_fp32 = dense_oracle.index_select(
        0,
        touched_rows.cpu(),
    )
    low_count_slots = torch.tensor([0, 1], dtype=torch.int64)
    exact_low_mismatches = int(
        (
            candidate_fp32.index_select(0, low_count_slots)
            != oracle_fp32.index_select(0, low_count_slots)
        ).sum().item()
    )
    if exact_low_mismatches:
        raise AssertionError(
            "owner-min count-1/2 rows differ from the dense FP32 oracle: "
            f"{exact_low_mismatches} values"
        )

    actual_owner_keys = owner_key.index_select(0, touched_rows).cpu()
    expected_owner_key_tensor = torch.tensor(
        [expected_owner_keys[row] for row, _count in row_counts],
        dtype=torch.int32,
    )
    owner_mismatches = int(
        (actual_owner_keys != expected_owner_key_tensor).sum().item()
    )
    if owner_mismatches:
        raise AssertionError(
            f"owner-min selected {owner_mismatches} non-minimum occurrences"
        )

    high_count_slots = torch.arange(2, len(row_counts), dtype=torch.int64)
    candidate_high_bf16 = candidate_fp32.index_select(
        0,
        high_count_slots,
    ).bfloat16()
    oracle_high_bf16 = oracle_fp32.index_select(
        0,
        high_count_slots,
    ).bfloat16()
    high_bf16_mismatches = int(
        (candidate_high_bf16 != oracle_high_bf16).sum().item()
    )

    def bf16_ordered_int(tensor):
        raw = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        magnitude = raw & 0x7FFF
        return torch.where(
            (raw & 0x8000) != 0,
            0x8000 - magnitude,
            0x8000 + magnitude,
        )

    max_bf16_ulp = int(
        (
            bf16_ordered_int(candidate_high_bf16)
            - bf16_ordered_int(oracle_high_bf16)
        ).abs().max().item()
    )
    max_fp32_abs = float((candidate_fp32 - oracle_fp32).abs().max().item())
    if not torch.allclose(candidate_fp32, oracle_fp32, rtol=2e-6, atol=2e-6):
        raise AssertionError(
            "owner-min aggregation exceeds the expected FP32 order tolerance: "
            f"max_abs={max_fp32_abs}"
        )
    print(
        "TRITON_DIRECT_SCRATCH_OWNER_MIN_MICROCHECK "
        f"counts={[count for _row, count in row_counts]} "
        f"count12_fp32_mismatch={exact_low_mismatches} "
        f"owner_mismatch={owner_mismatches} "
        f"count3plus_bf16_mismatch={high_bf16_mismatches} "
        f"count3plus_max_ulp={max_bf16_ulp} "
        f"max_fp32_abs={max_fp32_abs:.3e}"
    )
    print("TRITON_DIRECT_SCRATCH_OWNER_MIN_MICROCHECK_PASS")


def _run_triton_direct_scratch_pair_gpu_microcheck():
    """Check fused K=2 forward and both AOT side-effecting backwards."""
    if not torch.cuda.is_available():
        raise RuntimeError("direct-scratch pair microcheck requires CUDA")
    device = torch.device("cuda")
    n_rows, width, batch, sequence = 97, 32, 4, 11
    torch.manual_seed(0x2A17)
    torch.cuda.manual_seed(0x2A17)

    weight0 = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16().requires_grad_()
    weight1 = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16().requires_grad_()
    reference0 = weight0.detach().clone()
    reference1 = weight1.detach().clone()
    # Exercise repeated rows both within a warp and across Triton programs.
    # Row 7 is deliberately active in both halves: the numeric row id may
    # collide across a K=2 pair, but the two independent scratch tables must
    # never alias or exchange gradients.
    flat_occurrence = torch.arange(
        batch * sequence,
        dtype=torch.int64,
        device=device,
    )
    flat_indices0 = ((flat_occurrence * 7 + 3) % 19).clone()
    flat_indices1 = ((flat_occurrence * 11 + 5) % 23).clone()
    flat_indices0[4:12] = 7
    flat_indices1[7:16] = 7
    flat_indices1[::9] = flat_indices0[::9]
    indices0 = flat_indices0.view(batch, sequence)
    indices1 = flat_indices1.view(batch, sequence)

    # A nonzero sentinel verifies that every active row is cleared before the
    # atomic scatter and that inactive rows remain untouched.
    scratch_sentinel = 2048.0
    scratch0 = torch.full(
        (n_rows, width),
        scratch_sentinel,
        dtype=torch.float32,
        device=device,
    )
    scratch1 = torch.full_like(scratch0, scratch_sentinel)

    # Binary-fraction coefficients keep every duplicate-row FP32 sum exact,
    # independent of atomic arrival order, while differing by occurrence,
    # feature lane, and output half.
    position = flat_occurrence.view(batch, sequence, 1)
    lane = torch.arange(width * 2, device=device).view(1, 1, -1)
    occurrence_scale0 = torch.where(
        position % 2 == 0,
        2.0 ** -5,
        2.0 ** -6,
    )
    occurrence_scale1 = torch.where(
        position % 3 == 0,
        2.0 ** -7,
        2.0 ** -8,
    )
    lane_sign = torch.where(lane % 2 == 0, 1.0, -1.0)
    half_sign = torch.where(lane < width, 1.0, -1.0)
    coefficient0 = (occurrence_scale0 * lane_sign).bfloat16()
    coefficient1 = (occurrence_scale1 * half_sign).bfloat16()

    def pair_shared_loss_fn(
        table0,
        table1,
        rows0,
        rows1,
        grad_table0,
        grad_table1,
        coeff0,
        coeff1,
    ):
        pair = torch.ops.ngram_direct.embedding_pair(
            table0,
            table1,
            rows0,
            rows1,
            grad_table0,
            grad_table1,
        )
        # The shared trigram value is consumed by three layers in the model.
        # Two independent uses here force AOTAutograd to accumulate one common
        # grad_output before invoking the registered pair backward.
        loss = (pair * coeff0).sum() + (pair * coeff1).sum()
        return pair, loss

    compiled_pair_shared_loss = torch.compile(
        pair_shared_loss_fn,
        dynamic=False,
        fullgraph=True,
    )
    pair_output, pair_loss = compiled_pair_shared_loss(
        weight0,
        weight1,
        indices0,
        indices1,
        scratch0,
        scratch1,
        coefficient0,
        coefficient1,
    )

    reference_output = torch.cat(
        [
            F.embedding(indices0, reference0),
            F.embedding(indices1, reference1),
        ],
        dim=-1,
    )
    forward_mismatches = int((pair_output != reference_output).sum().item())
    if forward_mismatches:
        raise AssertionError(
            f"paired direct forward differs at {forward_mismatches} values"
        )

    pair_loss.backward()
    torch.cuda.synchronize()
    if weight0.grad is not None or weight1.grad is not None:
        raise AssertionError("paired direct embedding materialized a weight gradient")

    active0 = indices0.flatten()
    active1 = indices1.flatten()
    unique0 = torch.unique(active0)
    unique1 = torch.unique(active1)
    occurrence_grad = (coefficient0 + coefficient1).float()
    expected0 = torch.full_like(scratch0, scratch_sentinel)
    expected1 = torch.full_like(scratch1, scratch_sentinel)
    accumulated0 = torch.zeros_like(scratch0)
    accumulated1 = torch.zeros_like(scratch1)
    accumulated0.index_add_(
        0,
        active0,
        occurrence_grad[..., :width].reshape(-1, width),
    )
    accumulated1.index_add_(
        0,
        active1,
        occurrence_grad[..., width:].reshape(-1, width),
    )
    expected0.index_copy_(0, unique0, accumulated0.index_select(0, unique0))
    expected1.index_copy_(0, unique1, accumulated1.index_select(0, unique1))
    scratch0_mismatches = int((scratch0 != expected0).sum().item())
    scratch1_mismatches = int((scratch1 != expected1).sum().item())
    repeated0 = active0.numel() - unique0.numel()
    repeated1 = active1.numel() - unique1.numel()
    cross_pair_rows = torch.isin(unique0, unique1).sum().item()
    if not repeated0 or not repeated1 or not cross_pair_rows:
        raise AssertionError("paired direct collision coverage is inconsistent")
    print(
        "TRITON_DIRECT_SCRATCH_PAIR_MICROCHECK "
        f"forward_mismatch={forward_mismatches} "
        f"scratch0_mismatch={scratch0_mismatches} "
        f"scratch1_mismatch={scratch1_mismatches} "
        f"repeated0={repeated0} repeated1={repeated1} "
        f"cross_pair_rows={cross_pair_rows} "
        f"shared_uses=2"
    )
    if scratch0_mismatches or scratch1_mismatches:
        raise AssertionError("paired direct AOT scratch differs from dense oracle")
    print("TRITON_DIRECT_SCRATCH_PAIR_MICROCHECK_PASS")


def _run_triton_direct_compact_scratch_gpu_microcheck():
    """Bit-check compact FP32 sums, owner mapping, RMS update, and clearing."""
    if not torch.cuda.is_available():
        raise RuntimeError("compact direct-scratch microcheck requires CUDA")
    if not _DIRECT_SCRATCH_FUSED_CLEAR:
        raise RuntimeError("compact direct-scratch microcheck requires fused clear")
    device = torch.device("cuda")
    n_rows, width, batch, sequence = 97, 32, 4, 11
    n_occurrences = batch * sequence

    flat_position = torch.arange(n_occurrences, device=device)
    flat_rows = ((flat_position * 7 + 3) % 29).clone()
    flat_rows[4:12] = 7
    flat_rows[17:23] = 19
    flat_rows[::13] = 41
    indices = flat_rows.view(batch, sequence)
    unique_rows = torch.unique(flat_rows)
    if unique_rows.numel() >= n_occurrences:
        raise AssertionError("compact microcheck did not create duplicates")

    torch.manual_seed(0xC04FAC7)
    torch.cuda.manual_seed(0xC04FAC7)
    weight = (
        torch.rand(n_rows, width, device=device) - 0.5
    ).bfloat16().requires_grad_()
    compact_scratch = torch.zeros(
        n_occurrences,
        width,
        dtype=torch.float32,
        device=device,
    )
    owner_key = torch.empty(n_rows, dtype=torch.int32, device=device)

    # Binary fractions keep duplicate sums exactly representable in FP32, so
    # the check isolates mapping/lifecycle errors from atomic arrival order.
    position = flat_position.view(batch, sequence, 1)
    lane = torch.arange(width, device=device).view(1, 1, width)
    coefficient0 = (
        torch.where(position % 2 == 0, 2.0**-5, -(2.0**-6))
        * torch.where(lane % 2 == 0, 1.0, -1.0)
    ).bfloat16()
    coefficient1 = (
        torch.where(position % 3 == 0, 2.0**-7, 2.0**-8)
        * torch.where(lane % 3 == 0, -1.0, 1.0)
    ).bfloat16()

    def compact_shared_loss_fn(table, rows, scratch, owners, coeff0, coeff1):
        lookup = torch.ops.ngram_direct.embedding_compact(
            table,
            rows,
            scratch,
            owners,
        )
        # The real shared trigram output feeds three layers; two independent
        # uses here verify AOTAutograd combines uses before the side effect.
        return (lookup * coeff0).sum() + (lookup * coeff1).sum()

    compiled_loss = torch.compile(
        compact_shared_loss_fn,
        dynamic=False,
        fullgraph=True,
    )
    loss = compiled_loss(
        weight,
        indices,
        compact_scratch,
        owner_key,
        coefficient0,
        coefficient1,
    )
    loss.backward()
    torch.cuda.synchronize()
    if weight.grad is not None:
        raise AssertionError("compact direct embedding materialized weight.grad")

    occurrence_grad = (coefficient0 + coefficient1).bfloat16()
    dense_scratch = torch.zeros(
        n_rows,
        width,
        dtype=torch.float32,
        device=device,
    )
    _triton_direct_scratch_scatter(
        indices,
        occurrence_grad,
        dense_scratch,
    )
    torch.cuda.synchronize()

    unique_owner_keys = owner_key.index_select(0, unique_rows)
    owner_slots = (0x7FFFFFFF - unique_owner_keys).to(torch.int64)
    first_positions = torch.stack(
        [
            (flat_rows == row).nonzero(as_tuple=False)[0, 0]
            for row in unique_rows
        ]
    )
    owner_mismatches = int((owner_slots != first_positions).sum().item())
    compact_sums = compact_scratch.index_select(0, owner_slots)
    dense_sums = dense_scratch.index_select(0, unique_rows)
    sum_mismatches = int((compact_sums != dense_sums).sum().item())
    if owner_mismatches or sum_mismatches:
        raise AssertionError(
            "compact owner/sum mismatch: "
            f"owners={owner_mismatches} sums={sum_mismatches}"
        )

    candidate_parameter = weight.detach().clone()
    reference_parameter = candidate_parameter.clone()
    initial_moment = (
        torch.rand(n_rows, width, device=device) * 0.02
    ).bfloat16()
    candidate_moment = initial_moment.clone()
    reference_moment = initial_moment.clone()
    reference_marker = torch.zeros(n_rows, dtype=torch.int32, device=device)
    step, lr, beta2, eps = 17, 0.6, 0.999, 1e-10

    triton_lazy_rmsprop_compact_direct_scratch_rows_step(
        candidate_parameter,
        flat_rows,
        compact_scratch,
        owner_key,
        candidate_moment,
        step,
        lr,
        beta2,
        eps,
    )
    triton_lazy_rmsprop_direct_scratch_rows_step(
        reference_parameter,
        flat_rows,
        dense_scratch,
        reference_moment,
        reference_marker,
        step,
        lr,
        beta2,
        eps,
    )
    torch.cuda.synchronize()

    parameter_mismatches = int(
        (candidate_parameter != reference_parameter).sum().item()
    )
    moment_mismatches = int((candidate_moment != reference_moment).sum().item())
    compact_nonzero = int((compact_scratch != 0).sum().item())
    dense_nonzero = int((dense_scratch != 0).sum().item())
    claimed_rows = int((reference_marker == step).sum().item())
    print(
        "TRITON_DIRECT_COMPACT_SCRATCH_MICROCHECK "
        f"occurrences={n_occurrences} unique={unique_rows.numel()} "
        f"owner_mismatch={owner_mismatches} sum_mismatch={sum_mismatches} "
        f"p_mismatch={parameter_mismatches} v_mismatch={moment_mismatches} "
        f"compact_nonzero={compact_nonzero} dense_nonzero={dense_nonzero} "
        f"reference_claimed={claimed_rows}"
    )
    if parameter_mismatches or moment_mismatches:
        raise AssertionError("compact RMSProp differs from dense FP32 scratch")
    if compact_nonzero or dense_nonzero:
        raise AssertionError("compact/dense fused clear left stale gradients")
    if claimed_rows != unique_rows.numel():
        raise AssertionError("dense reference claimed the wrong active-row count")
    print("TRITON_DIRECT_COMPACT_SCRATCH_MICROCHECK_PASS")


if os.environ.get("TRITON_DIRECT_COMPACT_SCRATCH_MICROCHECK", "0") == "1":
    _run_triton_direct_compact_scratch_gpu_microcheck()
    sys.exit(0)
if os.environ.get("TRITON_LAZY_RMS_MICROCHECK", "0") == "1":
    _run_triton_lazy_rms_gpu_microcheck()
    sys.exit(0)
if os.environ.get("TRITON_UNCOALESCED_SPARSE_MICROCHECK", "0") == "1":
    _run_triton_uncoalesced_sparse_gpu_microcheck()
    sys.exit(0)
if _DIRECT_SCRATCH_FUSED_CLEAR_MICROCHECK:
    _run_triton_direct_scratch_fused_clear_gpu_microcheck()
    sys.exit(0)
if os.environ.get("TRITON_DIRECT_SCRATCH_GRAD_MICROCHECK", "0") == "1":
    _run_triton_direct_scratch_grad_gpu_microcheck()
    sys.exit(0)
if (
    os.environ.get("TRITON_DIRECT_SCRATCH_OWNER_MIN_MICROCHECK", "0")
    == "1"
):
    _run_triton_direct_scratch_owner_min_gpu_microcheck()
    sys.exit(0)
if os.environ.get("TRITON_DIRECT_SCRATCH_PAIR_MICROCHECK", "0") == "1":
    _run_triton_direct_scratch_pair_gpu_microcheck()
    sys.exit(0)


@torch.compile(
    dynamic=False,
    fullgraph=True,
    **_OPTIMIZER_COMPILE_KWARGS,
)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # RMSProp CPU tensors (no beta1 -- saves first moment VRAM)
        self._rmsprop_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        ngram_groups = [
            group for group in self.param_groups if group.get("is_ngram_ve", False)
        ]
        bigram_specs = [
            spec
            for group in ngram_groups
            for spec in group["active_index_specs"]
            if spec[0] == "bigram"
        ]
        trigram_specs = [
            spec
            for group in ngram_groups
            for spec in group["active_index_specs"]
            if spec[0] == "trigram"
        ]
        if not bigram_specs or not trigram_specs:
            raise RuntimeError("lazy n-gram RMSProp requires bigram and trigram specs")
        ngram_device = ngram_groups[0]["params"][0].device
        self._bigram_hash_primes = torch.tensor(
            [[spec[1], spec[2]] for spec in bigram_specs],
            dtype=torch.long,
            device=ngram_device,
        )
        self._trigram_hash_primes = torch.tensor(
            [[spec[1], spec[2], spec[3]] for spec in trigram_specs],
            dtype=torch.long,
            device=ngram_device,
        )
        self._bigram_table_size = bigram_specs[0][-1]
        if any(spec[-1] != self._bigram_table_size for spec in bigram_specs):
            raise RuntimeError("all bigram tables must have one common size")
        self._trigram_table_sizes = torch.tensor(
            [spec[-1] for spec in trigram_specs],
            dtype=torch.long,
            device=ngram_device,
        )
        if self._trigram_table_sizes.numel() != self._trigram_hash_primes.shape[0]:
            raise RuntimeError("trigram size/hash specification count mismatch")
        self.needs_ngram_batch = any(
            not group.get("sparse_grad", False) for group in ngram_groups
        )
        self._ngram_active_rows = None
        step_fns = {
            "adamw": self._step_adamw,
            "rmsprop": self._step_rmsprop,
            "muon": self._step_muon,
        }
        self._step_dispatch = tuple((step_fns[group["kind"]], group) for group in self.param_groups)
        dispatch_by_kind = {
            kind: tuple(item for item in self._step_dispatch if item[1]["kind"] == kind)
            for kind in ("adamw", "rmsprop", "muon")
        }
        self._optimizer_streams = (
            (torch.cuda.Stream(), dispatch_by_kind["adamw"]),
            (torch.cuda.Stream(), dispatch_by_kind["rmsprop"]),
            (torch.cuda.Stream(), dispatch_by_kind["muon"]),
        )

    @torch.no_grad()
    def set_ngram_batch(self, idx):
        """Record exact active n-gram rows for the pending optimizer step.

        This is called after backward but before ``x`` advances to the next
        loader slot.  Dense embedding gradients have already aggregated all
        repeated indices and hash collisions. Preserve the model's exact hash
        order here: the Triton path claims duplicates with a device marker,
        while the Inductor fallback explicitly de-duplicates each table.
        """
        if idx.ndim != 2:
            raise ValueError(f"expected [B, T] token batch, got {tuple(idx.shape)}")
        if not self.needs_ngram_batch:
            self._ngram_active_rows = None
            return
        bigram_rows, trigram_rows = ngram_active_rows_fused(
            idx,
            self._bigram_hash_primes,
            self._trigram_hash_primes,
            self._bigram_table_size,
            self._trigram_table_sizes,
        )
        self.set_ngram_active_rows(tuple(bigram_rows) + tuple(trigram_rows))

    @torch.no_grad()
    def set_ngram_active_rows(self, ordered_rows):
        """Attach model-produced rows to n-gram parameters in optimizer order."""
        if not self.needs_ngram_batch:
            self._ngram_active_rows = None
            return
        ordered_rows = tuple(ordered_rows)
        expected_rows = sum(
            len(group["params"])
            for group in self.param_groups
            if group.get("is_ngram_ve", False)
        )
        if len(ordered_rows) != expected_rows:
            raise RuntimeError(
                f"received {len(ordered_rows)} n-gram row tensors, expected {expected_rows}"
            )
        active_rows = {}
        row_i = 0
        for group in self.param_groups:
            if not group.get("is_ngram_ve", False):
                continue
            for parameter, spec in zip(group["params"], group["active_index_specs"]):
                if spec[0] not in ("bigram", "trigram"):
                    raise ValueError(f"unknown n-gram active-index kind: {spec[0]}")
                active_rows[parameter] = ordered_rows[row_i]
                row_i += 1
        if row_i != expected_rows:
            raise RuntimeError("n-gram active-row ordering is inconsistent")
        self._ngram_active_rows = active_rows

    def _active_rows_for_parameter(self, parameter):
        if self._ngram_active_rows is None:
            raise RuntimeError("set_ngram_batch(x) must run before optimizer.step()")
        try:
            return self._ngram_active_rows[parameter]
        except KeyError as exc:
            raise RuntimeError("missing active rows for n-gram parameter") from exc

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_rmsprop(self, group):
        """Lazy row-wise RMSProp for the hashed n-gram embedding tables."""
        if group["weight_decay"] != 0.0:
            raise RuntimeError(
                "lazy n-gram RMSProp currently requires weight_decay=0; "
                "otherwise untouched parameters also need delayed decay"
            )

        params = group["params"]
        specs = group["active_index_specs"]
        if len(params) != len(specs):
            raise RuntimeError("n-gram parameter/hash-spec ordering is inconsistent")
        direct_grad_scratch = group.get("direct_grad_scratch")
        direct_compact_owner_key = group.get("direct_compact_owner_key")
        if direct_grad_scratch is not None:
            if len(direct_grad_scratch) != len(params):
                raise RuntimeError("direct scratch/parameter ordering is inconsistent")
            if any(scratch is None for scratch in direct_grad_scratch):
                raise RuntimeError("a direct n-gram scratch table is uninitialized")
            if direct_compact_owner_key is None:
                direct_compact_owner_key = (None,) * len(params)
            if len(direct_compact_owner_key) != len(params):
                raise RuntimeError("compact owner/parameter ordering is inconsistent")

        group_step = group.get("_lazy_step", 0) + 1
        beta2 = float(group["beta2"])
        if not 0.0 < beta2 < 1.0:
            raise ValueError(f"lazy RMSProp beta2 must be in (0, 1), got {beta2}")
        group["_lazy_step"] = group_step

        self._rmsprop_step_t.fill_(group_step)
        self._rmsprop_lr_t.fill_(group["lr"])
        self._rmsprop_beta2_t.fill_(beta2)
        self._rmsprop_eps_t.fill_(group["eps"])

        for parameter_i, (p, spec) in enumerate(zip(params, specs)):
            direct_scratch = (
                None
                if direct_grad_scratch is None
                else direct_grad_scratch[parameter_i]
            )
            compact_owner_key = (
                None
                if direct_grad_scratch is None
                else direct_compact_owner_key[parameter_i]
            )
            # Preserve the default path's lazy state initialization exactly.
            # Direct mode is the intentional exception: its autograd formula
            # returns None for weight, so p.grad must remain absent.
            if direct_scratch is None and p.grad is None:
                continue
            if direct_scratch is not None and p.grad is not None:
                raise RuntimeError(
                    "direct n-gram backward unexpectedly materialized weight.grad"
                )
            state = self.state[p]
            if not state:
                state["step"] = group_step
                state["exp_avg_sq"] = torch.zeros_like(p)
                if _USE_TRITON_LAZY_RMS:
                    # One persistent marker per table row (~2 MiB/table).
                    # The monotonically increasing optimizer step is the
                    # generation id, so the marker never needs clearing.
                    if compact_owner_key is None:
                        state["row_marker"] = torch.zeros(
                            p.shape[0], dtype=torch.int32, device=p.device
                        )
                    if _NGRAM_UNCOALESCED_SPARSE:
                        # Deliberately uninitialized: the first kernel clears
                        # only rows present in the current COO gradient before
                        # any atomic accumulation reaches this scratch table.
                        state["grad_accum_fp32"] = torch.empty(
                            p.shape,
                            dtype=torch.float32,
                            device=p.device,
                        )
            else:
                state["step"] = group_step

            if direct_scratch is not None:
                if not _USE_TRITON_LAZY_RMS:
                    raise RuntimeError(
                        "NGRAM_DIRECT_SCRATCH_GRAD requires USE_TRITON_LAZY_RMS=1"
                    )
                row_indices = self._active_rows_for_parameter(p)
                # Backward writes scratch on the default stream.  step() makes
                # this RMS stream wait on that stream before reaching here;
                # the final wait also prevents the next backward from clearing
                # rows while this update still reads them.
                if compact_owner_key is None:
                    triton_lazy_rmsprop_direct_scratch_rows_step(
                        p,
                        row_indices,
                        direct_scratch,
                        state["exp_avg_sq"],
                        state["row_marker"],
                        group_step,
                        group["lr"],
                        beta2,
                        group["eps"],
                    )
                else:
                    triton_lazy_rmsprop_compact_direct_scratch_rows_step(
                        p,
                        row_indices,
                        direct_scratch,
                        compact_owner_key,
                        state["exp_avg_sq"],
                        group_step,
                        group["lr"],
                        beta2,
                        group["eps"],
                    )
                continue

            grad = p.grad
            if grad.layout == torch.sparse_coo:
                # Embedding sparse backward emits an uncoalesced COO tensor:
                # repeated tokens and hash collisions must be summed before
                # squaring, exactly as dense embedding backward does.
                if grad.sparse_dim() != 1 or grad.dense_dim() != 1:
                    raise RuntimeError(
                        "expected sparse embedding grad with one sparse row dimension"
                    )
                if _NGRAM_UNCOALESCED_SPARSE:
                    if not _USE_TRITON_LAZY_RMS:
                        raise RuntimeError(
                            "NGRAM_UNCOALESCED_SPARSE requires "
                            "USE_TRITON_LAZY_RMS=1"
                        )
                    # The underscored accessors intentionally preserve the
                    # original occurrence order; public indices()/values()
                    # require a costly global coalesce/sort first.
                    row_indices = grad._indices()[0].contiguous()
                    row_grad = grad._values().contiguous()
                    triton_lazy_rmsprop_uncoalesced_sparse_rows_step(
                        p,
                        row_indices,
                        row_grad,
                        state["grad_accum_fp32"],
                        state["exp_avg_sq"],
                        state["row_marker"],
                        group_step,
                        group["lr"],
                        beta2,
                        group["eps"],
                    )
                else:
                    grad = grad.coalesce()
                    # Break the sparse tensor view/base relationship before
                    # the dynamic compiled row kernel. Torch 2.9's automatic-
                    # dynamic analysis otherwise trips over COO value strides.
                    row_indices = grad.indices()[0].contiguous()
                    row_grad = grad.values().contiguous()
                    lazy_rmsprop_sparse_rows_step_dynamic(
                        p,
                        row_indices,
                        row_grad,
                        state["exp_avg_sq"],
                        self._rmsprop_step_t,
                        self._rmsprop_lr_t,
                        self._rmsprop_beta2_t,
                        self._rmsprop_eps_t,
                    )
            elif grad.layout == torch.strided:
                row_indices = self._active_rows_for_parameter(p)
                if _USE_TRITON_LAZY_RMS:
                    triton_lazy_rmsprop_dense_rows_step(
                        p,
                        row_indices,
                        grad,
                        state["exp_avg_sq"],
                        state["row_marker"],
                        group_step,
                        group["lr"],
                        beta2,
                        group["eps"],
                    )
                else:
                    # Exact Inductor reference/fallback. Duplicate index_copy
                    # writes are unordered, so de-duplicate before updating.
                    if _NGRAM_UNIQUE_ROWS:
                        row_indices = torch.unique(row_indices)
                    lazy_rmsprop_dense_rows_step_dynamic(
                        p,
                        row_indices,
                        grad,
                        state["exp_avg_sq"],
                        self._rmsprop_step_t,
                        self._rmsprop_lr_t,
                        self._rmsprop_beta2_t,
                        self._rmsprop_eps_t,
                    )
            else:
                raise RuntimeError(f"unsupported n-gram gradient layout: {grad.layout}")

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        current_stream = torch.cuda.current_stream()
        for stream, _ in self._optimizer_streams:
            stream.wait_stream(current_stream)
        for stream, dispatch in self._optimizer_streams:
            with torch.cuda.stream(stream):
                for step_fn, group in dispatch:
                    step_fn(group)
        for stream, _ in self._optimizer_streams:
            current_stream.wait_stream(stream)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = int(os.environ.get("ASPECT_RATIO", 96))  # model_dim = depth * ASPECT_RATIO (d8*96=768 -> dim=768, 6 heads)
HEAD_DIM = int(os.environ.get("HEAD_DIM", 128))  # target head dimension for attention
WINDOW_PATTERN = os.environ.get("WINDOW_PATTERN", "TTTL")  # 3 tiny + 1 long -- sandwich norm + warmdown=0.8 variant

# Optimization
TOTAL_BATCH_SIZE = int(os.environ.get("DEVICE_BATCH_SIZE", 72)) * TRAIN_SEQ_LEN * int(os.environ.get("GRAD_ACCUM", 1))
EMBEDDING_LR = float(os.environ.get("EMBEDDING_LR", 0.6))  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = float(os.environ.get("UNEMBEDDING_LR", 0.004))  # learning rate for lm_head (Adam)
MATRIX_LR = float(os.environ.get("MATRIX_LR", 0.04))  # learning rate for matrix parameters (Muon)
SCALAR_LR = float(os.environ.get("SCALAR_LR", 0.8))  # x0 Muon warmdown SCALAR_LR=0.8
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.1))  # baseline WD
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
DEMON_FINAL_BETA1 = float(os.environ.get("DEMON_FINAL_BETA1", 0.55))  # baseline Demon
NGRAM_VE_BETAS = (0.5, 0.999)  # RMSProp only uses beta2=0.999; higher beta2 preserves gradient history for sparse tables
NGRAM_VE_LR_SCALE = float(os.environ.get("NGRAM_VE_LR_SCALE", 1.0))  # RMSProp base LR scale
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.0))  # fraction of time budget for LR warmup
WARMDOWN_RATIO = float(os.environ.get("WARMDOWN_RATIO", 0.95))  # extend warmdown trend (0.80->0.85->0.90->0.95), warmdown starts at 5%
ADAM_WARMDOWN_RATIO = float(os.environ.get("ADAM_WARMDOWN_RATIO", 0.65))  # slightly longer Adam warmdown to match extended Muon warmdown
NGRAM_WARMDOWN_RATIO = 0.0  # no warmdown for bigram/trigram VE (sparse tables benefit from full-rate training)
FINAL_LR_FRAC = float(os.environ.get("FINAL_LR_FRAC", 0.05))  # restored FLR=0.05

# Model size
DEPTH = int(os.environ.get("DEPTH", 8))  # number of transformer layers
DEVICE_BATCH_SIZE = int(os.environ.get("DEVICE_BATCH_SIZE", 72))  # per-device batch size -- B200

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
_SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
# No autocast: model is natively BF16 -- eliminates FP32->BF16 cast overhead in compile graph
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
B200_BF16_PEAK_FLOPS = 2.25e15

tokenizer = Tokenizer.from_directory()
BOS_TOKEN_ID = tokenizer.get_bos_token_id()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = int(os.environ["MODEL_DIM"]) if os.environ.get("MODEL_DIM") else ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")
print(f"Training sequence length: {TRAIN_SEQ_LEN:,}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model.expand_bigram_tables_crn(
    int(os.environ.get("BIGRAM_CRN_MULT", "512")),
    _SEED,
)
model.expand_trigram_tables_crn(
    (
        int(os.environ.get("TRIGRAM_CRN_MULT_0", "2048")),
        int(os.environ.get("TRIGRAM_CRN_MULT_1", "2048")),
    ),
    _SEED,
)
# Cast entire model to BF16: enables removing autocast, simplifies compile graph
model.to(dtype=torch.bfloat16)
model.initialize_direct_ngram_scratch()
if _NGRAM_UNCOALESCED_SPARSE:
    if not model.ngram_sparse_grad:
        raise RuntimeError(
            "NGRAM_UNCOALESCED_SPARSE=1 requires NGRAM_SPARSE_GRAD=1"
        )
    if not _USE_TRITON_LAZY_RMS:
        raise RuntimeError(
            "NGRAM_UNCOALESCED_SPARSE=1 requires USE_TRITON_LAZY_RMS=1"
        )
if _NGRAM_DIRECT_SCRATCH_GRAD:
    if model.ngram_sparse_grad:
        raise RuntimeError(
            "NGRAM_DIRECT_SCRATCH_GRAD=1 requires dense embedding weights"
        )
    if not _USE_TRITON_LAZY_RMS:
        raise RuntimeError(
            "NGRAM_DIRECT_SCRATCH_GRAD=1 requires USE_TRITON_LAZY_RMS=1"
        )

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    ngram_ve_betas=NGRAM_VE_BETAS,
    ngram_ve_lr_scale=NGRAM_VE_LR_SCALE,
)
print(
    "N-gram optimizer: active-row lazy RMSProp; gradient layout="
    + (
        (
            (
                "direct K=2 pair gather/scatter -> persistent FP32 scratch "
                "(no weight.grad/COO)"
                if _DIRECT_SCRATCH_PAIR_FUSION
                else "direct backward -> owner-min FP32 scratch "
                "(duplicate-only atomics; no weight.grad/COO)"
            )
            if (_DIRECT_SCRATCH_PAIR_FUSION or _DIRECT_SCRATCH_OWNER_MIN)
            else "direct backward -> persistent dense/compact FP32 scratch (no weight.grad/COO)"
        )
        if _NGRAM_DIRECT_SCRATCH_GRAD
        else (
            "sparse COO (uncoalesced FP32 scatter)"
            if _NGRAM_UNCOALESCED_SPARSE
            else (
                "sparse COO (experimental)"
                if model.ngram_sparse_grad
                else "dense (compile-safe)"
            )
        )
    )
)
if _NGRAM_DIRECT_SCRATCH_GRAD:
    print(
        "Direct scratch clearing: "
        + (
            "startup zero + fused RMS post-consume clear"
            if _DIRECT_SCRATCH_FUSED_CLEAR
            else "per-backward active-row pre-clear"
        )
    )
if _USE_TRITON_LAZY_RMS:
    print(
        "N-gram active-row kernel: Triton atomic-dedup "
        f"BLOCK_R={_TRITON_RMS_BLOCK_R} BLOCK_D=512 "
        f"warps={_TRITON_RMS_NUM_WARPS} "
        f"round_profile={_TRITON_RMS_ROUND_PROFILE_NAME}"
    )

muon_groups = []
ngram_groups = []
x0_warmdown_groups = []
adam_groups = []
adam_demon_groups = []
muon_group_lrs = []
x0_group_lrs = []
adam_group_lrs = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        muon_groups.append(group)
        muon_group_lrs.append((group, group["initial_lr"]))
    elif group.get("is_ngram_ve", False):
        ngram_groups.append(group)
    elif group.get("is_x0_muon_warmdown", False):
        x0_warmdown_groups.append(group)
        x0_group_lrs.append((group, group["initial_lr"]))
    else:
        adam_groups.append(group)
        adam_group_lrs.append((group, group["initial_lr"]))
        if group.get("demon_beta1", False):
            adam_demon_groups.append((group, group["betas"][1]))

# Materialize every optimizer state before Inductor reserves CUDA-graph pools.
# The values exactly match each step function's lazy zero initialization; only
# allocation order changes, which is essential with this near-capacity model.
if _NGRAM_DIRECT_SCRATCH_GRAD:
    for group in ngram_groups:
        compact_owners = group.get("direct_compact_owner_key")
        if compact_owners is None:
            compact_owners = (None,) * len(group["params"])
        for p, compact_owner in zip(group["params"], compact_owners):
            state = optimizer.state[p]
            state["step"] = 0
            state["exp_avg_sq"] = torch.zeros_like(p)
            if _USE_TRITON_LAZY_RMS and compact_owner is None:
                state["row_marker"] = torch.zeros(
                    p.shape[0], dtype=torch.int32, device=p.device
                )

for group in optimizer.param_groups:
    if group["kind"] == "adamw":
        for p in group["params"]:
            state = optimizer.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
    elif group["kind"] == "muon" and group["params"]:
        p = group["params"][0]
        state = optimizer.state[p]
        group_num_params = len(group["params"])
        shape = p.shape
        state["momentum_buffer"] = torch.zeros(
            group_num_params, *shape, dtype=p.dtype, device=p.device
        )
        state_shape = (
            (group_num_params, shape[-2], 1)
            if shape[-2] >= shape[-1]
            else (group_num_params, 1, shape[-1])
        )
        state["second_momentum_buffer"] = torch.zeros(
            state_shape, dtype=p.dtype, device=p.device
        )

model = torch.compile(model, dynamic=False, fullgraph=True, **({"mode": os.environ["COMPILE_MODE"]} if os.environ.get("COMPILE_MODE") else {}))

_FAST_LOADER_HEADER = struct.Struct("<4sQqI")


class FastBatchPipeline:
    """Bounded CPU-process producer feeding independent pinned host slots."""

    def __init__(self, batch_size, sequence_len, depth=8):
        if sys.byteorder != "little":
            raise RuntimeError("fast-loader binary protocol currently requires little-endian host")
        if depth < 2:
            raise ValueError("FAST_LOADER_DEPTH must be at least 2")

        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.depth = depth
        self.host_slots = torch.empty(
            (depth, 2, batch_size, sequence_len),
            dtype=torch.long,
            pin_memory=True,
        )
        self.payload_nbytes = (
            self.host_slots[0].numel() * self.host_slots.element_size()
        )
        self._host_views = [
            memoryview(self.host_slots[slot].numpy()).cast("B")
            for slot in range(depth)
        ]
        self._free = queue.Queue(maxsize=depth)
        self._ready = queue.Queue(maxsize=depth)
        for slot in range(depth):
            self._free.put_nowait(slot)

        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr_tail = bytearray()
        self._slot_states = ["FREE"] * depth
        self._error = None
        self._closed = False
        self._prefetch_started = False
        self._last_progress = time.monotonic()
        self._acquire_timeout = float(os.environ.get("FAST_LOADER_TIMEOUT", "30"))
        if not math.isfinite(self._acquire_timeout) or self._acquire_timeout <= 0:
            raise ValueError("FAST_LOADER_TIMEOUT must be a positive finite number")

        worker_env = os.environ.copy()
        worker_env.update({
            "FAST_LOADER_BATCH_SIZE": str(batch_size),
            "FAST_LOADER_SEQUENCE_LEN": str(sequence_len),
        })
        self._proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), _FAST_LOADER_WORKER_FLAG],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            env=worker_env,
        )
        if (
            self._proc.stdin is None
            or self._proc.stdout is None
            or self._proc.stderr is None
        ):
            raise RuntimeError("failed to create fast-loader subprocess pipes")

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="fast-loader-stderr",
            daemon=True,
        )
        self._reader_thread = threading.Thread(
            target=self._read_frames,
            name="fast-loader-reader",
            daemon=True,
        )
        self._stderr_thread.start()
        self._reader_thread.start()
        atexit.register(self.close)

    def _drain_stderr(self):
        try:
            while True:
                chunk = self._proc.stderr.read(4096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail.extend(chunk)
                    if len(self._stderr_tail) > 65536:
                        del self._stderr_tail[:-65536]
        except (OSError, ValueError) as exc:
            if not self._stop.is_set():
                self._set_error(f"fast-loader stderr drain failed: {exc}")
                self._terminate_worker()

    def _stderr_text(self):
        with self._stderr_lock:
            return bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()

    def _set_error(self, message):
        with self._state_lock:
            if self._error is None:
                self._error = message

    def _mark_progress(self):
        with self._state_lock:
            self._last_progress = time.monotonic()

    def _terminate_worker(self):
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def _read_exact(self, stream, target, allow_clean_eof=False):
        view = memoryview(target).cast("B")
        offset = 0
        while offset < len(view):
            count = stream.readinto(view[offset:])
            if count is None:
                continue
            if count == 0:
                if allow_clean_eof and offset == 0:
                    return False
                raise EOFError(
                    f"fast-loader pipe ended after {offset}/{len(view)} frame bytes"
                )
            offset += count
            self._mark_progress()
        return True

    def _get_free_slot(self):
        while not self._stop.is_set():
            try:
                slot = self._free.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._state_lock:
                state = self._slot_states[slot]
                if state != "FREE":
                    raise RuntimeError(
                        f"fast-loader host slot {slot} was queued free in state {state}"
                    )
                self._slot_states[slot] = "FILLING"
            return slot
        return None

    def _return_unpublished_slot(self, slot):
        with self._state_lock:
            state = self._slot_states[slot]
            if state not in {"FILLING", "READY"}:
                raise RuntimeError(
                    f"cannot return fast-loader host slot {slot} from state {state}"
                )
            self._slot_states[slot] = "FREE"
        if not self._stop.is_set():
            self._free.put_nowait(slot)

    def _put_ready(self, item):
        slot = item[0]
        with self._state_lock:
            state = self._slot_states[slot]
            if state != "FILLING":
                raise RuntimeError(
                    f"cannot publish fast-loader host slot {slot} from state {state}"
                )
            self._slot_states[slot] = "READY"
        while not self._stop.is_set():
            try:
                self._ready.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        self._return_unpublished_slot(slot)
        return False

    def _read_frames(self):
        expected_sequence = 0
        held_slot = None
        try:
            while not self._stop.is_set():
                held_slot = self._get_free_slot()
                if held_slot is None:
                    return

                header_bytes = bytearray(_FAST_LOADER_HEADER.size)
                if not self._read_exact(
                    self._proc.stdout,
                    header_bytes,
                    allow_clean_eof=True,
                ):
                    self._return_unpublished_slot(held_slot)
                    held_slot = None
                    if self._stop.is_set():
                        return
                    returncode = self._proc.wait(timeout=2)
                    raise RuntimeError(
                        f"fast-loader worker exited unexpectedly with code {returncode}"
                    )

                magic, sequence, batch_epoch, payload_nbytes = _FAST_LOADER_HEADER.unpack(
                    header_bytes
                )
                if magic != _FAST_LOADER_MAGIC:
                    raise RuntimeError(f"invalid fast-loader frame magic: {magic!r}")
                if sequence != expected_sequence:
                    raise RuntimeError(
                        f"fast-loader frame sequence {sequence}, expected {expected_sequence}"
                    )
                if payload_nbytes != self.payload_nbytes:
                    raise RuntimeError(
                        f"fast-loader payload size {payload_nbytes}, expected {self.payload_nbytes}"
                    )

                self._read_exact(self._proc.stdout, self._host_views[held_slot])
                if not self._put_ready((held_slot, sequence, batch_epoch)):
                    return
                held_slot = None
                expected_sequence += 1
        except BaseException as exc:
            if held_slot is not None:
                try:
                    self._return_unpublished_slot(held_slot)
                except (queue.Full, RuntimeError):
                    pass
            if not self._stop.is_set():
                self._set_error(str(exc))
                self._terminate_worker()

    def start_prefetch(self):
        """Release the worker's post-frame-0 barrier exactly once."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("fast-loader pipeline is closed")
            if self._error is not None:
                raise RuntimeError(self._error)
            if self._prefetch_started:
                return
            self._prefetch_started = True
            self._last_progress = time.monotonic()
        try:
            written = self._proc.stdin.write(b"\x01")
            if written != 1:
                raise BrokenPipeError(
                    f"fast-loader start barrier wrote {written!r}/1 bytes"
                )
            self._proc.stdin.flush()
            self._proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._set_error(f"failed to release fast-loader start barrier: {exc}")
            self._terminate_worker()
            detail = self._stderr_text()
            suffix = f"\nworker stderr:\n{detail}" if detail else ""
            raise RuntimeError(
                f"failed to release fast-loader start barrier: {exc}{suffix}"
            ) from exc

    def acquire(self):
        acquire_started = time.monotonic()
        while not self._stop.is_set():
            with self._state_lock:
                error = self._error
            if error is not None:
                detail = self._stderr_text()
                if detail:
                    error = f"{error}\nworker stderr:\n{detail}"
                raise RuntimeError(error)
            try:
                item = self._ready.get(timeout=0.1)
            except queue.Empty:
                if self._proc.poll() is not None:
                    self._set_error(
                        f"fast-loader worker stopped with code {self._proc.returncode}"
                    )
                    continue
                with self._state_lock:
                    last_activity = max(self._last_progress, acquire_started)
                stalled_for = time.monotonic() - last_activity
                if stalled_for >= self._acquire_timeout:
                    self._set_error(
                        f"fast-loader made no pipe progress for {stalled_for:.1f}s "
                        f"(timeout {self._acquire_timeout:.1f}s)"
                    )
                    self._terminate_worker()
                continue

            slot = item[0]
            fatal_error = None
            with self._state_lock:
                state = self._slot_states[slot]
                if state != "READY":
                    raise RuntimeError(
                        f"acquired fast-loader host slot {slot} in state {state}"
                    )
                # A reader/stderr thread can publish a fatal error after the
                # pre-get check above. Do not lease one more batch once that
                # error is visible.
                fatal_error = self._error
                if fatal_error is None:
                    self._slot_states[slot] = "LEASED"
                    self._last_progress = time.monotonic()
                else:
                    self._slot_states[slot] = "FREE"
            if fatal_error is not None:
                self._free.put_nowait(slot)
                detail = self._stderr_text()
                if detail:
                    fatal_error = f"{fatal_error}\nworker stderr:\n{detail}"
                raise RuntimeError(fatal_error)
            return item
        raise RuntimeError("fast-loader pipeline is closed")

    def release(self, slot):
        if self._stop.is_set():
            return
        if not isinstance(slot, int) or not 0 <= slot < self.depth:
            raise ValueError(f"invalid fast-loader host slot {slot!r}")
        with self._state_lock:
            state = self._slot_states[slot]
            if state != "LEASED":
                raise RuntimeError(
                    f"cannot release fast-loader host slot {slot} from state {state}"
                )
            self._slot_states[slot] = "FREE"
        try:
            self._free.put_nowait(slot)
        except queue.Full as exc:
            with self._state_lock:
                self._slot_states[slot] = "LEASED"
            raise RuntimeError(
                f"fast-loader free queue overflow while releasing host slot {slot}"
            ) from exc

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._proc.poll() is None:
            self._terminate_worker()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3)
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self._reader_thread.join(timeout=3)
        self._stderr_thread.join(timeout=3)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass


FAST_LOADER_DEPTH = int(os.environ.get("FAST_LOADER_DEPTH", "8"))
fast_loader = FastBatchPipeline(
    DEVICE_BATCH_SIZE,
    TRAIN_SEQ_LEN,
    depth=FAST_LOADER_DEPTH,
)

FAST_LOADER_VERIFY_HASH_BATCHES = int(
    os.environ.get("FAST_LOADER_VERIFY_HASH_BATCHES", "0")
)
if FAST_LOADER_VERIFY_HASH_BATCHES < 0:
    raise ValueError("FAST_LOADER_VERIFY_HASH_BATCHES cannot be negative")

if FAST_LOADER_VERIFY_HASH_BATCHES:
    # Verification-only mode. Example:
    # FAST_LOADER_VERIFY_HASH_BATCHES=64 python train_fast_loader.py
    # This intentionally instantiates the official GPU loader and exits before
    # model training; it is never part of a benchmark run.
    import hashlib

    from prepare import make_dataloader

    def _hash_verify_tensor(digest, tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(memoryview(array).cast("B"))

    def _mismatch_detail(name, candidate, reference):
        if candidate.shape != reference.shape:
            return (
                f"{name} shape mismatch: fast={tuple(candidate.shape)}, "
                f"official={tuple(reference.shape)}"
            )
        mismatch = (candidate != reference).nonzero(as_tuple=False)
        if mismatch.numel() == 0:
            return f"{name} differs in metadata or shape"
        index = tuple(int(i) for i in mismatch[0].tolist())
        return (
            f"{name}{index}: fast={int(candidate[index])}, "
            f"official={int(reference[index])}"
        )

    official_loader = make_dataloader(
        tokenizer,
        DEVICE_BATCH_SIZE,
        TRAIN_SEQ_LEN,
        "train",
    )
    fast_digest = hashlib.sha256()
    official_digest = hashlib.sha256()
    fast_loader.start_prefetch()
    try:
        for verify_idx in range(FAST_LOADER_VERIFY_HASH_BATCHES):
            verify_slot = None
            try:
                verify_slot, verify_sequence, verify_epoch = fast_loader.acquire()
                if verify_sequence != verify_idx:
                    raise AssertionError(
                        f"fast-loader verification sequence {verify_sequence}, "
                        f"expected {verify_idx}"
                    )

                official_x, official_y, official_epoch = next(official_loader)
                torch.cuda.synchronize()
                official_x = official_x.detach().cpu().contiguous()
                official_y = official_y.detach().cpu().contiguous()
                fast_x = fast_loader.host_slots[verify_slot, 0]
                fast_y = fast_loader.host_slots[verify_slot, 1]

                if verify_epoch != official_epoch:
                    raise AssertionError(
                        f"batch {verify_idx} epoch mismatch: "
                        f"fast={verify_epoch}, official={official_epoch}"
                    )
                if fast_x.shape != official_x.shape or not torch.equal(fast_x, official_x):
                    raise AssertionError(
                        f"batch {verify_idx} "
                        + _mismatch_detail("x", fast_x, official_x)
                    )
                if fast_y.shape != official_y.shape or not torch.equal(fast_y, official_y):
                    raise AssertionError(
                        f"batch {verify_idx} "
                        + _mismatch_detail("y", fast_y, official_y)
                    )

                _hash_verify_tensor(fast_digest, fast_x)
                _hash_verify_tensor(fast_digest, fast_y)
                fast_digest.update(struct.pack("<q", int(verify_epoch)))
                _hash_verify_tensor(official_digest, official_x)
                _hash_verify_tensor(official_digest, official_y)
                official_digest.update(struct.pack("<q", int(official_epoch)))
            finally:
                if verify_slot is not None:
                    fast_loader.release(verify_slot)
    finally:
        fast_loader.close()

    fast_hash = fast_digest.hexdigest()
    official_hash = official_digest.hexdigest()
    if fast_hash != official_hash:
        raise AssertionError(
            f"verification hashes differ: fast={fast_hash}, official={official_hash}"
        )
    print(
        f"FAST_LOADER_BIT_EXACT batches={FAST_LOADER_VERIFY_HASH_BATCHES} "
        f"sha256={fast_hash}"
    )
    sys.exit(0)

# Two independent CUDA slots avoid the official loader's mutable-buffer alias.
# Only the first batch is staged before the measured loop, matching the original
# one-batch initialization rather than moving the full dataset outside budget.
gpu_batches = torch.empty(
    (2, 2, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN),
    dtype=torch.long,
    device=device,
)
copy_stream = torch.cuda.Stream()
copy_events = [torch.cuda.Event(), torch.cuda.Event()]
first_host_slot, first_sequence, epoch = fast_loader.acquire()
if first_sequence != 0:
    raise RuntimeError(f"first fast-loader frame was {first_sequence}, expected 0")
gpu_batches[0].copy_(fast_loader.host_slots[first_host_slot], non_blocking=True)
torch.cuda.synchronize()
fast_loader.release(first_host_slot)
current_gpu_slot = 0
x = gpu_batches[current_gpu_slot, 0]
y = gpu_batches[current_gpu_slot, 1]
print(
    f"Using exact ordered CPU-process loader "
    f"(depth={FAST_LOADER_DEPTH}, "
    f"host={fast_loader.host_slots.numel() * fast_loader.host_slots.element_size() / 2**20:.1f} MiB)"
)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)


def get_lr_multiplier(progress, warmdown_ratio=WARMDOWN_RATIO):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


MUON_PEAK_MOMENTUM = float(os.environ.get("MUON_PEAK_MOMENTUM", 0.95))
MUON_WARMDOWN_MOMENTUM = float(
    os.environ.get("MUON_WARMDOWN_MOMENTUM", 0.79)
)
# Reverse Demon for NorMuon beta2: INCREASE beta2 during warmdown for more stable variance normalization
MUON_BETA2_PEAK = 0.95  # standard beta2 during full-LR phase
MUON_BETA2_WARMDOWN = float(os.environ.get("MUON_BETA2_WARMDOWN", 0.97))  # target beta2 at end of warmdown
MUON_LR_BOOST = 1.0  # no LR boost
# VE RMSProp reverse-Demon: increase VE beta2 during last 30% of Muon warmdown
# Analogous to Muon's 0.95->0.97, but for ngram VE tables (0.999->0.9995)
NGRAM_VE_BETA2_WARMDOWN = float(
    os.environ.get("NGRAM_VE_BETA2_WARMDOWN", 0.9999)
)  # delayed VE beta2 ramp target (0.999->target over the last 30%)
def get_muon_momentum(step, progress=None):
    # Warmup: 0.85 -> 0.95 over 300 steps
    frac = min(step / 300, 1)
    base = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    # Quadratic Demon: back-loaded shape keeps peak momentum longer
    if progress is not None:
        warmdown_start = 1.0 - WARMDOWN_RATIO
        if progress > warmdown_start:
            wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
            base = MUON_PEAK_MOMENTUM + (wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    return base


def get_muon_beta2(progress):
    """Reverse beta2: increase beta2 during warmdown for more stable variance norm."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return MUON_BETA2_PEAK
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return MUON_BETA2_PEAK + wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)


def get_muon_lr_boost(progress):
    """Boost Muon LR during warmdown to compensate for higher beta2 reducing step size."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return 1.0
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return 1.0 + wd_frac * (MUON_LR_BOOST - 1.0)


def get_adam_beta1(progress, warmdown_ratio=ADAM_WARMDOWN_RATIO):
    """Forward Demon: decrease beta1 during warmdown for more responsive gradient following."""
    initial_beta1 = ADAM_BETAS[0]
    final_beta1 = DEMON_FINAL_BETA1
    warmdown_start = 1.0 - warmdown_ratio
    if progress < warmdown_start:
        return initial_beta1
    else:
        warmdown_progress = (progress - warmdown_start) / warmdown_ratio
        return initial_beta1 + (final_beta1 - initial_beta1) * warmdown_progress


# WD pulse: RECTANGULAR shape -- with 95% warmdown (starts at 5%), pulses shifted earlier
# Main pulse at 3% center, 2% total duration (1% half-width): fires at 2-4%, before warmdown onset at 5%
# Early pulse at 1.5% center, 1% total duration: fires at 1-2% progress
# Both pulses fire in the full-LR phase (0-5%), maintaining the pre-warmdown regularization timing
WD_PULSE_CENTER = 0.03   # shift main pulse to 3% (fires before warmdown at 5%)
WD_PULSE_HALF_WIDTH = 0.01  # 1% half-width: 2% total duration (tighter for earlier firing)
WD_PULSE_MAGNITUDE = 5.0  # try 5x main pulse (vs 8x) -- 5x optimal WITH Muon Demon, 8x WITHOUT; current setup HAS Demon
WD_EARLY_PULSE_CENTER = 0.015  # shift early pulse to 1.5%
WD_EARLY_PULSE_HALF_WIDTH = 0.005  # 0.5% half-width: 1% total duration
WD_EARLY_PULSE_MAGNITUDE = 3.0  # 3x early pulse (gentler, to initialize regularization)
# Mid-warmdown triangular pulse: fires at 80% total progress (= ~79% through warmdown)
# This is WITHIN the VE beta2 ramp zone (which starts at 71.5% total = 70% through warmdown)
# Hypothesis: VE beta2 stabilization provides a safety net for a mid-warmdown WD perturbation
WD_MID_PULSE_CENTER = float(os.environ.get("WD_MID_PULSE_CENTER", 0.80))  # default: 80% total progress
WD_MID_PULSE_HALF_WIDTH = 0.025  # 2.5% half-width: 5% total triangular duration
WD_MID_PULSE_MAGNITUDE = 4.0  # 4x magnitude (triangular shape -- less harsh than rectangular)

def get_weight_decay(progress):
    base_wd = WEIGHT_DECAY * (1 - progress)
    # Early small pulse: 3x spike at 2% progress (step ~65), 2% total duration
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        return base_wd * WD_EARLY_PULSE_MAGNITUDE  # RECTANGULAR early pulse
    # Main pulse: 8x rectangular spike at 5% progress (step ~163), 3% total duration
    dist = abs(progress - WD_PULSE_CENTER)
    if dist < WD_PULSE_HALF_WIDTH:
        return base_wd * WD_PULSE_MAGNITUDE  # RECTANGULAR main pulse
    # Mid-warmdown triangular pulse: fires within VE beta2 stabilization zone
    mid_dist = abs(progress - WD_MID_PULSE_CENTER)
    if mid_dist < WD_MID_PULSE_HALF_WIDTH:
        # Triangular: linear ramp up then down (proven optimal shape)
        local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
        bump = 2 * local if local < 0.5 else 2 * (1 - local)
        return base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
    return base_wd


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Table expansion and optimizer preallocation leave several GiB of fully
# unused allocator cache behind.  Release it before the first lazy-compiled
# forward so Inductor can reserve one contiguous activation buffer; this does
# not alter model state, data order, or the measured 300-second loop.
torch.cuda.empty_cache()

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
inv_time_budget = 1.0 / TIME_BUDGET
inv_muon_warmdown = 1.0 / WARMDOWN_RATIO
inv_adam_warmdown = 1.0 / ADAM_WARMDOWN_RATIO
muon_warmdown_start = 1.0 - WARMDOWN_RATIO
adam_warmdown_start = 1.0 - ADAM_WARMDOWN_RATIO

if grad_accum_steps != 1:
    raise RuntimeError(
        "train_fast_loader.py currently requires GRAD_ACCUM=1 so its two CUDA "
        "slots preserve the official one-batch-per-step consumption order"
    )

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    # Frame 0 is the only CPU batch prepared before timing. Releasing this
    # barrier after t0 lets the worker overlap all subsequent work with step 0.
    if step == 0:
        fast_loader.start_prefetch()

    # Fetching the bounded ready-queue item and staging its H2D remain inside
    # the measured step. CPU packing itself runs concurrently in the worker.
    staged_host_slot, next_sequence, next_epoch = fast_loader.acquire()
    expected_sequence = step + 1
    if next_sequence != expected_sequence:
        raise RuntimeError(
            f"fast-loader batch sequence {next_sequence}, expected {expected_sequence}"
        )
    next_gpu_slot = current_gpu_slot ^ 1
    with torch.cuda.stream(copy_stream):
        gpu_batches[next_gpu_slot].copy_(
            fast_loader.host_slots[staged_host_slot],
            non_blocking=True,
        )
        copy_events[next_gpu_slot].record(copy_stream)

    for _micro_step in range(grad_accum_steps):
        with autocast_ctx:
            model_output = model(x, y)
        if _REUSE_NGRAM_ROWS:
            loss, model_ngram_rows = model_output
        else:
            loss = model_output
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()

    # ``x`` is still the batch whose dense embedding gradients were just
    # produced.  Record its exact hash inputs before advancing to the staged
    # next batch; lazy RMSProp consumes them on its dedicated CUDA stream.
    if optimizer.needs_ngram_batch:
        if _REUSE_NGRAM_ROWS:
            optimizer.set_ngram_active_rows(model_ngram_rows)
        else:
            optimizer.set_ngram_batch(x)

    # Match the original loop's observable state: after backward, x/y/epoch
    # refer to the next official batch. The event prevents the default stream
    # from consuming that slot before its asynchronous copy is complete.
    torch.cuda.current_stream().wait_event(copy_events[next_gpu_slot])
    x = gpu_batches[next_gpu_slot, 0]
    y = gpu_batches[next_gpu_slot, 1]
    epoch = next_epoch

    # Progress and schedules (decoupled warmdown: Muon=0.9, Adam=0.7, Ngram VE=0.0)
    progress = min(total_training_time * inv_time_budget, 1.0)
    if progress < muon_warmdown_start:
        lrm_muon = 1.0
        muon_wd_frac = 0.0
    else:
        muon_wd_frac = (progress - muon_warmdown_start) * inv_muon_warmdown
        lrm_muon = ((1.0 - progress) * inv_muon_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC

    if progress < adam_warmdown_start:
        lrm_adam = 1.0
        adam_beta1 = ADAM_BETAS[0]
    else:
        adam_wd_frac = (progress - adam_warmdown_start) * inv_adam_warmdown
        lrm_adam = ((1.0 - progress) * inv_adam_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
        adam_beta1 = ADAM_BETAS[0] + (DEMON_FINAL_BETA1 - ADAM_BETAS[0]) * adam_wd_frac

    frac = min(step / 300, 1)
    muon_momentum = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    if progress > muon_warmdown_start:
        muon_momentum = MUON_PEAK_MOMENTUM + (muon_wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    muon_beta2 = MUON_BETA2_PEAK + muon_wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)
    muon_lr_boost = 1.0 + muon_wd_frac * (MUON_LR_BOOST - 1.0)
    # VE RMSProp reverse-Demon: DELAYED ramp (only last 30% of Muon warmdown)
    late_frac = max(0.0, (muon_wd_frac - 0.7) / 0.3)
    ve_beta2 = NGRAM_VE_BETAS[1] + late_frac * (NGRAM_VE_BETA2_WARMDOWN - NGRAM_VE_BETAS[1])

    base_wd = WEIGHT_DECAY * (1 - progress)
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        muon_weight_decay = base_wd * WD_EARLY_PULSE_MAGNITUDE
    else:
        dist = abs(progress - WD_PULSE_CENTER)
        if dist < WD_PULSE_HALF_WIDTH:
            muon_weight_decay = base_wd * WD_PULSE_MAGNITUDE
        else:
            mid_dist = abs(progress - WD_MID_PULSE_CENTER)
            if mid_dist < WD_MID_PULSE_HALF_WIDTH:
                local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
                bump = 2 * local if local < 0.5 else 2 * (1 - local)
                muon_weight_decay = base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
            else:
                muon_weight_decay = base_wd

    muon_lr = lrm_muon * muon_lr_boost
    if progress < muon_warmdown_start:
        for group in muon_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
    else:
        for group, initial_lr in muon_group_lrs:
            group["lr"] = initial_lr * muon_lr
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
        for group, initial_lr in x0_group_lrs:
            group["lr"] = initial_lr * lrm_muon
    if progress >= adam_warmdown_start:
        for group, initial_lr in adam_group_lrs:
            group["lr"] = initial_lr * lrm_adam
        for group, beta2 in adam_demon_groups:
            group["betas"] = (adam_beta1, beta2)
    # Update ngram VE RMSProp beta2 during warmdown (delayed reverse-Demon for sparse tables)
    if progress >= muon_warmdown_start and late_frac > 0.0:
        for group in ngram_groups:
            group["beta2"] = ve_beta2
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    # Device-wide synchronization above proves the async H2D no longer reads
    # this pinned slot, so the reader may safely refill it.
    fast_loader.release(staged_host_slot)
    current_gpu_slot = next_gpu_slot

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm_muon: {lrm_muon:.2f} lrm_adam: {lrm_adam:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up -- but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

fast_loader.close()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval is mandatory and always uses the fixed harness from prepare.py.
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / B200_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
