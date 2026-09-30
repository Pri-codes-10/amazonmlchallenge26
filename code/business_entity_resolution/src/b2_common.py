"""
b2_common.py — shared loading, candidate format, checkpointing and scoring for Block 2.

Design notes:
- Every loader supports pyarrow predicate pushdown on country_norm / entity_role, so the whole
  10.3M-row pool is never materialised when one country's ~4-6M rows are needed.
- Sparse text features use hashed n-grams in chunks: TfidfVectorizer.fit_transform on 6M+ rows
  transiently spikes to 3-5x its final matrix size, hashing + chunking avoids that.
- Both top-k search functions merge a running top-k over POOL blocks, because a full
  (query_batch x pool_size) score matrix does not fit in memory (2000 x 6.2M float32 = ~50GB).
"""

import os
import json
import time
import math
import glob
import gc
import hashlib
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import er_eval as ev

# ---------------------------------------------------------------------------
# Paths — computed at import time from $BER_WORKSPACE (the notebooks set it in their first cell)
# ---------------------------------------------------------------------------
#   <workspace>/block1/<er_eval.B1_VERSION>/{records,labels}
#   <workspace>/block2/{candidates,markers,artifacts,reports,prod}
WORKSPACE_ROOT = ev.workspace_root()
BLOCK1_ROOT = f"{WORKSPACE_ROOT}/block1/{ev.B1_VERSION}"
BLOCK2_ROOT = f"{WORKSPACE_ROOT}/block2"
CANDIDATES_DIR = f"{BLOCK2_ROOT}/candidates"
MARKERS_DIR = f"{BLOCK2_ROOT}/markers"
ARTIFACTS_DIR = f"{BLOCK2_ROOT}/artifacts"     # fine-tuned bi-encoder checkpoints
REPORTS_DIR = f"{BLOCK2_ROOT}/reports"
# Scratch copies of the record tables: cheap to rebuild, so they live on local disk.
LOCAL_CACHE = os.environ.get("BER_B2_CACHE") or os.path.join(tempfile.gettempdir(), "ber_b2_cache")


# ---------------------------------------------------------------------------
# Timing / logging
# ---------------------------------------------------------------------------

def _gpu_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None


@contextmanager
def step(name, reset_gpu_peak=True):
    """Times a block and reports elapsed seconds + peak GPU memory.

        with b2.step("A1 char n-gram forward search"):
            ...
    """
    try:
        import torch
        if reset_gpu_peak and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] >>> START  {name}")
    try:
        yield
    finally:
        dt = time.time() - t0
        gm = _gpu_mem_gb()
        gm_str = f", peak_gpu={gm:.2f}GB" if gm is not None else ""
        print(f"[{time.strftime('%H:%M:%S')}] <<< DONE   {name}  ({dt:.1f}s{gm_str})")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Done markers — resumability after an interrupted run
# ---------------------------------------------------------------------------

def marker_path(name):
    return f"{MARKERS_DIR}/_DONE_{name}"


def is_done(name):
    return os.path.exists(marker_path(name))


def mark_done(name, meta=None):
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(marker_path(name), "w") as f:
        json.dump({"ts": time.time(), **(meta or {})}, f)
    print(f"marked done: {name}")


# ---------------------------------------------------------------------------
# Loading Block 1 records
# ---------------------------------------------------------------------------

RECORD_COLUMNS_DEFAULT = [
    "entity_id", "source", "split", "country_norm",
    "name_norm", "addr_norm",
    "name_latin", "addr_latin", "name_core", "name_skeleton", "name_aka",
    "legal_suffix", "is_domain_name", "name_script", "addr_script",
    "postal_code", "street_number", "has_landmark",
    "name_missing", "addr_missing",
    "name_n_tokens", "addr_n_tokens",
    "cluster_id", "entity_role",
]


def _shard_dir(split, source):
    return f"{BLOCK1_ROOT}/records/{split}_{source.lower()}"


def load_records(split, source, columns=None, country=None, roles=None, local_cache=True):
    """Load one split/source's shards (e.g. split='train', source='S2').

    country: optional country_norm filter ('INDIA' | 'US' | 'FRANCE').
    roles:   optional list of entity_role values to keep.
    Applies pyarrow predicate pushdown and caches the filtered frame on local disk,
    so re-running a cell does not re-read and re-filter every shard.
    """
    cols = columns or RECORD_COLUMNS_DEFAULT
    key_str = f"{split}|{source}|{country}|{sorted(roles) if roles else 'ALL'}|{sorted(cols)}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()[:16]
    cache_path = f"{LOCAL_CACHE}/{split}_{source}_{cache_key}.parquet"
    if local_cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    filters = []
    if country is not None:
        filters.append(("country_norm", "=", country))
    if roles is not None:
        filters.append(("entity_role", "in", roles))

    shard_dir = _shard_dir(split, source)
    files = sorted(glob.glob(f"{shard_dir}/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No shards found under {shard_dir}")

    frames = []
    for f in files:
        tbl = pq.read_table(f, columns=cols, filters=filters or None)
        if tbl.num_rows:
            frames.append(tbl.to_pandas())
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    if local_cache:
        os.makedirs(LOCAL_CACHE, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def load_pool(country, split="train", extra_cols=None):
    """Full S2+S3 pool for one country/split — ALL roles included
    (POOL/matched-train-role/TEST), per the Block 1 contract: retrieve against
    the full pool regardless of which role the query belongs to."""
    cols = list(dict.fromkeys(RECORD_COLUMNS_DEFAULT + (extra_cols or [])))
    s2 = load_records(split, "S2", columns=cols, country=country)
    s3 = load_records(split, "S3", columns=cols, country=country)
    pool = pd.concat([s2, s3], ignore_index=True)
    return pool


def load_labels():
    pp = pd.read_parquet(f"{BLOCK1_ROOT}/labels/positive_pairs.parquet")
    roles = pd.read_parquet(f"{BLOCK1_ROOT}/labels/entity_roles.parquet")
    return pp, roles


# ---------------------------------------------------------------------------
# Block-wise top-K search over the record pool (dense embeddings, torch)
# ---------------------------------------------------------------------------

def dense_blockwise_topk(query_emb, pool_emb, k, query_batch=4000, pool_block=200_000,
                          device=None, verbose=True, log_every_seconds=15):
    """
    query_emb, pool_emb: numpy arrays (n, dim), assumed L2-normalized so a dot
    product is cosine similarity. Returns (topk_idx, topk_score) numpy arrays
    of shape (n_queries, k).

    verbose=True prints progress: a silent multi-minute block loop is
    indistinguishable from a hang.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool_t = torch.from_numpy(pool_emb).to(device=device, dtype=torch.float16)
    n_q = query_emb.shape[0]
    n_p = pool_emb.shape[0]
    out_idx = np.full((n_q, k), -1, dtype=np.int64)
    out_score = np.full((n_q, k), -np.inf, dtype=np.float32)

    n_query_batches = math.ceil(n_q / query_batch)
    n_pool_blocks = math.ceil(n_p / pool_block)
    if verbose:
        print(f"    dense_blockwise_topk: {n_q:,} queries x {n_p:,} pool rows, "
              f"device={device}, {n_query_batches} query batch(es) x "
              f"{n_pool_blocks} pool block(s) = {n_query_batches * n_pool_blocks} "
              f"block matmul(s) total")

    t_start = time.time()
    t_last_log = t_start
    blocks_done = 0
    total_blocks = n_query_batches * n_pool_blocks

    for qi, qstart in enumerate(range(0, n_q, query_batch)):
        q_chunk = torch.from_numpy(query_emb[qstart:qstart + query_batch]).to(
            device=device, dtype=torch.float16)
        n_qc = q_chunk.shape[0]
        best_score = torch.full((n_qc, k), -float("inf"), device=device, dtype=torch.float32)
        best_idx = torch.full((n_qc, k), -1, device=device, dtype=torch.int64)

        for pstart in range(0, n_p, pool_block):
            pool_chunk = pool_t[pstart:pstart + pool_block]
            scores = (q_chunk @ pool_chunk.T).float()  # (n_qc, block_size)
            block_size = scores.shape[1]
            eff_k = min(k, block_size)
            top_score, top_local_idx = torch.topk(scores, eff_k, dim=1)
            top_idx = top_local_idx + pstart

            merged_score = torch.cat([best_score, top_score], dim=1)
            merged_idx = torch.cat([best_idx, top_idx], dim=1)
            best_score, order = torch.topk(merged_score, k, dim=1)
            best_idx = torch.gather(merged_idx, 1, order)

            blocks_done += 1
            now = time.time()
            if verbose and now - t_last_log >= log_every_seconds:
                elapsed = now - t_start
                rate = blocks_done / elapsed if elapsed > 0 else 0
                remaining = (total_blocks - blocks_done) / rate if rate > 0 else float("nan")
                print(f"    query batch {qi + 1}/{n_query_batches} -- "
                      f"{blocks_done}/{total_blocks} block matmuls done "
                      f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")
                t_last_log = now

        out_idx[qstart:qstart + n_qc] = best_idx.cpu().numpy()
        out_score[qstart:qstart + n_qc] = best_score.cpu().numpy()

    if verbose:
        print(f"    dense_blockwise_topk done in {time.time() - t_start:.1f}s")

    return out_idx, out_score


# ---------------------------------------------------------------------------
# Label table access
# ---------------------------------------------------------------------------

def _get_s1_id_series(roles_df):
    """The S1-id column of entity_roles.parquet: tries the known names, then the index,
    and fails loudly rather than guessing."""
    for cand in ("source1_entity_id", "s1_id", "entity_id"):
        if cand in roles_df.columns:
            return roles_df[cand].astype(str)
    idx = roles_df.index
    if idx.name in ("source1_entity_id", "s1_id", "entity_id") or idx.dtype == object:
        return pd.Series(idx.astype(str), index=roles_df.index)
    raise KeyError("no S1-id column in entity_roles.parquet "
                   f"(columns: {list(roles_df.columns)}, index name: {idx.name})")
