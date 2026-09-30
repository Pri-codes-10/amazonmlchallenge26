"""
b2_prod.py — Block 2 candidate generation: the per-country fine-tuned multilingual-e5-small
bi-encoder (dense forward search) run over every S1 query the later blocks need.

Reuses b2_common's loaders and dense_blockwise_topk unchanged.

Notebooks that use it:
    02_block2_candidategen_india.ipynb   INDIA train roles + INDIA test          (GPU)
    02_block2_candidategen_us.ipynb      US + every other country, train + test  (GPU)
    02_block2_candidategen_merge.ipynb   verify, coverage, PC report, K_FINAL,   (CPU)
                                         merged manifest = what Block 3/4 read

Each notebook writes only inside its own country folder prod/<COUNTRY>/..., and a shard counts as
done only once its done-*.json exists (written last), so an interrupted run resumes.
"""

import os
import re
import gc
import tempfile
import json
import glob
import time
import math
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

import b2_common as b2

# ---------------------------------------------------------------------------
# Constants — the production decision, in one place
# ---------------------------------------------------------------------------

PROD_VERSION = "v1"
PROD_ROOT = f"{b2.BLOCK2_ROOT}/prod"
LOCAL_EMB_CACHE = os.environ.get("BER_B2_EMB_CACHE") or os.path.join(tempfile.gettempdir(), "ber_b2_emb_cache")

BASE_MODEL = "intfloat/multilingual-e5-small"
CHANNEL = "dense_finetuned"
VARIANT = f"e5_finetuned_prod_{PROD_VERSION}"

K_GEN = 50               # stored per query; downstream truncates to K_FINAL (rank <= K)
DEFAULT_K_FINAL = 30     # provisional; the merge notebook records the real choice
EVAL_K_LIST = (10, 20, 30, 50)

TRAIN_ROLE_ORDER = ["CE-FIT", "CE-DEV", "TREE-FIT", "TREE-DEV", "CALIBRATION", "HOLDOUT"]
TEST_ROLE = "TEST"
UNBIASED_ROLES = ["TREE-FIT", "TREE-DEV", "CALIBRATION", "HOLDOUT"]  # never seen by the bi-encoder

CHECKPOINT_FOR = {"INDIA": "INDIA", "US": "US"}   # any other country uses DEFAULT_CHECKPOINT
DEFAULT_CHECKPOINT = "US"                          # Latin-script transfer (FRANCE and any other country)

SHARD_QUERIES = 200_000
PROD_COLS = ["entity_id", "source", "country_norm", "entity_role", "name_norm", "addr_norm"]
# production only needs these 6 of Block 1's 24 record columns -> several GB less RAM
ENCODE_BATCH = 256
QUERY_BATCH = 4000
POOL_BLOCK = 200_000


def use_smoke_root(on=True):
    """Redirect every write to prod_smoke/ (small end-to-end test run)."""
    global PROD_ROOT
    PROD_ROOT = f"{b2.BLOCK2_ROOT}/prod_smoke" if on else f"{b2.BLOCK2_ROOT}/prod"
    print(f"PROD_ROOT -> {PROD_ROOT}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def country_root(country):
    return f"{PROD_ROOT}/{country}"


def shard_dir(country, split, role):
    return f"{country_root(country)}/candidates/{split}/{role}"


def _shard_files(country, split, role, i):
    d = shard_dir(country, split, role)
    return {
        "cand": f"{d}/cand-{i:05d}.parquet",
        "queries": f"{d}/queries-{i:05d}.parquet",
        "pos": f"{d}/pos-{i:05d}.parquet",
        "done": f"{d}/done-{i:05d}.json",
    }


def _robust_write(write_fn, path, tries=8, base_delay=2.0):
    """Write with the parent directory re-asserted and a backoff retry, so a transient
    filesystem error costs one shard rather than the whole role."""
    d = os.path.dirname(path)
    last_err = None
    for attempt in range(tries):
        try:
            os.makedirs(d, exist_ok=True)
            write_fn(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"write reported success but {path} is not visible yet")
            return
        except (FileNotFoundError, OSError) as e:
            last_err = e
            delay = base_delay * (attempt + 1)
            print(f"  [retry {attempt + 1}/{tries}] write to {path} failed ({e!r}); "
                  f"re-asserting the directory and retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"failed to write {path} after {tries} attempts: {last_err}")


def merged_root():
    return f"{PROD_ROOT}/merged"


def checkpoint_key_for(country):
    return CHECKPOINT_FOR.get(country, DEFAULT_CHECKPOINT)


def checkpoint_path(key):
    return f"{b2.ARTIFACTS_DIR}/e5_finetuned_{key}"


# ---------------------------------------------------------------------------
# Markers (country-scoped — no shared markers folder anywhere in production)
# ---------------------------------------------------------------------------

def marker_path(country, name):
    return f"{country_root(country)}/markers/_DONE_{name}"


def is_done(country, name):
    return os.path.exists(marker_path(country, name))


def mark_done(country, name, meta=None):
    path = marker_path(country, name)
    payload = {"ts": time.time(), **(meta or {})}

    def _write(p, payload=payload):
        with open(p, "w") as f:
            json.dump(payload, f, default=float)
    _robust_write(_write, path)
    print(f"marked done: {country}/{name}")


def require_markers(pairs):
    """pairs: list of (country, marker_name). Raises unless every producer stage is committed."""
    missing = [(c, n) for c, n in pairs if not is_done(c, n)]
    if missing:
        raise FileNotFoundError(f"Block 2 stages not finished: {missing} - "
                                "run the india and us candidate-generation notebooks first")
    print("all markers present.")


# ---------------------------------------------------------------------------
# Preflight — run before anything heavy
# ---------------------------------------------------------------------------

def check_duplicate_names(root):
    """Flags ' (1)'-style auto-renamed duplicates left by an interrupted copy."""
    if not os.path.isdir(root):
        return []
    names = os.listdir(root)
    dupes = [n for n, c in Counter(names).items() if c > 1]
    suffixed = [n for n in names if re.search(r" \(\d+\)(\.\w+)?$", n)]
    return sorted(set(dupes + suffixed))


def verify_checkpoint(key):
    """Fails loudly unless the fine-tuned checkpoint is present and complete."""
    p = checkpoint_path(key)
    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {p}\n"
            "Put the fine-tuned bi-encoder checkpoints in that folder (see README).")
    files = [f for f in glob.glob(f"{p}/**/*", recursive=True) if os.path.isfile(f)]
    has_weights = any(f.endswith(("model.safetensors", "pytorch_model.bin")) for f in files)
    has_modules = os.path.exists(f"{p}/modules.json")
    if not (has_weights and has_modules):
        raise FileNotFoundError(
            f"Checkpoint at {p} looks incomplete (weights={has_weights}, "
            f"modules.json={has_modules}); candidates would silently differ from the chosen model.")
    sig = sorted((os.path.relpath(f, p), os.path.getsize(f)) for f in files)
    fingerprint = hashlib.md5(json.dumps(sig).encode()).hexdigest()[:12]
    size_mb = sum(s for _, s in sig) / 1e6
    return {"key": key, "path": p, "fingerprint": fingerprint, "size_mb": round(size_mb, 1)}


def preflight(checkpoint_keys):
    """GPU/RAM info, duplicate-folder scan, checkpoint verification."""
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            print(f"GPU      : {pr.name} ({pr.total_memory / 1e9:.1f} GB)")
        else:
            print("GPU      : NONE — embedding the record pool needs a GPU")
    except ImportError:
        print("torch not importable")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"Host RAM : {vm.total / 1e9:.1f} GB total, {vm.available / 1e9:.1f} GB free")
    except ImportError:
        pass

    problems = []
    for root in (b2.BLOCK2_ROOT, b2.ARTIFACTS_DIR, PROD_ROOT):
        d = check_duplicate_names(root)
        if d:
            problems.append(f"{root}: duplicate / auto-renamed entries {d}")
    for msg in problems:
        print("WARNING:", msg)
    if problems:
        print("-> remove the duplicated entries above before continuing.")

    infos = {}
    for key in checkpoint_keys:
        infos[key] = verify_checkpoint(key)
        print(f"checkpoint {key}: {infos[key]['path']}  "
              f"({infos[key]['size_mb']} MB, fingerprint {infos[key]['fingerprint']})")
    return infos


def ensure_prod_root():
    os.makedirs(PROD_ROOT, exist_ok=True)
    print(f"prod root: {PROD_ROOT}")


def ensure_country_dirs(country):
    for sub in ("markers", "candidates"):
        os.makedirs(f"{country_root(country)}/{sub}", exist_ok=True)


# ---------------------------------------------------------------------------
# Data discovery + loading
# ---------------------------------------------------------------------------

def countries_in(split, source="S1"):
    """Distinct country_norm values in a split/source (reads one column only)."""
    files = sorted(glob.glob(f"{b2._shard_dir(split, source)}/part-*.parquet"))
    vals = set()
    for f in files:
        col = pq.read_table(f, columns=["country_norm"]).column(0)
        vals |= set(pc.unique(col).to_pylist())
    has_null = None in vals
    return sorted(v for v in vals if v is not None), has_null


def load_pool_sorted(country, split):
    """b2.load_pool, de-duplicated and sorted by entity_id so row order (and so
    the embedding cache key) is deterministic across runs."""
    pool = pd.concat([b2.load_records(split, src, columns=PROD_COLS, country=country)
                      for src in ("S2", "S3")], ignore_index=True)
    pool["entity_id"] = pool["entity_id"].astype(str)
    n0 = len(pool)
    pool = pool.drop_duplicates("entity_id").sort_values("entity_id").reset_index(drop=True)
    if len(pool) != n0:
        print(f"WARNING: dropped {n0 - len(pool):,} duplicate entity_ids from the pool")
    return pool


def prepare_positives(positive_pairs):
    """positive_pairs.parquet -> (s1_id, cand_id, cross_script) with string ids."""
    pp = positive_pairs.rename(columns={"source1_entity_id": "s1_id",
                                         "matched_entity_id": "cand_id"})
    keep = ["s1_id", "cand_id"] + (["cross_script"] if "cross_script" in pp.columns else [])
    pp = pp[keep].copy()
    pp["s1_id"] = pp["s1_id"].astype(str)
    pp["cand_id"] = pp["cand_id"].astype(str)
    if "cross_script" not in pp.columns:
        pp["cross_script"] = False
    return pp.drop_duplicates(["s1_id", "cand_id"])


# ---------------------------------------------------------------------------
# Model + embeddings
# ---------------------------------------------------------------------------

def dense_texts(df, prefix):
    """The checkpoint was fine-tuned on exactly this text format — do not change it."""
    return (prefix + df["name_norm"].fillna("").astype(str) + " | " +
            df["addr_norm"].fillna("").astype(str)).tolist()


def load_model(ckpt_info):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(ckpt_info["path"], device=device)
    print(f"loaded {ckpt_info['key']} checkpoint on {device}")
    return model


def encode(model, texts, batch_size=None, chunk_size=500_000, label="texts"):
    """fp32 compute, fp16 storage, L2-normalized."""
    batch_size = batch_size or ENCODE_BATCH
    out, t0, n = [], time.time(), len(texts)
    for s in range(0, n, chunk_size):
        emb = model.encode(texts[s:s + chunk_size], batch_size=batch_size,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        out.append(emb.astype(np.float16))
        done = min(s + chunk_size, n)
        el = time.time() - t0
        eta = el / done * (n - done) if done else float("nan")
        print(f"    encoded {label}: {done:,}/{n:,}  ({el:.0f}s, ~{eta:.0f}s left)")
    if not out:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float16)
    return np.concatenate(out, axis=0) if len(out) > 1 else out[0]


def pool_embeddings(model, ckpt_info, country, split, pool, cache_dir=None):
    """Embeds the pool once per (country, split, checkpoint), cached on local disk
    (several GB per country, cheap to rebuild)."""
    cache_dir = cache_dir or LOCAL_EMB_CACHE
    ids = pool["entity_id"].to_numpy()
    key_src = f"{country}|{split}|{ckpt_info['fingerprint']}|{len(ids)}|{ids[:500].tolist()}|{ids[-500:].tolist()}"
    key = hashlib.md5(key_src.encode()).hexdigest()[:16]
    path = f"{cache_dir}/pool_{country}_{split}_{key}.npy"
    if os.path.exists(path):
        emb = np.load(path)
        print(f"pool embeddings from local cache: {path} {emb.shape}")
        return emb
    emb = encode(model, dense_texts(pool, "passage: "), label=f"{country} {split} pool")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, emb)
    return emb


# ---------------------------------------------------------------------------
# Shard writing — vectorized (no per-row Python loops at 40M+ rows)
# ---------------------------------------------------------------------------

def _const_col(value, n):
    return pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, dtype=np.int32)),
                                          pa.array([value], type=pa.string()))


def build_shard(q_ids, idx, sc, pool_ids, pool_src, pool_index, country, split, role,
                positives=None):
    """Turns one (queries x K) search result into the candidate table, plus a
    per-query sidecar and (train only) a per-positive-pair sidecar.

    label (train): 1 iff (s1_id, cand_id) is in positive_pairs, else 0. Null on test.
    pos sidecar: one row per labeled positive of these queries, with rank_found
    (0 = not retrieved in top-K_GEN) and in_pool (False = the matched record is not
    in this country's pool, i.e. unretrievable by construction).
    """
    n_q, k = idx.shape
    flat = idx.ravel()
    valid = flat >= 0
    rows_q = np.repeat(np.arange(n_q, dtype=np.int64), k)[valid]
    cidx = flat[valid].astype(np.int64)
    rank = np.tile(np.arange(1, k + 1, dtype=np.int32), n_q)[valid]
    score = sc.ravel()[valid].astype(np.float32)
    n_rows = len(cidx)
    n_cands = valid.reshape(n_q, k).sum(axis=1).astype(np.int32)

    label = None
    pos_df = None
    if positives is not None:
        n_pool = np.int64(len(pool_ids))
        q_index = pd.Index(q_ids)
        qpos = positives[positives["s1_id"].isin(set(q_ids))]
        qi = q_index.get_indexer(qpos["s1_id"].to_numpy()).astype(np.int64)
        pj = pool_index.get_indexer(qpos["cand_id"].to_numpy()).astype(np.int64)
        in_pool = pj >= 0
        pos_codes = qi[in_pool] * n_pool + pj[in_pool]
        cand_codes = rows_q * n_pool + cidx
        is_pos = np.isin(cand_codes, pos_codes)
        label = is_pos.astype(np.int8)
        lookup = pd.Series(rank[is_pos], index=cand_codes[is_pos])
        all_codes = np.where(in_pool, qi * n_pool + np.maximum(pj, 0), -1)
        rank_found = np.array(lookup.reindex(all_codes).fillna(0).astype(np.int32), copy=True)
        rank_found[~in_pool] = 0
        pos_df = pd.DataFrame({
            "s1_id": qpos["s1_id"].to_numpy(),
            "cand_id": qpos["cand_id"].to_numpy(),
            "cross_script": qpos["cross_script"].astype(bool).to_numpy(),
            "in_pool": in_pool,
            "rank_found": rank_found,
            "role": role, "country": country,
        })

    table = pa.table({
        "s1_id": pa.array(q_ids[rows_q], type=pa.string()),
        "cand_id": pa.array(pool_ids[cidx], type=pa.string()),
        "source": pa.array(pool_src[cidx], type=pa.string()).dictionary_encode(),
        "channel": _const_col(CHANNEL, n_rows),
        "variant": _const_col(VARIANT, n_rows),
        "score": pa.array(score, type=pa.float32()),
        "rank": pa.array(rank, type=pa.int32()),
        "role": _const_col(role, n_rows),
        "country": _const_col(country, n_rows),
        "split": _const_col(split, n_rows),
        "label": (pa.array(label, type=pa.int8()) if label is not None
                  else pa.nulls(n_rows, type=pa.int8())),
    })
    queries_df = pd.DataFrame({"s1_id": q_ids, "n_cands": n_cands})
    top1 = sc[:, 0][np.isfinite(sc[:, 0])] if n_q else np.array([])
    stats = {
        "n_queries": int(n_q),
        "n_rows": int(n_rows),
        "n_queries_no_cands": int((n_cands == 0).sum()),
        "top1_score_pcts": (np.percentile(top1, [5, 25, 50, 75, 95]).round(4).tolist()
                            if len(top1) else None),
    }
    if pos_df is not None:
        stats["n_pos"] = int(len(pos_df))
        stats["n_pos_in_pool"] = int(pos_df["in_pool"].sum())
        for kk in EVAL_K_LIST:
            if kk <= k:
                stats[f"n_pos_found_at_{kk}"] = int(((pos_df["rank_found"] > 0) &
                                                     (pos_df["rank_found"] <= kk)).sum())
    return table, queries_df, pos_df, stats


def run_role(model, ckpt_info, country, split, role, pool, pool_emb, positives=None,
             k=None, shard_queries=None, query_batch=None, pool_block=None, max_queries=None):
    """Forward dense search for every S1 of (country, split, role), sharded by
    query so a disconnect only costs the shard in flight. Defaults resolve at
    call time, so config cells can set b2p.SHARD_QUERIES etc."""
    k = k or K_GEN
    shard_queries = shard_queries or SHARD_QUERIES
    query_batch = query_batch or QUERY_BATCH
    pool_block = pool_block or POOL_BLOCK
    name = f"{split}_{role}"
    if is_done(country, name):
        print(f"[skip] {country}/{name} already done")
        return
    t_role = time.time()
    q = b2.load_records(split, "S1", columns=PROD_COLS, country=country, roles=[role])
    q["entity_id"] = q["entity_id"].astype(str)
    q = q.drop_duplicates("entity_id").sort_values("entity_id").reset_index(drop=True)
    if max_queries is not None:
        q = q.iloc[:max_queries]
    n = len(q)
    print(f"\n=== {country} {split} {role}: {n:,} queries vs {len(pool):,} pool rows ===")
    if n == 0 or len(pool) == 0:
        if n and not len(pool):
            print(f"WARNING: {n:,} queries but an EMPTY pool — they get no candidates")
        _write_empty_role(country, split, role, q["entity_id"].to_numpy())
        mark_done(country, name, {"n_queries": n, "n_rows": 0, "empty_pool": len(pool) == 0})
        return

    pool_ids = pool["entity_id"].to_numpy()
    pool_src = pool["source"].astype(str).to_numpy()
    pool_index = pd.Index(pool_ids)
    d = shard_dir(country, split, role)
    os.makedirs(d, exist_ok=True)
    n_shards = math.ceil(n / shard_queries)
    tot_rows = 0
    for si in range(n_shards):
        files = _shard_files(country, split, role, si)
        if os.path.exists(files["done"]):
            print(f"  [skip] shard {si + 1}/{n_shards} already written")
            continue
        t0 = time.time()
        qs = q.iloc[si * shard_queries:(si + 1) * shard_queries]
        q_ids = qs["entity_id"].to_numpy()
        q_emb = encode(model, dense_texts(qs, "query: "), label=f"shard {si + 1}/{n_shards} queries")
        idx, sc = b2.dense_blockwise_topk(q_emb, pool_emb, k, query_batch, pool_block)
        table, qdf, pos_df, stats = build_shard(q_ids, idx, sc, pool_ids, pool_src, pool_index,
                                                country, split, role, positives)
        _robust_write(lambda p, t=table: pq.write_table(t, p, compression="zstd"), files["cand"])
        _robust_write(lambda p, d=qdf: d.to_parquet(p, index=False), files["queries"])
        if pos_df is not None:
            _robust_write(lambda p, d=pos_df: d.to_parquet(p, index=False), files["pos"])
        stats.update({"shard": si, "k": k, "checkpoint": ckpt_info["key"],
                      "checkpoint_fingerprint": ckpt_info["fingerprint"],
                      "elapsed_s": round(time.time() - t0, 1)})

        def _write_done(p, stats=stats):
            with open(p, "w") as f:
                json.dump(stats, f)
        _robust_write(_write_done, files["done"])       # commit point: written last
        tot_rows += stats["n_rows"]
        msg = f"  shard {si + 1}/{n_shards}: {stats['n_rows']:,} rows in {stats['elapsed_s']}s"
        if "n_pos" in stats and stats["n_pos"]:
            f30 = stats.get("n_pos_found_at_30", 0) / stats["n_pos"]
            msg += f"  | recall@30 on this shard: {f30 * 100:.2f}%"
        print(msg)
        del q_emb, idx, sc, table, qdf, pos_df
        gc.collect()
    mark_done(country, name, {"n_queries": n, "n_shards": n_shards,
                              "elapsed_s": round(time.time() - t_role, 1)})
    write_country_manifest(country)


def _write_empty_role(country, split, role, q_ids):
    """Queries that cannot get candidates still get a queries sidecar, so the
    merge notebook's coverage check reports them instead of missing them."""
    if len(q_ids) == 0:
        return
    files = _shard_files(country, split, role, 0)
    empty_idx = np.full((len(q_ids), 1), -1, dtype=np.int64)
    empty_sc = np.full((len(q_ids), 1), -np.inf, dtype=np.float32)
    table, qdf, _, stats = build_shard(q_ids, empty_idx, empty_sc, np.array([], dtype=object),
                                       np.array([], dtype=object), pd.Index([]),
                                       country, split, role, None)
    _robust_write(lambda p: pq.write_table(table, p, compression="zstd"), files["cand"])
    _robust_write(lambda p: qdf.to_parquet(p, index=False), files["queries"])
    stats.update({"shard": 0, "k": 0, "empty_pool": True})

    def _write_done(p, stats=stats):
        with open(p, "w") as f:
            json.dump(stats, f)
    _robust_write(_write_done, files["done"])


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def collect_shards(country):
    """Scans done-*.json files (the commit markers) for one country."""
    out = []
    for done in sorted(glob.glob(f"{country_root(country)}/candidates/*/*/done-*.json")):
        with open(done) as f:
            st = json.load(f)
        d = os.path.dirname(done)
        split, role = d.split("/")[-2], d.split("/")[-1]
        i = int(re.search(r"done-(\d+)\.json$", done).group(1))
        files = _shard_files(country, split, role, i)
        out.append({"country": country, "split": split, "role": role, "shard": i,
                    "cand_path": files["cand"], "queries_path": files["queries"],
                    "pos_path": files["pos"] if os.path.exists(files["pos"]) else None,
                    **{k: v for k, v in st.items() if k != "shard"}})
    return out


def write_country_manifest(country):
    shards = collect_shards(country)
    m = {"prod_version": PROD_VERSION, "country": country, "k_gen": K_GEN,
         "channel": CHANNEL, "variant": VARIANT, "base_model": BASE_MODEL,
         "checkpoint": checkpoint_key_for(country), "written_ts": time.time(),
         "shards": shards}
    path = f"{country_root(country)}/manifest.json"

    def _write(p, m=m):
        with open(p, "w") as f:
            json.dump(m, f, indent=1, default=float)
    _robust_write(_write, path)
    return path


def produced_countries():
    """Country folders that exist under PROD_ROOT (excluding merged/)."""
    if not os.path.isdir(PROD_ROOT):
        return []
    return sorted(n for n in os.listdir(PROD_ROOT)
                  if os.path.isdir(f"{PROD_ROOT}/{n}") and n != "merged"
                  and not re.search(r" \(\d+\)$", n))


# ---------------------------------------------------------------------------
# Verification, coverage, pair completeness (used by the merge notebook)
# ---------------------------------------------------------------------------

def verify_shards(shards):
    """Every committed shard's files exist and the parquet row count matches."""
    problems = []
    for s in shards:
        tag = f"{s['country']}/{s['split']}/{s['role']}/shard{s['shard']}"
        if not os.path.exists(s["cand_path"]):
            problems.append(f"{tag}: candidate file missing")
            continue
        n = pq.ParquetFile(s["cand_path"]).metadata.num_rows
        if n != s["n_rows"]:
            problems.append(f"{tag}: parquet has {n:,} rows, done.json says {s['n_rows']:,}")
        if not os.path.exists(s["queries_path"]):
            problems.append(f"{tag}: queries sidecar missing")
        if s["split"] != "test" and s["n_rows"] > 0 and s.get("pos_path") is None:
            problems.append(f"{tag}: train shard without positives sidecar")
    return problems


def produced_queries(shards):
    frames = []
    for s in shards:
        qdf = pd.read_parquet(s["queries_path"])
        qdf["country"], qdf["split"], qdf["role"] = s["country"], s["split"], s["role"]
        frames.append(qdf)
    if not frames:
        return pd.DataFrame(columns=["s1_id", "n_cands", "country", "split", "role"])
    return pd.concat(frames, ignore_index=True)


def expected_queries(split):
    """What SHOULD have been queried: every S1 of the split, with country + role."""
    df = b2.load_records(split, "S1", columns=["entity_id", "country_norm", "entity_role"])
    df = df.rename(columns={"entity_id": "s1_id", "country_norm": "country",
                            "entity_role": "role"})
    df["s1_id"] = df["s1_id"].astype(str)
    return df.drop_duplicates("s1_id")


def coverage_report(expected, produced, k_needed=None):
    """Missing queries, duplicated queries, role/country mismatches, short lists."""
    k_needed = k_needed or DEFAULT_K_FINAL
    e = expected.set_index("s1_id")
    dup = produced[produced.duplicated("s1_id", keep=False)]
    p = produced.drop_duplicates("s1_id").set_index("s1_id")
    missing = e.index.difference(p.index)
    extra = p.index.difference(e.index)
    both = e.index.intersection(p.index)
    role_mismatch = int((e.loc[both, "role"].astype(str) != p.loc[both, "role"].astype(str)).sum())
    ctry_mismatch = int((e.loc[both, "country"].astype(str) != p.loc[both, "country"].astype(str)).sum())
    short = p[p["n_cands"] < k_needed]
    rep = {
        "n_expected": int(len(e)), "n_produced": int(len(p)),
        "n_missing": int(len(missing)), "n_extra": int(len(extra)),
        "n_duplicated_across_shards": int(dup["s1_id"].nunique()),
        "n_role_mismatch": role_mismatch, "n_country_mismatch": ctry_mismatch,
        f"n_with_fewer_than_{k_needed}_cands": int(len(short)),
        "n_with_zero_cands": int((p["n_cands"] == 0).sum()),
        "missing_by_country_role": (e.loc[missing].groupby(["country", "role"]).size()
                                    .rename("n").reset_index().to_dict("records")
                                    if len(missing) else []),
    }
    return rep


def load_pos_sidecars(shards):
    frames = [pd.read_parquet(s["pos_path"]) for s in shards if s.get("pos_path")]
    if not frames:
        return pd.DataFrame(columns=["s1_id", "cand_id", "cross_script", "in_pool",
                                     "rank_found", "role", "country"])
    return pd.concat(frames, ignore_index=True)


def pc_from_positives(pos, k_list=EVAL_K_LIST, bucket_map=None):
    """Pair completeness from the positives sidecars — same definitions as
    pair recall, macro per S1, cross/same script and match-count bucket."""
    res = {"overall": {}, "cross_script": {}, "same_script": {}, "by_match_bucket": {},
           "n_pairs": int(len(pos)), "n_s1_with_matches": int(pos["s1_id"].nunique()),
           "pos_in_pool_frac": float(pos["in_pool"].mean()) if len(pos) else float("nan")}
    if not len(pos):
        return res
    pos = pos.copy()
    if bucket_map is not None:
        pos["match_bucket"] = pos["s1_id"].map(bucket_map)
    for k in k_list:
        hit = (pos["rank_found"] > 0) & (pos["rank_found"] <= k)
        res["overall"][k] = {"pair_recall": float(hit.mean()),
                             "macro_recall": float(hit.groupby(pos["s1_id"]).mean().mean())}
        for lab, want in (("cross_script", True), ("same_script", False)):
            m = pos["cross_script"] == want
            res[lab][k] = {"pair_recall": float(hit[m].mean()) if m.any() else float("nan"),
                           "n_pairs": int(m.sum())}
        if bucket_map is not None:
            for b, idx in pos.groupby("match_bucket").groups.items():
                res["by_match_bucket"].setdefault(str(b), {})[k] = {
                    "pair_recall": float(hit.loc[idx].mean()), "n_pairs": int(len(idx))}
    return res


def print_pc(res, title):
    print(f"\n=== {title}  (pairs={res['n_pairs']:,}, S1={res['n_s1_with_matches']:,}, "
          f"in-pool={res['pos_in_pool_frac'] * 100:.2f}%) ===")
    print(f"{'K':>4} | {'pair':>8} | {'macro':>8} | {'cross':>8} | {'same':>8}")
    for k, ov in res["overall"].items():
        cs = res["cross_script"][k]["pair_recall"]
        ss = res["same_script"][k]["pair_recall"]
        print(f"{k:>4} | {ov['pair_recall'] * 100:7.2f}% | {ov['macro_recall'] * 100:7.2f}% | "
              f"{cs * 100:7.2f}% | {ss * 100:7.2f}%")


def suggest_k_final(pos, target=0.995, roles=UNBIASED_ROLES, k_list=EVAL_K_LIST):
    """Smallest K whose pooled pair recall on the roles the bi-encoder never saw
    reaches `target`. CE-FIT is excluded on purpose (see contract)."""
    sub = pos[pos["role"].isin(roles)]
    for k in k_list:
        r = ((sub["rank_found"] > 0) & (sub["rank_found"] <= k)).mean()
        if r >= target:
            return k, float(r)
    k = max(k_list)
    return k, float(((sub["rank_found"] > 0) & (sub["rank_found"] <= k)).mean())


def write_merged_manifest(all_shards, k_final, pc_report, coverage, checkpoints):
    os.makedirs(merged_root(), exist_ok=True)
    m = {"prod_version": PROD_VERSION, "k_gen": K_GEN, "k_final": int(k_final),
         "channel": CHANNEL, "variant": VARIANT, "base_model": BASE_MODEL,
         "checkpoints": checkpoints, "written_ts": time.time(),
         "coverage": coverage, "shards": all_shards}
    with open(f"{merged_root()}/manifest.json", "w") as f:
        json.dump(m, f, indent=1, default=float)
    with open(f"{merged_root()}/pc_report.json", "w") as f:
        json.dump(pc_report, f, indent=1, default=float)
    print(f"merged manifest -> {merged_root()}/manifest.json  (K_FINAL={k_final})")


# ---------------------------------------------------------------------------
# Downstream reader API (Block 3 / Block 4 import b2_prod and use these)
# ---------------------------------------------------------------------------

def load_merged_manifest(path=None):
    with open(path or f"{merged_root()}/manifest.json") as f:
        return json.load(f)


def select_shards(manifest, split, roles=None, countries=None):
    return [s for s in manifest["shards"]
            if s["split"] == split
            and (roles is None or s["role"] in roles)
            and (countries is None or s["country"] in countries)]


def iter_candidates(split, roles=None, countries=None, k=None, columns=None, manifest=None):
    """Yields (shard_meta, DataFrame) one shard at a time — the memory-safe way to
    process tens of millions of rows. k defaults to the manifest's K_FINAL."""
    m = manifest or load_merged_manifest()
    k = m["k_final"] if k is None else k
    for s in select_shards(m, split, roles, countries):
        t = pq.read_table(s["cand_path"], columns=columns, filters=[("rank", "<=", k)])
        yield s, t.to_pandas()


def read_candidates(split, roles=None, countries=None, k=None, columns=None, manifest=None):
    """Everything matching, concatenated. Fine for one role/country; for all of
    train prefer iter_candidates."""
    frames = [df for _, df in iter_candidates(split, roles, countries, k, columns, manifest)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def missed_positives(roles=None, countries=None, k=None, manifest=None):
    """Labeled positives NOT retrieved within top-k (rank_found == 0 or > k).
    For Block 4 training-set augmentation on CE-FIT only — never inject these
    into evaluation roles (that would overstate end-to-end recall)."""
    m = manifest or load_merged_manifest()
    k = m["k_final"] if k is None else k
    pos = load_pos_sidecars(select_shards(m, "train", roles, countries))
    return pos[(pos["rank_found"] == 0) | (pos["rank_found"] > k)].reset_index(drop=True)
