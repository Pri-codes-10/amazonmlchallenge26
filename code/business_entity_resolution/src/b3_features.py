"""
b3_features.py - Block 3: pair features.

    f(S1 record, candidate record, Block 2 retrieval context) -> feature vector      (deterministic)

Reads candidates ONLY through b2_prod (merged manifest) and Block 1 prepared records. Never reads a
label-bearing column into a feature (see LEAK_COLS / check_no_leak). Labels are passed through untouched.

Groups:
  G1 retrieval | G2 name similarity | G3 alias | G4 legal suffix | G5 address | G6 format/script
  G7 source/meta | G8 candidate-list context | G9 cross-list exclusivity (extension, ablated like the rest)

Output: <ROOT>/block3/<version>/features/<split>/<ROLE>/<COUNTRY>/part-XXXXX.parquet (+ done-XXXXX.json),
one part per Block 2 shard, same part index.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

import er_eval as ev

FEATURE_CODE_VERSION = "b3-v1.0"
HASH_BITS = 22
TOKEN_PATTERN = r"(?u)\b\w+\b"
TRAIN_ROLES = ["CE-FIT", "CE-DEV", "TREE-FIT", "TREE-DEV", "CALIBRATION", "HOLDOUT"]
FEATURE_ROLES = ["TREE-FIT", "TREE-DEV", "CALIBRATION", "HOLDOUT"]  # + TEST on the test split

# Columns that must NEVER be loaded into a feature table (leakage rules).
LEAK_COLS = {"cluster_id", "entity_role", "group_id", "stratum", "n_matches", "n_s2", "n_s3",
             "is_singleton", "has_cross_script", "match_bucket", "cross_script", "s1_name_script",
             "match_name_script", "matched_source"}

REC_COLS = ["entity_id", "country_norm", "name_latin", "name_core", "name_skeleton", "name_aka",
            "legal_suffix", "is_domain_name", "name_script", "name_is_mixed_script", "name_was_romanized",
            "addr_latin", "postal_code", "street_number", "has_landmark", "name_missing", "addr_missing",
            "name_has_embedded_phone", "addr_has_embedded_phone", "name_n_tokens", "addr_is_multi_location"]
assert not (set(REC_COLS) & LEAK_COLS)

PASS_COLS = ["s1_id", "cand_id", "role", "country", "split", "label"]
CAT_FEATURES = ["country", "source", "cand_name_script"]

# --------------------------------------------------------------------------------------------
# Feature registry (name, group, dtype, description) - order is the output column order
# --------------------------------------------------------------------------------------------
FEATURES: list[tuple[str, str, str, str]] = []


def _reg(name, group, dtype, desc):
    FEATURES.append((name, group, dtype, desc))


# G1 retrieval
_reg("dense_score", "G1", "float32", "Block 2 bi-encoder cosine")
_reg("dense_rank", "G1", "int16", "Block 2 rank (1 = best)")
_reg("score_gap_top1", "G1", "float32", "score - S1's top-1 score")
_reg("score_ratio_top1", "G1", "float32", "score / S1's top-1 score")
_reg("score_gap_next", "G1", "float32", "score - next candidate's score (NaN for last)")
_reg("score_gap_prev", "G1", "float32", "previous candidate's score - score (NaN for rank 1)")
_reg("n_cands", "G1", "int16", "candidates in this S1's list (<= K_FINAL)")
_reg("is_rank1", "G1", "int8", "rank == 1")
# G2 name similarity
for fld in ("latin", "core", "skel"):
    _reg(f"name_{fld}_ratio", "G2", "float32", f"fuzz.ratio on name_{fld} (0-1)")
    _reg(f"name_{fld}_tset", "G2", "float32", f"fuzz.token_set_ratio on name_{fld}")
    _reg(f"name_{fld}_tsort", "G2", "float32", f"fuzz.token_sort_ratio on name_{fld}")
    _reg(f"name_{fld}_jw", "G2", "float32", f"Jaro-Winkler on name_{fld}")
    _reg(f"name_{fld}_lev", "G2", "float32", f"normalized Levenshtein similarity on name_{fld}")
    _reg(f"name_{fld}_eq", "G2", "int8", f"exact equality of name_{fld} (both non-empty)")
_reg("name_core_partial", "G2", "float32", "fuzz.partial_ratio on name_core")
_reg("name_first_tok_eq", "G2", "int8", "first token of name_core equal (brand anchor)")
_reg("name_last_tok_eq", "G2", "int8", "last token of name_core equal")
_reg("skel_first_tok_eq", "G2", "int8", "first token of name_skeleton equal")
_reg("name_ntok_absdiff", "G2", "int16", "|name_n_tokens difference|")
_reg("name_len_ratio", "G2", "float32", "min/max char length of name_latin")
_reg("name_tok_jaccard", "G2", "float32", "token Jaccard on name_latin")
_reg("name_idf_jaccard", "G2", "float32", "IDF-weighted Jaccard on name_latin tokens")
_reg("name_idf_cover_min", "G2", "float32", "shared IDF mass / smaller side's IDF mass (containment)")
_reg("name_idf_s1_unmatched", "G2", "float32", "share of S1 name IDF mass missing from candidate")
_reg("name_idf_cand_unmatched", "G2", "float32", "share of candidate name IDF mass missing from S1")
_reg("name_rarest_shared_idf", "G2", "float32", "IDF of the rarest shared name token (0 = none)")
# G3 alias
_reg("alias_best_tset", "G3", "float32", "best token_set over {core, aka} x {core, aka}")
_reg("alias_gain", "G3", "float32", "alias_best_tset - name_core_tset")
_reg("aka_n", "G3", "int8", "sides with a non-empty name_aka (0/1/2)")
# G4 legal suffix
_reg("suffix_state", "G4", "int8", "0 both missing, 1 one missing, 2 equal, 3 both present & different")
# G5 address
_reg("addr_tset", "G5", "float32", "token_set_ratio on addr_latin")
_reg("addr_ratio", "G5", "float32", "fuzz.ratio on addr_latin")
_reg("addr_tok_jaccard", "G5", "float32", "token Jaccard on addr_latin")
_reg("addr_idf_jaccard", "G5", "float32", "IDF-weighted Jaccard on addr_latin")
_reg("addr_idf_cover_min", "G5", "float32", "shared IDF mass / smaller side's mass")
_reg("addr_idf_s1_unmatched", "G5", "float32", "share of S1 addr IDF mass missing from candidate")
_reg("addr_idf_cand_unmatched", "G5", "float32", "share of candidate addr IDF mass missing from S1")
_reg("addr_rarest_shared_idf", "G5", "float32", "IDF of the rarest shared address token")
_reg("postal_state", "G5", "int8", "0 both missing, 1 one missing, 2 equal, 3 edit-distance 1, 4 different")
_reg("street_state", "G5", "int8", "0 both missing, 1 one missing, 2 equal, 3 different")
_reg("landmark_n", "G5", "int8", "sides with has_landmark")
_reg("s1_addr_missing", "G5", "int8", "S1 address missing")
_reg("cand_addr_missing", "G5", "int8", "candidate address missing (strong positive in EDA)")
_reg("multi_loc_n", "G5", "int8", "sides with addr_is_multi_location")
# G6 format / script
_reg("name_cross_script", "G6", "int8", "name_script differs (computed from records)")
_reg("cand_name_script", "G6", "category", "candidate name_script")
_reg("cand_name_romanized", "G6", "int8", "candidate name_was_romanized")
_reg("name_mixed_n", "G6", "int8", "sides with mixed-script names")
_reg("domain_n", "G6", "int8", "sides with is_domain_name")
_reg("core_nospace_ratio", "G6", "float32", "fuzz.ratio of name_core with spaces removed (domain stems)")
_reg("name_phone_n", "G6", "int8", "sides with an embedded phone in the name")
_reg("addr_phone_n", "G6", "int8", "sides with an embedded phone in the address")
_reg("name_missing_n", "G6", "int8", "sides with a missing name")
# G7 source / meta
_reg("source", "G7", "category", "candidate source S2/S3")
_reg("country", "G7", "category", "country (open set; ablate for FRANCE transfer)")
# G8 candidate-list context
_reg("name_sim_rank_in_list", "G8", "int16", "rank of name_core_tset among this S1's candidates")
_reg("name_sim_n_ge90", "G8", "int16", "candidates in the list with name_core_tset >= 0.90")
_reg("name_sim_gap_best", "G8", "float32", "best name_core_tset in list - this one")
_reg("addr_sim_gap_best", "G8", "float32", "best addr_tset in list - this one")
_reg("name_addr_n_strong", "G8", "int16", "candidates with name_core_tset >= 0.9 and addr_tset >= 0.7")
# G9 cross-list exclusivity (needs the per split x country pre-pass)
_reg("cand_fanin", "G9", "int16", "how many S1 lists (whole split, all roles) contain this candidate")
_reg("cand_is_best_s1", "G9", "int8", "this S1 has the highest retrieval score for the candidate")
_reg("cand_best_margin", "G9", "float32", "if best: score - runner-up S1 score; else score - best S1 score")

FEATURE_NAMES = [f[0] for f in FEATURES]
GROUPS = sorted({f[1] for f in FEATURES})
GROUP_FEATURES = {g: [f[0] for f in FEATURES if f[1] == g] for g in GROUPS}


def feature_set_hash(names=None) -> str:
    names = names or FEATURE_NAMES
    spec = [(n, g, d) for n, g, d, _ in FEATURES if n in set(names)]
    return hashlib.sha1(json.dumps([FEATURE_CODE_VERSION, spec]).encode()).hexdigest()[:12]


def feature_spec() -> dict:
    return {"code_version": FEATURE_CODE_VERSION, "feature_set_hash": feature_set_hash(),
            "hash_bits": HASH_BITS, "token_pattern": TOKEN_PATTERN,
            "features": [{"name": n, "group": g, "dtype": d, "description": s} for n, g, d, s in FEATURES]}


# --------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------
def ver_dir(version: str) -> str:
    return os.path.join(ev.ROOT, "block3", version)


def part_dir(version, split, role, country):
    return os.path.join(ver_dir(version), "features", split, role, country)


def prepare_folders(version, plan: dict):
    """Create every output folder once, before the build starts.
    plan = {split: {"roles": [...], "countries": [...]}}"""
    root = ver_dir(version)
    for sub in ("vocab", "g9", "reports", "reports/per_s1", "markers"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for split, p in plan.items():
        for c in p["countries"]:
            os.makedirs(os.path.join(root, "g9", split, c), exist_ok=True)
            for r in p["roles"]:
                os.makedirs(part_dir(version, split, r, c), exist_ok=True)
    with open(os.path.join(root, "feature_spec.json"), "w") as fh:
        json.dump(feature_spec(), fh, indent=1)
    with open(os.path.join(root, "markers", "_FOLDERS_READY"), "w") as fh:
        json.dump({"plan": plan, "time": time.strftime("%F %T")}, fh)
    print("folders ready under", root)


def require_folders_ready(version):
    p = os.path.join(ver_dir(version), "markers", "_FOLDERS_READY")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing - run prepare_folders() first")


# --------------------------------------------------------------------------------------------
# IDF (unsupervised, fitted on TRAIN pool text only, then frozen)
# --------------------------------------------------------------------------------------------
def _hv():
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(n_features=2 ** HASH_BITS, token_pattern=TOKEN_PATTERN, lowercase=False,
                             alternate_sign=False, binary=True, norm=None, dtype=np.float32)


def fit_idf(version, batch_size=1_000_000):
    """Document frequencies of name_latin / addr_latin tokens over the TRAIN S2+S3 pool (all
    countries, all roles incl. POOL). No labels involved. Written once, reused for test."""
    hv = _hv()
    df = {"name_latin": np.zeros(2 ** HASH_BITS, np.int64), "addr_latin": np.zeros(2 ** HASH_BITS, np.int64)}
    n = 0
    t0 = time.time()
    for src in ("s2", "s3"):
        files = sorted(glob.glob(ev.b1_path("records", f"train_{src}", "part-*.parquet")))
        for b in ds.dataset(files, format="parquet").to_batches(columns=list(df), batch_size=batch_size):
            n += b.num_rows
            for col in df:
                x = np.asarray(b.column(col).to_pandas().fillna("").astype(str).to_numpy(dtype=object))
                X = hv.transform(x)
                df[col] += np.bincount(X.indices, minlength=2 ** HASH_BITS)
            print(f"  idf: {n:,} pool rows ({time.time() - t0:.0f}s)", end="\r")
    out = os.path.join(ver_dir(version), "vocab")
    os.makedirs(out, exist_ok=True)
    for col, d in df.items():
        idf = (np.log((1.0 + n) / (1.0 + d)) + 1.0).astype(np.float32)
        np.save(os.path.join(out, f"idf_{col}.npy"), idf)
    json.dump({"n_docs": n, "hash_bits": HASH_BITS, "fitted_on": "train S2+S3 pool",
               "time": time.strftime("%F %T")}, open(os.path.join(out, "idf_meta.json"), "w"))
    print(f"\nidf fitted on {n:,} train pool records -> {out}")


def load_idf(version):
    d = os.path.join(ver_dir(version), "vocab")
    return (np.load(os.path.join(d, "idf_name_latin.npy")), np.load(os.path.join(d, "idf_addr_latin.npy")))


# --------------------------------------------------------------------------------------------
# Record tables (one per side, per split x country)
# --------------------------------------------------------------------------------------------
def _s(df, c):
    return df[c].astype(object).where(df[c].notna(), "").astype(str).to_numpy(dtype=object)


def _flag(df, c):
    return df[c].fillna(False).astype(bool).to_numpy().astype(np.int8)


def _hash(arr):
    return pd.util.hash_array(np.asarray(arr, dtype=object))


def _first_last(arr):
    n = len(arr)
    f = np.empty(n, dtype=object)
    l = np.empty(n, dtype=object)
    for i, s in enumerate(arr):
        t = s.split()
        f[i] = t[0] if t else ""
        l[i] = t[-1] if t else ""
    return f, l


def check_no_leak(columns):
    bad = set(columns) & LEAK_COLS
    if bad:
        raise AssertionError(f"LEAKAGE GUARD: label-bearing columns loaded into a feature table: {bad}")


class RecordTable:
    """Column arrays + hashed token matrices for fast pairwise gathers."""

    def __init__(self, df: pd.DataFrame, key: np.ndarray, idf_name, idf_addr, hv=None):
        check_no_leak(df.columns)
        hv = hv or _hv()
        self.index = pd.Index(np.asarray(key, dtype=object))
        if not self.index.is_unique:
            raise ValueError("record keys are not unique")
        self.n = len(df)
        for c in ("name_latin", "name_core", "name_skeleton", "name_aka", "addr_latin",
                  "postal_code", "street_number"):
            setattr(self, c, _s(df, c))
        self.legal_h = _hash(_s(df, "legal_suffix"))
        self.legal_has = np.array([bool(x) for x in _s(df, "legal_suffix")])
        self.name_script = _s(df, "name_script")
        for c in ("is_domain_name", "name_is_mixed_script", "name_was_romanized", "has_landmark",
                  "name_missing", "addr_missing", "name_has_embedded_phone", "addr_has_embedded_phone",
                  "addr_is_multi_location"):
            setattr(self, c, _flag(df, c))
        self.name_n_tokens = df["name_n_tokens"].fillna(0).to_numpy().astype(np.int16)
        self.name_len = np.fromiter((len(s) for s in self.name_latin), np.int32, self.n)
        self.core_nospace = np.array([s.replace(" ", "") for s in self.name_core], dtype=object)
        cf, cl = _first_last(self.name_core)
        sf, _ = _first_last(self.name_skeleton)
        self.core_first_h, self.core_last_h, self.skel_first_h = _hash(cf), _hash(cl), _hash(sf)
        self.core_first_ok = np.array([bool(x) for x in cf])
        self.skel_first_ok = np.array([bool(x) for x in sf])
        for c in ("name_latin", "name_core", "name_skeleton"):
            setattr(self, c + "_h", _hash(getattr(self, c)))
        self.name_tok = hv.transform(self.name_latin).tocsr()
        self.addr_tok = hv.transform(self.addr_latin).tocsr()
        self.name_w = np.asarray(self.name_tok @ idf_name).ravel()
        self.addr_w = np.asarray(self.addr_tok @ idf_addr).ravel()
        self.name_ntok_set = np.diff(self.name_tok.indptr)
        self.addr_ntok_set = np.diff(self.addr_tok.indptr)

    def pos(self, keys) -> np.ndarray:
        p = self.index.get_indexer(np.asarray(keys, dtype=object))
        if (p < 0).any():
            raise KeyError(f"{int((p < 0).sum())} ids not found in record table (e.g. {np.asarray(keys)[p < 0][:3]})")
        return p


def pool_key(source, entity_id):
    return (pd.Series(source).astype(str).to_numpy(dtype=object) + "|" +
            pd.Series(entity_id).astype(str).to_numpy(dtype=object))


def load_tables(split, country, idf):
    """(S1 table, pool table) for one split x country."""
    t0 = time.time()
    hv = _hv()
    s1 = ev.read_records(split, "s1", REC_COLS, countries=[country])
    S = RecordTable(s1, s1["entity_id"].to_numpy(dtype=object), *idf, hv=hv)
    parts = []
    for src in ("S2", "S3"):
        d = ev.read_records(split, src, REC_COLS, countries=[country])
        d["_src"] = src
        parts.append(d)
    pool = pd.concat(parts, ignore_index=True)
    P = RecordTable(pool, pool_key(pool["_src"], pool["entity_id"]), *idf, hv=hv)
    print(f"  tables {split}/{country}: S1 {S.n:,}  pool {P.n:,}  ({time.time() - t0:.0f}s)")
    return S, P


# --------------------------------------------------------------------------------------------
# Pairwise helpers
# --------------------------------------------------------------------------------------------
def _cp(a, b, scorer, valid, scale=1.0):
    out = np.full(len(a), np.nan, np.float32)
    idx = np.flatnonzero(valid)
    if len(idx):
        out[idx] = process.cpdist(a[idx], b[idx], scorer=scorer, workers=-1, dtype=np.float32) * scale
    return out


def _nonempty(arr):
    return np.fromiter((bool(s) for s in arr), bool, len(arr))


def _token_overlap(Xa, Xb, idf, wa, wb, na, nb):
    I = Xa.multiply(Xb).tocsr()
    I.eliminate_zeros()
    n_inter = np.diff(I.indptr).astype(np.float32)
    J = I.copy()
    J.data = idf[J.indices].astype(np.float32)
    w_inter = np.asarray(J.sum(axis=1)).ravel().astype(np.float32)
    rarest = np.asarray(J.max(axis=1).todense()).ravel().astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        union_n = na + nb - n_inter
        jac = np.where(union_n > 0, n_inter / union_n, np.nan)
        union_w = wa + wb - w_inter
        idf_jac = np.where(union_w > 0, w_inter / union_w, np.nan)
        mn = np.minimum(wa, wb)
        cover = np.where(mn > 0, w_inter / mn, np.nan)
        s1_un = np.where(wa > 0, 1 - w_inter / wa, np.nan)
        c_un = np.where(wb > 0, 1 - w_inter / wb, np.nan)
    return (jac.astype(np.float32), idf_jac.astype(np.float32), cover.astype(np.float32),
            s1_un.astype(np.float32), c_un.astype(np.float32), rarest)


def _groups(s1_sorted):
    codes = pd.factorize(s1_sorted, sort=False)[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return codes, starts


def _grp_max(x, codes, starts):
    return np.maximum.reduceat(x, starts)[codes]


def _grp_sum(x, codes, starts):
    return np.add.reduceat(x, starts)[codes]


def _grp_rank_desc(x, codes, starts):
    n = len(x)
    order = np.lexsort((-x, codes))
    r = np.empty(n, np.int16)
    r[order] = (np.arange(n) - starts[codes[order]] + 1).astype(np.int16)
    return r


# --------------------------------------------------------------------------------------------
# G9 pre-pass (per split x country, over ALL roles of the split)
# --------------------------------------------------------------------------------------------
def g9_path(version, split, country):
    return os.path.join(ver_dir(version), "g9", split, country, "g9.parquet")


def g9_prepass(version, b2p, split, country, P: RecordTable, k: int):
    """Label-free: for each candidate record, how many S1 lists (top-k) contain it, and the best /
    runner-up retrieval score across those S1s. Uses every role of the split (mirrors test)."""
    roles = TRAIN_ROLES if split == "train" else ["TEST"]
    cnt = np.zeros(P.n, np.int32)
    best = np.full(P.n, -np.inf, np.float32)
    second = np.full(P.n, -np.inf, np.float32)
    t0 = time.time()
    nrows = 0
    for meta, df in b2p.iter_candidates(split, roles=roles, countries=[country], k=k):
        if len(df) == 0:
            continue
        pos = P.pos(pool_key(df["source"], df["cand_id"]))
        sc = df["score"].to_numpy(np.float32)
        nrows += len(df)
        cnt += np.bincount(pos, minlength=P.n).astype(np.int32)
        order = np.lexsort((-sc, pos))
        ps, ss = pos[order], sc[order]
        first = np.r_[True, ps[1:] != ps[:-1]]
        fi = np.flatnonzero(first)
        nxt = fi + 1
        has2 = nxt < len(ps)
        has2[has2] = ps[nxt[has2]] == ps[fi[has2]]
        sb_pos, sb = ps[fi], ss[fi]
        s2 = np.full(len(fi), -np.inf, np.float32)
        s2[has2] = ss[nxt[has2]]
        bo, so = best[sb_pos], second[sb_pos]
        best[sb_pos] = np.maximum(bo, sb)
        second[sb_pos] = np.maximum(np.minimum(bo, sb), np.maximum(so, s2))
    keep = cnt > 0
    out = pd.DataFrame({"key": P.index.to_numpy()[keep], "fanin": cnt[keep], "best": best[keep],
                        "second": second[keep]})
    path = g9_path(version, split, country)
    out.to_parquet(path, index=False)
    json.dump({"rows": nrows, "k": k, "cands": int(keep.sum()), "elapsed_s": round(time.time() - t0, 1)},
              open(path.replace(".parquet", ".done.json"), "w"))
    print(f"  g9 {split}/{country}: {nrows:,} rows, {keep.sum():,} distinct candidates, "
          f"fan-in mean {cnt[keep].mean():.2f} max {cnt.max()} ({time.time() - t0:.0f}s)")
    return out


def load_g9(version, split, country, P: RecordTable):
    path = g9_path(version, split, country)
    if not os.path.exists(path.replace(".parquet", ".done.json")):
        raise FileNotFoundError(f"G9 pre-pass missing for {split}/{country}: run the G9 stage first")
    g = pd.read_parquet(path)
    pos = P.pos(g["key"].to_numpy(dtype=object))
    cnt = np.zeros(P.n, np.int32)
    best = np.full(P.n, np.nan, np.float32)
    second = np.full(P.n, np.nan, np.float32)
    cnt[pos] = g["fanin"].to_numpy()
    best[pos] = g["best"].to_numpy()
    sec = g["second"].to_numpy().astype(np.float32)
    second[pos] = np.where(np.isfinite(sec), sec, np.nan)
    return cnt, best, second


# --------------------------------------------------------------------------------------------
# The feature function
# --------------------------------------------------------------------------------------------
def featurize(df: pd.DataFrame, S: RecordTable, P: RecordTable, idf, g9=None) -> pd.DataFrame:
    """One Block 2 shard (rank <= K_FINAL) -> one feature frame, same rows (re-sorted by s1, rank)."""
    idf_name, idf_addr = idf
    df = df.sort_values(["s1_id", "rank"], kind="stable").reset_index(drop=True)
    n = len(df)
    si = S.pos(df["s1_id"].to_numpy(dtype=object))
    ci = P.pos(pool_key(df["source"], df["cand_id"]))
    codes, starts = _groups(df["s1_id"].to_numpy(dtype=object))
    F = {}

    # ---- G1 retrieval
    score = df["score"].to_numpy(np.float32)
    rank = df["rank"].to_numpy(np.int32)
    top1 = _grp_max(score, codes, starts)
    last = np.r_[codes[1:] != codes[:-1], True]
    first = np.r_[True, codes[1:] != codes[:-1]]
    nxt = np.r_[score[1:], np.nan].astype(np.float32)
    nxt[last] = np.nan
    prv = np.r_[np.nan, score[:-1]].astype(np.float32)
    prv[first] = np.nan
    F["dense_score"] = score
    F["dense_rank"] = rank.astype(np.int16)
    F["score_gap_top1"] = (score - top1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["score_ratio_top1"] = np.where(top1 > 0, score / top1, np.nan).astype(np.float32)
    F["score_gap_next"] = (score - nxt).astype(np.float32)
    F["score_gap_prev"] = (prv - score).astype(np.float32)
    F["n_cands"] = _grp_sum(np.ones(n, np.int32), codes, starts).astype(np.int16)
    F["is_rank1"] = (rank == 1).astype(np.int8)

    # ---- G2 name similarity
    for fld, col in (("latin", "name_latin"), ("core", "name_core"), ("skel", "name_skeleton")):
        a, b = getattr(S, col)[si], getattr(P, col)[ci]
        v = _nonempty(a) & _nonempty(b)
        F[f"name_{fld}_ratio"] = _cp(a, b, fuzz.ratio, v, 0.01)
        F[f"name_{fld}_tset"] = _cp(a, b, fuzz.token_set_ratio, v, 0.01)
        F[f"name_{fld}_tsort"] = _cp(a, b, fuzz.token_sort_ratio, v, 0.01)
        F[f"name_{fld}_jw"] = _cp(a, b, JaroWinkler.normalized_similarity, v)
        F[f"name_{fld}_lev"] = _cp(a, b, Levenshtein.normalized_similarity, v)
        F[f"name_{fld}_eq"] = (v & (getattr(S, col + "_h")[si] == getattr(P, col + "_h")[ci])).astype(np.int8)
        if fld == "core":
            core_a, core_b, core_v = a, b, v
    F["name_core_partial"] = _cp(core_a, core_b, fuzz.partial_ratio, core_v, 0.01)
    okf = S.core_first_ok[si] & P.core_first_ok[ci]
    F["name_first_tok_eq"] = (okf & (S.core_first_h[si] == P.core_first_h[ci])).astype(np.int8)
    F["name_last_tok_eq"] = (okf & (S.core_last_h[si] == P.core_last_h[ci])).astype(np.int8)
    oks = S.skel_first_ok[si] & P.skel_first_ok[ci]
    F["skel_first_tok_eq"] = (oks & (S.skel_first_h[si] == P.skel_first_h[ci])).astype(np.int8)
    F["name_ntok_absdiff"] = np.abs(S.name_n_tokens[si].astype(np.int32) - P.name_n_tokens[ci]).astype(np.int16)
    la, lb = S.name_len[si], P.name_len[ci]
    with np.errstate(divide="ignore", invalid="ignore"):
        F["name_len_ratio"] = np.where(np.maximum(la, lb) > 0, np.minimum(la, lb) / np.maximum(la, lb),
                                       np.nan).astype(np.float32)
    (F["name_tok_jaccard"], F["name_idf_jaccard"], F["name_idf_cover_min"], F["name_idf_s1_unmatched"],
     F["name_idf_cand_unmatched"], F["name_rarest_shared_idf"]) = _token_overlap(
        S.name_tok[si], P.name_tok[ci], idf_name, S.name_w[si], P.name_w[ci],
        S.name_ntok_set[si].astype(np.float32), P.name_ntok_set[ci].astype(np.float32))

    # ---- G3 alias
    sa, pa_ = S.name_aka[si], P.name_aka[ci]
    sa_ok, pa_ok = _nonempty(sa), _nonempty(pa_)
    base = F["name_core_tset"]
    c1 = _cp(sa, core_b, fuzz.token_set_ratio, sa_ok & _nonempty(core_b), 0.01)
    c2 = _cp(core_a, pa_, fuzz.token_set_ratio, pa_ok & _nonempty(core_a), 0.01)
    c3 = _cp(sa, pa_, fuzz.token_set_ratio, sa_ok & pa_ok, 0.01)
    with np.errstate(invalid="ignore"):
        stack = np.vstack([base, c1, c2, c3])
        allnan = np.isnan(stack).all(axis=0)
        best = np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(stack), -1, stack), axis=0))
    F["alias_best_tset"] = best.astype(np.float32)
    F["alias_gain"] = (best - np.where(np.isnan(base), 0, base)).astype(np.float32)
    F["aka_n"] = (sa_ok.astype(np.int8) + pa_ok.astype(np.int8)).astype(np.int8)

    # ---- G4 legal suffix
    ha, hb = S.legal_has[si], P.legal_has[ci]
    st = np.zeros(n, np.int8)
    st[ha ^ hb] = 1
    both = ha & hb
    eq = S.legal_h[si] == P.legal_h[ci]
    st[both & eq] = 2
    st[both & ~eq] = 3
    F["suffix_state"] = st

    # ---- G5 address
    aa, ab = S.addr_latin[si], P.addr_latin[ci]
    av = _nonempty(aa) & _nonempty(ab)
    F["addr_tset"] = _cp(aa, ab, fuzz.token_set_ratio, av, 0.01)
    F["addr_ratio"] = _cp(aa, ab, fuzz.ratio, av, 0.01)
    (F["addr_tok_jaccard"], F["addr_idf_jaccard"], F["addr_idf_cover_min"], F["addr_idf_s1_unmatched"],
     F["addr_idf_cand_unmatched"], F["addr_rarest_shared_idf"]) = _token_overlap(
        S.addr_tok[si], P.addr_tok[ci], idf_addr, S.addr_w[si], P.addr_w[ci],
        S.addr_ntok_set[si].astype(np.float32), P.addr_ntok_set[ci].astype(np.float32))
    pa1, pb1 = S.postal_code[si], P.postal_code[ci]
    ha, hb = _nonempty(pa1), _nonempty(pb1)
    st = np.zeros(n, np.int8)
    st[ha ^ hb] = 1
    both = np.flatnonzero(ha & hb)
    if len(both):
        dist = process.cpdist(pa1[both], pb1[both], scorer=Levenshtein.distance, workers=-1)
        st[both] = np.where(dist == 0, 2, np.where(dist == 1, 3, 4))
    F["postal_state"] = st
    sa1, sb1 = S.street_number[si], P.street_number[ci]
    ha, hb = _nonempty(sa1), _nonempty(sb1)
    st = np.zeros(n, np.int8)
    st[ha ^ hb] = 1
    both = ha & hb
    eqs = sa1 == sb1
    st[both & eqs] = 2
    st[both & ~eqs] = 3
    F["street_state"] = st
    F["landmark_n"] = (S.has_landmark[si] + P.has_landmark[ci]).astype(np.int8)
    F["s1_addr_missing"] = S.addr_missing[si]
    F["cand_addr_missing"] = P.addr_missing[ci]
    F["multi_loc_n"] = (S.addr_is_multi_location[si] + P.addr_is_multi_location[ci]).astype(np.int8)

    # ---- G6 format / script
    na_, nb_ = S.name_script[si], P.name_script[ci]
    F["name_cross_script"] = ((na_ != nb_) & _nonempty(na_) & _nonempty(nb_)).astype(np.int8)
    F["cand_name_script"] = np.where(_nonempty(nb_), nb_, "empty").astype(object)
    F["cand_name_romanized"] = P.name_was_romanized[ci]
    F["name_mixed_n"] = (S.name_is_mixed_script[si] + P.name_is_mixed_script[ci]).astype(np.int8)
    F["domain_n"] = (S.is_domain_name[si] + P.is_domain_name[ci]).astype(np.int8)
    xa, xb = S.core_nospace[si], P.core_nospace[ci]
    F["core_nospace_ratio"] = _cp(xa, xb, fuzz.ratio, _nonempty(xa) & _nonempty(xb), 0.01)
    F["name_phone_n"] = (S.name_has_embedded_phone[si] + P.name_has_embedded_phone[ci]).astype(np.int8)
    F["addr_phone_n"] = (S.addr_has_embedded_phone[si] + P.addr_has_embedded_phone[ci]).astype(np.int8)
    F["name_missing_n"] = (S.name_missing[si] + P.name_missing[ci]).astype(np.int8)

    # ---- G7 source / meta
    F["source"] = df["source"].astype(str).to_numpy(dtype=object)
    F["country"] = df["country"].astype(str).to_numpy(dtype=object)

    # ---- G8 candidate-list context (within this S1's list; lists are never split across shards)
    ns = np.where(np.isnan(F["name_core_tset"]), -1.0, F["name_core_tset"]).astype(np.float32)
    ad = np.where(np.isnan(F["addr_tset"]), -1.0, F["addr_tset"]).astype(np.float32)
    F["name_sim_rank_in_list"] = _grp_rank_desc(ns, codes, starts)
    F["name_sim_n_ge90"] = _grp_sum((ns >= 0.9).astype(np.int32), codes, starts).astype(np.int16)
    F["name_sim_gap_best"] = (_grp_max(ns, codes, starts) - ns).astype(np.float32)
    F["addr_sim_gap_best"] = np.where(ad < 0, np.nan, _grp_max(ad, codes, starts) - ad).astype(np.float32)
    F["name_addr_n_strong"] = _grp_sum(((ns >= 0.9) & (ad >= 0.7)).astype(np.int32), codes, starts).astype(np.int16)

    # ---- G9 exclusivity
    if g9 is not None:
        cnt, bst, sec = g9
        fan = cnt[ci]
        b, s2 = bst[ci], sec[ci]
        is_best = score >= b - 1e-6
        F["cand_fanin"] = np.minimum(fan, 32767).astype(np.int16)
        F["cand_is_best_s1"] = is_best.astype(np.int8)
        F["cand_best_margin"] = np.where(is_best, score - s2, score - b).astype(np.float32)
    else:
        F["cand_fanin"] = np.full(n, -1, np.int16)
        F["cand_is_best_s1"] = np.full(n, -1, np.int8)
        F["cand_best_margin"] = np.full(n, np.nan, np.float32)

    missing = [f for f in FEATURE_NAMES if f not in F]
    assert not missing, missing
    out = df[[c for c in PASS_COLS if c in df.columns]].copy()
    for name, _, dtype, _ in FEATURES:
        v = F[name]
        out[name] = v if dtype == "category" else np.asarray(v).astype(dtype)
    return out


# --------------------------------------------------------------------------------------------
# Shard runner (resumable, worker-partitioned, commit = done json written last)
# --------------------------------------------------------------------------------------------
_warned_fallback = set()


def shard_index(meta, fallback: int) -> int:
    get = (lambda k: meta.get(k)) if isinstance(meta, dict) else (lambda k: getattr(meta, k, None))
    for k in ("shard", "shard_idx", "shard_index", "part", "part_idx", "index"):
        v = get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    for k in ("path", "file", "cand_path", "cand_file", "files"):
        v = get(k)
        if v is not None:
            m = re.search(r"(\d+)\.parquet", str(v))
            if m:
                return int(m.group(1))
    if "fallback" not in _warned_fallback:
        print("WARNING: shard meta has no shard index/path; using iteration order. Paste the probe "
              "output from Stage 1 so the notebook can read the real index.")
        _warned_fallback.add("fallback")
    return fallback


def run_features(version, b2p, split, roles, countries, k, worker=0, n_workers=1, use_g9=True,
                 max_shards=None, smoke_rows=None, tables=None):
    """Compute features for every committed Block 2 shard of (split, roles, countries).
    Worker w handles shards with index % n_workers == w. Re-running skips committed parts."""
    idf = load_idf(version)
    fh = feature_set_hash()
    summary = []
    for country in countries:
        S = P = g9 = None
        counters = {}
        done_here = 0
        for meta, df in b2p.iter_candidates(split, roles=roles, countries=[country], k=k):
            role = str(df["role"].iloc[0]) if len(df) else str((meta or {}).get("role", roles[0]))
            counters[role] = counters.get(role, -1) + 1
            sid = shard_index(meta, counters[role])
            if sid % n_workers != worker:
                continue
            od = part_dir(version, split, role, country)
            done_p = os.path.join(od, f"done-{sid:05d}.json")
            if os.path.exists(done_p):
                continue
            if max_shards is not None and done_here >= max_shards:
                break
            if S is None:
                S, P = tables if tables is not None else load_tables(split, country, idf)
                g9 = load_g9(version, split, country, P) if use_g9 else None
            if smoke_rows:
                keep = df["s1_id"].drop_duplicates().iloc[: max(1, smoke_rows // max(k, 1))]
                df = df[df["s1_id"].isin(keep)]
            t0 = time.time()
            if len(df):
                feats = featurize(df, S, P, idf, g9)
                os.makedirs(od, exist_ok=True)
                feats.to_parquet(os.path.join(od, f"part-{sid:05d}.parquet"), index=False, compression="zstd")
            el = time.time() - t0
            info = {"n_rows": int(len(df)), "n_s1": int(df["s1_id"].nunique()) if len(df) else 0,
                    "feature_set_hash": fh, "k": k, "elapsed_s": round(el, 1), "use_g9": use_g9,
                    "rows_per_min": int(len(df) / max(el, 1e-6) * 60), "smoke": bool(smoke_rows),
                    "b2_meta": {kk: str(vv) for kk, vv in (meta.items() if isinstance(meta, dict) else [])}}
            json.dump(info, open(done_p, "w"))
            done_here += 1
            summary.append({"split": split, "role": role, "country": country, "shard": sid} | info)
            print(f"  {split}/{role}/{country} part {sid:05d}: {len(df):,} rows in {el:.0f}s "
                  f"({info['rows_per_min']:,} rows/min)")
    return pd.DataFrame(summary)


def committed_parts(version, split, roles, countries=None):
    out = []
    for role in roles:
        base = os.path.join(ver_dir(version), "features", split, role)
        cs = countries or (sorted(os.listdir(base)) if os.path.isdir(base) else [])
        for c in cs:
            for dp in sorted(glob.glob(os.path.join(base, c, "done-*.json"))):
                pp = dp.replace("done-", "part-").replace(".json", ".parquet")
                if os.path.exists(pp):
                    out.append(pp)
    return out


def load_features(version, split, roles, countries=None, columns=None, s1_frac=None) -> pd.DataFrame:
    """Read committed parts only. s1_frac subsamples BY S1 (deterministic hash, same everywhere)."""
    parts = committed_parts(version, split, roles, countries)
    if not parts:
        raise FileNotFoundError(f"no committed Block 3 parts for {split} {roles} {countries}")
    frames = []
    for p in parts:
        d = pq.read_table(p, columns=columns).to_pandas()
        if s1_frac is not None:
            d = d[ev.in_s1_subsample(d["s1_id"].to_numpy(), s1_frac)]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    for c in CAT_FEATURES:
        if c in out.columns:
            out[c] = out[c].astype(str)
    if "name_cross_script" in out.columns:
        out["pair_cross_script"] = out["name_cross_script"].astype(np.int8)
    return out


def write_manifest(version, plan):
    man = {"feature_set_hash": feature_set_hash(), "code_version": FEATURE_CODE_VERSION,
           "time": time.strftime("%F %T"), "parts": {}}
    for split, p in plan.items():
        for r in p["roles"]:
            for c in p["countries"]:
                parts = committed_parts(version, split, [r], [c])
                rows = sum(json.load(open(x.replace("part-", "done-").replace(".parquet", ".json")))["n_rows"]
                           for x in parts)
                man["parts"][f"{split}/{r}/{c}"] = {"n_parts": len(parts), "n_rows": rows}
    json.dump(man, open(os.path.join(ver_dir(version), "manifest.json"), "w"), indent=1)
    return man


# --------------------------------------------------------------------------------------------
# Checks: leakage scan, redundancy, error analysis
# --------------------------------------------------------------------------------------------
def single_feature_scan(df: pd.DataFrame, features, max_rows=2_000_000, seed=0) -> pd.DataFrame:
    """Univariate ROC-AUC / PR-AUC of each feature alone (direction-free). A near-perfect single
    feature (>= 0.99) is a leakage suspect; investigate before any bake-off run."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = df.sample(min(len(df), max_rows), random_state=seed) if len(df) > max_rows else df
    y = d["label"].astype(int).to_numpy()
    rows = []
    for f in features:
        x = d[f]
        if f in CAT_FEATURES or x.dtype == object:
            rate = d.groupby(f)["label"].mean()
            v = x.map(rate).to_numpy(np.float64)
        else:
            v = x.to_numpy(np.float64)
            v = np.where(np.isnan(v), np.nanmin(v) - 1 if np.isfinite(np.nanmin(v)) else -1, v)
        try:
            auc = roc_auc_score(y, v)
            flip = auc < 0.5
            auc = max(auc, 1 - auc)
            ap = average_precision_score(y, -v if flip else v)
        except ValueError:
            auc, ap = np.nan, np.nan
        rows.append({"feature": f, "group": dict((n, g) for n, g, _, _ in FEATURES).get(f, "?"),
                     "roc_auc": auc, "pr_auc": ap, "nan_rate": float(df[f].isna().mean()),
                     "LEAK_SUSPECT": bool(auc >= 0.99)})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def redundant_pairs(df: pd.DataFrame, features, thr=0.97, max_rows=500_000) -> pd.DataFrame:
    num = [f for f in features if f not in CAT_FEATURES]
    d = df[num].sample(min(len(df), max_rows), random_state=0)
    c = d.rank().corr().abs()
    out = [(a, b, c.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if c.loc[a, b] >= thr]
    return pd.DataFrame(out, columns=["feat_a", "feat_b", "spearman"]).sort_values("spearman", ascending=False)


def error_examples(dev: pd.DataFrame, prob: np.ndarray, threshold: float, n: int = 50, split="train"):
    """Top false positives (highest prob, label 0) and false negatives (label 1, below threshold),
    with raw names/addresses for reading. Use these to invent the next feature."""
    d = dev[["s1_id", "cand_id", "source", "country", "label", "dense_rank"]].copy() if "source" in dev.columns \
        else dev[["s1_id", "cand_id", "country", "label", "dense_rank"]].copy()
    d["prob"] = prob
    fp = d[(d.label == 0) & (d.prob >= threshold)].nlargest(n, "prob")
    fn = d[(d.label == 1) & (d.prob < threshold)].nsmallest(n, "prob")
    ids1 = set(fp.s1_id) | set(fn.s1_id)
    ids2 = set(fp.cand_id) | set(fn.cand_id)
    s1 = ev.read_records(split, "s1", ["entity_id", "name_raw", "addr_raw"], ids=ids1)
    s1 = s1.drop_duplicates("entity_id").set_index("entity_id")
    pool = pd.concat([ev.read_records(split, s, ["entity_id", "name_raw", "addr_raw"], ids=ids2) for s in ("S2", "S3")])
    pool = pool.drop_duplicates("entity_id").set_index("entity_id")

    def enrich(x):
        x = x.copy()
        x["s1_name"] = x.s1_id.map(s1.name_raw)
        x["s1_addr"] = x.s1_id.map(s1.addr_raw)
        x["cand_name"] = x.cand_id.map(pool.name_raw)
        x["cand_addr"] = x.cand_id.map(pool.addr_raw)
        return x
    return enrich(fp), enrich(fn)
