"""
b4_ctx.py - Block 4 context signals consumed by Block 5.

Label-free features built ONLY from cross-encoder logits (V2 = production, V1 = second CE) plus Block 2's
candidate lists. Four levels, each one reading across more rows than the last:

  BASE  S1-list context of the CE score:  v2_logit, v2_gap_top, v2_rank, v2_n_pos, v2_n_scored
  L0    pair level, two CEs:    v1_logit, ens_logit (mean of available CEs), ce_min, ce_absdiff, ce_sign_agree
  L1    S1-list level:          v2_gap_next, v2_share, v1_gap_top, v1_rank, ens_gap_top, ens_rank, ens_gap_next,
                                ens_share, ens_entropy, ens_z, rank_shift (Block 2 rank - CE rank)
  L2    record-claimant level:  over EVERY S1 of the split (all roles) whose list contains this S2/S3 record:
                                rec_rank, rec_gap_best, rec_n_pos, rec_share, mutual_best

L2 is the CE's own view of "who else claims this record" (Block 3's G9 is the same idea on the bi-encoder
score). Train has exactly one owner per record, so the trees can LEARN the one-owner rule from it.
Only ranks / gaps / shares are used (never raw claimant counts except n_pos) because test has every S1
competing while a train role has only a fraction of them. parity_report() checks the shift explicitly.

Output: <ROOT>/block4/ctx/<version>/<split>/<ROLE>/<COUNTRY>.parquet   (one file per split x role x
        country, written to a tmp name then renamed, plus <COUNTRY>.done.json written last)
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CTX_CODE_VERSION = "b4ctx-v1.0"

BASE_FEATS = ["v2_logit", "v2_gap_top", "v2_rank", "v2_n_pos", "v2_n_scored"]
L0_FEATS = ["v1_logit", "ens_logit", "ce_min", "ce_absdiff", "ce_sign_agree"]
L1_FEATS = ["v2_gap_next", "v2_share", "v1_gap_top", "v1_rank", "ens_gap_top", "ens_rank", "ens_gap_next",
            "ens_share", "ens_entropy", "ens_z", "rank_shift"]
L2_FEATS = ["rec_rank", "rec_gap_best", "rec_n_pos", "rec_share", "mutual_best"]
LEVELS = {"BASE": BASE_FEATS, "L0": L0_FEATS, "L1": L1_FEATS, "L2": L2_FEATS}
V1_DEPENDENT = {"v1_logit", "ce_min", "ce_absdiff", "ce_sign_agree", "v1_gap_top", "v1_rank"}

TRAIN_ALL_ROLES = ["CE-FIT", "CE-DEV", "TREE-FIT", "TREE-DEV", "CALIBRATION", "HOLDOUT"]
FEATURE_ROLES = {"train": ["TREE-FIT", "TREE-DEV", "CALIBRATION", "HOLDOUT"], "test": ["TEST"]}


# --------------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------------
def read_logits(logits_root: str, split: str, roles, country: str, col_out: str) -> pd.DataFrame:
    """Committed CE logit parts under <logits_root>/<split>/<ROLE>/<COUNTRY>/part-*.parquet (b4_ce layout).
    A part counts only if its done-*.json exists; if a folder has parts but NO done files at all (e.g. a
    notebook that wrote consolidated files), all parts are read and a warning is printed."""
    frames = []
    for role in roles:
        d = os.path.join(logits_root, split, role, country)
        if not os.path.isdir(d):
            continue
        dones = sorted(glob.glob(os.path.join(d, "done-*.json")))
        if dones:
            parts = [p.replace("done-", "part-").replace(".json", ".parquet") for p in dones]
            parts = [p for p in parts if os.path.exists(p)]
        else:
            parts = sorted(glob.glob(os.path.join(d, "*.parquet")))
            if parts:
                print(f"  WARNING: {d} has parquet files but no done-*.json; reading all {len(parts)}")
        for p in parts:
            t = pq.read_table(p, columns=["s1_id", "cand_id", "ce_logit"]).to_pandas()
            frames.append(t)
    if not frames:
        return pd.DataFrame({"s1_id": pd.Series(dtype=object), "cand_id": pd.Series(dtype=object),
                             col_out: pd.Series(dtype=np.float32)})
    out = pd.concat(frames, ignore_index=True).rename(columns={"ce_logit": col_out})
    out[col_out] = out[col_out].astype(np.float32)
    dup = out.duplicated(["s1_id", "cand_id"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (s1_id, cand_id) logits under {logits_root}/{split}/{country}")
    return out


def read_b2(b2p, split: str, roles, country: str, k: int) -> pd.DataFrame:
    cols = ["s1_id", "cand_id", "source", "rank", "role"]
    frames = []
    for _meta, df in b2p.iter_candidates(split, roles=roles, countries=[country], k=k):
        if len(df):
            frames.append(df[[c for c in cols if c in df.columns]].copy())
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["rank"] = out["rank"].astype(np.int32)
    return out


# --------------------------------------------------------------------------------------------
# Grouped statistics (numpy, O(n log n); NaN values do not take part and get NaN outputs)
# --------------------------------------------------------------------------------------------
def group_stats(codes: np.ndarray, x: np.ndarray) -> dict:
    """Per-row statistics of x within its group (codes). Rows with NaN x are excluded from every group
    aggregate and receive NaN. Returns arrays aligned to the input rows."""
    n = len(x)
    valid = ~np.isnan(x)
    out = {k: np.full(n, np.nan, np.float32) for k in
           ("max", "gap_top", "rank", "gap_next", "other_best", "share", "n_valid", "n_pos", "mean", "std",
            "entropy")}
    vi = np.flatnonzero(valid)
    if len(vi) == 0:
        return out
    c, v = codes[vi], x[vi].astype(np.float64)
    order = np.lexsort((-v, c))
    cs, vs = c[order], v[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    gid = np.cumsum(start) - 1
    starts = np.flatnonzero(start)
    size = np.diff(np.r_[starts, len(cs)])
    gmax = vs[starts]
    rank = np.arange(len(cs)) - starts[gid] + 1
    second = np.full(len(starts), np.nan)
    has2 = size >= 2
    second[has2] = vs[starts[has2] + 1]
    # best among the OTHER members: if I am rank 1 -> second best, else -> the max
    other_best = np.where(rank == 1, second[gid], gmax[gid])
    nxt = np.r_[vs[1:], np.nan]
    last = np.r_[start[1:], True]
    nxt[last] = np.nan
    e = np.exp(vs - gmax[gid])
    esum = np.add.reduceat(e, starts)
    share = e / esum[gid]
    ent = np.add.reduceat(-share * np.log(np.maximum(share, 1e-300)), starts)
    gsum = np.add.reduceat(vs, starts)
    gsq = np.add.reduceat(vs * vs, starts)
    mean = gsum / size
    std = np.sqrt(np.maximum(gsq / size - mean * mean, 0.0))
    npos = np.add.reduceat((vs > 0).astype(np.float64), starts)
    back = vi[order]
    out["max"][back] = gmax[gid]
    out["gap_top"][back] = gmax[gid] - vs
    out["rank"][back] = rank
    out["gap_next"][back] = vs - nxt
    out["other_best"][back] = other_best
    out["share"][back] = share
    out["n_valid"][back] = size[gid]
    out["n_pos"][back] = npos[gid]
    out["mean"][back] = mean[gid]
    out["std"][back] = std[gid]
    out["entropy"][back] = ent[gid]
    return out


def _record_codes(df: pd.DataFrame) -> np.ndarray:
    """Integer code per (source, cand_id) record without building 60M python strings."""
    h = pd.util.hash_array(df["cand_id"].astype(str).to_numpy(dtype=object))
    src = pd.factorize(df["source"].astype(str))[0].astype(np.uint64) if "source" in df.columns \
        else np.zeros(len(df), np.uint64)
    return pd.factorize(h ^ (src * np.uint64(0x9E3779B97F4A7C15)))[0]


# --------------------------------------------------------------------------------------------
# Feature builder for one (split, country)
# --------------------------------------------------------------------------------------------
def build_ctx(base: pd.DataFrame, has_v1: bool) -> pd.DataFrame:
    """base: Block 2 rows of EVERY role of the split (s1_id, cand_id, source, rank, role) left-joined with
    v2_logit (+ v1_logit). Returns base + all context columns. Rows with no logit (past K_CE, or a role
    that was never CE-scored) get NaN features and take no part in any group."""
    df = base.reset_index(drop=True)
    x2 = df["v2_logit"].to_numpy(np.float64)
    x1 = df["v1_logit"].to_numpy(np.float64) if has_v1 else np.full(len(df), np.nan)
    ens = np.where(np.isnan(x1), x2, np.where(np.isnan(x2), x1, (x1 + x2) / 2.0))
    s1c = pd.factorize(df["s1_id"])[0]
    F = {}
    # ---- BASE (v2 list context)
    g2 = group_stats(s1c, x2)
    F["v2_logit"] = x2.astype(np.float32)
    F["v2_gap_top"] = g2["gap_top"]
    F["v2_rank"] = g2["rank"]
    F["v2_n_pos"] = g2["n_pos"]
    F["v2_n_scored"] = g2["n_valid"]
    # ---- L0
    F["v1_logit"] = x1.astype(np.float32)
    F["ens_logit"] = ens.astype(np.float32)
    both = ~np.isnan(x1) & ~np.isnan(x2)
    F["ce_min"] = np.where(both, np.minimum(x1, x2), np.nan).astype(np.float32)
    F["ce_absdiff"] = np.where(both, np.abs(x1 - x2), np.nan).astype(np.float32)
    F["ce_sign_agree"] = np.where(both, ((x1 > 0) == (x2 > 0)).astype(np.float32), np.nan).astype(np.float32)
    # ---- L1
    F["v2_gap_next"] = g2["gap_next"]
    F["v2_share"] = g2["share"]
    g1 = group_stats(s1c, x1)
    F["v1_gap_top"] = g1["gap_top"]
    F["v1_rank"] = g1["rank"]
    ge = group_stats(s1c, ens)
    F["ens_gap_top"] = ge["gap_top"]
    F["ens_rank"] = ge["rank"]
    F["ens_gap_next"] = ge["gap_next"]
    F["ens_share"] = ge["share"]
    F["ens_entropy"] = ge["entropy"]
    with np.errstate(invalid="ignore", divide="ignore"):
        F["ens_z"] = np.where(ge["std"] > 1e-6, (ens - ge["mean"]) / ge["std"], 0.0).astype(np.float32)
    F["ens_z"][np.isnan(ens)] = np.nan
    F["rank_shift"] = (df["rank"].to_numpy(np.float32) - ge["rank"]).astype(np.float32)
    # ---- L2 (record claimants, across every S1 of the split in this country)
    rc = _record_codes(df)
    gr = group_stats(rc, ens)
    F["rec_rank"] = gr["rank"]
    F["rec_gap_best"] = (ens - gr["other_best"]).astype(np.float32)   # >0: margin over runner-up; <0: below best
    F["rec_n_pos"] = gr["n_pos"]
    F["rec_share"] = gr["share"]
    npos_list = np.maximum(np.nan_to_num(ge["n_pos"], nan=0.0), 1.0)
    F["mutual_best"] = np.where(np.isnan(ens), np.nan,
                                ((gr["rank"] == 1) & (ge["rank"] <= npos_list)).astype(np.float32)).astype(np.float32)
    for k, v in F.items():
        df[k] = np.asarray(v, dtype=np.float32)
    return df


def ctx_path(root: str, version: str, split: str, role: str, country: str) -> str:
    return os.path.join(root, "block4", "ctx", version, split, role, f"{country}.parquet")


def run_ctx(root, version, b2p, split, countries, k, v2_root, v1_root=None, all_roles=None,
            overwrite=False) -> list[dict]:
    """Build and write context features for every FEATURE role of `split`, per country. The claimant groups
    (L2) use ALL roles of the split that have logits, so claimant density is as close to test as possible."""
    all_roles = all_roles or (TRAIN_ALL_ROLES if split == "train" else ["TEST"])
    out_roles = FEATURE_ROLES[split]
    stats = []
    for country in countries:
        todo = [r for r in out_roles
                if overwrite or not os.path.exists(ctx_path(root, version, split, r, country).replace(".parquet", ".done.json"))]
        if not todo:
            print(f"  ctx {split}/{country}: already committed, skip")
            continue
        t0 = time.time()
        base = read_b2(b2p, split, all_roles, country, k)
        if len(base) == 0:
            print(f"  ctx {split}/{country}: no Block 2 rows")
            continue
        l2 = read_logits(v2_root, split, all_roles, country, "v2_logit")
        base = base.merge(l2, on=["s1_id", "cand_id"], how="left")
        has_v1 = False
        if v1_root:
            l1 = read_logits(v1_root, split, all_roles, country, "v1_logit")
            has_v1 = len(l1) > 0
            base = base.merge(l1, on=["s1_id", "cand_id"], how="left") if has_v1 else base.assign(v1_logit=np.nan)
        else:
            base["v1_logit"] = np.nan
        cov = (base.assign(_v2=base["v2_logit"].notna(), _v1=base["v1_logit"].notna())
               .groupby("role")[["_v2", "_v1"]].mean().round(4).to_dict("index"))
        feats = build_ctx(base, has_v1)
        cols = ["s1_id", "cand_id"] + BASE_FEATS + L0_FEATS + L1_FEATS + L2_FEATS
        for role in todo:
            part = feats.loc[feats["role"] == role, cols]
            p = ctx_path(root, version, split, role, country)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            part.to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, p)
            info = {"rows": int(len(part)), "has_v1": has_v1, "logit_coverage_by_role": cov,
                    "claimant_roles": all_roles, "code_version": CTX_CODE_VERSION, "time": time.strftime("%F %T")}
            json.dump(info, open(p.replace(".parquet", ".done.json"), "w"), default=str)
            stats.append({"split": split, "role": role, "country": country, "rows": len(part)})
        print(f"  ctx {split}/{country}: {len(base):,} rows over roles {sorted(base['role'].unique())}, "
              f"v1={has_v1}, wrote {todo} in {time.time() - t0:.0f}s; coverage {cov}")
    return stats


def load_ctx(root, version, split, role, countries, columns=None) -> pd.DataFrame:
    frames = []
    for c in countries:
        p = ctx_path(root, version, split, role, c)
        if not os.path.exists(p.replace(".parquet", ".done.json")):
            raise FileNotFoundError(f"ctx not committed: {p} (run Stage 2 for {split}/{c})")
        frames.append(pq.read_table(p, columns=columns).to_pandas())
    return pd.concat(frames, ignore_index=True)


def ctx_has_v1(root, version, split, role, countries) -> bool:
    ok = []
    for c in countries:
        p = ctx_path(root, version, split, role, c).replace(".parquet", ".done.json")
        if os.path.exists(p):
            ok.append(json.load(open(p)).get("has_v1", False))
    return bool(ok) and all(ok)


# --------------------------------------------------------------------------------------------
# Parity: HOLDOUT (train) vs TEST, same countries. Drop features that shift.
# --------------------------------------------------------------------------------------------
def parity_report(root, version, countries=("INDIA", "US"), feats=None, n=300_000, ks_max=0.10, seed=0):
    from scipy.stats import ks_2samp
    feats = feats or (BASE_FEATS + L0_FEATS + L1_FEATS + L2_FEATS)
    rows = []
    for c in countries:
        try:
            a = load_ctx(root, version, "train", "HOLDOUT", [c], columns=feats)
            b = load_ctx(root, version, "test", "TEST", [c], columns=feats)
        except FileNotFoundError as e:
            print("  parity:", e)
            continue
        a = a.sample(min(n, len(a)), random_state=seed)
        b = b.sample(min(n, len(b)), random_state=seed)
        for f in feats:
            x, y = a[f].dropna().to_numpy(), b[f].dropna().to_numpy()
            if len(x) < 100 or len(y) < 100:
                rows.append({"country": c, "feature": f, "ks": np.nan, "nan_holdout": a[f].isna().mean(),
                             "nan_test": b[f].isna().mean()})
                continue
            ks = ks_2samp(x, y).statistic
            rows.append({"country": c, "feature": f, "ks": float(ks),
                         "p50_holdout": float(np.median(x)), "p50_test": float(np.median(y)),
                         "p99_holdout": float(np.quantile(x, .99)), "p99_test": float(np.quantile(y, .99)),
                         "nan_holdout": float(a[f].isna().mean()), "nan_test": float(b[f].isna().mean())})
    rep = pd.DataFrame(rows)
    if rep.empty:
        return rep, []
    worst = rep.groupby("feature")["ks"].max()
    nan_gap = (rep["nan_test"] - rep["nan_holdout"]).abs().groupby(rep["feature"]).max()
    drop = sorted(set(worst[worst > ks_max].index) | set(nan_gap[nan_gap > 0.05].index))
    return rep, drop


def france_view(root, version, feats=("ens_logit", "ce_absdiff", "rec_n_pos", "rec_gap_best", "mutual_best"), n=300_000):
    """Label-free look at FRANCE vs US test on the signals that matter for the one-owner problem."""
    out = []
    for c in ("US", "INDIA", "FRANCE"):
        try:
            d = load_ctx(root, version, "test", "TEST", [c], columns=list(feats))
        except FileNotFoundError:
            continue
        d = d.sample(min(n, len(d)), random_state=0)
        row = {"country": c}
        for f in feats:
            v = d[f].dropna()
            row[f + "_p50"] = float(v.median()) if len(v) else np.nan
            row[f + "_p90"] = float(v.quantile(.9)) if len(v) else np.nan
        out.append(row)
    return pd.DataFrame(out)
