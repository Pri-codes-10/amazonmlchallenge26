"""
er_eval.py - shared evaluation and Block 1 record access for Blocks 3-6.

One implementation of the competition metric (S1-macro F0.5 with the singleton rule: an S1 with no
true match scores 1.0 if nothing is accepted and 0.0 on any accept; an S1 with matches scores F0.5
of the accepted set against ALL its true matches, so retrieval misses cost recall), plus the global
threshold search, slice breakdowns, paired bootstrap, run log and the fixed proxy CatBoost.

Labels are read ONLY here, for evaluation. Nothing in this module produces a feature.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# The single definition of the Block 1 output subfolder: notebook 01 writes block1/<B1_VERSION>/,
# every later block reads it from there.
B1_VERSION = "v2"
ROOT: str | None = None


# --------------------------------------------------------------------------------------------
# Paths / records
# --------------------------------------------------------------------------------------------
def workspace_root() -> str:
    """The workspace every block reads and writes; the notebooks set $BER_WORKSPACE to it."""
    root = os.environ.get("BER_WORKSPACE")
    if not root:
        raise FileNotFoundError("set os.environ['BER_WORKSPACE'] to the workspace folder "
                                "before importing the pipeline modules (see any notebook's first cell)")
    return root.rstrip("/")


def set_root(root: str) -> str:
    global ROOT
    ROOT = root
    return root


def b1_path(*parts) -> str:
    assert ROOT, "call er_eval.set_root(...) first"
    return os.path.join(ROOT, "block1", B1_VERSION, *parts)


def read_records(split: str, source: str, columns, countries=None, roles=None, ids=None) -> pd.DataFrame:
    """Block 1 prepared records for one (split, source), optionally filtered by country / role.
    Filtering happens inside pyarrow, so only the requested rows and columns are materialised."""
    d = b1_path("records", f"{split}_{source.lower()}")
    files = sorted(glob.glob(os.path.join(d, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(d)
    dset = ds.dataset(files, format="parquet")
    flt = None
    if countries is not None:
        flt = ds.field("country_norm").isin(list(countries))
    if roles is not None:
        f2 = ds.field("entity_role").isin(list(roles))
        flt = f2 if flt is None else (flt & f2)
    if ids is not None:
        f3 = ds.field("entity_id").isin(list(ids))
        flt = f3 if flt is None else (flt & f3)
    return dset.to_table(columns=list(columns), filter=flt).to_pandas()


# --------------------------------------------------------------------------------------------
# Deterministic S1 subsample (identical in every notebook / session / block)
# --------------------------------------------------------------------------------------------
def s1_hash_unit(ids) -> np.ndarray:
    """Stable hash of each id mapped to [0, 1). Same id -> same value everywhere."""
    h = pd.util.hash_array(np.asarray(ids, dtype=object))
    return (h % np.uint64(1_000_000)).astype(np.float64) / 1e6


def in_s1_subsample(ids, frac: float | None) -> np.ndarray:
    if frac is None or frac >= 1.0:
        return np.ones(len(ids), dtype=bool)
    return s1_hash_unit(ids) < frac


# --------------------------------------------------------------------------------------------
# Evaluation universe: every S1 of the evaluated roles (singletons included)
# --------------------------------------------------------------------------------------------
def load_eval_universe(roles, countries=None, s1_frac=None) -> pd.DataFrame:
    """One row per train S1 in `roles`: s1_id, country, n_true, has_cross, singleton.
    n_true counts ALL true matches (retrieved or not)."""
    s1 = read_records("train", "s1", ["entity_id", "country_norm"], countries=countries, roles=roles)
    s1 = s1.rename(columns={"entity_id": "s1_id", "country_norm": "country"}).drop_duplicates("s1_id")
    s1 = s1[in_s1_subsample(s1["s1_id"].to_numpy(), s1_frac)]
    pp = pd.read_parquet(b1_path("labels", "positive_pairs.parquet"),
                         columns=["source1_entity_id", "cross_script"])
    g = pp.groupby("source1_entity_id").agg(n_true=("cross_script", "size"),
                                            has_cross=("cross_script", "max"))
    u = s1.join(g, on="s1_id")
    u["n_true"] = u["n_true"].fillna(0).astype(np.int32)
    u["has_cross"] = u["has_cross"].fillna(False).astype(bool)
    u["singleton"] = u["n_true"] == 0
    u["country"] = u["country"].astype(object)
    return u.reset_index(drop=True)


def f_from_counts(tp, npred, ntrue, beta: float = 0.5) -> np.ndarray:
    """Per-S1 F-beta with the competition's singleton rule."""
    b2 = beta * beta
    tp = np.asarray(tp, dtype=np.float64)
    npred = np.asarray(npred, dtype=np.float64)
    ntrue = np.asarray(ntrue, dtype=np.float64)
    f = np.zeros(len(tp), dtype=np.float64)
    single = ntrue == 0
    f[single] = (npred[single] == 0).astype(np.float64)
    m = (~single) & (tp > 0)
    p = tp[m] / npred[m]
    r = tp[m] / ntrue[m]
    f[m] = (1 + b2) * p * r / (b2 * p + r)
    return f


class S1Scorer:
    """Scores candidate pairs against a fixed S1 universe."""

    def __init__(self, universe: pd.DataFrame):
        self.u = universe.reset_index(drop=True)
        self.index = pd.Index(self.u["s1_id"].to_numpy(dtype=object))
        assert self.index.is_unique
        self.n_true = self.u["n_true"].to_numpy(np.int64)
        self.U = len(self.u)

    def codes(self, s1_ids) -> np.ndarray:
        c = self.index.get_indexer(np.asarray(s1_ids, dtype=object))
        if (c < 0).any():
            raise ValueError(f"{int((c < 0).sum())} pair rows have an S1 outside the eval universe "
                             "(wrong roles/countries/s1_frac?)")
        return c

    def counts(self, codes, pred, label):
        npred = np.bincount(codes, weights=pred.astype(np.float64), minlength=self.U)
        tp = np.bincount(codes, weights=(pred & (label == 1)).astype(np.float64), minlength=self.U)
        return tp, npred

    def per_s1_f(self, codes, score, label, t, beta=0.5) -> np.ndarray:
        tp, npred = self.counts(codes, score >= t, label)
        return f_from_counts(tp, npred, self.n_true, beta)

    def check_labels(self, codes, label):
        pos = np.bincount(codes, weights=(label == 1).astype(np.float64), minlength=self.U)
        bad = pos > self.n_true
        if bad.any():
            raise ValueError(f"{int(bad.sum())} S1 have more labelled candidates than true matches")
        return pos

    def best_threshold(self, codes, score, label, beta=0.5, n_coarse=200, n_fine=60):
        """Global threshold maximising mean per-S1 F-beta. Two-pass quantile grid."""
        score = np.asarray(score, dtype=np.float64)
        q = np.concatenate([np.linspace(0.0, 0.80, 41), np.linspace(0.80, 0.9995, n_coarse)])
        grid = np.unique(np.quantile(score, q))
        grid = np.append(grid, np.nextafter(score.max(), np.inf))  # "accept nothing"
        res = [(t, self.per_s1_f(codes, score, label, t, beta).mean()) for t in grid]
        i = int(np.argmax([r[1] for r in res]))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)]
        if hi > lo:
            fine = np.linspace(lo, hi, n_fine)
            res += [(t, self.per_s1_f(codes, score, label, t, beta).mean()) for t in fine]
        best_t, best = max(res, key=lambda r: r[1])
        curve = pd.DataFrame(res, columns=["threshold", "s1_macro_f05"]).sort_values("threshold")
        return float(best_t), float(best), curve


def _pair_metrics(score, label):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    if label.sum() == 0 or label.sum() == len(label):
        return {"pr_auc": float("nan"), "pair_f05_best": float("nan")}
    ap = average_precision_score(label, score)
    p, r, _ = precision_recall_curve(label, score)
    f = 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12)
    return {"pr_auc": float(ap), "pair_f05_best": float(np.nanmax(f))}


def evaluate(scorer: S1Scorer, pairs: pd.DataFrame, score_col: str = "score",
             pair_slice_cols=("country", "pair_cross_script"), threshold: float | None = None,
             beta: float = 0.5) -> tuple[dict, np.ndarray]:
    """Competition metric + breakdowns.

    pairs: one row per candidate pair with s1_id, label, <score_col> (+ optional slice columns).
    threshold: None = search the best global threshold on these pairs (TREE-DEV / CE-DEV use);
               a float = apply a locked threshold (CALIBRATION -> HOLDOUT use).
    Returns (report dict, per-S1 F vector aligned with scorer.u)."""
    codes = scorer.codes(pairs["s1_id"].to_numpy())
    label = pairs["label"].to_numpy().astype(np.int8)
    score = pairs[score_col].to_numpy().astype(np.float64)
    score = np.where(np.isnan(score), -np.inf, score)
    scorer.check_labels(codes, label)
    if threshold is None:
        t = scorer.best_threshold(codes, score, label, beta)[0]
    else:
        t = float(threshold)
    f = scorer.per_s1_f(codes, score, label, t, beta)
    u = scorer.u
    rep = {
        "threshold": t,
        "s1_macro_f05": float(f.mean()),
        "n_s1": int(scorer.U),
        "n_pairs": int(len(pairs)),
        "retrieval_recall_ceiling": float((scorer.check_labels(codes, label).sum()) / max(u.n_true.sum(), 1)),
        "slices_s1": {},
        "slices_pair": {},
    }
    fin = np.isfinite(score)
    rep.update(_pair_metrics(score[fin], label[fin]))
    for name, mask in [("singleton", u.singleton.to_numpy()),
                       ("matched", ~u.singleton.to_numpy()),
                       ("cross_script_s1", u.has_cross.to_numpy()),
                       ("same_script_s1", (~u.has_cross.to_numpy()) & (~u.singleton.to_numpy()))]:
        if mask.any():
            rep["slices_s1"][name] = {"s1_macro_f05": float(f[mask].mean()), "n_s1": int(mask.sum())}
    for c in sorted(u["country"].dropna().unique()):
        m = (u["country"] == c).to_numpy()
        rep["slices_s1"][f"country={c}"] = {"s1_macro_f05": float(f[m].mean()), "n_s1": int(m.sum())}
        for name, extra in [("singleton", u.singleton.to_numpy()), ("cross_script_s1", u.has_cross.to_numpy())]:
            mm = m & extra
            if mm.any():
                rep["slices_s1"][f"country={c}|{name}"] = {"s1_macro_f05": float(f[mm].mean()),
                                                           "n_s1": int(mm.sum())}
    for col in pair_slice_cols:
        if col in pairs.columns:
            v = pairs[col].to_numpy()
            for val in pd.unique(v):
                m = (v == val) & fin
                if m.sum() > 0:
                    rep["slices_pair"][f"{col}={val}"] = _pair_metrics(score[m], label[m]) | {"n": int(m.sum())}
    # accept-count diagnostics (how many candidates the model accepts per S1)
    tp, npred = scorer.counts(codes, score >= t, label)
    rep["accepted_per_s1_mean"] = float(npred.mean())
    rep["singletons_with_false_accept"] = int(((scorer.n_true == 0) & (npred > 0)).sum())
    return rep, f


def rerank_metrics(pairs: pd.DataFrame, score_col: str) -> dict:
    """For S1 with >=1 positive in its list: is a true match at position 1 / within 3 when the list
    is re-sorted by score_col, compared with the Block 2 (bi-encoder) order."""
    d = pairs[["s1_id", "rank", "label", score_col]].copy()
    has_pos = d.groupby("s1_id")["label"].transform("max") == 1
    d = d[has_pos]
    out = {}
    for name, key, asc in [("biencoder", "rank", True), ("model", score_col, False)]:
        d = d.sort_values(["s1_id", key], ascending=[True, asc], kind="stable")
        pos = d.groupby("s1_id").cumcount()
        top1 = d[pos == 0].groupby("s1_id")["label"].max()
        top3 = d[pos < 3].groupby("s1_id")["label"].max()
        out[f"{name}_hit@1"] = float(top1.mean())
        out[f"{name}_hit@3"] = float(top3.mean())
    out["n_s1_with_pos_in_list"] = int(d["s1_id"].nunique())
    return out


def bootstrap_ci(f: np.ndarray, n: int = 1000, seed: int = 0, chunk: int = 50):
    rng = np.random.default_rng(seed)
    means = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(f), size=(min(chunk, n - s), len(f)))
        means.append(f[idx].mean(axis=1))
    m = np.concatenate(means)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_bootstrap(f_base: np.ndarray, f_new: np.ndarray, n: int = 1000, seed: int = 0, chunk: int = 50):
    """Delta (new - base) of S1-macro F0.5 with a 95% CI, resampling S1s. Both vectors must be
    aligned to the same universe (same roles, countries, s1_frac)."""
    assert len(f_base) == len(f_new)
    d = f_new - f_base
    rng = np.random.default_rng(seed)
    means = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, n - s), len(d)))
        means.append(d[idx].mean(axis=1))
    m = np.concatenate(means)
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "p_not_better": float((m <= 0).mean()),
            "verdict": ("BETTER" if np.percentile(m, 2.5) > 0 else
                        "WORSE" if np.percentile(m, 97.5) < 0 else "WITHIN NOISE")}


# --------------------------------------------------------------------------------------------
# Run log
# --------------------------------------------------------------------------------------------
def universe_fingerprint(scorer: S1Scorer) -> str:
    return hashlib.sha1("\n".join(scorer.u["s1_id"].astype(str)).encode()).hexdigest()[:12]


def log_run(report_dir: str, run_id: str, info: dict, f_vec: np.ndarray, scorer: S1Scorer):
    os.makedirs(os.path.join(report_dir, "per_s1"), exist_ok=True)
    np.save(os.path.join(report_dir, "per_s1", f"{run_id}.npy"), f_vec.astype(np.float32))
    rec = {"run_id": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "universe": universe_fingerprint(scorer)} | info
    with open(os.path.join(report_dir, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    return rec


def load_runs(report_dir: str) -> pd.DataFrame:
    p = os.path.join(report_dir, "runs.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return pd.json_normalize(rows)


def compare_runs(report_dir: str, base_id: str, new_id: str, **kw):
    fa = np.load(os.path.join(report_dir, "per_s1", f"{base_id}.npy")).astype(np.float64)
    fb = np.load(os.path.join(report_dir, "per_s1", f"{new_id}.npy")).astype(np.float64)
    return paired_bootstrap(fa, fb, **kw)


def summary_table(report_dir: str) -> pd.DataFrame:
    """Compact leaderboard of every logged run (what to send back after each round)."""
    r = load_runs(report_dir)
    if r.empty:
        return r
    keep = ["run_id", "variant", "n_features", "s1_macro_f05", "ci95_low", "ci95_high", "pr_auc",
            "threshold", "slices_s1.country=INDIA.s1_macro_f05", "slices_s1.country=US.s1_macro_f05",
            "slices_s1.cross_script_s1.s1_macro_f05", "slices_s1.singleton.s1_macro_f05",
            "delta_vs_ref.delta", "delta_vs_ref.verdict"]
    cols = [c for c in keep if c in r.columns]
    return r[cols].rename(columns=lambda c: c.replace("slices_s1.", "").replace(".s1_macro_f05", ""))


# --------------------------------------------------------------------------------------------
# Proxy CatBoost (fixed across variants - only the feature list changes)
# --------------------------------------------------------------------------------------------
PROXY_PARAMS = dict(depth=6, learning_rate=0.1, iterations=1000, od_type="Iter", od_wait=50,
                    eval_metric="Logloss", random_seed=42, verbose=200, allow_writing_files=False)


def _prep_X(df, features, cat_features):
    X = df[features].copy()
    for c in features:
        if c in cat_features:
            X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X


def train_proxy(train_df: pd.DataFrame, dev_df: pd.DataFrame, features, cat_features=(),
                task_type: str = "CPU", params: dict | None = None):
    """Fixed-parameter CatBoost: train on TREE-FIT, early-stop on TREE-DEV (Logloss).
    Returns (model, dev probabilities, feature importance DataFrame)."""
    from catboost import CatBoostClassifier, Pool
    features = list(features)
    cats = [c for c in features if c in set(cat_features)]
    p = dict(PROXY_PARAMS) | {"task_type": task_type} | (params or {})
    if task_type == "CPU":
        p.setdefault("thread_count", -1)
    ptr = Pool(_prep_X(train_df, features, cats), train_df["label"].astype(int).to_numpy(), cat_features=cats)
    pdv = Pool(_prep_X(dev_df, features, cats), dev_df["label"].astype(int).to_numpy(), cat_features=cats)
    model = CatBoostClassifier(**p)
    model.fit(ptr, eval_set=pdv, use_best_model=True)
    prob = model.predict_proba(pdv)[:, 1]
    imp = model.get_feature_importance(prettified=True)
    return model, prob, imp


def proxy_run(run_id: str, train_df, dev_df, features, cat_features, scorer: S1Scorer, report_dir: str,
              variant: str = "", ref_run: str | None = None, task_type: str = "CPU", params=None,
              extra: dict | None = None):
    """Train the proxy, score TREE-DEV with the competition metric, log everything."""
    t0 = time.time()
    model, prob, imp = train_proxy(train_df, dev_df, features, cat_features, task_type, params)
    ev = dev_df[["s1_id", "label"] + [c for c in ("country", "pair_cross_script", "rank") if c in dev_df.columns]].copy()
    ev["prob"] = prob
    rep, f = evaluate(scorer, ev, "prob")
    lo, hi = bootstrap_ci(f)
    info = {"variant": variant, "n_features": len(features), "features": list(features),
            "best_iteration": int(model.get_best_iteration() or 0), "train_rows": int(len(train_df)),
            "elapsed_s": round(time.time() - t0, 1), "ci95_low": lo, "ci95_high": hi,
            "importance_top15": imp.head(15).values.tolist()} | rep | (extra or {})
    if ref_run and os.path.exists(os.path.join(report_dir, "per_s1", f"{ref_run}.npy")):
        info["delta_vs_ref"] = compare_runs_vec(report_dir, ref_run, f) | {"ref": ref_run}
    log_run(report_dir, run_id, info, f, scorer)
    return model, prob, info


def compare_runs_vec(report_dir, base_id, f_new):
    fa = np.load(os.path.join(report_dir, "per_s1", f"{base_id}.npy")).astype(np.float64)
    return paired_bootstrap(fa, f_new)


