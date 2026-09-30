"""
b5_catboost.py - Block 5: one CatBoost pair model on Block 3 features + Block 4 context
(b4_ctx levels BASE / L0 / L1 / L2), one consolidated parquet of probabilities per
(split, role, country), and the Block 6 cross-fit accept policy.

Stages hand off only through files under <ROOT>/block5/<out_version>/, each written to a tmp name
and renamed with a marker written last, so a re-run skips what is already committed.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import b4_ctx as cx

B5_CODE_VERSION = "b5-catboost-1.0"
CFG: dict = {}
PASS = ["s1_id", "cand_id", "source", "country", "role", "label", "dense_rank"]
EXTRA_OUT = ["name_core_tset", "addr_tset", "cand_fanin", "cand_is_best_s1", "cand_best_margin",
             "v2_logit", "v1_logit", "ens_logit", "ce_absdiff", "rec_rank", "rec_gap_best", "mutual_best"]


def configure(**kw) -> dict:
    """root, b3_version, b3_selected (json path or None), ctx_version, out_version,
    task_type ('GPU'|'CPU'), countries_train, countries_test."""
    CFG.clear()
    CFG.update(dict(out_version="final", ctx_version="v1", task_type="GPU", countries_train=["INDIA", "US"],
                    countries_test=["INDIA", "US", "FRANCE"], b3_selected=None))
    CFG.update(kw)
    return CFG


def P(*parts) -> str:
    return os.path.join(CFG["root"], "block5", CFG["out_version"], *parts)


def mark(path: str, info: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(dict(info or {}, time=time.strftime("%F %T")), open(tmp, "w"), default=str, indent=1)
    os.replace(tmp, path)


def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------
def b3_feature_list(b3) -> list[str]:
    sel = CFG.get("b3_selected")
    if sel and os.path.exists(sel):
        j = json.load(open(sel))
        feats = j["features"] if isinstance(j, dict) else list(j)
        return [f for f in feats if f in b3.FEATURE_NAMES]
    return list(b3.FEATURE_NAMES)


def ctx_levels(has_v1: bool) -> dict:
    lv = {k: list(v) for k, v in cx.LEVELS.items()}
    if not has_v1:
        for k in lv:
            lv[k] = [f for f in lv[k] if f not in cx.V1_DEPENDENT]
    return lv


def variant_features(b3_feats, levels: dict, upto: str) -> list[str]:
    order = ["BASE", "L0", "L1", "L2"]
    out = list(b3_feats)
    for lv in order[: order.index(upto) + 1]:
        out += levels[lv]
    return out


def load_role(b3, split, role, countries, s1_frac=None, b3_cols=None) -> pd.DataFrame:
    """Block 3 features (committed parts) inner-joined with Block 4 context on (s1_id, cand_id)."""
    t0 = time.time()
    cols = None if b3_cols is None else list(dict.fromkeys(["s1_id", "cand_id", "role", "country", "split", "label",
                                                            "source", "dense_rank"] + list(b3_cols)))
    f = b3.load_features(CFG["b3_version"], split, [role], countries, columns=cols, s1_frac=s1_frac)
    c = cx.load_ctx(CFG["root"], CFG["ctx_version"], split, role, countries)
    n = len(f)
    df = f.merge(c, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    miss = df["v2_logit"].isna().mean()
    if len(c) < 0.98 * n:
        print(f"  WARNING {split}/{role}: ctx has {len(c):,} rows for {n:,} Block 3 rows")
    for col in b3.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    print(f"  loaded {split}/{role} {countries}: {len(df):,} rows, {df['s1_id'].nunique():,} S1, "
          f"v2 logit missing {miss:.2%} ({time.time() - t0:.0f}s)")
    return df


# --------------------------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------------------------
def scorer_for(ev, df: pd.DataFrame, s1_frac=None):
    """Universe = every S1 of the role(s)/countries present in df (singletons with no rows included).
    Pass the same s1_frac the rows were subsampled with."""
    roles = sorted(df["role"].astype(str).unique())
    countries = sorted(df["country"].astype(str).unique())
    return ev.S1Scorer(ev.load_eval_universe(roles, countries=countries, s1_frac=s1_frac))


def s1_macro(ev, scorer, df, prob, threshold=None):
    codes = scorer.codes(df["s1_id"].to_numpy())
    lab = df["label"].to_numpy().astype(np.int8)
    p = np.where(np.isnan(prob), -np.inf, prob)
    if threshold is None:
        t, best, _ = scorer.best_threshold(codes, p, lab)
    else:
        t = float(threshold)
    f = scorer.per_s1_f(codes, p, lab, t)
    return float(f.mean()), t, f


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
CB_DEFAULT = dict(depth=8, learning_rate=0.08, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5,
                  border_count=254, iterations=4000)


def _cb_X(df, feats, cats):
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X


def fit_cb(train, dev, feats, cats, params, seed=0, od_wait=150, verbose=500):
    from catboost import CatBoostClassifier, Pool
    cats = [c for c in feats if c in set(cats)]
    p = dict(CB_DEFAULT) | dict(params or {})
    p.update(random_seed=seed, od_type="Iter", od_wait=od_wait, eval_metric="Logloss",
             task_type=CFG.get("task_type", "GPU"), verbose=verbose, allow_writing_files=False)
    if p["task_type"] == "GPU":
        p.setdefault("devices", "0")
    else:
        p.setdefault("thread_count", -1)
    m = CatBoostClassifier(**p)
    m.fit(Pool(_cb_X(train, feats, cats), train["label"].astype(int).to_numpy(), cat_features=cats),
          eval_set=Pool(_cb_X(dev, feats, cats), dev["label"].astype(int).to_numpy(), cat_features=cats),
          use_best_model=True)
    return m


def pred_cb(m, df, feats, cats):
    from catboost import Pool
    cats = [c for c in feats if c in set(cats)]
    return m.predict_proba(Pool(_cb_X(df, feats, cats), cat_features=cats))[:, 1]


# --------------------------------------------------------------------------------------------
# Stage: HPO (one Optuna study per session, merged by file)
# --------------------------------------------------------------------------------------------
def hpo_worker(ev, b3, session, fit, dev, scorer, feats, minutes=40, max_trials=30, seed_base=100,
               warm_params=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = P("hpo", session)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "trials.jsonl")
    done = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
    sampler = optuna.samplers.TPESampler(seed=seed_base + ord(session[0]), n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for r in done:                                          # resume: replay finished trials into the study
        try:
            study.add_trial(optuna.trial.create_trial(params=r["params"], distributions=_dists(),
                                                      value=r["value"]))
        except Exception:
            pass
    if not done:
        study.enqueue_trial(_clip_params(warm_params or {}))
    t_end = time.time() + minutes * 60

    def objective(trial):
        params = dict(depth=trial.suggest_int("depth", 6, 10),
                      learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                      l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                      random_strength=trial.suggest_float("random_strength", 0.2, 5.0, log=True),
                      bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                      border_count=trial.suggest_categorical("border_count", [128, 254]))
        t0 = time.time()
        m = fit_cb(fit, dev, feats, b3.CAT_FEATURES, params | {"iterations": 3000}, seed=0, verbose=0)
        prob = pred_cb(m, dev, feats, b3.CAT_FEATURES)
        s, t, _ = s1_macro(ev, scorer, dev, prob)
        rec = {"session": session, "trial": trial.number, "params": params, "value": s, "threshold": t,
               "best_iteration": int(m.get_best_iteration() or 0), "elapsed_s": round(time.time() - t0)}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  hpo {session} trial {trial.number}: {s:.6f}  {params}  ({rec['elapsed_s']}s)")
        return s

    n_left = max_trials - len(done)
    while n_left > 0 and time.time() < t_end:
        study.optimize(objective, n_trials=1)
        n_left -= 1
    mark(os.path.join(d, "_DONE"), {"n_trials": max_trials - n_left})


def _dists():
    import optuna.distributions as D
    return {"depth": D.IntDistribution(6, 10), "learning_rate": D.FloatDistribution(0.03, 0.2, log=True),
            "l2_leaf_reg": D.FloatDistribution(1.0, 30.0, log=True),
            "random_strength": D.FloatDistribution(0.2, 5.0, log=True),
            "bagging_temperature": D.FloatDistribution(0.0, 1.0),
            "border_count": D.CategoricalDistribution([128, 254])}


def _clip_params(p):
    base = {k: CB_DEFAULT[k] for k in ("depth", "learning_rate", "l2_leaf_reg", "random_strength",
                                       "bagging_temperature", "border_count")}
    base.update({k: v for k, v in p.items() if k in base})
    base["depth"] = int(min(max(base["depth"], 6), 10))
    base["learning_rate"] = float(min(max(base["learning_rate"], 0.03), 0.2))
    base["l2_leaf_reg"] = float(min(max(base["l2_leaf_reg"], 1.0), 30.0))
    base["random_strength"] = float(min(max(base["random_strength"], 0.2), 5.0))
    base["bagging_temperature"] = float(min(max(base["bagging_temperature"], 0.0), 1.0))
    base["border_count"] = 254 if base["border_count"] not in (128, 254) else base["border_count"]
    return base


def hpo_merge(sessions, deadline_min=0):
    """Best trial over every session's log. Sessions without _DONE are waited for up to deadline_min."""
    t0 = time.time()
    while deadline_min and time.time() - t0 < deadline_min * 60:
        if all(os.path.exists(P("hpo", s, "_DONE")) for s in sessions):
            break
        time.sleep(20)
    rows = []
    for s in sessions:
        p = P("hpo", s, "trials.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        raise FileNotFoundError("no HPO trials found")
    tab = pd.DataFrame(rows).sort_values("value", ascending=False)
    best = tab.iloc[0]
    iters = int(best["best_iteration"])
    params = dict(best["params"]) | {"iterations": max(int(iters * 1.25), 500)}
    mark(P("hpo", "best.json"), {"params": params, "value": float(best["value"]), "session": best["session"],
                                 "n_trials": len(tab)})
    return tab, params


# --------------------------------------------------------------------------------------------
# Stage: score jobs (one consolidated parquet per split x role x country)
# --------------------------------------------------------------------------------------------
def score_path(split, role, country):
    return P("scores", split, role, f"{country}.parquet")


def load_scores(split, role, countries) -> pd.DataFrame:
    fr = []
    for c in countries:
        p = score_path(split, role, c)
        if not os.path.exists(p.replace(".parquet", ".done.json")):
            raise FileNotFoundError(p)
        fr.append(pd.read_parquet(p))
    return pd.concat(fr, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Block 6: cross-fit accept policy on CALIBRATION
# --------------------------------------------------------------------------------------------
def policy_accept(df, prob, pol):
    """thr: p >= t.   thr_top: p >= t, or the S1's top candidate with p >= t_top (t_top < t)."""
    acc = prob >= pol["t"]
    if pol.get("kind") == "thr_top":
        top = df.assign(_p=prob).groupby("s1_id")["_p"].transform("max").to_numpy() == prob
        acc = acc | (top & (prob >= pol["t_top"]))
    return acc


def _fit_policy(ev, scorer, df, prob, mask_s1):
    codes = scorer.codes(df["s1_id"].to_numpy())
    lab = df["label"].to_numpy().astype(np.int8)
    sel_rows = mask_s1[codes]
    t, _, _ = scorer.best_threshold(codes[sel_rows], prob[sel_rows], lab[sel_rows])
    best = ({"kind": "thr", "t": t}, None)
    for kind, pol in [("thr", {"kind": "thr", "t": t})] + [
            ("thr_top", {"kind": "thr_top", "t": t2, "t_top": tt})
            for t2 in (t, t + 0.05, t + 0.1) for tt in (t - 0.1, t - 0.2, t - 0.3) if tt > 0.05]:
        acc = policy_accept(df, prob, pol)
        tp, npred = scorer.counts(codes, acc, lab)
        f = ev.f_from_counts(tp, npred, scorer.n_true)
        v = f[mask_s1].mean()
        if best[1] is None or v > best[1]:
            best = (pol, v)
    return best[0]


def crossfit_decision(ev, scorer, df, prob, folds=2):
    """Per-S1 F vector where each S1 is scored with a policy fitted on the OTHER fold(s); plus the final
    policy fitted on all of CALIBRATION."""
    fold = (ev.s1_hash_unit(scorer.u["s1_id"].to_numpy()) * folds).astype(int)
    codes = scorer.codes(df["s1_id"].to_numpy())
    lab = df["label"].to_numpy().astype(np.int8)
    f_cross = np.zeros(scorer.U)
    pols = []
    for g in range(folds):
        pol = _fit_policy(ev, scorer, df, prob, fold != g)
        acc = policy_accept(df, prob, pol)
        tp, npred = scorer.counts(codes, acc, lab)
        f = ev.f_from_counts(tp, npred, scorer.n_true)
        f_cross[fold == g] = f[fold == g]
        pols.append(pol)
    final = _fit_policy(ev, scorer, df, prob, np.ones(scorer.U, bool))
    return f_cross, final, pols
