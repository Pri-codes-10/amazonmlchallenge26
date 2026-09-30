"""
b4_ce.py - Block 4: the multilingual cross-encoder.

Reads candidates ONLY through b2_prod (merged manifest) and record text from Block 1.
  * CE-FIT  -> gradient updates (training pairs, incl. missed-positive injection, in_pool only)
  * CE-DEV  -> checkpoint / variant selection (all rows at rank <= K_FINAL, no sampling)
  * TREE-FIT, TREE-DEV, CALIBRATION, HOLDOUT, TEST -> inference only, one checkpoint for all

Output: <ROOT>/block4/<version>/ce_logits/<split>/<ROLE>/<COUNTRY>/part-XXXXX.parquet
        (s1_id, cand_id, ce_logit float32 raw, ce_model), 1:1 with Block 2 shards.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import er_eval as ev

ALLOWED_LICENSES = {"mit", "apache-2.0"}

# The one cross-encoder in the pipeline: the Block 2 backbone family with a new 1-logit head.
# $BER_MODEL_DIR overrides the repo id with an already-downloaded local directory (offline runs).
VARIANTS = {
    "V2": dict(repo="intfloat/multilingual-e5-small", lr=5e-5, batch=64, grad_accum=1),
}

TRAIN_DEFAULTS = dict(max_len=160, epochs=1, max_train_pairs=2_000_000, warmup_frac=0.05, weight_decay=0.01,
                      eval_every=4000, eval_s1_frac=0.10, eval_batch=512, seed=42, grad_clip=1.0)


# --------------------------------------------------------------------------------------------
# Paths / folders
# --------------------------------------------------------------------------------------------
def ver_dir(version):
    return os.path.join(ev.ROOT, "block4", version)


def out_dir(version, kind, split, role, country):
    return os.path.join(ver_dir(version), kind, split, role, country)


def prepare_folders(version, plan: dict, kinds=("ce_logits",)):
    """Create every output folder once, before any inference starts."""
    root = ver_dir(version)
    for sub in ("artifacts", "reports", "reports/per_s1", "markers", "hf"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for kind in kinds:
        for split, p in plan.items():
            for r in p["roles"]:
                for c in p["countries"]:
                    os.makedirs(out_dir(version, kind, split, r, c), exist_ok=True)
    json.dump({"plan": plan, "kinds": list(kinds), "time": time.strftime("%F %T")},
              open(os.path.join(root, "markers", f"_FOLDERS_READY_{'_'.join(k.replace('/', '-') for k in kinds)}"), "w"))
    print("folders ready under", root, "for", kinds)


# --------------------------------------------------------------------------------------------
# Model fetch with licence gate (weights cached under the workspace; inference is then offline)
# --------------------------------------------------------------------------------------------
def fetch_model(variant: str, version: str):
    """Download the backbone into <ROOT>/block4/<version>/hf/ after checking its licence and size.
    Set $BER_MODEL_DIR to an already-downloaded directory to skip the network entirely."""
    repo = os.environ.get("BER_MODEL_DIR") or VARIANTS[variant]["repo"]
    if os.path.isdir(repo):
        return repo, {"repo": repo, "license": "local", "revision": "local"}
    from huggingface_hub import model_info, snapshot_download
    info = model_info(repo)
    lic = None
    cd = getattr(info, "card_data", None)
    if cd is not None:
        lic = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
    if not lic:
        tags = [t for t in (info.tags or []) if t.startswith("license:")]
        lic = tags[0].split(":", 1)[1] if tags else None
    lic_norm = str(lic).lower() if lic else "unknown"
    if lic_norm not in ALLOWED_LICENSES:
        raise RuntimeError(f"{repo}: license '{lic}' is not MIT/Apache-2.0 - not allowed by the rules")
    n_params = None
    try:
        n_params = int(info.safetensors.total) if getattr(info, "safetensors", None) else None
    except Exception:
        pass
    if n_params is not None and n_params > 8e9:
        raise RuntimeError(f"{repo}: {n_params:,} params > 8B")
    local = os.path.join(ver_dir(version), "hf", repo.replace("/", "__"), info.sha)
    if not os.path.exists(os.path.join(local, "_COMPLETE")):
        snapshot_download(repo_id=repo, revision=info.sha, local_dir=local,
                          ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot",
                                           "flax_model*", "tf_model*", "*.gguf"])
        open(os.path.join(local, "_COMPLETE"), "w").write(info.sha)
    meta = {"repo": repo, "license": lic_norm, "revision": info.sha, "n_params": n_params, "local_dir": local}
    print(f"{variant}: {repo} @ {info.sha[:10]}  license={lic_norm}  params={n_params}")
    return local, meta


# --------------------------------------------------------------------------------------------
# Record text (Block 1)
# --------------------------------------------------------------------------------------------
def _txt(df, fmt):
    s = lambda c: df[c].astype(object).where(df[c].notna(), "").astype(str)
    t = s("name_norm") + " | " + s("addr_norm")
    if fmt == "B":
        t = t + " || " + s("name_latin") + " | " + s("addr_latin")
    return t.str.strip().to_numpy(dtype=object)


class TextStore:
    """Pair text in the bi-encoder's format: "{name_norm} | {addr_norm}" (A), + romanized (B)."""

    def __init__(self, split: str, countries, fmt: str = "A"):
        cols = ["entity_id", "name_norm", "addr_norm", "name_script"] + (["name_latin", "addr_latin"] if fmt == "B" else [])
        self.fmt = fmt
        s1 = ev.read_records(split, "s1", cols, countries=countries)
        self.s1_index = pd.Index(s1["entity_id"].astype(str).to_numpy(dtype=object))
        self.s1_text = _txt(s1, fmt)
        self.s1_script = s1["name_script"].astype(str).to_numpy(dtype=object)
        parts = []
        for src in ("S2", "S3"):
            d = ev.read_records(split, src, cols, countries=countries)
            d["_src"] = src
            parts.append(d)
        pool = pd.concat(parts, ignore_index=True)
        ids = pool["entity_id"].astype(str).to_numpy(dtype=object)
        self.pool_index = pd.Index(pool["_src"].to_numpy(dtype=object) + "|" + ids)
        self.id_only = pd.Index(ids)
        self.id_only_unique = self.id_only.is_unique
        self.pool_text = _txt(pool, fmt)
        self.pool_script = pool["name_script"].astype(str).to_numpy(dtype=object)
        print(f"  text {split} {countries} fmt={fmt}: S1 {len(s1):,}  pool {len(pool):,}")

    def pool_pos(self, df):
        ids = df["cand_id"].astype(str).to_numpy(dtype=object)
        pos = np.full(len(df), -1, np.int64)
        has = df["source"].notna().to_numpy() if "source" in df.columns else np.zeros(len(df), bool)
        if has.any():
            key = df["source"].astype(str).to_numpy(dtype=object)[has] + "|" + ids[has]
            pos[has] = self.pool_index.get_indexer(key)
        if (~has).any():
            if not self.id_only_unique:
                raise ValueError("candidate rows without `source` and entity ids not unique across S2/S3")
            pos[~has] = self.id_only.get_indexer(ids[~has])
        return pos

    def pair_texts(self, df):
        si = self.s1_index.get_indexer(df["s1_id"].astype(str).to_numpy(dtype=object))
        ci = self.pool_pos(df)
        bad = (si < 0) | (ci < 0)
        if bad.any():
            raise KeyError(f"{int(bad.sum())} pairs have ids missing from Block 1 records")
        cross = (self.s1_script[si] != self.pool_script[ci]).astype(np.int8)
        return self.s1_text[si], self.pool_text[ci], cross


# --------------------------------------------------------------------------------------------
# Candidate reading
# --------------------------------------------------------------------------------------------
CAND_COLS = ["s1_id", "cand_id", "source", "score", "rank", "role", "country", "split", "label"]


def load_pairs(b2p, split, roles, countries, k, s1_frac=None) -> pd.DataFrame:
    frames = []
    for meta, df in b2p.iter_candidates(split, roles=roles, countries=countries, k=k):
        if s1_frac is not None:
            df = df[ev.in_s1_subsample(df["s1_id"].to_numpy(), s1_frac)]
        frames.append(df[[c for c in CAND_COLS if c in df.columns]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CAND_COLS)


def build_train_pairs(b2p, countries, k_read=50, k_train=30, n_neg_top=15, n_neg_rand=3, s1_frac=None,
                      inject_missed=True, seed=42) -> pd.DataFrame:
    """Per CE-FIT S1: every positive in its list (rank <= k_read) + injected missed positives
    (in_pool only); negatives = the n_neg_top highest-ranked non-matches within k_train plus
    n_neg_rand random non-matches from lower in the list."""
    rng = np.random.default_rng(seed)
    frames = []
    for meta, df in b2p.iter_candidates("train", roles=["CE-FIT"], countries=countries, k=k_read):
        if s1_frac is not None:
            df = df[ev.in_s1_subsample(df["s1_id"].to_numpy(), s1_frac)]
        if not len(df):
            continue
        df = df[[c for c in CAND_COLS if c in df.columns]]
        pos = df[df["label"] == 1].assign(kind="pos")
        neg = df[df["label"] == 0].sort_values(["s1_id", "rank"], kind="stable")
        r = neg.groupby("s1_id").cumcount().to_numpy()
        hard = neg[(r < n_neg_top) & (neg["rank"].to_numpy() <= k_train)].assign(kind="hard_neg")
        rest = neg[r >= n_neg_top].copy()
        rest["_u"] = rng.random(len(rest))
        rest = rest[rest.groupby("s1_id")["_u"].rank(method="first") <= n_neg_rand].drop(columns="_u")
        frames += [pos, hard, rest.assign(kind="rand_neg")]
    pairs = pd.concat(frames, ignore_index=True)
    n_inj = 0
    if inject_missed:
        m = b2p.missed_positives(roles=["CE-FIT"])
        if len(m):
            if "in_pool" in m.columns:
                m = m[m["in_pool"].astype(bool)]
            if countries is not None and "country" in m.columns:
                m = m[m["country"].isin(countries)]
            if s1_frac is not None:
                m = m[ev.in_s1_subsample(m["s1_id"].to_numpy(), s1_frac)]
            if "source" not in m.columns:  # pos sidecar has no source; take it from the labels file
                pp = pd.read_parquet(ev.b1_path("labels", "positive_pairs.parquet"),
                                     columns=["source1_entity_id", "matched_entity_id", "matched_source"])
                pp = pp.rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id",
                                        "matched_source": "source"})
                pp["source"] = pp["source"].astype(str).str.extract(r"([23])", expand=False).map({"2": "S2", "3": "S3"})
                m = m.merge(pp, on=["s1_id", "cand_id"], how="left")
            m = m.assign(label=1, kind="injected_pos", rank=0, score=np.nan)
            have = set(zip(pairs["s1_id"], pairs["cand_id"]))
            m = m[[(a, b) not in have for a, b in zip(m["s1_id"], m["cand_id"])]]
            n_inj = len(m)
            pairs = pd.concat([pairs, m[[c for c in pairs.columns if c in m.columns]]], ignore_index=True)
    pairs = pairs.drop_duplicates(["s1_id", "cand_id"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"train pairs: {len(pairs):,}  ({pairs['s1_id'].nunique():,} S1)  kinds: "
          f"{pairs['kind'].value_counts().to_dict()}  injected={n_inj:,}")
    return pairs


# --------------------------------------------------------------------------------------------
# Torch helpers
# --------------------------------------------------------------------------------------------
def _device_and_amp():
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dev, amp = torch.device("cpu"), None
    return dev, amp


def load_model(path, for_training=True):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, ignore_mismatched_sizes=True)
    dev, amp = _device_and_amp()
    model.to(dev)
    model.train(for_training)
    return model, tok, dev, amp


def measure_token_lengths(tok, a, b, sample=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a), size=min(sample, len(a)), replace=False)
    enc = tok(list(a[idx]), list(b[idx]), truncation=False)
    L = np.array([len(x) for x in enc["input_ids"]])
    q = {f"p{p}": int(np.percentile(L, p)) for p in (50, 90, 95, 99, 99.9)}
    q["suggested_max_len"] = int(min(256, 8 * math.ceil(np.percentile(L, 99) / 8)))
    return q


def score_texts(model, tok, a, b, max_len, batch_size, dev, amp, prefetch=3, progress_every=0):
    """Length-sorted, dynamically padded inference. Returns raw logits (float32) in input order."""
    import torch
    n = len(a)
    out = np.empty(n, np.float32)
    if n == 0:
        return out
    lens = np.fromiter((len(x) + len(y) for x, y in zip(a, b)), np.int64, n)
    order = np.argsort(lens, kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]

    def enc(idx):
        return idx, tok(list(a[idx]), list(b[idx]), truncation="longest_first", max_length=max_len,
                        padding=True, return_tensors="pt")
    model.eval()
    t0 = time.time()
    with torch.inference_mode(), cf.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(enc, batches[i]) for i in range(min(prefetch, len(batches)))]
        nxt = len(futs)
        for bi in range(len(batches)):
            idx, e = futs[bi].result()
            if nxt < len(batches):
                futs.append(pool.submit(enc, batches[nxt]))
                nxt += 1
            e = {k: v.to(dev, non_blocking=True) for k, v in e.items()}
            if amp is not None:
                with torch.autocast(device_type=dev.type, dtype=amp):
                    lg = model(**e).logits
            else:
                lg = model(**e).logits
            out[idx] = lg.float().squeeze(-1).cpu().numpy()
            futs[bi] = None
            if progress_every and bi % progress_every == 0 and bi:
                done = (bi + 1) * batch_size
                print(f"    scored {done:,}/{n:,}  {done / (time.time() - t0):,.0f} pairs/s", end="\r")
    return out


def eval_ce(model, tok, dev_df, texts: TextStore, scorer, max_len, batch_size, device, amp, threshold=None):
    a, b, cross = texts.pair_texts(dev_df)
    t0 = time.time()
    logit = score_texts(model, tok, a, b, max_len, batch_size, device, amp, progress_every=200)
    el = time.time() - t0
    e = dev_df[["s1_id", "label", "rank", "country"]].copy()
    e["pair_cross_script"] = cross
    e["ce_logit"] = logit
    rep, f = ev.evaluate(scorer, e, "ce_logit", threshold=threshold)
    rep |= ev.rerank_metrics(e, "ce_logit")
    rep["throughput_pairs_s"] = float(len(e) / max(el, 1e-6))
    rep["logit_std"] = float(np.std(logit))
    if rep["logit_std"] < 1e-2:
        print(f"  WARNING: logit std {rep['logit_std']:.2e} - the model has collapsed to a constant "
              "(lower lr / more warmup, check labels); do not select this checkpoint")
    return rep, f, logit


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------
def train_ce(variant, model_path, model_meta, pairs, texts_train: TextStore, dev_df, texts_dev: TextStore,
             scorer_dev_sub, version, cfg=None, run_tag=None, fmt="A"):
    """Pointwise BCE on one logit. Periodic CE-DEV (subsample) evaluation; keeps the checkpoint with
    the best S1-macro F0.5. Returns (best_dir, history)."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    cfg = dict(TRAIN_DEFAULTS) | {k: VARIANTS[variant][k] for k in ("lr", "batch", "grad_accum")} | (cfg or {})
    run_tag = run_tag or variant
    adir = os.path.join(ver_dir(version), "artifacts", run_tag)
    os.makedirs(adir, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    model, tok, dev, amp = load_model(model_path, True)
    a, b, _ = texts_train.pair_texts(pairs)
    y = pairs["label"].to_numpy(np.float32)
    rng = np.random.default_rng(cfg["seed"])
    n_total = min(len(pairs), int(cfg["max_train_pairs"])) * int(cfg["epochs"])
    order = np.concatenate([rng.permutation(len(pairs)) for _ in range(int(cfg["epochs"]) + 1)])[:n_total]
    bs, ga = int(cfg["batch"]), int(cfg["grad_accum"])
    n_micro = n_total // bs
    n_steps = max(1, n_micro // ga)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = get_linear_schedule_with_warmup(opt, int(cfg["warmup_frac"] * n_steps), n_steps)
    scaler = torch.amp.GradScaler("cuda") if (amp is not None and amp == torch.float16) else None
    lossf = torch.nn.BCEWithLogitsLoss()
    hist, best, best_dir = [], -1.0, os.path.join(adir, "best")
    dev_sub = dev_df[ev.in_s1_subsample(dev_df["s1_id"].to_numpy(), cfg["eval_s1_frac"])]
    t0, run_loss, step = time.time(), 0.0, 0
    print(f"{run_tag}: {n_total:,} train pairs, {n_steps:,} optimizer steps, bs={bs}x{ga}, lr={cfg['lr']}, "
          f"max_len={cfg['max_len']}, eval every {cfg['eval_every']} steps on {len(dev_sub):,} CE-DEV pairs")

    def do_eval(tag):
        nonlocal best
        rep, f, _ = eval_ce(model, tok, dev_sub, texts_dev, scorer_dev_sub, cfg["max_len"], cfg["eval_batch"], dev, amp)
        h = {"step": step, "tag": tag, "elapsed_min": round((time.time() - t0) / 60, 1),
             "train_loss": run_loss, "s1_macro_f05": rep["s1_macro_f05"], "pr_auc": rep["pr_auc"],
             "model_hit@1": rep["model_hit@1"], "biencoder_hit@1": rep["biencoder_hit@1"],
             "throughput_pairs_s": rep["throughput_pairs_s"]}
        hist.append(h)
        print(f"\n  [eval {tag}] step {step}: S1-macro F0.5 {rep['s1_macro_f05']:.4f}  PR-AUC {rep['pr_auc']:.4f}  "
              f"hit@1 {rep['model_hit@1']:.4f} (bi-enc {rep['biencoder_hit@1']:.4f})")
        if rep["s1_macro_f05"] > best:
            best = rep["s1_macro_f05"]
            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
            print(f"  -> new best, saved {best_dir}")
        model.train()

    model.train()
    for mi in range(n_micro):
        idx = order[mi * bs:(mi + 1) * bs]
        e = tok(list(a[idx]), list(b[idx]), truncation="longest_first", max_length=cfg["max_len"],
                padding=True, return_tensors="pt")
        e = {k: v.to(dev, non_blocking=True) for k, v in e.items()}
        yt = torch.from_numpy(y[idx]).to(dev)
        if amp is not None:
            with torch.autocast(device_type=dev.type, dtype=amp):
                lg = model(**e).logits.squeeze(-1)
        else:
            lg = model(**e).logits.squeeze(-1)
        loss = lossf(lg.float(), yt) / ga
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        run_loss = 0.98 * run_loss + 0.02 * float(loss) * ga if mi else float(loss) * ga
        if (mi + 1) % ga == 0:
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                rate = (mi + 1) * bs / (time.time() - t0)
                eta = (n_micro - mi - 1) * bs / max(rate, 1e-6) / 60
                print(f"  step {step}/{n_steps}  loss {run_loss:.4f}  {rate:,.0f} pairs/s  ETA {eta:.0f} min", end="\r")
            if step % cfg["eval_every"] == 0:
                do_eval("periodic")
    do_eval("final")
    meta = {"variant": variant, "run_tag": run_tag, "base_model": model_meta, "text_format": fmt,
            "train_config": cfg, "n_train_pairs": int(n_total), "history": hist, "best_dev_sub_s1_macro_f05": best,
            "time": time.strftime("%F %T")}
    json.dump(meta, open(os.path.join(adir, "meta.json"), "w"), indent=1, default=str)
    return best_dir, hist


# --------------------------------------------------------------------------------------------
# Inference runner (resumable, worker-partitioned, 1:1 with Block 2 shards)
# --------------------------------------------------------------------------------------------
def _shard_index(meta, fallback):
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
    return fallback


def run_inference(version, b2p, model_dir, ce_model_id, split, roles, countries, k_ce, worker=0, n_workers=1,
                  kind="ce_logits", s1_frac=None, fmt="A", max_len=160, batch_size=512, model=None):
    """Score rank <= k_ce rows of every committed Block 2 shard; worker w takes shard % n_workers == w."""
    if model is None:
        model, tok, dev, amp = load_model(model_dir, for_training=False)
    else:
        model, tok, dev, amp = model
    summary = []
    for country in countries:
        texts = None
        counters = {}
        for meta, df in b2p.iter_candidates(split, roles=roles, countries=[country], k=k_ce):
            role = str(df["role"].iloc[0]) if len(df) else str((meta or {}).get("role", roles[0]))
            counters[role] = counters.get(role, -1) + 1
            sid = _shard_index(meta, counters[role])
            if sid % n_workers != worker:
                continue
            od = out_dir(version, kind, split, role, country)
            done_p = os.path.join(od, f"done-{sid:05d}.json")
            if os.path.exists(done_p):
                continue
            if s1_frac is not None:
                df = df[ev.in_s1_subsample(df["s1_id"].to_numpy(), s1_frac)]
            if texts is None:
                texts = TextStore(split, [country], fmt)
            t0 = time.time()
            if len(df):
                a, b, _ = texts.pair_texts(df)
                logit = score_texts(model, tok, a, b, max_len, batch_size, dev, amp, progress_every=500)
                out = pd.DataFrame({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(),
                                    "ce_logit": logit.astype(np.float32), "ce_model": ce_model_id})
                assert not out.duplicated(["s1_id", "cand_id"]).any()
                os.makedirs(od, exist_ok=True)
                out.to_parquet(os.path.join(od, f"part-{sid:05d}.parquet"), index=False, compression="zstd")
            el = time.time() - t0
            info = {"n_rows": int(len(df)), "k_ce": k_ce, "ce_model": ce_model_id, "elapsed_s": round(el, 1),
                    "pairs_per_s": float(len(df) / max(el, 1e-6)), "s1_frac": s1_frac, "worker": worker}
            json.dump(info, open(done_p, "w"))
            summary.append({"split": split, "role": role, "country": country, "shard": sid} | info)
            print(f"\n  {kind} {split}/{role}/{country} part {sid:05d}: {len(df):,} pairs in {el / 60:.1f} min "
                  f"({info['pairs_per_s']:,.0f}/s)")
    return pd.DataFrame(summary)


def load_logits(version, kind, split, roles, countries=None) -> pd.DataFrame:
    frames = []
    for role in roles:
        base = os.path.join(ver_dir(version), kind, split, role)
        cs = countries or (sorted(os.listdir(base)) if os.path.isdir(base) else [])
        for c in cs:
            for dp in sorted(glob.glob(os.path.join(base, c, "done-*.json"))):
                pp = dp.replace("done-", "part-").replace(".json", ".parquet")
                if os.path.exists(pp):
                    frames.append(pq.read_table(pp).to_pandas())
    if not frames:
        raise FileNotFoundError(f"no committed logits for {kind} {split} {roles}")
    out = pd.concat(frames, ignore_index=True)
    assert not out.duplicated(["s1_id", "cand_id"]).any(), "duplicate (s1_id, cand_id) in logits"
    return out
