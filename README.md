# Rank 512 ML Challenge 2026: Business Entity Resolution Solution
Out of 30000 submissions

**Team Name:** Ninja mutant turtles <br>
**Team Mates:** [Mohikshit Ghorai](https://github.com/psycocodes) , [Priyansh Ghosh](https://github.com/Pri-codes-10) , [Debjit Bhunia](https://github.com/Dpythonl) <br>
**Submission Date:** 2026-09-29

---

## 1. Executive Summary

We resolve each `source1` entity against `source2`/`source3` with a four-stage cascade: a
per-country fine-tuned multilingual bi-encoder retrieves 20 candidates per query, a fine-tuned
cross-encoder re-scores every retained pair, a CatBoost model combines those logits with 75
hand-crafted pair features and 18 label-free "context" signals, and an entity-level resolver enforces
the one-owner-per-record constraint that pairwise scoring cannot see. The key innovation is handling
the **cross-script** problem in data preparation rather than in the matcher: 6.96% of true pairs have
the two sides written in different scripts, and character TF-IDF retrieves only **1.82%** of them —
romanizing with a local character table lifts that to **58.33%**, and an added phonetic consonant
skeleton key closes most of the remaining spelling drift. Every decision threshold is fitted on a
dedicated CALIBRATION split by cross-fit, with a HOLDOUT split read exactly once for reporting.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA that drove the design:

- **Cross-script pairs are the dominant recall risk.** 6.96% of ground-truth pairs pair a non-Latin
  name (Devanagari, Tamil, Gujarati, Bengali, Kannada, Oriya, Telugu, Malayalam, Gurmukhi) with a
  Latin one. Naive character n-grams recover 1.82% of them. Generic transliteration alone is not
  enough: for Tamil, Malayalam and Gurmukhi, a phonetic romanizer garbled almost the whole name (a
  recognizable business word survived in only 0.64% / 0.73% / 0.83% of names, versus 71.9–85.9% for
  every other script). We therefore built a native-script business-vocabulary table from **181,341
  verified cross-script name pairs across 34,011 ground-truth clusters** and apply it *before*
  romanization. A follow-up scan found a narrower variant of the same bug in addresses: 248,552
  addresses (>1% of the dataset) are otherwise plain Latin but carry the state name in local script
  ("… Chennai தமிழ்நாடு").
- **Remaining drift is vowel/spelling noise** ("Shree" vs "sri", "Traders" vs "tredrs"), so we emit a
  consonant-skeleton key that folds both spellings to the same pattern (`sr blj trdrs`).
- **Legal forms and honorifics are high-frequency noise.** `pvt ltd`, `llp`, `sarl`, `M/s.` and
  friends appear in most names and must be stripped to a canonical form to expose the brand anchor.
  Two spelled-out variants alone (`elelpi` → `llp`, `pra li` → `pvt ltd`) explained nearly all of a
  5.6–20.7% legal-suffix detection gap across six scripts.
- **Addresses are much weaker evidence than names.** Postal codes are nearly always absent for INDIA,
  so they can never be *required*; addresses also collapse many distinct businesses onto one
  multi-tenant building.
- **Missing / junk fields are common** (`na`, `n/a`, `null`, `xxx`, `0`, `test`), and are normalised
  to empty rather than matched on.
- **Countries are an open set.** Unknown labels pass through normalised and are never dropped; FRANCE
  appears in test with **no training labels at all**.
- **Records are claimed at most once.** Across 7.64M ground-truth pairs, every S2/S3 record belongs to
  at most one S1 — 0 exceptions. Pairwise scoring cannot exploit this; a resolver stage can.
- **The metric punishes over-merging on singletons.** An S1 with no true match scores 1.0 only if
  nothing is accepted, and 0.0 on any accept, so precision on no-match queries is worth as much as
  recall everywhere else.

### 2.2 Solution Strategy

Six sequential blocks, each communicating only through files on disk, so any stage can be re-run
independently:

1. **Block 1 — preparation.** Normalise, romanize, build the skeleton/alias/legal-suffix keys, align
   the ground truth, and assign **entity-level** roles (CE-FIT 30%, CE-DEV 5%, TREE-FIT 30%,
   TREE-DEV 10%, CALIBRATION 10%, HOLDOUT 15%), stratified by country × script group × match-count
   bucket × has-cross-script and seeded. Roles are assigned per *entity group*, so no S1 and its
   matches ever straddle two roles.
2. **Block 2 — candidate generation (blocking).** Per-country fine-tuned bi-encoder, dense forward
   search within the same country.
3. **Block 3 — pair features.** 75 deterministic features in 9 groups, plus a frozen token-IDF fitted
   on the train pool only.
4. **Block 4 — cross-encoder.** A second, jointly-encoding model re-scores every retained pair, then
   its logits are turned into label-free list- and record-level context features.
5. **Block 5 — pair model.** One CatBoost model over Block 3 + Block 4 features.
6. **Block 6 — decision + resolution.** Accept threshold, entity resolver, both TSVs.

**Approach Type:** Hybrid — dense-retrieval blocking + cross-encoder re-ranking + gradient-boosted
classifier + a constraint-based entity resolver.

**Core Innovation:** Three things, in order of contribution.
(a) *Cross-script recovery in Block 1* — GT-derived native-script vocabulary substitution applied
before romanization, plus a phonetic skeleton key, turning a 1.82% cross-script retrieval rate into a
usable one.
(b) *Record-claimant context features (level "L2")* — for every candidate record, its rank, positive
count and mutual-best status **across every S1 in the split that also listed it**. This lets the trees
learn the one-owner-per-record rule from data. It was the largest single modelling gain in our
ablation (0.985632 → 0.987105 S1-macro F0.5). Crucially these are expressed only as ranks, gaps and
shares, never raw claimant counts, because a train role contains a fraction of the S1s while test has
all of them competing — a count would shift distribution between train and test. A **parity check**
(Kolmogorov–Smirnov > 0.10 or NaN-rate gap > 5pp between HOLDOUT and TEST) then drops any signal that
still shifts; it removed `rec_gap_best` and `rec_share`.
(c) *An explicit entity resolver* after the pairwise decision, which enforces one owner per record and
vetoes the two failure modes pairwise features cannot see: multi-tenant buildings and chain/template
names.

---

## 3. Candidate Generation (Blocking)

Dense retrieval with a per-country fine-tuned `intfloat/multilingual-e5-small` bi-encoder. Queries and
pool records are encoded as `"{name_norm} | {addr_norm}"` in **native script** (the cross-script work
of Block 1 feeds the lexical keys; the encoder is multilingual and benefits from the original text),
L2-normalised, fp16, and searched with a block-wise running top-k merge over the pool so a
(queries × 6.2M pool) score matrix is never materialised.

- **Blocking keys used:** a learned dense embedding of `name_norm | addr_norm`, hard-gated by
  `country_norm` (records never cross countries). Block 1 additionally emits `name_latin`,
  `name_core`, `name_skeleton`, `name_aka` and `postal_code`; a Block 1 key bake-off on CE-DEV
  compared them (`name_latin` ∪ `name_skeleton` was the best lexical union) and the dense encoder was
  chosen over them on recall. INDIA and US each have their own checkpoint; **FRANCE and any other
  country use the US checkpoint** (Latin-script transfer), since they have no labels to fine-tune on.
- **Candidate pairs generated:** `K_GEN = 50` stored per query, truncated downstream to
  **`K_FINAL = 20`**. On the test set that is **34,650,880 candidate pairs over 1,732,544 `source1`
  entities** — exactly 20 per entity, no entity left without candidates. This is the scored set and is
  what `output/candidate_pairs.tsv` contains.
- **How we ensured true matches were not lost:**
  - Recall is measured as *pair completeness* per role, from a positives sidecar written next to every
    candidate shard, and read **only on the four roles the bi-encoder never saw** (TREE-FIT, TREE-DEV,
    CALIBRATION, HOLDOUT). CE-FIT recall is knowingly optimistic — the encoder was fine-tuned on 90%
    of it — and is excluded from every decision.
  - `K_FINAL` is chosen as the smallest K whose pooled recall on those unbiased roles reaches a
    **0.995 target**; the achieved figure per role and country is written to
    `workspace/block2/prod/merged/pc_report.json` by the merge notebook.
  - HOLDOUT pair completeness is treated as the **hard recall ceiling**: no later block can recover a
    pair missed here, and this is stated in the report rather than hidden.
  - Coverage is asserted, not assumed: every S1 must be queried exactly once, with the right role and
    country, and every committed shard's row count must match its `done-*.json`.
  - For countries without labels, where recall is unmeasurable, we compare the median top-1 retrieval
    score against US test as a distribution-shift sanity check.
  - Positives the retriever missed are injected into the **cross-encoder's training set** (in-pool
    only) so the re-ranker still learns from hard positives. This improves the model; it does not
    recover those pairs at inference, and we do not count it as recall.

---

## 4. Matching Model

Two learned models feed one classifier.

**Cross-encoder (Block 4).** `intfloat/multilingual-e5-small` with a fresh single-logit
classification head, trained with pointwise binary cross-entropy on **CE-FIT pairs only**: every
positive of an S1 within its top 50, the positives blocking missed (injected, in-pool only), its 15
hardest non-matches within the top 30, and 3 random deeper non-matches. One epoch, at most 2M pairs,
lr 5e-5, batch 64, warmup 5%, weight decay 0.01, grad clip 1.0, seed 42; `max_len` measured as the
99th percentile of CE-DEV pair lengths. Every 4,000 steps it is evaluated on 10% of CE-DEV S1s and the
best S1-macro-F0.5 checkpoint is kept. **CE-DEV only selects the checkpoint; it never contributes
gradients.** One checkpoint then scores every role and the test set.

**Features used** — 93 in total (75 hand-crafted + 20 context, minus 2 dropped by the parity check):

- **Name features (30, group G2):** token-set / token-sort / partial ratios, Jaro–Winkler and
  normalised Levenshtein over three name views (`name_latin`, `name_core` = brand anchor with the
  legal suffix stripped, `name_skeleton` = phonetic consonant skeleton), exact-match flags,
  first/last-token equality, token-count and length ratios, plain and **IDF-weighted** Jaccard,
  IDF coverage of the rarer side, IDF mass left unmatched on each side, and the IDF of the rarest
  shared token. The IDF is fitted on the train S2+S3 pool only and frozen for test.
- **Alias features (3, G3):** best similarity against the `d/b/a`, `a/k/a`, `f/k/a` alias extracted in
  Block 1, and the gain it provides over the primary name.
- **Legal-form feature (1, G4):** a categorical agree / disagree / one-side-missing state over the
  canonicalised legal suffix.
- **Address features (14, G5):** token-set ratio, plain and IDF-weighted Jaccard, IDF coverage and
  unmatched mass, rarest shared token IDF, a **postal-code state** and a **street-number state**
  (agree / conflict / unknown — never a hard requirement, since PINs are nearly always absent for
  INDIA), landmark and multi-location counts, and missing-side flags.
- **Format / script features (9, G6):** whether the pair is cross-script, the candidate's script
  (categorical), whether it was romanized, mixed-script and bare-web-domain flags, a space-insensitive
  core ratio, embedded-phone-number flags, and a missing-name count.
- **Retrieval context (8, G1):** the bi-encoder score, its rank, gaps and ratio to the list's top,
  gaps to the neighbouring ranks, list size, and an is-rank-1 flag.
- **List context (5, G8):** the pair's name-similarity rank within its own candidate list, how many
  candidates score ≥ 90, gaps to the best name and address similarity, and a count of candidates with
  both strong.
- **Cross-list exclusivity (3, G9):** how many other S1 lists contain this candidate record, whether
  this S1 is that record's best claimant, and by what margin — the bi-encoder-side analogue of L2.
- **Source / country (2, G7):** both categorical.
- **Cross-encoder context (20 → 18 kept):** the raw logit plus its list context (gap to top, rank,
  number of positive logits, number scored, gap to next, share, entropy, z-score, and the shift
  between bi-encoder rank and cross-encoder rank), and the **record-claimant level**: `rec_rank`,
  `rec_n_pos`, `mutual_best`. `rec_gap_best` and `rec_share` were dropped by the HOLDOUT-vs-TEST
  parity check.

Features that depend on a *second* cross-encoder exist in the code but are dropped automatically,
because the final pipeline ships one. Labels are never features: a fixed leak-column list is enforced
in Block 3, and any single feature reaching ROC-AUC ≥ 0.99 is investigated before any model is fitted.

**Model type:** a single **CatBoost** classifier (no ensembling, no second model family). Trained on
**TREE-FIT** only, early-stopped on **TREE-DEV** log-loss (`od_wait` 150, 6000-iteration ceiling),
seed 0. Parameters: depth 8, learning rate 0.08, `l2_leaf_reg` 3.0, random strength 1.0, bagging
temperature 0.5, 254 borders. A time-boxed Optuna search is wired in but **did not complete**, so the
submitted model used these fixed defaults — stated here rather than presented as tuned. CALIBRATION,
HOLDOUT and TEST are never trained on. FRANCE is never trained on at all (no labels); the same model
scores it, with `country` as a categorical feature.

**Threshold selection method:** S1-macro F0.5 optimisation on **CALIBRATION only**, by 2-fold
cross-fit — each S1 is scored with a policy fitted on the other fold, so the reported calibration
figure is not fitted on itself; the shipped policy is then refitted on all of CALIBRATION. The search
is a two-pass quantile grid over a global probability threshold, and also considers a
"threshold + always accept the S1's top candidate above a lower bar" variant, keeping whichever scores
better. The entity resolver's knobs are tuned by the same cross-fit and are adopted for INDIA/US
**only if that cross-fit is not worse** than the plain accept rule. FRANCE, having no labels, uses a
fixed preset (`dropall2`: an S1 that loses a contested record keeps nothing from it, crowd and
template vetoes off) followed by emptying any S1 whose only accepted pair sits below the threshold —
chosen from France's *label-free* output profile against INDIA/US, not tuned on any label.

**Entity resolver.** France-aware name/address keys (accents folded, street abbreviations unified,
leading and mid-string legal forms stripped, house number found in either address order, postal code
parsed from free text, administrative-region words never used as street evidence), then: a **crowding
veto** (if an S1 sits in a building shared by ≥ 3 distinct S1 business names, an accept additionally
requires the *names* to agree), a **template veto** (if an S1's name key is shared by ≥ 3 S1s — a
chain or template name — an accept is vetoed when the *addresses* conflict), and **one-owner
arbitration** (a record accepted for ≥ 2 S1s keeps at most one: the top S1 by probability if it leads
by ≥ 0.15 and its name+address evidence is not weaker than the runner-up's; else the S1 with clearly
stronger evidence if its probability ≥ 0.5; else nobody).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9871** on TREE-DEV (20% S1 subsample) for the shipped feature set.
  The context-level ablation that selected it: BASE 0.985523 → L0 0.985542 → L1 0.985632 →
  **L2 0.987105**. Held-out leaderboard score for the corresponding submission: **0.978**
  (progression across submissions: 0.965 → 0.975 → 0.976 → 0.977 → 0.978).
  The ~0.009 gap between TREE-DEV and the leaderboard is expected and we do not paper over it: the
  test set has *every* S1 competing for each record while a train role holds only a fraction of them,
  and FRANCE is scored with a model that never saw a French label.
  Per-role and per-country breakdowns — including the single HOLDOUT read — are written to
  `workspace/block6/reports/holdout_report.json` by Block 6.

- **Common false positives (wrong merges):** two modes dominate, and the resolver exists specifically
  because pairwise features cannot see either.
  1. **Multi-tenant buildings.** Many distinct businesses share one address string, so strong address
     agreement plus weak name agreement merges unrelated entities. Addressed by the crowding veto
     (≥ 3 distinct S1 names at one address ⇒ names must agree).
  2. **Chain and template names.** Identical or near-identical names at different locations (e.g.
     "Defense Club SARL"); strong name agreement plus conflicting addresses. Addressed by the template
     veto.
  A third, smaller mode is **singleton over-acceptance**: an S1 with no true match accepting its
  best-looking candidate, which costs a full 1.0 on that S1 under the metric. This is why Block 3's
  diagnostics track "singletons with a false accept" explicitly and why the FRANCE rule empties S1s
  whose only accepted pair is weak.

- **Common false negatives (missed matches):**
  1. **Retrieval misses** — pairs never in the top 20. This is a hard floor: HOLDOUT pair completeness
     bounds the achievable recall and no later block can recover them.
  2. **Residual cross-script drift** — the cases the native-vocabulary table and the phonetic skeleton
     still do not fold together, which is why cross-script S1s are reported as their own slice
     throughout.
  3. **One-sided or junk addresses**, where the only discriminating evidence is a short, generic name.
  4. **Resolver arbitration losses** — where two S1s contest one record and neither leads by enough,
     the record is dropped for both, trading recall for the precision the metric rewards.

  Ranked examples with the model's probability, country, retrieval rank and both raw name/address
  strings are regenerated by Block 3's final stage into
  `workspace/block3/v1/reports/top_false_positives.csv` and `top_false_negatives.csv`.

---

## 6. Conclusion

We treated this as a recall problem in data preparation and a precision problem in decision-making:
cross-script normalisation and a fine-tuned bi-encoder get the true match into a 20-candidate list,
and a cross-encoder, a CatBoost model with record-claimant context, and a constraint-based resolver
decide which of those 20 to accept. The two lessons that mattered most were that a generic phonetic
romanizer silently destroys some scripts — only checking it against ground-truth cross-script pairs
revealed it — and that any feature crossing the train/test boundary must be expressed as a rank or a
share rather than a count, with an explicit distribution-parity check to catch the ones that still
drift. Keeping CALIBRATION and HOLDOUT strictly separated, and reading HOLDOUT exactly once, is what
made the offline numbers trustworthy enough to act on.

---

## Appendix

### A. Code Artefacts

Complete and runnable, under `code/business_entity_resolution/` in this zip:

```
code/business_entity_resolution/
├── README.md              setup, run order, licences, reproducibility caveat
├── requirements.txt       Python >= 3.10
├── src/                   8 notebooks + 8 modules, flat
├── data/                  supplied by the grader, read-only: train/, test/, validate_submission.py
└── workspace/             created by the notebooks: block1/ … block6/
```

**Entry points — run the notebooks in `src/` in this order.** Every notebook's first cell is
identical: it locates the folder containing `src/`, creates `workspace/`, and exports the environment
variables the modules read *before* importing any of them. There are no absolute paths, no cloud-drive
or notebook-host dependencies, and nothing is written outside `workspace/` except temporary scratch
files and the Hugging Face model cache (`workspace/hf_cache`).

| # | notebook | module(s) | hardware |
|---|---|---|---|
| 1 | `01_block1.ipynb` | `er_eval` | CPU, high RAM |
| 2 | `02_block2_candidategen_india.ipynb` → `_us.ipynb` → `_merge.ipynb` | `b2_common`, `b2_prod` | GPU, GPU, CPU |
| 3 | `03_block3.ipynb` | `b3_features` | CPU, high RAM |
| 4 | `04_block4.ipynb` | `b4_ce`, `b4_ctx` | GPU |
| 5 | `05_block5.ipynb` | `b5_catboost` | GPU or CPU |
| 6 | `06_block6.ipynb` | `b6_resolve` | CPU |

`06_block6.ipynb` produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` under
`workspace/block6/`, and validates them with both an inline validator and the organisers'
`data/validate_submission.py`. Assembling this archive is a separate packaging step, so the pipeline
writes nothing but its own outputs; Block 6's last stage prints the exact commands.

Every stage is **idempotent**: a unit of work counts as done only once its `done-*.json` exists
(written last), so a re-run skips committed shards, Block 4 skips training when the checkpoint exists,
and Block 5 reuses a saved model when the parameters and feature list match.

**Fair play.** No external data and no external APIs. Two models, both well under the 8B-parameter
limit: `intfloat/multilingual-e5-small` (MIT, ~118M parameters) for the bi-encoder and the
cross-encoder, and CatBoost (Apache-2.0) for the pair model. Romanization uses `anyascii`, a local
character table (ISC licence). `b4_ce.fetch_model` refuses to download anything that is not MIT or
Apache-2.0, or larger than 8B parameters. No weights are committed; the first run downloads the
backbone from Hugging Face, and `BER_MODEL_DIR` points at a pre-downloaded copy for an offline run.

**Reproducibility, stated plainly.** Training is not bit-reproducible across machines: CatBoost GPU
and CPU differ, as do library versions and thread order. A fresh run reproduces the submitted file
closely but not necessarily byte for byte. The hyper-parameter search never completed, so the
submitted model used the documented CatBoost defaults.

### B. Additional Results

Reports regenerated by a full run, all under `workspace/`:

| file | contents |
|---|---|
| `block1/<v>/reports/block1_qa_report.json` | every QA and leakage check as PASS/FAIL: row conservation, id uniqueness and prefixes, roles within the allowed set, romanization completeness, "positive pairs never cross roles", "merged entity groups share one role", `cluster_id` agrees with `positive_pairs` |
| `block1/<v>/reports/block2_key_bakeoff.json` | recall@K per blocking key, split by same-script vs cross-script — the 1.82% / 58.33% figures above |
| `block2/prod/merged/pc_report.json` | pair completeness per country × role and pooled over the unbiased roles; the `K_FINAL` recall/volume trade-off table |
| `block3/v1/reports/runs.jsonl` | every feature-group ablation with its per-S1 F0.5 vector, so any two runs can be compared by paired bootstrap; plus per-slice scores (country, cross-script, singleton), PR-AUC, accepted-per-S1 and feature importances |
| `block3/v1/reports/top_false_{positives,negatives}.csv` | the 50 worst errors of each kind, with raw text |
| `block4/v1/reports/ce_dev_report_V2.json` | full CE-DEV metrics, including cross-encoder hit@1 versus bi-encoder hit@1 |
| `block4/v1/reports/coverage_check.json` | per (split, role, country): candidate rows in vs logits out, asserted equal |
| `block5/final/parity.json` | the HOLDOUT-vs-TEST parity check and the signals it dropped |
| `block6/reports/policy.json`, `resolver.json` | the accept policy and resolver verdict, with their CALIBRATION cross-fit scores |
| `block6/reports/holdout_report.json` | the single HOLDOUT read: accept-rule-only vs with-resolver, overall and per country, with a paired bootstrap between them |
| `block6/reports/validation_regenerated.json` | inline validator result plus the organisers' validator's exit code and output |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
