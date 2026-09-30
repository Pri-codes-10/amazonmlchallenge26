# Business Entity Resolution

Matches each `source1` business entity to its duplicates in `source2` / `source3`, per country
(India, US, France). Six notebooks turn the raw TSVs into `matching_results.tsv` and
`candidate_pairs.tsv`. Metric: S1-macro F0.5, where an S1 with no true match scores 1.0 only if
nothing is accepted.

## Layout

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
├── src/                 notebooks 01 … 06 + the modules, flat
├── data/                supplied by the grader, read-only: train/, test/, validate_submission.py
└── workspace/           created by the notebooks: block1/ … block6/
```

`data/train/` holds `train_source{1,2,3}.tsv` and `train_ground_truth.tsv`; `data/test/` holds
`test_source{1,2,3}.tsv`. Nothing is written outside `workspace/` except temporary scratch files.

## Setup

```sh
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA build
pip install -r requirements.txt
jupyter lab            # then open src/ and run the notebooks in order
```

Every notebook's first cell finds the folder containing `src/` by walking up from the working
directory (override with `BER_HOME`), creates `workspace/`, and exports `BER_WORKSPACE` and
`HF_HOME` before importing anything. Run notebooks from `src/` or from the repository root — both
work. Re-running is safe: each stage skips what it already committed (`done-*.json` is written last),
and Block 5 reuses a saved model when the parameters and feature list match.

## Run order

| # | notebook | hardware | writes under `workspace/` |
|---|---|---|---|
| 1 | `01_block1.ipynb` | CPU, high RAM | `block1/v2/records/`, `labels/`, `reports/` — normalised records, ground truth, entity roles |
| 2 | `02_block2_candidategen_india.ipynb` → `_us.ipynb` → `_merge.ipynb` | GPU, GPU, CPU | `block2/prod/<COUNTRY>/candidates/`, `block2/prod/merged/manifest.json` (`k_final`) |
| 3 | `03_block3.ipynb` | CPU, high RAM | `block3/v1/features/`, `vocab/`, `reports/selected_features.json` |
| 4 | `04_block4.ipynb` | GPU | `block4/v1/artifacts/V2/`, `ce_logits/`, `block4/ctx/v1/` |
| 5 | `05_block5.ipynb` | GPU or CPU | `block5/final/model/`, `scores/`, `features_final.json` |
| 6 | `06_block6.ipynb` | CPU | `block6/output/*.tsv`, `block6/reports/` |

Block 2's fine-tuned bi-encoder checkpoints are expected at
`workspace/block2/artifacts/e5_finetuned_{INDIA,US}`; France and any other country use the US
checkpoint.

## Outputs

`workspace/block6/output/matching_results.tsv` and `candidate_pairs.tsv`, with reports in
`workspace/block6/reports/`. Block 6 runs its own inline validator and, when
`data/validate_submission.py` exists, the organisers' validator (its `--help` is printed together
with the exact command used). Building `<team_name>submission.zip` is a separate packaging step —
Block 6's last stage prints the commands.

## Models and licences

Only `intfloat/multilingual-e5-small` (MIT, ~118M parameters) for both the bi-encoder and the
cross-encoder, and CatBoost (Apache-2.0) for the pair model. No external data or APIs. The first run
downloads the backbone from Hugging Face into `workspace/hf_cache` and `workspace/block4/v1/hf/`, so
it needs internet once; `b4_ce.fetch_model` refuses anything that is not MIT / Apache-2.0 or larger
than 8B parameters. To run offline, point `BER_MODEL_DIR` at an already-downloaded copy of the model
directory. Weights are never committed.

## Reproducibility

Training is not bit-reproducible across machines: CatBoost GPU and CPU differ, and so do library
versions and thread order. 