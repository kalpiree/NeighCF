# NeighCF

Code for **NeighCF: Neighborhood-aware Collaborative Filtering for LLM-based Sequential Recommendation**, sixteen baselines, and RQ1–RQ4.

| Methods | Implementations |
| --- | --- |
| NeighCF | Neighbor retrieval, collaborative prompts, candidate-likelihood scoring and training |
| GRU4Rec, Caser, SASRec, LightGCN | Sequential and graph recommenders |
| LLaMA2, GPT-4 | Zero-shot candidate ranking |
| TALLRec, MoRec | Recommendation tuning and text-modality recommendation |
| CoRAL, LLM-CF, DRDT, RecICL | Collaborative retrieval, augmentation, reflection and in-context learning |
| LLaRA, LLM-SRec, FCRLLM, UCGRec | Additional LLM-based comparisons |

## Setup

Use Python 3.11 from the repository root. On CUDA hosts, install the appropriate PyTorch 2.4.1 CUDA build before the requirements. TALLRec uses a separate environment.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -B scripts/check_setup.py --dependencies
python -B scripts/smoke.py
```

Add `--gpu` to the setup check to inspect CUDA. The smoke test uses synthetic CPU inputs.

Keep data, model weights, upstream checkouts and outputs outside this repository:

```sh
export NEIGHCF_WORKSPACE="$(cd .. && pwd)/neighcf-workspace"
mkdir -p "$NEIGHCF_WORKSPACE"
cp configs/project.example.json "$NEIGHCF_WORKSPACE/project.json"
python scripts/fetch_baselines.py --directory "$NEIGHCF_WORKSPACE/sources" --fetch
python scripts/fetch_baselines.py --directory "$NEIGHCF_WORKSPACE/sources" --verify
```

In `project.json`, set `workspace` to `"."` and fill in assets and budgets. Asset/output paths resolve against the workspace; NeighCF configuration paths resolve against this repository. Teacher paths accept `{seed}` and `{dataset}`. Models must be local Hugging Face snapshots.

## Data and protocol

Each canonical dataset contains `items.json`, `train.jsonl`, `valid.jsonl`, `test.jsonl`, `train_interactions.jsonl`, `manifest.json` and `splits.json`.

- `items.json` maps string item IDs to objects containing `title`.
- Query rows contain `query_id`, `user_id`, `history`, `history_timestamps`, `timestamp`, `target_id` and `candidate_ids`.
- Training events contain `user_id`, `item_id`, `timestamp` and `split: "train"`.
- Timestamps are milliseconds; histories contain strictly earlier events. Candidate IDs are unique and contain the target exactly once.

Experiments use **100 training candidates and 51 evaluation candidates**, fixed across methods and seeds 42–46. The manuscript used 100 evaluation candidates. Cohorts, NeighCF and CoRAL are reconstructions. Baseline configurations and manifests record backbone, domain and scoring adaptations. Keep results from different protocols separate.

For MovieLens, prepare a 100-candidate parent and derive the evaluation view:

```sh
python data_preparation/prepare_movielens.py RAW_ML100K NEW_PARENT_DIRECTORY
python data_preparation/audit_frozen.py NEW_PARENT_DIRECTORY --raw-dir RAW_ML100K
python data_preparation/derive_candidates.py derive --source NEW_PARENT_DIRECTORY --output NEW_CANONICAL_DIRECTORY --expected-candidates 51 --seed 42
python data_preparation/derive_candidates.py audit --source NEW_PARENT_DIRECTORY --output NEW_CANONICAL_DIRECTORY --expected-candidates 51 --seed 42
```

For Steam/Goodreads, use `data_preparation/prepare_review_datasets.py` (`--help` lists inputs), then `audit_review_bundle.py` and the same candidate derivation. Training rows stay unchanged.

For LastFM, run `python datasets/run_lastfm.py --root EXTERNAL_LASTFM_WORKSPACE --release "$PWD" --plan`. Remove `--plan` to prepare first artist listens from LastFM-1K. Source terms: `datasets/lastfm-source.json`.

## Features and model configuration

```sh
python integrations/fcrllm/features.py --data CANONICAL_DATA --model-path LOCAL_LLAMA2 --output NEW_FCR_FEATURES --expected-candidates 51 --device cuda:0
python scripts/prepare_neighcf.py --fcr-features NEW_FCR_FEATURES --train-events CANONICAL_DATA/train_interactions.jsonl --output NEW_NEIGHCF_FEATURES --clusters 16 --seed 42
```

Centre fitting uses training-observed items; catalog embeddings stay frozen. Prepare M = 4, 8, 12, 16 and 20 centres for RQ3.

`integrations/neighcf/example.config.json` and `example.training.json` configure retrieval, prompts, scoring and optimization:

- `hard_text_reference` selects neighbors by threshold. `soft_embedding_gate` weights a fixed score-ranked context of up to `soft_context_k` neighbors. This differentiable embedding gate is a reconstruction choice; discrete retrieval receives no gradient.
- The default `literal_positive` index retains the full eligible pool. Optional `user_top_clusters` and `user_mass_threshold` policies filter that pool after computing causal profiles.
- Defaults retain the target history and each neighbor's three most recent items before its observed next item. Prompt/sequence limits are 3,072/4,096 tokens; oversized inputs fail. Neighbor-score ties use string user-ID order.
- `full_model` trains the generator and enabled gate; `gate_only` freezes the generator. Defaults: Adam, 5 epochs, effective batch 128, learning rate 1e-5, 10% warmup, cosine decay, validation NDCG@10 selection.
- FP32 full 7B training needs roughly 112 GB before activations, exceeding one 96 GB H20. LoRA/quantization are not implemented; large-model NeighCF GPU training is untested.

Select hyperparameters on validation data. Llama-2 features, k-means and candidate-title likelihood scoring are reconstruction choices.

## Training and evaluation

```sh
python -m integrations.neighcf.run prompt --config CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --queries QUERIES --output NEW_PROMPT_DIRECTORY
python -m integrations.neighcf.run train --config CONFIG --training-config TRAINING_CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --train TRAIN --valid VALID --test TEST --model-path LOCAL_MODEL --variant full --device cuda:0 --output NEW_RUN_DIRECTORY
python -m integrations.neighcf.run score --config CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --queries TEST --model-path LOCAL_MODEL --checkpoint RUN/best.pt --variant full --device cuda:0 --output NEW_SCORE_DIRECTORY
```

Variants: `full`, `without_collaborative`, `fixed_neighbors`, `global_pool`. The two ablations require `full_model`. Protocol metadata comes from `splits.json`; otherwise set `--dataset-name` and `--protocol-sha256`. Validation selects the test checkpoint.

```sh
python evaluate.py evaluate PREDICTIONS_JSONL --metadata METADATA_JSON --output NEW_METRICS_JSON --expected-candidates 51
python evaluate.py aggregate SEED42_METRICS SEED43_METRICS SEED44_METRICS SEED45_METRICS SEED46_METRICS --output NEW_AGGREGATE_JSON
```

Metadata: `model`, `dataset`, `seed`, `protocol_sha256`. Predictions: `query_id`, `user_id`, `candidate_ids`, `target_id`, aligned finite `scores`. Metrics average over queries; MRR covers all candidates. Ties use string item ID. Aggregate matching protocols/evaluation sets separately per model and variant.

## Baselines

For `llara`, `llmsrec`, `fcrllm`, `ucgrec`: `python integrations/METHOD/run.py --help`. Other adapters: `python -m integrations.METHOD.run --help`.

| Adapter | Required inputs and settings |
| --- | --- |
| `traditional` | Pinned BSARec source; `--model gru4rec`, `caser` or `sasrec`; explicit epoch budget and `--expected-candidates 51` |
| `lightgcn` | Pinned source, canonical training edges and explicit epoch budget |
| `morec` | Pinned SIGIR 2023 text-modality MoRec source, local BERT and its example configuration |
| `zeroshot` | Local LLaMA2-7B or an explicit GPT-4 snapshot; API execution requires `--execute-api` and `--max-requests` |
| `tallrec` | Separate `requirements-tallrec.txt` environment, one visible CUDA GPU, local LLaMA weights and instruction-tuned rank-8 LoRA adapter; native binary AUC selects the checkpoint |
| `llmcf` | Pinned source, `--num-neg`, CoT/item embedding assets and their provenance manifest |
| `recicl` | Pinned source, local model and `--item-kind movie\|book\|game\|artist`; recent same-user examples |
| `drdt` | Pinned source, domain and earlier training demonstrations; local model or explicitly budgeted API access |
| `coral` | Observed training ratings, user/item embeddings, provenance, local model and backend-call budget |
| `llara`, `llmsrec`, `ucgrec` | Validation-selected teacher for every dataset and seed |
| `fcrllm` | Llama-2 title-feature bundle |
| `ucgrec` | Completed 128-example pilot with verified restore; non-Steam policies also require semantic features and their manifest |

Teacher options: `python integrations/llmsrec/run.py teacher --help`; keep weights, metadata and completion receipts together. UCGRec features: `python integrations/ucgrec/features.py --help`. LLM-CF/RecICL require explicit learning rate, weight decay, optimizer and schedule. Assets are not included.

## Research questions

```sh
python scripts/rq.py plan --project "$NEIGHCF_WORKSPACE/project.json" --rq 1 --datasets steam --models neighcf sasrec
python scripts/rq.py run --project "$NEIGHCF_WORKSPACE/project.json" --id rq1/steam/neighcf/seed-42
python scripts/rq.py run --project "$NEIGHCF_WORKSPACE/project.json" --id rq1/steam/neighcf/seed-42 --resume
```

Planning lists commands and prerequisites. `run --all` executes filtered tasks sequentially. Set `traditional_epochs`, baseline settings, assets and `tallrec.python` first.

| RQ | Tasks |
| --- | --- |
| RQ1 | NeighCF and sixteen baselines |
| RQ2 | Full NeighCF, without collaborative information, fixed neighbors |
| RQ3 | M = 4/8/12/16/20, K = 2/4/6/8/10, activity-group analysis |
| RQ4 | Retrieval and prompt timing for full, global-pool and fixed-neighbor variants |

RQ3 requires `rq3_threshold.value` and a validation receipt (`split`, `selected_tau`, `validation_sha256`, `protocol_sha256`). K controls `soft_context_k`; `activity_threshold` groups users by untruncated history length.

RQ4 times retrieval/prompts on CPU over five repetitions, excluding inference, loading and feature preparation. For inference timing, see `python scripts/benchmark.py --help`.

## Resume and outputs

Resume with the original command/output path and `--resume`; preserve source, data, configuration, model, runtime and seed. Uncertain API requests require `--retry-uncertain`. Scheduling and allocations are external.

NeighCF checkpoints retain model/gate parameters, optimizer, scheduler, RNG, progress and validation selection with SHA-256 receipts. Keep outputs separate from inputs/source. Load only trusted checkpoints; restoration uses PyTorch pickle.

Selected-model test outputs are `test.metrics.json`, `test.ranks.jsonl` and `test.scores.jsonl`. Standalone scoring produces `metrics.json`, `ranks.jsonl` and `scores.jsonl`.
