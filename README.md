# NeighCF

Implementation of **NeighCF: Neighborhood-aware Collaborative Filtering for LLM-based Sequential Recommendation**, sixteen comparison methods, and RQ1–RQ4. NeighCF is reconstructed from the manuscript; its implementation choices are specified below.

## Methods

| Method | Implementation |
| --- | --- |
| NeighCF | Causal neighbor retrieval, prompts, candidate-likelihood scoring and training |
| LightGCN | Native graph propagation, BPR objective and train-only graph |
| LLaMA2-zero-shot, GPT-4-zero-shot | Explicit generated-ranking prompt, frozen candidates and response journals |
| TALLRec | Native recommendation tuning, binary AUC selection and canonical candidate ranking |
| MoRec | SIGIR 2023 text-modality BERT/SASRec end-to-end BCE variant |
| CoRAL | DDPG reconstruction with observed ratings and a local frozen LLM |
| LLM-CF | Native SASREC_aug with supplied CoT and item-text features |
| DRDT | Native sequential and collaborative reflection methods |
| RecICL | Native no-CF prediction methods with recent same-user examples |
| GRU4Rec, Caser, SASRec | Pinned BSARec implementations with validation selection and checkpoint resume |
| LLaRA | Native architecture with canonical data and candidate evaluation adapters |
| LLM-SRec | Native architecture with a Llama-2 backbone substitution |
| FCRLLM | Native architecture with Llama-2 title features |
| UCGRec | Native top-k implementation with temporal graphs and domain-specific semantic features |

The twelve manuscript baselines use modality-based MoRec from SIGIR 2023. LLaRA, LLM-SRec, FCRLLM and UCGRec are additional comparisons. Native source revisions and file hashes are recorded in `third_party/UPSTREAM.json`.

## Setup

Use Python 3.11. Run commands from the repository root. TALLRec uses the separate environment described below.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -B scripts/check_setup.py --dependencies
python -B scripts/smoke.py
```

Dependencies pin PyTorch 2.4.1. Install its CUDA build for the compute host before installing the requirements; CUDA 12.4 wheels use the PyTorch `cu124` index. `bitsandbytes` is enabled on Linux x86-64. Add `--gpu` to `check_setup.py` to inspect CUDA availability. The smoke command uses synthetic CPU inputs.

Keep data, weights, native checkouts and outputs in an external workspace:

```sh
export NEIGHCF_WORKSPACE="$(cd .. && pwd)/neighcf-workspace"
mkdir -p "$NEIGHCF_WORKSPACE"
cp configs/project.example.json "$NEIGHCF_WORKSPACE/project.json"
python scripts/fetch_baselines.py --directory "$NEIGHCF_WORKSPACE/sources" --fetch
python scripts/fetch_baselines.py --directory "$NEIGHCF_WORKSPACE/sources" --verify
```

Without `--fetch` or `--verify`, the fetcher prints a plan. Fetching includes any assets tracked in the upstream repositories; keep those checkouts external. Supply backbone weights as local Hugging Face snapshot directories.

Edit `project.json` and set `workspace` to `"."`. Asset paths and `output_root` resolve against that workspace. `neighcf_config` and `training_config` resolve against the code root. Teacher paths support `{seed}` and `{dataset}`. Set the local model path, dataset paths, feature paths and training budgets before running.

## Data

Each canonical dataset contains `items.json`, `train.jsonl`, `valid.jsonl`, `test.jsonl`, `train_interactions.jsonl`, `manifest.json` and `splits.json`.

- Items: an object mapping item IDs to objects containing `title`.
- Queries: `query_id`, `user_id`, `history`, `history_timestamps`, `timestamp`, `target_id` and `candidate_ids`.
- Neighbor events: `user_id`, `item_id`, `timestamp` and `split: "train"`.
- Embeddings: a JSON object mapping every catalog item ID to a vector.
- Centres: a JSON array of equal-dimensional vectors.

Canonical timestamps are milliseconds. Histories contain only strictly earlier events. Candidate IDs are unique, retain their frozen order and contain the target once. The configured protocol uses **100 training candidates and 51 evaluation candidates**; the manuscript used 100 for evaluation.

Prepare a 100-candidate parent, then derive its evaluation view:

```sh
python data_preparation/prepare_movielens.py RAW_ML100K NEW_PARENT_DIRECTORY
python data_preparation/audit_frozen.py NEW_PARENT_DIRECTORY --raw-dir RAW_ML100K
python data_preparation/derive_candidates.py derive --source NEW_PARENT_DIRECTORY --output NEW_CANONICAL_DIRECTORY --expected-candidates 51 --seed 42
python data_preparation/derive_candidates.py audit --source NEW_PARENT_DIRECTORY --output NEW_CANONICAL_DIRECTORY --expected-candidates 51 --seed 42
```

Steam and Goodreads use `prepare_review_datasets.py` with `--dataset steam` or `--dataset goodreads-history`, plus `--reviews`, `--metadata` and `--output`. Audit with `audit_review_bundle.py`. Apply the same candidate derivation afterward; it preserves training rows. Keep candidate sets fixed across models and training seeds.

For LastFM:

```sh
python datasets/run_lastfm.py --root EXTERNAL_LASTFM_WORKSPACE --release "$PWD" --plan
```

Removing `--plan` downloads and prepares the recorded LastFM-1K source. This uses timestamped first artist listens and a different cohort from the manuscript. Source attribution and terms are in `datasets/lastfm-source.json` and `data_preparation/lastfm-official-README.txt`. The supplied preparation recipes do not reproduce unavailable original cohort files.

## Features and teachers

Generate the FCRLLM embedding bundle and fit NeighCF centres:

```sh
python integrations/fcrllm/features.py --data CANONICAL_DATA --model-path LOCAL_LLAMA2 --output NEW_FCR_FEATURES --expected-candidates 51 --device cuda:0
python scripts/prepare_neighcf.py --fcr-features FCR_FEATURES --train-events CANONICAL_DATA/train_interactions.jsonl --output NEW_NEIGHCF_M16_FEATURES --clusters 16 --seed 42
```

Centre fitting uses training-observed items only. Catalog embeddings remain frozen. The Llama-2 encoder and seeded k-means recipe are reconstruction choices. Prepare separate centre bundles for M = 4, 8, 12, 16 and 20 for RQ3.

LLaRA, LLM-SRec and UCGRec need a validation-selected teacher for each dataset and seed:

```sh
python integrations/llmsrec/run.py teacher --data CANONICAL_DATA --source LLM_SREC_SOURCE --output NEW_TEACHER_DIRECTORY --dataset-name DATASET_LABEL --seed 42 --expected-candidates 51 --device cuda:0
```

Repeat for seeds 42–46. Keep `teacher.pth`, `teacher.metadata.json` and `teacher.completion.json` together. Example path: `teachers/movielens/seed-{seed}/teacher.pth`.

UCGRec requires a completed pilot with `--pilot-examples 128` and verified checkpoint restore. Set `ucgrec_pilot` to its completion receipt. Non-Steam policies also require semantic features and their manifest. `integrations/ucgrec/features.py --help` lists the local description-model, sentence-encoder, prompt and policy arguments. Feature preparation retains raw text and token IDs. It can restore one missing final object brace when recorded end-of-turn evidence occurs before the generation cap; other malformed descriptions fail validation.

## NeighCF configuration

`integrations/neighcf/example.config.json` and `example.training.json` define the configuration schemas. Five epochs and effective batch size 128 follow the manuscript. Select the learning rate, warmup, threshold policy, context limits, regularization and activity cutoff using validation data.

### Retrieval and prompts

Soft memberships use `softmax(-distance² / sigma²)`. User profiles average historical memberships. Active clusters combine the user's top clusters with candidate memberships above `theta`. Similarity combines masked cosine and shared membership mass by geometric mean.

The default `literal_positive` index uses `z_v[c] > 0`. Softmax memberships are positive, so it includes every eligible user and provides no pruning benefit. `user_top_clusters` and `user_mass_threshold` are optional sparse reconstruction policies. All policies currently recompute causal profiles before filtering.

Retrieval excludes the query user and events at or after the query timestamp. A neighbor needs two earlier events: the last becomes its observed next item and the preceding prefix supplies its history. Profiles use the complete eligible prefix. Ties use string item-ID order.

### Scoring and training modes

`hard_text_reference` selects thresholded neighbors and constructs textual prompts. `soft_embedding_gate` adds a differentiable reconstruction: a score-ranked context of `soft_context_k` neighbors, with each neighbor's prompt-token embeddings scaled by `sigmoid((S - tau) / temperature)`. This embedding gate is an additional implementation choice, not a specified manuscript mechanism. Discrete retrieval receives no gradient.

The size regularizer sums gates over the full retrieved pool, including neighbors omitted from the prompt context. `gate_eta` controls its weight. Thresholds can use one fixed scalar, an activity-logistic function or per-user parameters. With `train_gate=false`, they stay fixed. For a fixed threshold, `threshold_value` and `gate_initial_threshold` must agree.

Candidate scoring tokenizes the prompt and each `title + candidate_terminator` suffix separately, then sums or averages teacher-forced suffix log probabilities. Training applies cross entropy over candidate scores. Tokenization, normalization and terminator are configurable reconstruction choices. Oversized prompts fail instead of being truncated.

`full_model` trains the backbone and enabled gate with Adam and warmup/cosine scheduling. `gate_only` freezes the backbone. LoRA and quantization are not implemented. Gradient clipping is optional.

| Variant | Behavior |
| --- | --- |
| `full` | Configured retrieval and neighbor context |
| `without_collaborative` | Omits neighbor text and gates; requires `full_model` |
| `fixed_neighbors` | Uses the available top `fixed_neighbor_k` neighbors with unit weights; requires `full_model` |
| `global_pool` | Uses all eligible users; equivalent to `full` with `literal_positive` |

### Commands

```sh
python -m integrations.neighcf.run audit --config CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --queries QUERIES --output NEW_AUDIT_DIRECTORY
python -m integrations.neighcf.run prompt --config CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --queries QUERIES --output NEW_PROMPT_DIRECTORY
python -m integrations.neighcf.run train --config CONFIG --training-config TRAINING_CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --train TRAIN --valid VALID --test TEST --model-path LOCAL_MODEL --variant full --device cuda:0 --output NEW_RUN_DIRECTORY
python -m integrations.neighcf.run score --config CONFIG --items ITEMS --embeddings EMBEDDINGS --centres CENTRES --events EVENTS --queries TEST --model-path LOCAL_MODEL --checkpoint RUN/best.pt --variant full --device cuda:0 --output NEW_SCORE_DIRECTORY
```

Training and scoring read protocol metadata from the query directory's `splits.json`. Otherwise supply `--dataset-name` and `--protocol-sha256`. Hard-reference scoring without a checkpoint also requires `--seed`. `--test` runs after validation selects the best epoch. `audit` and `prompt` require no model.

Loading is local-only, uses FP32 and disables remote model code. Full FP32 7B training needs roughly **112 GB** for parameters, gradients and Adam moments before activations and candidate graphs, exceeding a single 96 GB H20. Large-model GPU training remains untested. Local verification used synthetic CPU fixtures with PyTorch 2.2.2; the production environment pins 2.4.1.

## Baseline commands

Use direct script entry points for the newer adapters:

```sh
python integrations/llara/run.py --help
python integrations/llmsrec/run.py --help
python integrations/fcrllm/run.py --help
python integrations/ucgrec/run.py --help
```

Traditional models:

```sh
python -m integrations.traditional.run audit --model sasrec --source BSAREC_SOURCE --data CANONICAL_DATA --dataset DATASET_LABEL --output NEW_AUDIT_DIRECTORY --expected-candidates 51 --device cpu
python -m integrations.traditional.run train --model sasrec --source BSAREC_SOURCE --data CANONICAL_DATA --dataset DATASET_LABEL --output NEW_RUN_DIRECTORY --epochs CHOSEN_EPOCHS --expected-candidates 51 --dimension 64 --batch-size 256 --learning-rate 0.001
```

Choose `gru4rec`, `caser` or `sasrec`. Set the epoch budget explicitly. Defaults include sequence length 50 and validation NDCG@10 selection. Record the training budget, sequence length and regularization choice with each experiment.

### LightGCN and modality MoRec

```sh
python -m integrations.lightgcn.run train --source LIGHTGCN_SOURCE --data CANONICAL_DATA --dataset DATASET_LABEL --output NEW_RUN --epochs CHOSEN_EPOCHS --seed 42 --device cuda
python -m integrations.morec.run train --source MOREC_SOURCE --data CANONICAL_DATA --encoder-path LOCAL_BERT --config integrations/morec/example.config.json --output NEW_RUN --seed 42 --device cuda:0
```

LightGCN deduplicates canonical training histories and targets into graph edges. Its native BPR sampler excludes known training items. Every evaluation user needs training edges. Default dimension is 64 with three graph layers; choose the epoch budget explicitly.

MoRec loads the original `bce_text/main-end2end` model from [IDvs.MoRec](https://github.com/westlake-repl/IDvs.MoRec). Titles supply the text modality. It uses each user's latest training prefix, shifted sequence BCE and unseen negatives; validation NDCG@10 selects the epoch. Its example configuration records sequence length 20, BERT title width 30 and a 150-epoch budget. Both commands automatically evaluate the selected model on the fixed test candidates.

### Zero-shot inference and DRDT

```sh
python -m integrations.zeroshot.run plan --backend llama2 --data CANONICAL_DATA
python -m integrations.zeroshot.run run --backend llama2 --data CANONICAL_DATA --model-path LOCAL_LLAMA2 --device cuda:0 --output NEW_RUN
python -m integrations.zeroshot.run export --backend gpt4 --model-id GPT4_SNAPSHOT --data CANONICAL_DATA --output NEW_REQUEST_DIRECTORY
python -m integrations.drdt.run audit --source DRDT_SOURCE --data CANONICAL_DATA --domain Books
python -m integrations.drdt.run score --source DRDT_SOURCE --data CANONICAL_DATA --domain Books --backend local --model-path LOCAL_MODEL --device cuda:0 --output NEW_RUN
```

Zero-shot prompts contain chronological titles and numbered candidates. A normally completed JSON permutation supplies ordinal scores. This prompt is an explicit candidate-ranking adaptation; no target labels enter it. Local LLaMA2 requires the 7B architecture. GPT-4 execution requires an explicit `gpt-4` or `gpt-4-*` model ID, `OPENAI_API_KEY`, `--execute-api` and a positive `--max-requests` limit. Prefer a fixed snapshot for repeatability. Offline `--responses FILE` rows contain `query_id`, the exported `request_sha256`, and `response` with `text`, `model` and `finish_reason`. Keep that file outside the output directory. Greedy local runs across seeds need not yield different predictions.

DRDT preserves the pinned sequential-reflection and collaborative-reflection calls. It requires an earlier other-user training demonstration with the same history endpoint and enough history for the selected reflection depth. Supported domains include `ml-1m`, `Games`, `Books` and the explicit LastFM wording adaptation `artists-native-books-adaptation-v1`. Local inference is a backbone substitution. API use requires `--backend api --allow-api --api-model PINNED_MODEL --max-api-calls BUDGET`. Both integrations journal completed calls. Resume keeps confirmed responses; an interrupted request with uncertain completion requires `--retry-uncertain` before another API attempt.

### TALLRec

Create a separate CUDA environment with `requirements-tallrec.txt`. Its native pins are Transformers 4.28.0, PEFT 0.3.0 and bitsandbytes 0.37.2; resolved runtime versions are recorded at launch and checked on resume. Set `CUDA_VISIBLE_DEVICES` to one allocated GPU. Supply local LLaMA-compatible weights and the original instruction-tuned rank-8 LoRA adapter.

```sh
python -m integrations.tallrec.run train --source TALLREC_SOURCE --data CANONICAL_DATA --dataset DATASET_LABEL --domain movie --adaptation implicit-history-positive-one-unobserved-negative-v1 --model-path LOCAL_LLAMA --instruction-adapter LOCAL_INSTRUCTION_LORA --output NEW_TRAIN_RUN --epochs 5 --batch-size 128 --device cuda
python -m integrations.tallrec.run evaluate --source TALLREC_SOURCE --data CANONICAL_DATA --dataset DATASET_LABEL --domain movie --run TRAIN_RUN --output NEW_EVALUATION_RUN --device cuda
```

The explicit implicit-feedback adaptation treats observed history as positive and adds one frozen unobserved negative per training query. Domains are `movie`, `book`, `item` or `artist`. Native binary AUC selects the checkpoint. Evaluation preserves the source's full-vocabulary softmax followed by Yes/No softmax. PEFT loading and validation-cadence compatibility fixes preserve native training. `prepare` exports and validates inputs without training. Full resume requires Trainer optimizer, scheduler, scaler, RNG and adapter files; adapter weights alone are insufficient.

### LLM-CF and RecICL

Both use `audit` or `train`, five epochs by default, and explicit `--learning-rate`, `--weight-decay`, `--optimizer adam|adamw` and `--schedule constant|warmup_cosine`.

```sh
python -m integrations.llmcf.run --help
python -m integrations.recicl.run --help
```

LLM-CF requires `--source`, `--data`, `--assets` and `--num-neg`. It loads the pinned `SASREC_aug` architecture, native sampled-softmax loss and CoT reconstruction term. Assets comprise `bank.jsonl`, `cot_embeddings.npy`, `item_embeddings.npy` and `manifest.json`. The manifest uses schema `llmcf-authentic-augmentation-v1`, the pinned `source_commit`, canonical `protocol_sha256`, ordered `item_ids`, file `sha256` mappings and nonempty `generator_model`, `generator_revision`, `embedding_model`, `embedding_revision`, `generation_description`. Each bank row has a unique `example_id`, `source_query_id`, `candidate_id`, binary `label`, `cot_index` and actual `cot_text`. Item and CoT embeddings have 768 dimensions. Generate these with the upstream preparation workflow; the adapter validates and consumes them. Retrieval selects two positive and two negative training examples, excludes the current query and equal/future examples of the same user, then applies seeded ordering.

RecICL requires `--source`, `--data`, `--model-path` and `--item-kind movie|book|game|artist`. It uses the exact native no-CF forward/scoring methods, raw-Yes-logit BCE and frozen input embeddings. The canonical adaptation uses up to four recent same-user positive demonstrations, ten-item histories, one frozen negative per training query and static candidate ranking. The domain prompt, full-model tuning and validation NDCG@10 selection are explicit adaptations. The optional HashGNN variant and original streaming evaluation are separate upstream workflows. Both adapters require `--output` for training and automatically score the selected model on test.

### CoRAL

```sh
python -m integrations.coral.run audit --config integrations/coral/example.config.json --data CANONICAL_DATA --ratings RATINGS_JSONL --embeddings EMBEDDINGS_JSON --embedding-provenance PROVENANCE_JSON
python -m integrations.coral.run train --config CONFIG --data CANONICAL_DATA --ratings RATINGS_JSONL --embeddings EMBEDDINGS_JSON --embedding-provenance PROVENANCE_JSON --model-path LOCAL_MODEL --output NEW_RUN --device cuda:0 --max-backend-calls BUDGET
```

The [CoRAL reconstruction](https://arxiv.org/abs/2403.06447) uses DDPG actor/critic retrieval, Euclidean user/item selection, replay, target networks, OU exploration and changes in absolute prediction error as rewards. It allows ten training retrieval steps with error-based stopping and uses five evaluation steps. Configuration specifies all numerical choices. The transition MLP learns through critic TD gradients over recomputed histories, with a separate target transition; this is an explicit reconstruction choice.

Ratings require `user_id`, `item_id`, finite `rating`, millisecond `timestamp` and `split: "train"`, matching the pinned canonical training interactions. Both observed preference classes must exist. Missing interactions never supply dislikes. Embeddings are `{ "users": {ID: VECTOR}, "items": {ID: VECTOR} }`. Provenance binds `fit_split: "train"`, `protocol_sha256`, `ratings_sha256`, `embeddings_sha256` and `source_method`. This variant uses supplied frozen training embeddings, a local LLM's full-vocabulary Yes probability and canonical ranking with validation selection. Model/backbone and embedding preparation choices must accompany reported results. Training includes selected-policy test evaluation.

## Research questions

```sh
python scripts/rq.py plan --project "$NEIGHCF_WORKSPACE/project.json" --rq 1 --datasets steam --models neighcf sasrec
python scripts/rq.py run --project "$NEIGHCF_WORKSPACE/project.json" --id rq1/steam/neighcf/seed-42
python scripts/rq.py run --project "$NEIGHCF_WORKSPACE/project.json" --id rq1/steam/neighcf/seed-42 --resume
```

Planning reports commands and missing prerequisites. `run --all` executes the filtered tasks sequentially after preflight. Training uses seeds 42–46. Set `traditional_epochs` and model-specific `baselines` options explicitly. Configure native source paths, authentic feature/rating assets and `tallrec.python`. The TALLRec task contains separate training and evaluation steps. Paid API methods remain blocked until explicitly enabled or supplied with offline responses.

| RQ | Tasks |
| --- | --- |
| RQ1 | Comparison of NeighCF and sixteen baselines |
| RQ2 | Full NeighCF, without collaborative information, fixed neighbors |
| RQ3 | M = 4/8/12/16/20, K = 2/4/6/8/10, activity-group analysis |
| RQ4 | Retrieval and prompt timing for full, global-pool and fixed-neighbor variants |

RQ3 requires a validation-selected `rq3_threshold.value` and receipt with `split: "valid"`, `selected_tau`, `validation_sha256` and `protocol_sha256`. The planner freezes this threshold. K controls `soft_context_k`. Set `activity_threshold`; grouping uses `history_length_before_truncation` and completed RQ1 test ranks.

RQ4 performs five CPU timing repetitions per variant. It excludes model inference, loading, feature preparation and initial membership computation. To measure inference too, use `scripts/benchmark.py` with the exact configuration and assets, plus `--model-path`, `--checkpoint`, `--checkpoint-variant` and `--device cuda:0`. Create its external output directory first. Its `--queries` input is the canonical test JSONL and `--output` is a new timing JSONL file.

## Evaluation

```sh
python evaluate.py evaluate PREDICTIONS_JSONL --metadata METADATA_JSON --output NEW_METRICS_JSON --expected-candidates 51
python evaluate.py aggregate SEED42_METRICS SEED43_METRICS SEED44_METRICS SEED45_METRICS SEED46_METRICS --output NEW_AGGREGATE_JSON
```

Metadata contains `model`, `dataset`, `seed` and `protocol_sha256`. Predictions contain `query_id`, `user_id`, `candidate_ids`, `target_id` and aligned finite `scores`. Metrics average over queries. Ties use descending score, then ascending string item ID; MRR covers all candidates. Aggregation checks protocol, model, evaluation-set identity and unique seeds. Aggregate each variant separately.

## Resume

Append `--resume` to the original training or journaled inference command and output path. Keep source, data, configuration, model, tokenizer, runtime and seed unchanged. Checkpoints from another source tree require separate compatibility verification.

Checkpoints retain model/gate parameters, Adam state, scheduler, RNG, progress and validation selection. NeighCF publishes SHA-256 receipts and retains a previous checkpoint for recovery. `latest.pt` can fall back to the verified previous copy; corrupt `best.pt` is rejected. Load only trusted checkpoints because optimizer/RNG restoration uses PyTorch pickle. Keep output directories separate from inputs and source, without symlink ancestors.

NeighCF writes `manifest.json`, `latest.pt`, `best.pt`, checksum receipts, progress and completion records. Test outputs are `test.metrics.json`, `test.ranks.jsonl` and `test.scores.jsonl`. Standalone scoring uses `metrics.json`, `ranks.jsonl` and `scores.jsonl`.

A cluster scheduler must separately arrange termination signals and a new allocation invoking `--resume`. This package contains no allocation-renewal service.

## Attribution

- [LLaRA](https://github.com/ljy0ustc/LLaRA): `9733f732399e95e236f4f2e1d1139c1f0437090e`
- [LLM-SRec](https://github.com/Sein-Kim/LLM-SRec): `b81019ca655fb759cee895924b8b6c7cc0f0cce9`
- [FCRLLM](https://github.com/Byungmoon-Heo/FCRLLM): `8656e45aa6162241d47cb4d57b3e6d5928b01886`
- [UCGRec](https://github.com/KimHanBeul/UCGRec): `586cbf476d432deb13d6f7e9aa93b8636c8eb60f`
- [BSARec](https://github.com/yehjin-shin/BSARec): `c80bdc0f1a889b78fe9a27e2058389cc84caea84`

Additional source identities and available license texts are in `third_party/UPSTREAM.json` and `third_party/licenses`. Native source remains in external checkouts; each manifest entry records its license status. Model and dataset terms apply separately. CPU fixtures verify setup, scoring and recovery behavior. Full GPU training, pretrained-model inference and paid API execution have not been qualified for the added methods.
