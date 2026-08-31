# Shopping Copilot — SCOPE Agent

## Project Overview

**SCOPE** stands for **Stateful Conversational Offline Product Engine**. It is a
multi-turn conversational search agent for a frozen catalog of 50,000 clothing,
shoe, and jewelry products. The agent tracks evolving customer preferences, asks
one useful clarification question per turn, and returns a ranked Top 10 at the
same time.

SCOPE is offline-first and combines:

- deterministic session state and explicit intent-override handling;
- weighted SQLite FTS5/Porter BM25 retrieval;
- local `all-MiniLM-L6-v2` semantic retrieval;
- weighted reciprocal-rank fusion;
- `MATCH / VIOLATION / UNKNOWN` constraint evidence;
- exact-phrase reranking and an explicit hard-constraint gate.

```text
User message
    -> conversation state and clarification
    -> BM25 Top 160 + MiniLM Dense Top 200
    -> weighted reciprocal-rank fusion
    -> constraint-aware reranking and hard gate
    -> Top 10 recommendations
```

The official submission path requires no external LLM, API key, hosted vector
database, or network access during inference. A guarded OpenAI-compatible API
adapter is retained as an optional extension, but it is disabled for the
submitted SCOPE results.

### Results compared with the official Baseline

Both systems use the unchanged official evaluator and identical fixed split IDs.
Higher Hit Rate, MRR, and Technical Score are better; lower MTTC is better.

| Split and model | Sessions | Hit Rate@10 | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|---:|
| Development — official Baseline | 160 | 0.112500 | 0.055598 | 9.931250 | 0.094304 |
| Development — SCOPE Agent | 160 | **0.925000** | **0.594204** | **3.243750** | **0.795886** |
| Validation — official Baseline | 40 | 0.175000 | 0.117778 | 9.325000 | 0.156333 |
| Validation — SCOPE Agent | 40 | **0.950000** | **0.554573** | **2.825000** | **0.804872** |

## Setup and Installation Instructions

### 1. Clone the submission branch

```bash
git clone --branch HELLSJ \
  https://github.com/blanknamezzz/techjam-conversational-search.git
cd techjam-conversational-search
```

### 2. Create a Python environment

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Prepare the frozen catalog

Download `catalog.jsonl.gz` from the official participant-kit GitHub Release,
verify it with the published `SHA256SUMS`, and place the extracted file at
`data/catalog.jsonl`:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The catalog is intentionally excluded from Git because it is a large official
data asset.

### 4. Prepare the local semantic model and index

Build the 384-dimensional product embedding cache once:

```bash
python scripts/build_dense_index.py --batch-size 128
```

The command stores the local MiniLM model and approximately 73 MiB product vector
matrix under `artifacts/`. The directory is excluded from ordinary Git. For an
offline scoring environment, build these artifacts before network access is
disabled or unpack a separately provided model-assets archive into the repository
root.

If the Dense artifacts are unavailable, the Agent remains runnable and falls
back to sparse retrieval, but the reported SCOPE score will not be reproduced.

### 5. Optional external API configuration

The external language-model adapter is not required for SCOPE. To experiment
with it locally, create a Git-ignored `.env` containing only locally managed
credentials:

```dotenv
TECHJAM_LLM_API_KEY=replace-with-your-own-key
TECHJAM_LLM_BASE_URL=https://your-provider.example/v1
TECHJAM_LLM_MODEL=your-model-name
TECHJAM_LLM_TIMEOUT_SECONDS=15
```

Never commit a real API key. Missing or invalid API configuration falls back to
the offline search path.

## Steps to Reproduce the Results

The experiment runner uses `v6` as the stable internal configuration identifier
for the submitted SCOPE Agent. It does not represent a separate submitted model.

### 1. Run all regression tests

```bash
python -m unittest discover -s tests -v
```

### 2. Reproduce the official Baseline

```bash
python scripts/run_experiment.py \
  --variant baseline \
  --split-ids data/splits/task_a_dev_ids.txt \
  --output reports/experiments/baseline_dev.json

python scripts/run_experiment.py \
  --variant baseline \
  --split-ids data/splits/task_a_validation_ids.txt \
  --output reports/experiments/baseline_validation.json
```

Expected aggregate results:

| Split | Hit Rate@10 | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|
| Development | 0.112500 | 0.055598 | 9.931250 | 0.094304 |
| Validation | 0.175000 | 0.117778 | 9.325000 | 0.156333 |

### 3. Reproduce SCOPE

```bash
python scripts/run_experiment.py \
  --variant v6 \
  --split-ids data/splits/task_a_dev_ids.txt \
  --output reports/experiments/scope_dev.json

python scripts/run_experiment.py \
  --variant v6 \
  --split-ids data/splits/task_a_validation_ids.txt \
  --output reports/experiments/scope_validation.json
```

Expected aggregate results when the local Dense artifacts are present:

| Split | Hit Rate@10 | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|
| Development | 0.925000 | 0.594204 | 3.243750 | 0.795886 |
| Validation | 0.950000 | 0.554573 | 2.825000 | 0.804872 |

### 4. Run the official full public-set evaluator

```bash
python -m evaluator.local_evaluator
```

This writes the aggregate and per-session output to the Git-ignored
`results.json` file. The official evaluator, public labels, metric formula, and
catalog are not modified.

### 5. Run the deterministic multi-turn demo

```bash
python scripts/demo_session.py
```

The demo shows state accumulation, simultaneous clarification and recommendation,
and an explicit intent override.

## Limitations and Future Improvements

- The fixed public validation split contains only 40 sessions, so the private
  organizer set remains the decisive generalization test.
- Deterministic extraction may miss complex negation, comparison, and relational
  requirements.
- The final reranker is an interpretable weighted model rather than a trained
  cross-encoder or learning-to-rank system.
- Dense model and index artifacts are not stored in ordinary Git; they must be
  built during setup or supplied as a documented release asset.
- The optional online language-model path adds latency, token cost, and provider
  availability risk, so the official model remains offline-first.

With more time, we would evaluate a compact cross-encoder on a larger frozen
holdout, learn calibrated retrieval weights, add more varied human-written
conversation tests, and investigate privacy-reviewed profile personalization.

## Team Member Contributions

The five team members contributed equally, with complementary ownership across
the complete retrieval and evaluation pipeline.

| Team member | Contribution share | Primary contribution |
|---|---:|---|
| Ziyuan Wang | 20% | Technical lead, Agent orchestration, module integration, and final runtime contract |
| Chengxuan Li | 20% | Catalog processing, BM25 retrieval, structured search, and data validation |
| Junrui Li | 20% | Intent recognition, constraint extraction, conversation state, and clarification policy |
| Haoming Li | 20% | Local semantic retrieval, embedding index construction, and RRF candidate fusion |
| Boyu Du | 20% | Constraint-aware reranking, evaluator integration, regression tests, and experiment analysis |
