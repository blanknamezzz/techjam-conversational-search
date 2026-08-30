# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Agent

Python 3.10 or later is recommended. Install the enhanced Agent dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Build the local 384-dimensional product embedding cache once:

```bash
python3 scripts/build_dense_index.py --batch-size 128
```

The generated model and approximately 73 MiB vector matrix are stored under
`artifacts/` and ignored by Git. The Agent automatically falls back to sparse
retrieval if those local files are absent.

Run the official evaluator without modifying its scoring logic:

```bash
python3 -m evaluator.local_evaluator
```

The command writes per-session results and aggregate metrics to `results.json`.
Do not edit the evaluator or public labels when reporting a score.

The original weak BM25 starter is retained in `starter/baseline_agent.py`. Its
official score is Hit Rate@10 `0.125`, MRR `0.068034`, and MTTC `9.81`; see
`docs/baseline_results.json`.

## Enhanced Architecture

The default `starter.Agent` is an offline-first conversational hybrid retriever:

```text
conversation state and clarification
  -> weighted BM25 Top 160 + local MiniLM Dense Top 160
  -> weighted reciprocal-rank fusion
  -> MATCH / VIOLATION / UNKNOWN plus exact-phrase constraint reranking
  -> explicit hard-constraint gate
  -> catalog-valid Top 10
```

Three reproducible variants are available:

| Variant | Purpose |
|---|---|
| `v1` | Stateful clarification and enhanced BM25 only |
| `v2` | V1 plus equal-weight Dense RRF, retained as an ablation |
| `v3` | Weighted Hybrid retrieval and constraint reranking |
| `v4` | V3 with post-override clarification through turn 8 |
| `v5` | V4 plus exact-constraint phrase evidence |
| `v6` | Stronger exact evidence; current default |
| `v7` | V6 plus confidence-gated online LLM understanding; optional |

Run a development-split experiment:

```bash
python3 scripts/run_experiment.py \
  --variant v6 \
  --split-ids data/splits/task_a_dev_ids.txt \
  --output reports/experiments/v3_dev.json
```

Run all tests:

```bash
python3 -m unittest discover -v -s tests -p "test_*.py"
```

Run a three-turn demo including an intent override:

```bash
python3 scripts/demo_session.py
```

The optional online query-rewrite example in `starter/optional_api.py` is fully
disabled unless V7 is explicitly selected and all required environment values
are present. V6 remains the offline default.

### Optional V7 API experiment

Copy `.env.example` to the Git-ignored `.env` file and insert a newly generated
key, the provider's OpenAI-compatible base URL, and a model name:

```bash
cp .env.example .env
```

```dotenv
TECHJAM_LLM_API_KEY=replace-with-a-new-key
TECHJAM_LLM_BASE_URL=https://your-provider.example/v1
TECHJAM_LLM_MODEL=your-model-name
TECHJAM_LLM_TIMEOUT_SECONDS=15
```

Run the V7 development experiment:

```bash
python3 scripts/run_experiment.py \
  --variant v7 \
  --split-ids data/splits/task_a_dev_ids.txt \
  --output reports/experiments/v7_api_dev.json
```

If credentials are missing, invalid, rate-limited, or time out, V7 falls back
to V6 for that turn. Never paste a real key into Python, Markdown, JSON, Git, or
chat messages. See `reports/final_method.md` for results, costs, limitations,
and fallback behavior.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  enhanced offline Agent entry point
starter/baseline_agent.py         unchanged weak BM25 reference
scripts/build_dense_index.py      local embedding build command
scripts/run_experiment.py         split-aware experiment runner
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
