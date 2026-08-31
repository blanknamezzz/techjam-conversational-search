# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

This branch contains an offline-first hybrid implementation. The default V6
agent combines BM25, local sentence embeddings, conversation-state tracking,
RRF fusion, and constraint-aware reranking. V9 optionally adds an external LLM
only for intent and constraint understanding; the LLM never selects product IDs.

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
  -> weighted BM25 Top 160 + local MiniLM Dense Top 200
  -> weighted reciprocal-rank fusion
  -> MATCH / VIOLATION / UNKNOWN plus exact-phrase constraint reranking
  -> explicit hard-constraint gate
  -> catalog-valid Top 10
```

Two maintained variants are available:

| Variant | Purpose |
|---|---|
| `v6` | Offline-first hybrid search; current default and final validated model |
| `v9` | V6 plus confidence-gated LLM understanding for exploratory or complex turns |

### Current results

The fixed public split contains 160 development sessions and 40 validation
sessions. These metrics are session averages; the private 800-session organizer
set is not available locally.

| Variant and split | Hit Rate@10 | MRR | MTTC | Technical score |
|---|---:|---:|---:|---:|
| Official weak baseline, all 200 public sessions | 0.1250 | 0.068034 | 9.8100 | 0.106710 |
| V6, 160-session development split | 0.9250 | 0.594204 | 3.2438 | 0.795886 |
| V6, 40-session validation split | 0.9500 | 0.554573 | 2.8250 | 0.804872 |
| V9 historical LLM run, 160-session development split | 0.9375 | 0.559896 | 3.2625 | 0.791469 |

The final V6 pass corrected taxonomy phrases being misread as hard constraints,
moved the open-ended must-have clarification ahead of the size fallback, expanded
Dense recall to 200 candidates, and reduced Dense influence after an explicit
intent override. Generated local evaluation data and experiment JSON files are
Git-ignored. V9 remains an opt-in online experiment rather than a required
runtime dependency; its historical result predates the final V6 robustness pass
and should not be treated as a direct final comparison.

Run a development-split experiment:

```bash
python3 scripts/run_experiment.py \
  --variant v6 \
  --split-ids data/splits/task_a_dev_ids.txt \
  --output reports/experiments/v6_dev.json
```

Run all tests:

```bash
python3 -m unittest discover -v -s tests -p "test_*.py"
```

Run a three-turn demo including an intent override:

```bash
python3 scripts/demo_session.py
```

The optional online query-rewrite client in `starter/optional_api.py` is fully
disabled unless V9 is explicitly selected and all required environment values
are present. V6 remains the offline default.

### Optional V9 API experiment

Create a Git-ignored `.env` file in the repository root and insert a newly
generated key, the provider's OpenAI-compatible base URL, and a model name.
Never put real credentials in `.env.example`, source code, reports, commits, or
chat messages.

```dotenv
TECHJAM_LLM_API_KEY=replace-with-a-new-key
TECHJAM_LLM_BASE_URL=https://your-provider.example/v1
TECHJAM_LLM_MODEL=your-model-name
TECHJAM_LLM_TIMEOUT_SECONDS=15
```

Run the best development-scoring online experiment:

```bash
python3 scripts/run_experiment.py \
  --variant v9 \
  --split-ids data/splits/task_a_dev_ids.txt \
  --output reports/experiments/v9_api_dev.json
```

If credentials are missing, invalid, rate-limited, or time out, V9 falls back
to V6 for that turn. Never paste a real key into Python, Markdown, JSON, Git, or
chat messages. See `reports/Shopping Copilot (SCOPE Agent).md` for the submitted
architecture, results, limitations, and fallback behavior.

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

## Team Contributions

Replace the member labels with the team's real names before the final Devpost
submission.

| Team member | Primary contribution |
|---|---|
| Member A | Technical lead, Agent orchestration, integration, and final runtime contract |
| Member B | Catalog processing, BM25, structured retrieval, and data validation |
| Member C | Intent recognition, constraint extraction, conversation state, and clarification policy |
| Member D | Local semantic retrieval, embedding index, and RRF candidate fusion |
| Member E | Constraint-aware reranking, evaluation, regression tests, and experiment analysis |

## Files

```text
data/public_set.jsonl             200 labeled development sessions
data/splits/                      fixed 160/40 development-validation split
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  enhanced offline Agent entry point
starter/baseline_agent.py         unchanged weak BM25 reference
starter/state_tracker.py          deterministic multi-turn intent state
starter/catalog.py                in-memory catalog and SQLite FTS5 BM25
starter/dense_retriever.py        local MiniLM embedding retrieval
starter/fusion.py                 weighted reciprocal-rank fusion
starter/reranker.py               constraint evidence and final ranking
starter/optional_api.py           guarded OpenAI-compatible V9 client
scripts/build_dense_index.py      local embedding build command
scripts/run_experiment.py         split-aware experiment runner
scripts/build_synthetic_eval_set.py local-only robustness-set generator
scripts/select_hard_synthetic_eval_set.py frozen tune/holdout selector
evaluator/local_evaluator.py      public-set simulator and scorer
reports/Shopping Copilot (SCOPE Agent).md English project description for submission
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
