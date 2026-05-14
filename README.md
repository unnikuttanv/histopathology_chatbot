# Owkin Take-Home — Agentic Data Assistant

A Streamlit chat agent that lets a non-technical stakeholder query a small
gene/cancer expression dataset in natural language. The agent orchestrates
six tools over the dataset via an LLM (**Gemini cloud by default**, Ollama
local as fallback, OpenAI as opt-in), with pre-LLM safety filters, an audit
log, and a strict refusal taxonomy.

---

## Intended use and limitations

**Research and exploration tool only.** Operates on a small de-identified
research dataset of population-level gene expression medians.

**Not a medical device.** The agent refuses any question that falls into
the following categories:

| Category | Example shape |
| --- | --- |
| `diagnosis` | "Does this patient have disease X?" |
| `risk_stratification` | "Which patients are high risk?" |
| `prognosis` | "Predict mortality / readmission / survival." |
| `therapeutic_decision_support` | "Who needs treatment?" |
| `treatment_selection` | "Which drug should we use?" / "Rank these drugs." |
| `clinical_decision_support` | "Recommend the next clinical action." |
| `patient_interpretation` | "Interpret this lab result." |
| `triage` | "Prioritise these patients." |
| `biomarker_validation` | "Is gene X a validated target for Y?" |
| `off_label_inference` | "Does this median mean drug Z works?" |

The agent also refuses any input containing patient-identifiable data
(emails, phone numbers, dates of birth, MRN-like patterns, name-adjacent
clinical phrasing). PII detection is best-effort, not a clinical-grade DLP.

---

## Provider modes

| `LLM_PROVIDER` | Behaviour | Required env |
| --- | --- | --- |
| `gemini` (default) | Use Gemini if `GEMINI_API_KEY` set; **fall back to local Ollama** if not. | `GEMINI_API_KEY` (optional) |
| `openai` (opt-in) | Use OpenAI. Missing key is a config error (no fallback). | `OPENAI_API_KEY` |
| `ollama` (explicit local) | Always use local Ollama. The privacy-preserving mode. | Ollama installed locally |

> Privacy note: the default sends prompts + tool results to Google. The
> dataset itself is not transmitted. For a fully local run, set
> `LLM_PROVIDER=ollama` or simply leave `GEMINI_API_KEY` blank.

---

## Quick start (Docker)

1. Copy the env template:

   ```bash
   cp .env.example .env
   ```

2. Pick a provider:
   - **Gemini (default):** put your `GEMINI_API_KEY` in `.env`.
   - **OpenAI (opt-in):** set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.
   - **Ollama (local):** set `LLM_PROVIDER=ollama` (or leave keys blank).
     Install [Ollama](https://ollama.com/) and run `ollama pull llama3.1:8b`.

3. Build and run:

   ```bash
   docker compose up --build
   ```

4. Open <http://localhost:8501>.

On Linux hosts using the Ollama path, replace `host.docker.internal` with
your host's IP in `docker-compose.yml` (or use
`--add-host=host.docker.internal:host-gateway`).

## Quick start (without Docker)

```bash
python -m venv .venv
. .venv/Scripts/activate    # Windows
# or: source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Switching providers at runtime

Edit `.env` and restart the app. The sidebar shows which provider is
actually active — if `LLM_PROVIDER=gemini` but no key is set, the sidebar
will say "Using Ollama (GEMINI_API_KEY missing)".

---

## Example questions

**In scope** (the agent will answer):

- "What can you help me with?"
- "What cancers do you cover?"
- "What are the targets for lung cancer?"
- "Show median expression for the genes in breast."
- "Top 5 genes by expression in pancreatic cancer."
- "Compare breast and prostate."
- "Plot the top 3 expressions in glioblastoma."

**Unknown cancer** (graceful handling):

- "What about esophageal cancer?" → returns `did_you_mean` / `available`.

**Refused** (with category):

- "Which patients are high risk?" → `risk_stratification`
- "Recommend the next clinical action." → `clinical_decision_support`
- "Interpret this lab result." → `patient_interpretation`
- "Contact jane@example.com about this." → PII refusal.

---

## Design choices

- **Native function calling, not prompt-and-parse.** All three providers
  expose structured tool calling; we use it directly. Less brittle than
  parsing JSON out of free-form text.
- **Provider abstraction.** The agent loop is provider-agnostic. Each
  provider owns translation between our canonical OpenAI-style history and
  its native format.
- **Tools as the source of truth.** The model has no prior knowledge in
  scope. Numbers come from tool results; unknown inputs return structured
  errors (`did_you_mean` / `available`) so the model can self-correct.
- **Five anti-hallucination layers.** Hard rules in the system prompt;
  `temperature=0`; tools-only data; bounded loop (`MAX_STEPS=8`); visible
  tool-call trace in the UI.
- **Two-line safety posture.** Pre-LLM regex filters (PII + clinical
  taxonomy) plus the system-prompt taxonomy. Defense in depth.
- **Audit log.** One JSONL file per session in `logs/`, with a unified
  schema. Used for reproducibility and traceability.

---

## Audit log

Each chat session writes one append-only file: `logs/YYYY-MM-DDTHHMMSSZ.jsonl`.
One line per turn (normal or refusal). Schema:

```json
{
  "ts": "2026-05-14T12:34:56.789Z",
  "turn": 3,
  "user_message": "top 5 in lung",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "tool_calls": [{"name": "...", "args": {}, "result": {}}],
  "final_response": "...",
  "step_count": 2,
  "step_limit_hit": false,
  "refusal_category": null,
  "redacted_patterns": null
}
```

The log contains conversation text — don't share it without review. PII
refusals redact `user_message` to `<redacted>` and list matched patterns
in `redacted_patterns`.

---

## Testing

```bash
python -m pytest -q
```

All tests are offline (no network). Coverage:

- `tests/test_data.py` — dataset loader
- `tests/test_tools.py` — six tools + dispatch + TOOL_SPEC
- `tests/test_safety.py` — PII patterns + clinical taxonomy gold cases
- `tests/test_audit.py` — JSONL log shape + PII redaction
- `tests/test_providers_base.py` / `_ollama.py` / `_gemini.py` / `_openai.py`
  — provider translation with mocked clients
- `tests/test_agent_loop.py` — full agent loop with `FakeProvider`
- `tests/test_config.py` — env loading + fallback resolution + system prompt content

---

## Project structure

```
.
├── app.py              # Streamlit UI
├── agent.py            # provider-agnostic agent loop
├── tools.py            # 6 tools + TOOL_SPEC + DISPATCH
├── data.py             # loads CSV once
├── config.py           # env, system prompt, build_provider()
├── safety.py           # PII + clinical pre-filters + refusal taxonomy
├── audit.py            # JSONL per-session audit log
├── providers/
│   ├── base.py         # LLMProvider Protocol, dataclasses
│   ├── gemini.py       # google-genai (default)
│   ├── openai.py       # openai (opt-in)
│   ├── ollama.py       # ollama (local fallback)
│   └── fake.py         # FakeProvider for tests
├── tests/              # full offline test suite
├── owkin_take_home_data.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Limitations and next steps

Out of scope for this POC, but the right shape for them is in place:

- Replace regex-based safety filters with validated classifiers.
- Tamper-evident audit log with retention policy.
- Label data-grounded vs general-knowledge answers with a domain expert in
  the loop, rather than hard-refusing all general biology.
- Streaming responses.
- Evaluation harness with gold-standard Q&A pairs.
- Persistence beyond the running process.
- QMS alignment if any clinical use is ever contemplated.
