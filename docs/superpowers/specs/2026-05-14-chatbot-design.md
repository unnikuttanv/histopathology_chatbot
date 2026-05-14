# Chatbot — Agentic Data Assistant Design

**Date:** 2026-05-14
**Status:** Revised — providers reordered (Gemini default / Ollama fallback / OpenAI opt-in)
**Context:** Take-home exercise. Build a proof-of-concept agentic product that lets non-technical stakeholders interact in natural language with a small gene/cancer expression dataset, orchestrating two provided functions (`get_targets`, `get_expressions`) and a few helper tools.

---

## 1. Goals & non-goals

**Goals**
- Natural-language chat interface usable by a non-technical stakeholder.
- Orchestrate the two provided functions over `dataset.csv`.
- Add a small set of helper tools (comparison, top-N, chart) that demonstrate multi-tool reasoning.
- Strong grounding — answers come from the dataset, not the model's prior knowledge.
- Runnable locally on Mac/Windows, no GPU, Docker-first.
- **LLM provider hierarchy:** Gemini cloud is the **default** (best demo experience, strong tool calling). Ollama local is the **fallback** — used automatically when `LLM_PROVIDER=gemini` but no API key is configured, or when the user explicitly sets `LLM_PROVIDER=ollama` for a privacy-preserving local run. OpenAI is **opt-in** (`LLM_PROVIDER=openai`).

**Non-goals (POC scope)**
- Production-grade auth, multi-user state, persistence beyond the running process.
- Free-form code execution over the dataframe.
- A second LLM as a post-hoc fact-checker.
- **Clinical use of any kind.** No diagnosis, prognosis, risk stratification, treatment selection, drug recommendation/ranking, clinical decision support, patient-specific interpretation, triage/prioritisation, or biomarker validation. See §7.5 for the full taxonomy and how it's enforced.
- Acceptance of patient-identifiable input. See §7.5 PII pre-filter.

---

## 2. Dataset

`dataset.csv` — 81 rows, 3 columns: `cancer_indication`, `gene`, `median_value`.

Ten cancer indications present:
`breast, colorectal, gastric, glioblastoma, lung, melanoma, ovarian, pancreatic, prostate, renal`.

**Esophageal is intentionally absent** — used as a test for unknown-cancer handling.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                 │
│   - persistent safety banner + first-run modal (§7.5)   │
│   - chat input / message history (st.session_state)     │
│   - "New conversation" reset button                     │
│   - renders text + matplotlib charts inline             │
│   - collapsible "Tool calls" trace per assistant turn   │
└──────────────────────┬──────────────────────────────────┘
                       │ user message
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Safety pre-filters  (safety.py)  ── run BEFORE LLM     │
│   1. detect_pii(text)         → refuse + redacted log   │
│   2. detect_clinical_question → refuse + categorised log│
└──────────────────────┬──────────────────────────────────┘
                       │ passes
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Agent loop  (agent.py) — provider-agnostic             │
│   while provider returns tool calls:                    │
│     for each call: dispatch -> tools.DISPATCH[name]     │
│     append tool result, ask provider again              │
│   return final text + figures + trace                   │
│   → audit.log_turn(...)                                 │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌────────────┬───┴───┬────────────┐
    ▼            ▼       ▼            ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌─────────────────┐
│ gemini  │ │ openai │ │ ollama │ │ build_provider  │
│(default)│ │(opt-in)│ │(local  │ │ resolves which  │
│         │ │        │ │fallback)│ │ to use         │
└─────────┘ └────────┘ └────────┘ └─────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Tools  (tools.py) — all read df from data.py           │
│   • get_targets(cancer)         ← given                 │
│   • get_expressions(genes)      ← given                 │
│   • list_cancers()              ← helper                │
│   • compare_cancers(a, b)       ← helper                │
│   • top_genes(cancer, n)        ← helper                │
│   • plot_expressions(mapping)   ← chart                 │
└─────────────────────────────────────────────────────────┘

data.py    — loads dataset.csv once into a module-level df
config.py  — loads env (LLM_PROVIDER, GEMINI_API_KEY, OLLAMA_MODEL, OLLAMA_HOST),
             system prompt, MAX_STEPS
audit.py   — append-only JSONL per session (logs/YYYY-MM-DD-HHMMSS.jsonl)
```

Four small modules + a `providers/` package. `agent.py` has no Streamlit imports and is testable headless.

---

## 4. File layout

```
.
├── app.py              # Streamlit UI only
├── agent.py            # provider-agnostic agent loop
├── tools.py            # tool functions + TOOL_SPEC + DISPATCH
├── data.py             # loads CSV once
├── config.py           # .env, system prompt, MAX_STEPS
├── safety.py           # PII pre-filter, clinical-question pre-filter, refusal taxonomy
├── audit.py            # JSONL audit logging per session
├── providers/
│   ├── __init__.py
│   ├── base.py         # LLMProvider Protocol, ProviderResponse, Call
│   ├── gemini.py       # google-generativeai (default)
│   ├── openai.py       # openai python client (opt-in)
│   ├── ollama.py       # ollama python client (local fallback)
│   └── fake.py         # FakeProvider for agent-loop tests
├── logs/               # JSONL audit logs (gitignored)
├── tests/
│   ├── test_tools.py
│   ├── test_agent_loop.py            # uses FakeProvider
│   ├── test_providers_base.py        # Protocol + dataclasses + FakeProvider
│   ├── test_providers_ollama.py      # Ollama translation, mocked client
│   ├── test_providers_gemini.py      # Gemini translation, mocked client
│   ├── test_providers_openai.py      # OpenAI translation, mocked client
│   ├── test_safety.py                # PII patterns + clinical refusal taxonomy
│   ├── test_audit.py                 # JSONL log shape and redaction
│   └── test_config.py                # env loading + fallback logic
├── dataset.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore           # excludes logs/, .env
└── README.md
```

---

## 5. Tool contracts

| Tool | Args | Returns | Purpose |
|---|---|---|---|
| `get_targets` | `cancer_name: str` | `list[str]` or `{"error", "did_you_mean"}` | Given. |
| `get_expressions` | `genes: list[str]` | `dict[str, float]` | Given. |
| `list_cancers` | — | `list[str]` | Grounds "what cancers do you cover?" |
| `compare_cancers` | `cancer_a: str, cancer_b: str` | `{"shared": [...], "only_a": [...], "only_b": [...]}` | Single named tool reduces set-math hallucination. |
| `top_genes` | `cancer_name: str, n: int = 5` | `list[{gene, median_value}]` sorted desc | Common stakeholder ask. |
| `plot_expressions` | `expressions: dict[str, float], title: str` | `{"chart_id": str}` + figure on side-channel | Forces the agent to fetch values via `get_expressions` first. |

**Design choices**

- `plot_expressions` takes the expression dict, not a cancer name. Forces visible orchestration (`get_expressions` → `plot_expressions`) and ensures the chart matches any filter the user requested.
- `list_cancers` exists so the model can ground itself cheaply instead of guessing.
- Tools never raise to the agent; they return `{"error": ...}` or `{"warning": ...}`. The model sees these as normal outputs and recovers naturally.
- Validation lives in each tool. Unknown cancer → `{"error": "unknown cancer 'X'", "available": [...10 cancers...]}`.

**Tool schema and implementation are colocated in `tools.py`** — one source file, one diff to add a tool, no schema/code drift.

---

## 6. Agent loop

```python
def run_turn(history, user_message) -> AgentResponse:
    # Safety pre-filters run BEFORE any LLM call. See §7.5.
    pii = safety.detect_pii(user_message)
    if pii.matched:
        audit.log_refusal(category="pii", redacted=True, patterns=pii.patterns)
        return AgentResponse(safety.PII_REFUSAL, figures=[], trace=[],
                             refusal_category="pii")

    clinical = safety.detect_clinical_question(user_message)
    if clinical.matched:
        audit.log_refusal(category=clinical.category, user_message=user_message)
        return AgentResponse(safety.refusal_for(clinical.category),
                             figures=[], trace=[],
                             refusal_category=clinical.category)

    history.append(user_msg(user_message))
    figures, trace = [], []

    for step in range(MAX_STEPS):           # MAX_STEPS = 8
        resp = provider.chat(history, tools=TOOL_SPEC)
        if not resp.function_calls:
            history.append(resp.raw_assistant_content)
            audit.log_turn(user_message, trace, resp.text, provider_name)
            return AgentResponse(resp.text, figures, trace)

        history.append(resp.raw_assistant_content)
        tool_msgs = []
        for call in resp.function_calls:
            result, fig = dispatch(call.name, call.args)
            if fig is not None: figures.append(fig)
            trace.append({"name": call.name, "args": call.args, "result": result})
            tool_msgs.append(tool_result_msg(call.name, result))
        history.append(tool_msgs)

    audit.log_turn(user_message, trace, STEP_LIMIT_MSG, provider_name, step_limit_hit=True)
    return AgentResponse(STEP_LIMIT_MSG, figures, trace)
```

**Behaviour**
- **Parallel tool calls in one step** are supported. `MAX_STEPS` counts rounds, not individual calls.
- **History is the source of truth.** Stored in `st.session_state["history"]`; "New conversation" resets it.
- **Step limit hit** → return whatever we have plus a step-limit message; trace still rendered.
- **Unknown tool name** → dispatcher returns `{"error": "unknown tool 'X', available: [...]"}`, model self-corrects.
- **Tool raises** (unexpected) → caught, returned as `{"error": str(e)}`.

---

## 7. Grounding & anti-hallucination controls

Five layers:

1. **System prompt** with hard rules:
   - "You answer only from the tool outputs in this conversation. You do not have prior knowledge about which genes belong to which cancers, or about expression values."
   - "If you state a number, it must come from a tool result earlier in this conversation."
   - "Never invent gene names, cancer types, or values."
   - **Out-of-scope refusal** — full taxonomy and refusal templates in §7.5. The system prompt embeds the category list with example phrasings; the agent refuses and cites the matched category.
2. **Temperature = 0.0** for deterministic dispatch and verbatim numbers.
3. **Tools as the only source of truth.** Unknown inputs return structured errors with `did_you_mean` / `available`. Numerical outputs are exact floats from the DataFrame.
4. **Bounded agent loop.** `MAX_STEPS = 8` per user turn. Stop and surface a clear message if exceeded.
5. **Visible tool trace** in the UI. Every assistant turn has a collapsible expander showing each tool call name, arguments, and result. Transparency makes drift detectable.

**Explicitly not in scope:** a second LLM as fact-checker, RAG over an external knowledge base.

**Production hardening notes (for README):** at production scale, drop the hard refusal of general biology and instead label data-grounded vs. general-knowledge answers, with a domain expert in the loop. Out of scope for the POC.

---

## 7.5 Scope boundaries, intended use, and ethical posture

This system handles oncology-adjacent data. Even as a POC, it must be unambiguous about what it is, what it isn't, and what it will refuse.

### Intended use

> *Research and exploration tool for an internal gene/cancer expression dataset of population-level gene expression medians. **Not a medical device.** Not for diagnosis, prognosis, risk stratification, treatment selection, drug recommendation, clinical decision support, individual-patient interpretation, triage, or biomarker validation.*

This statement appears in the README, the UI sidebar "About" panel, and is summarised in the persistent banner.

### UI safety surfaces

- **Persistent banner** at the top of the chat: *"Research tool only — not a medical device. Not for diagnosis, prognosis, or treatment decisions."* Always visible, cannot be dismissed.
- **First-run modal** before chat input is enabled. Shows the full intended-use statement and requires the user to click "I understand — this is a research tool only" to proceed. Acknowledgement is stored in `st.session_state["safety_ack"]` and re-prompted on every "New conversation".
- **Welcome message** (the assistant's first turn) includes the generalisability caveat once: *"Values shown are population-level medians of unspecified cohort and provenance — they are not individual-patient signals."*
- **Sidebar "About" panel** shows the intended-use statement and the active LLM provider.

### Refusal taxonomy

Enforced in three places: the pre-LLM clinical filter in `safety.py`, the system prompt, and `tests/test_safety.py` gold-standard cases. Every refusal cites its category and redirects to a related in-scope question.

| Category | Example question shape | Why refused |
| --- | --- | --- |
| `diagnosis` | "Does this patient have disease X?" | Diagnosis |
| `risk_stratification` | "Which patients are high risk?" | Clinical risk stratification |
| `prognosis` | "Predict mortality / readmission / progression." | Prognosis |
| `therapeutic_decision_support` | "Who needs treatment?" | Therapeutic decision support |
| `treatment_selection` | "Which drug should we use?" / "Rank these drugs." | Treatment selection / drug ranking |
| `clinical_decision_support` | "Recommend the next clinical action." | Clinical decision support |
| `patient_interpretation` | "Interpret this lab result / patient record." | Patient-specific interpretation |
| `triage` | "Prioritise these patients." | Triage / workflow risk |
| `biomarker_validation` | "Is gene X a validated target for cancer Y?" | Validation claims beyond data |
| `off_label_inference` | "Does this median value mean responsive to drug Z?" | Off-label inference beyond data |

**Refusal template** (per category): *"This system doesn't provide {category description}. It only summarises population-level expression medians from the take-home dataset. You can ask, for example: '{redirect example}'."*

### Pre-LLM clinical filter (defense in depth)

`safety.detect_clinical_question(text) -> ClinicalMatch` runs before any LLM call. Lightweight regex / keyword matcher over the user message. Match categories use compact patterns such as:

- `risk_stratification`: `\b(which|what|who)\b.*\b(patient|patients|case|cases)\b.*\b(high|low)?\s*risk\b`
- `diagnosis`: `\bdoes\b.*\bpatient\b.*\bhave\b` or `\bdiagnos(e|is)\b.*\bpatient\b`
- `treatment_selection` / `clinical_decision_support`: `\b(recommend|should we|next action|prescribe|treat)\b.*\b(patient|case)\b`
- `triage`: `\b(prioritis|prioritiz|triage|rank)\b.*\b(patient|case)\b`
- `patient_interpretation`: `\b(interpret|explain)\b.*\b(this|the)\s+(lab|result|record|chart|scan)\b`
- `prognosis`: `\b(predict|forecast)\b.*\b(mortality|readmission|survival|outcome|progression)\b`

Matches return a fixed refusal citing the category, log a refusal event to the audit JSONL, and never reach the LLM. False positives are recoverable — the refusal text invites the user to rephrase against the dataset.

This is **best-effort, not a clinical-grade safety classifier.** The system prompt's enumerated taxonomy is the second line; tests pin both.

### PII pre-filter

`safety.detect_pii(text) -> PIIMatch` runs before any LLM call. Light regex over the user message:

- Email: standard pattern
- Phone: NANP + international common patterns
- Date of birth: `\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b` (and ISO variant)
- MRN-like patterns: `\b(MRN|Patient\s*ID|Medical\s*Record)\b[:#\s]+\S+`
- Name-adjacent clinical signal: `(?i)\bmr\.?\s+\w+|mrs\.?\s+\w+|patient\s+[A-Z]\w+` (heuristic only)

If matched: input is **not** sent to the LLM, **not** written to the audit log (we record only a redacted note: `{"event": "refusal", "category": "pii", "patterns": ["email", "dob"], "redacted": true}`). Returns a fixed refusal. README states this is best-effort and not a substitute for clinical-grade DLP.

### Audit log

`audit.log_turn(...)` writes one JSONL line per turn to `logs/YYYY-MM-DD-HHMMSS.jsonl` (one file per session, set at app start). **Single unified schema** for normal and refusal turns — refusals just set `refusal_category` and skip LLM-side fields:

```json
{
  "ts": "2026-05-14T12:34:56.789Z",
  "turn": 3,
  "user_message": "top 5 in lung",
  "provider": "ollama",
  "model": "llama3.1:8b",
  "tool_calls": [
    {"name": "top_genes", "args": {"cancer_name": "lung", "n": 5}, "result": {...}}
  ],
  "final_response": "...",
  "step_count": 2,
  "step_limit_hit": false,
  "refusal_category": null,
  "redacted_patterns": null
}
```

**Refusal turns** keep the same fields:

- `tool_calls`: `[]`, `step_count`: `0`, `step_limit_hit`: `false`.
- `final_response`: the refusal text shown to the user.
- `refusal_category`: one of `"pii"` or any §7.5 clinical category.
- **PII refusal additionally:** `user_message` set to `"<redacted>"`, `redacted_patterns` set to the list of pattern names that matched (e.g. `["email", "dob"]`). The original message is never written to disk.
- **Clinical refusal:** `user_message` is preserved (no PII was detected), `redacted_patterns` is `null`.

The log is local-only, append-only, no rotation; `logs/` is gitignored. README states the log's purpose (research-grade traceability for reproducibility), location, and that it contains conversation text — so it should not be shared without review.

### Production hardening notes

For real clinical use, the refusal taxonomy and PII filter would need to be replaced by validated classifiers, the audit log would need tamper-evidence and retention policy, the disclaimer would need legal review, and the system would require a Quality Management System aligned with the relevant regulatory class. None of that is in scope for this POC; the structure here is the right *shape* so those upgrades are localised changes.

---

## 8. LLM provider abstraction

```python
# providers/base.py
@dataclass
class Call:
    name: str
    args: dict

@dataclass
class ProviderResponse:
    text: str | None
    function_calls: list[Call]
    raw_assistant_content: Any  # native format, kept for history

class LLMProvider(Protocol):
    def chat(self, history, tools) -> ProviderResponse: ...
```

Each provider owns translation between our normalized history and its native format. The agent loop is single-purpose: orchestrate calls, build a trace, return text + figures.

**Providers**

- **`providers/gemini.py`** — **default** (`LLM_PROVIDER=gemini`). Uses `google-generativeai`, native function calling. Requires `GEMINI_API_KEY`. Best demo experience, strongest tool calling at lowest cost.
- **`providers/openai.py`** — **opt-in** (`LLM_PROVIDER=openai`). Uses the `openai` Python client. Requires `OPENAI_API_KEY`. Our canonical history format is already OpenAI-shaped, so this provider is the thinnest — no translation layer.
- **`providers/ollama.py`** — **local fallback** (`LLM_PROVIDER=ollama`, or automatic when a cloud provider has no key). Uses the `ollama` Python client against a local Ollama server (`OLLAMA_HOST=http://localhost:11434`). Default model `llama3.1:8b` (tool-use capable, CPU-runnable). The privacy-preserving mode.

**Resolution order in `config.build_provider()`:**

1. Read `LLM_PROVIDER` from env (default: `gemini`).
2. If `gemini` and `GEMINI_API_KEY` is set → use Gemini.
3. If `gemini` and `GEMINI_API_KEY` is missing → **log a warning and fall back to Ollama**. The UI sidebar reflects which provider is actually in use.
4. If `openai` → require `OPENAI_API_KEY`. No fallback (explicit opt-in; missing key is a configuration error, fail loud).
5. If `ollama` → use Ollama directly.

This makes a fresh checkout "just work" for the reviewer: drop a Gemini key in `.env` and run; or skip the key, install Ollama, and get a local run.

**Defensive parsing** (Ollama in particular — smaller models are flakier at tool dispatch; Gemini and OpenAI inherit the same protections). Each case is handled at a specific layer:

- **Malformed JSON in tool args** (tool layer) → dispatch returns `{"error": "bad arguments", "received": "..."}` to the model as a normal tool result; the model retries with corrected args on the next loop iteration. Counts as one normal step.
- **Unparseable provider response** (provider layer) → if `providers/ollama.py` cannot extract either a clean tool call list or final text from the model's output, it returns `ProviderResponse(text="<unparseable response>", function_calls=[])` and the agent loop ends that turn. No silent loop.
- **Skipped tool call, answered from memory** → caught by the system prompt's hard rule (the answer will be flagged in the visible trace as having no tool calls; reviewer can see it).
- **Unknown tool name** → dispatcher returns `{"error": "unknown tool 'X', available: [...]"}`; model self-corrects on the next step.

---

## 9. UI behaviour

- **Persistent safety banner** at the top of the chat (see §7.5): *"Research tool only — not a medical device. Not for diagnosis, prognosis, or treatment decisions."* Always visible, not dismissable.
- **First-run modal** before chat input is enabled. Shows the full intended-use statement; requires "I understand" click. Re-prompted on every "New conversation". Sets `st.session_state["safety_ack"] = True`.
- **Cold start:** once acknowledged, render a fixed welcome assistant message:
  *"Hi — I'm a data agent for the gene/cancer dataset (10 cancer types, expression medians). **Heads up:** values shown are population-level medians of unspecified cohort and provenance — not individual-patient signals. Try: 'What targets are in lung cancer?', 'Compare breast and prostate', 'Top 5 genes in pancreatic'."*
- **Chat input:** standard `st.chat_input`, disabled until `safety_ack` is set. On submit, call `agent.run_turn(history, msg)`.
- **Assistant turn rendering:** message text → any returned matplotlib figures → collapsible "🔍 Tool calls" expander listing each `{name, args, result}`. Refusal turns render the refusal text and a small "Why this was refused" expander citing the category.
- **"New conversation" button** in the sidebar → clears `st.session_state["history"]` and `safety_ack`; first-run modal re-appears.
- **Sidebar "About" panel** with the full intended-use statement and the active LLM provider.
- **Provider indicator** in the sidebar showing which LLM is actually active (Gemini cloud / OpenAI cloud / Ollama local). When the requested provider was unavailable and we fell back to Ollama, the sidebar shows the fallback reason ("GEMINI_API_KEY missing — using Ollama").
- **Capability question handling:** "How can you help me?" is answered via the system prompt + a `list_cancers()` call so the response is grounded.

---

## 10. Configuration

`.env.example`:

```
# LLM provider: "gemini" (default), "openai" (opt-in), or "ollama" (local fallback)
LLM_PROVIDER=gemini

# Gemini (default) — leave key blank to auto-fall-back to Ollama
GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-flash

# OpenAI (opt-in) — required when LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Ollama (local fallback, or LLM_PROVIDER=ollama for explicit privacy mode)
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
```

`config.py` exposes `SYSTEM_PROMPT`, `MAX_STEPS = 8`, `TEMPERATURE = 0.0`, `load_provider_config()`, and `build_provider()` (which implements the resolution order in §8).

---

## 11. Testing

- **`tests/test_tools.py`** — pytest unit tests per tool: known cancer, unknown cancer + `did_you_mean`, empty gene list, `top_genes` ordering, `compare_cancers` set logic. Pure functions, no LLM.
- **`tests/test_agent_loop.py`** — agent loop with a `FakeProvider` that emits scripted function calls. Covers: single call → final answer, multi-step orchestration, parallel calls in one step, step-limit handling, unknown-tool dispatch path, **early-exit on safety refusals (PII and clinical) before any provider call**. No network. Runs in <1s.
- **`tests/test_providers_base.py`** — `Call`, `ProviderResponse`, and `FakeProvider` semantics.
- **`tests/test_providers_ollama.py`** — Ollama translation with a mocked client. Covers: tool calls present, plain text, JSON-string args, malformed args, unparseable response.
- **`tests/test_providers_gemini.py`** — Gemini translation with a mocked client. Covers: text response, function call response, exception path.
- **`tests/test_providers_openai.py`** — OpenAI translation with a mocked client. Covers: tool calls present, plain text, JSON-string args (always string in OpenAI), exception path.
- **`tests/test_config.py`** — `load_provider_config()` defaults + `build_provider()` resolution order: Gemini with key → Gemini; Gemini without key → Ollama fallback; OpenAI without key → raises; explicit `ollama` → Ollama.
- **`tests/test_safety.py`** — safety filters:
  - PII pre-filter: each pattern (email, phone, DOB, MRN, name-adjacent) detected; legitimate dataset questions ("top 5 in lung", "compare breast and prostate") do **not** trigger.
  - Clinical pre-filter: each gold-standard example question from the §7.5 taxonomy triggers a refusal with the correct category. Each refusal text cites its category and offers a redirect example.
  - Negative cases: dataset-shaped questions like "Which genes are in lung?" do not match any clinical pattern.
- **`tests/test_audit.py`** — JSONL log shape and redaction:
  - Normal turn logs all fields (ts, turn, user_message, provider, model, tool_calls, final_response, step_count, step_limit_hit, refusal_category=null).
  - PII-refused turn logs only redacted metadata, never the original message.
  - Clinical-refused turn logs the original message and the matched category.
  - File is append-only and one-line-per-event.

Run: `pytest -q`. Target: all green, no network required.

---

## 12. Docker

`Dockerfile` — `python:3.11-slim`, copy code, `pip install -r requirements.txt`, `EXPOSE 8501`, `CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]`.

`docker-compose.yml` — single service, mounts `.env`, port 8501. Ollama is expected on the host (`OLLAMA_HOST=http://host.docker.internal:11434`); compose file documents this so the reviewer doesn't have to bundle Ollama into the image.

---

## 13. README contents (outline)

1. What it is — one-paragraph product description.
2. **Intended use and limitations** (top of README, before the install steps): the full statement from §7.5, the refusal taxonomy table, and an explicit "not a medical device" line.
3. **Provider modes** — default Gemini (cloud), fallback Ollama (local), opt-in OpenAI. Quick table showing which env vars unlock which mode.
4. Quick start with Docker (`docker compose up`) — works with any provider; default Gemini path needs `GEMINI_API_KEY` in `.env`.
5. Quick start without Docker (`pip install -r requirements.txt && streamlit run app.py`).
6. **Privacy-preserving local mode** — how to set `LLM_PROVIDER=ollama` (or leave keys blank to auto-fall-back), Ollama install instructions.
7. **Switching to OpenAI** — `.env` change + `OPENAI_API_KEY`.
8. Example questions (covering the checklist test cases including the esophageal unknown) and example questions the agent will refuse (one per refusal category).
9. Design choices — why three providers, why native function calling, the five anti-hallucination layers, the §7.5 safety posture (intended use, taxonomy, pre-filters, audit log), what's out of scope for a POC and why (e.g. no second-LLM fact-checker, no clinical reasoning, best-effort PII detection only).
10. Audit log — location (`logs/...jsonl`), schema, the fact that it contains conversation text, retention/sharing guidance.
11. Testing — `pytest -q`.
12. Limitations & next steps — production hardening (validated safety classifier instead of regex, tamper-evident logs, label grounded vs general-knowledge answers, domain expert in the loop, QMS for clinical use), persistence, observability, evaluation harness.

---

## 14. Requirements checklist mapping

| Requirement | Met by |
|---|---|
| Answers "How can you help me?" | Welcome message + system-prompt capability description + `list_cancers()` grounding (§9) |
| Gets lung cancer genes | `get_targets("lung")` (§5) |
| Gets breast cancer median expression | `get_targets` → `get_expressions` orchestration (§5–6) |
| Esophageal cancer (unknown) | `did_you_mean` / `available` path (§5) |
| Uses provided CSV | `data.py` (§3) |
| Uses provided functions | `get_targets`, `get_expressions` kept verbatim (§5) |
| Natural-language interface | Streamlit chat (§3, §9) |
| Runnable locally | `pip install && streamlit run` (§13) |
| README included | §13 |
| Docker preferred | `Dockerfile` + `docker-compose.yml` (§12) |
| No GPU required | Ollama runs CPU; Gemini is API (§8) |
| Python | All modules (§4) |
| Reliability — known + unknown | `did_you_mean`, step limit, tool errors-as-data (§5, §6) |
| Reproducibility | Temperature 0.0; deterministic dispatch (§7) |
| Portability Mac/Windows | Python + Streamlit + Docker (§12) |
| Performance — small dataset | 81 rows in pandas; loads instantly |
| Memory under 16 GB | Llama3.1:8b ≈ 5 GB RAM; well under 16 GB |
| Security — no arbitrary execution | Fixed `DISPATCH` table; no code-exec tool (§5) |
| Privacy — no external API by default | **Partially met.** Default mode is cloud (Gemini) for demo convenience. **Privacy-preserving mode is one config change:** `LLM_PROVIDER=ollama` (or simply leave `GEMINI_API_KEY` blank to auto-fall-back to Ollama). PII pre-filter blocks patient data from reaching ANY provider, even the default (§7.5). Documented prominently in README (§13). |
| Maintainability | 4 modules + `providers/` package, colocated schema + impl (§3–4) |
| Testability | Three test files; `FakeProvider` for loop tests (§11) |
| Explainability | README design choices section (§13) |
| Clinical safety / ethics | Intended-use statement + UI banner + first-run modal + refusal taxonomy + pre-LLM clinical filter + PII pre-filter + audit log (§7.5, §6, §9) |
| Traceability / auditability | JSONL audit log per session with full turn schema (§7.5, §11) |

---

## 15. Open questions / future work

- Evaluation harness (gold-standard Q&A pairs, automated scoring) — out of scope for POC, noted as next step.
- Streaming responses for better UX — Streamlit supports it, deferred for simplicity.
- Multi-turn refinement (clarifying questions from the agent when input is ambiguous) — partially handled by `did_you_mean`, could be richer.
- Persistence of conversation history beyond the running process — explicitly out of scope.
