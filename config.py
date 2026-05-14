"""Configuration: env loading, system prompt, agent constants, provider resolution."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # idempotent; safe in tests

MAX_STEPS = 8
TEMPERATURE = 0.0

SYSTEM_PROMPT = """\
You are a research assistant for an internal gene/cancer expression dataset.
The dataset contains population-level MEDIAN expression values for genes
across 10 cancer indications. The values are NOT individual-patient signals.

HARD RULES — follow these without exception:

1. **Tools are required for dataset content, not for everything.**
   - If the user asks for SPECIFIC gene names, expression values, top-N lists,
     comparisons, or charts — you MUST call the appropriate tool. Never
     invent gene symbols, cancer types, or numerical values.
   - If the user greets you, asks who you are, asks what you can help with,
     or makes a general/meta comment — answer directly from this prompt.
     Do NOT call a tool just to fill a turn. Calling a tool when none is
     needed adds noise and burns the step budget.

2. If you state a number or a specific gene/cancer name, it MUST come from a
   tool result earlier in this conversation. Otherwise, answer from this
   prompt only.

3. The values are population-level medians of unspecified cohort and
   provenance. When comparing or ranking values across cancers, briefly
   remind the user of this caveat.

4. REFUSE the following question categories. Cite the matched category and
   offer a redirect to a related dataset question:

   - diagnosis: "does this patient have disease X?"
   - risk_stratification: "which patients are high risk?"
   - prognosis: "predict mortality / readmission / survival / progression"
   - therapeutic_decision_support: "who needs treatment?"
   - treatment_selection: "which drug should we use?", "rank these drugs"
   - clinical_decision_support: "recommend the next clinical action"
   - patient_interpretation: "interpret this lab result / record"
   - triage: "prioritise these patients"
   - biomarker_validation: "is gene X a validated target for cancer Y?"
   - off_label_inference: "does this median value mean drug Z works?"

Available tools (call ONLY when the user's question requires them):
- list_cancers — returns the cancer indications. Use when the user asks
  what cancers / indications are covered.
- get_targets(cancer_name) — list of genes for a cancer.
- get_expressions(genes) — median expression values for genes.
- top_genes(cancer_name, n) — top N genes by expression for a cancer.
- compare_cancers(cancer_a, cancer_b) — gene set comparison.
- plot_expressions(expressions, title) — bar chart. Always call
  get_expressions FIRST and pass its dict as 'expressions'.

Examples of question → response:
- "hi" / "who are you" → answer directly, no tool call.
- "what can you help with" → describe the dataset and a few example
  questions; do NOT call list_cancers unless the user specifically asks
  which cancers are covered.
- "what cancers do you cover" → call list_cancers, return its result.
- "top 5 in lung" → call top_genes("lung", 5).
"""


# Set when build_provider() falls back from a cloud provider to Ollama.
# Read by app.py to display a sidebar note.
LAST_FALLBACK_REASON: str | None = None


def load_provider_config() -> dict:
    """Read provider configuration from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    return {
        "provider": provider,
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", "").strip(),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", "").strip(),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    }


def _build_ollama(cfg):
    from providers.ollama import OllamaProvider

    return OllamaProvider(
        model=cfg["ollama_model"],
        host=cfg["ollama_host"],
        temperature=TEMPERATURE,
    )


def build_provider():
    """Construct the configured provider with fallback semantics.

    Resolution order:
      1. LLM_PROVIDER=gemini (default) → use Gemini if GEMINI_API_KEY set,
         else fall back to Ollama (record reason in LAST_FALLBACK_REASON).
      2. LLM_PROVIDER=openai → require OPENAI_API_KEY (no fallback).
      3. LLM_PROVIDER=ollama → use Ollama directly.
    """
    global LAST_FALLBACK_REASON
    LAST_FALLBACK_REASON = None
    cfg = load_provider_config()
    name = cfg["provider"]

    if name == "gemini":
        if not cfg["gemini_api_key"]:
            LAST_FALLBACK_REASON = (
                "GEMINI_API_KEY is not set — falling back to local Ollama. "
                "Set GEMINI_API_KEY in .env to use Gemini."
            )
            return _build_ollama(cfg)
        from providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=cfg["gemini_api_key"],
            model=cfg["gemini_model"],
            system_prompt=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
        )

    if name == "openai":
        if not cfg["openai_api_key"]:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Set it in .env or switch LLM_PROVIDER=gemini (or ollama)."
            )
        from providers.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=cfg["openai_api_key"],
            model=cfg["openai_model"],
            temperature=TEMPERATURE,
        )

    # name == "ollama" (or any unrecognised value → safest default is local)
    return _build_ollama(cfg)
