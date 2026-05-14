"""Streamlit chat UI for the Owkin take-home agent."""
from __future__ import annotations

import streamlit as st

import audit
import config
from agent import run_turn

BANNER = (
    "Research tool only — not a medical device. "
    "Not for diagnosis, prognosis, or treatment decisions."
)

INTENDED_USE = (
    "**Intended use.** Research and exploration tool for an internal Owkin "
    "take-home dataset of population-level gene expression medians. "
    "**Not a medical device.** Not for diagnosis, prognosis, risk "
    "stratification, treatment selection, drug recommendation, clinical "
    "decision support, individual-patient interpretation, triage, or "
    "biomarker validation."
)

WELCOME = (
    "Hi — I'm a data agent for the gene/cancer dataset (10 cancer types, "
    "expression medians). **Heads up:** values shown are population-level "
    "medians of unspecified cohort and provenance — not individual-patient "
    "signals.\n\n"
    "Try: 'What targets are in lung cancer?', 'Compare breast and prostate', "
    "'Top 5 genes in pancreatic'."
)


def _init_session():
    if "audit_initialised" not in st.session_state:
        audit.init_session("logs")
        st.session_state.audit_initialised = True
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.display = []
    if "safety_ack" not in st.session_state:
        st.session_state.safety_ack = False
    if "provider" not in st.session_state:
        try:
            st.session_state.provider = config.build_provider()
            st.session_state.provider_error = None
        except Exception as e:
            st.session_state.provider = None
            st.session_state.provider_error = str(e)


def _reset_conversation():
    st.session_state.history = []
    st.session_state.display = []
    st.session_state.safety_ack = False
    audit.init_session("logs")
    st.session_state.audit_initialised = True


def _render_sidebar():
    with st.sidebar:
        st.markdown("### About")
        st.markdown(INTENDED_USE)
        provider = st.session_state.provider
        if provider is not None:
            st.markdown(f"**Active LLM:** `{provider.name}` (`{provider.model}`)")
        if config.LAST_FALLBACK_REASON:
            st.info(config.LAST_FALLBACK_REASON, icon="ℹ️")
        if st.session_state.get("provider_error"):
            st.error(st.session_state.provider_error)
        if st.button("New conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()


def _render_safety_modal():
    st.warning(BANNER)
    with st.container(border=True):
        st.markdown("### Before you start")
        st.markdown(INTENDED_USE)
        if st.button("I understand — this is a research tool only", type="primary"):
            st.session_state.safety_ack = True
            st.session_state.display.append({"role": "assistant", "text": WELCOME})
            st.rerun()


def _render_history():
    for msg in st.session_state.display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            for fig in msg.get("figures", []) or []:
                st.pyplot(fig, clear_figure=False)
            trace = msg.get("trace") or []
            if trace:
                with st.expander("Tool calls"):
                    for i, call in enumerate(trace, 1):
                        st.markdown(f"**{i}. `{call['name']}`**")
                        st.json({"args": call["args"], "result": call["result"]})
            if msg.get("refusal_category"):
                with st.expander("Why this was refused"):
                    st.markdown(
                        f"Category: `{msg['refusal_category']}`. "
                        f"See the README for the full refusal taxonomy."
                    )


def main():
    st.set_page_config(page_title="Owkin Data Agent", page_icon=":dna:")
    _init_session()

    st.markdown(f":red-background[{BANNER}]")
    st.title("Owkin Data Agent")

    _render_sidebar()

    if st.session_state.get("provider_error"):
        st.error(
            f"Provider initialization failed: {st.session_state.provider_error}. "
            "Edit `.env` and restart."
        )
        return

    if not st.session_state.safety_ack:
        _render_safety_modal()
        return

    _render_history()

    user_message = st.chat_input("Ask about the dataset")
    if not user_message:
        return

    st.session_state.display.append({"role": "user", "text": user_message})
    with st.spinner("Thinking..."):
        resp = run_turn(
            st.session_state.history,
            user_message,
            provider=st.session_state.provider,
            max_steps=config.MAX_STEPS,
            system_prompt=config.SYSTEM_PROMPT,
        )
    st.session_state.display.append(
        {
            "role": "assistant",
            "text": resp.text,
            "figures": resp.figures,
            "trace": resp.trace,
            "refusal_category": resp.refusal_category,
        }
    )
    st.rerun()


if __name__ == "__main__":
    main()
