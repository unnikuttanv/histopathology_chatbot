# Roadmap

Planned work, roughly in priority order. Effort estimates are rough.

## Now — robustness

- [ ] **Streaming responses** (~4 h): stream tokens and tool-call progress into the Streamlit chat instead of waiting for the full answer.
- [ ] **Conversation persistence** (~4 h): keep chat history across restarts (SQLite), with a "new session" control.
- [ ] **Agent regression tests** (~5 h): a fixed set of questions with the expected tool calls and refusal categories, run in CI against a mocked provider so prompt or tool changes can't silently break behaviour.
- [ ] **CI** (~2 h): GitHub Actions running ruff + pytest on every push.

## Next — safety & auditability

- [ ] **Better safety filters** (~6 h): replace the regex PII / clinical-taxonomy filters with a tested classifier, keeping the regex layer as a fallback; expand the safety test cases.
- [ ] **Tamper-evident audit log** (~4 h): hash-chained JSONL entries and a configurable retention policy.
- [ ] **Answer provenance labels** (~4 h): mark each answer as data-grounded vs. general knowledge in the UI, instead of hard-refusing all general biology questions.

## Later — usability

- [ ] **Bring-your-own dataset** (~6 h): upload a CSV, infer the schema, and generate the tool layer from it, so the assistant works on any tabular dataset.
- [ ] **Charts** (~4 h): a plotting tool so the agent can return simple charts alongside numbers.
- [ ] Docs: architecture diagram of the agent loop and provider abstraction.
