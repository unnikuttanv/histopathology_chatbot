import importlib

import pytest


def test_max_steps_is_8():
    import config
    importlib.reload(config)
    assert config.MAX_STEPS == 8


def test_temperature_is_zero():
    import config
    importlib.reload(config)
    assert config.TEMPERATURE == 0.0


def test_system_prompt_includes_all_refusal_categories():
    import config
    importlib.reload(config)
    for cat in [
        "diagnosis", "risk_stratification", "prognosis",
        "therapeutic_decision_support", "treatment_selection",
        "clinical_decision_support", "patient_interpretation",
        "triage", "biomarker_validation", "off_label_inference",
    ]:
        assert cat in config.SYSTEM_PROMPT, f"missing category in prompt: {cat}"


def test_system_prompt_includes_grounding_rules():
    import config
    importlib.reload(config)
    assert "tool" in config.SYSTEM_PROMPT.lower()
    assert "invent" in config.SYSTEM_PROMPT.lower()
    assert "median" in config.SYSTEM_PROMPT.lower()


def test_load_provider_config_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    import config
    importlib.reload(config)
    cfg = config.load_provider_config()
    assert cfg["provider"] == "gemini"


def test_load_provider_config_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import config
    importlib.reload(config)
    cfg = config.load_provider_config()
    assert cfg["provider"] == "openai"
    assert cfg["openai_api_key"] == "sk-test"


def test_build_provider_gemini_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    import config
    importlib.reload(config)
    monkeypatch.setattr(
        "providers.gemini.GeminiProvider",
        lambda **kw: type("P", (), {"name": "gemini", "model": kw["model"]})(),
    )
    p = config.build_provider()
    assert p.name == "gemini"


def test_build_provider_gemini_no_key_falls_back_to_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import config
    importlib.reload(config)
    monkeypatch.setattr(
        "providers.ollama.OllamaProvider",
        lambda **kw: type("P", (), {"name": "ollama", "model": kw["model"]})(),
    )
    p = config.build_provider()
    assert p.name == "ollama"
    assert config.LAST_FALLBACK_REASON is not None
    assert "GEMINI_API_KEY" in config.LAST_FALLBACK_REASON


def test_build_provider_openai_no_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import config
    importlib.reload(config)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.build_provider()


def test_build_provider_explicit_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    import config
    importlib.reload(config)
    monkeypatch.setattr(
        "providers.ollama.OllamaProvider",
        lambda **kw: type("P", (), {"name": "ollama", "model": kw["model"]})(),
    )
    p = config.build_provider()
    assert p.name == "ollama"
