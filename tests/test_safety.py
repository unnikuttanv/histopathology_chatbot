import safety


# --- PII ---


def test_pii_detects_email():
    m = safety.detect_pii("contact jane@example.com about this")
    assert m.matched
    assert "email" in m.patterns


def test_pii_detects_phone():
    m = safety.detect_pii("call (415) 555-0123 if urgent")
    assert m.matched
    assert "phone" in m.patterns


def test_pii_detects_dob_us_format():
    m = safety.detect_pii("DOB 03/15/1972")
    assert m.matched
    assert "dob" in m.patterns


def test_pii_detects_dob_iso_format():
    m = safety.detect_pii("born on 1972-03-15")
    assert m.matched
    assert "dob" in m.patterns


def test_pii_detects_mrn():
    m = safety.detect_pii("MRN: 1234567 needs review")
    assert m.matched
    assert "mrn" in m.patterns


def test_pii_detects_name_adjacent():
    m = safety.detect_pii("patient Smith was seen today")
    assert m.matched
    assert "name_adjacent" in m.patterns


def test_pii_no_match_on_dataset_question():
    assert not safety.detect_pii("what are the top 5 genes in lung cancer").matched


def test_pii_no_match_on_capability_question():
    assert not safety.detect_pii("how can you help me").matched


def test_pii_refusal_message_is_a_string():
    assert isinstance(safety.PII_REFUSAL, str)
    assert "patient" in safety.PII_REFUSAL.lower()


# --- Clinical taxonomy gold cases ---

CLINICAL_GOLD_CASES = [
    ("Does this patient have lung cancer?", "diagnosis"),
    ("Which patients are high risk?", "risk_stratification"),
    ("Predict mortality for this cohort.", "prognosis"),
    ("Who needs treatment?", "therapeutic_decision_support"),
    ("Which drug should we use for breast cancer?", "treatment_selection"),
    ("Recommend the next clinical action.", "clinical_decision_support"),
    ("Interpret this lab result.", "patient_interpretation"),
    ("Prioritise these patients.", "triage"),
    ("Is BRCA1 a validated target for breast?", "biomarker_validation"),
    ("Does this expression value mean the drug works?", "off_label_inference"),
]


def test_clinical_filter_triggers_on_each_gold_case():
    for text, expected_category in CLINICAL_GOLD_CASES:
        m = safety.detect_clinical_question(text)
        assert m.matched, f"should match {text!r}"
        assert m.category == expected_category, (
            f"{text!r}: expected {expected_category}, got {m.category}"
        )


def test_clinical_filter_does_not_match_dataset_questions():
    benign = [
        "what cancers do you cover",
        "list the genes in lung",
        "top 5 in pancreatic",
        "compare breast and prostate",
        "how can you help me",
        "show me a chart of expressions in lung",
    ]
    for text in benign:
        assert not safety.detect_clinical_question(text).matched, text


def test_refusal_for_returns_text_with_category_and_redirect():
    msg = safety.refusal_for("diagnosis")
    assert "diagnosis" in msg.lower()
    assert "you can ask" in msg.lower() or "for example" in msg.lower()


def test_refusal_for_each_category_returns_unique_text():
    seen = set()
    for _, cat in CLINICAL_GOLD_CASES:
        seen.add(safety.refusal_for(cat))
    assert len(seen) == len(CLINICAL_GOLD_CASES)
