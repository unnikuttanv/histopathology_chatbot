"""Pre-LLM safety filters and refusal taxonomy.

Two filters run BEFORE any LLM call:

1. detect_pii(text)              — patient-identifiable input
2. detect_clinical_question(text)— clinical-question shapes refused
                                    by the §7.5 taxonomy

Both are best-effort regex/keyword matchers. They are the first
line of defense — the system prompt embeds the same taxonomy as
the second line, and tests pin both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
    ),
    "dob": re.compile(
        r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
        r"|\b(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b"
    ),
    "mrn": re.compile(
        r"\b(?:MRN|Patient\s*ID|Medical\s*Record)\b[:#\s]+\S+", re.IGNORECASE
    ),
    "name_adjacent": re.compile(
        r"\b(?:mr|mrs|ms|dr)\.?\s+[A-Z]\w+"
        r"|\bpatient\s+[A-Z]\w+",
        re.IGNORECASE,
    ),
}


@dataclass
class PIIMatch:
    matched: bool
    patterns: list[str] = field(default_factory=list)


def detect_pii(text: str) -> PIIMatch:
    if not isinstance(text, str) or not text:
        return PIIMatch(matched=False)
    hits = [name for name, pat in PII_PATTERNS.items() if pat.search(text)]
    return PIIMatch(matched=bool(hits), patterns=hits)


PII_REFUSAL = (
    "I can't accept patient-identifiable information. This system only "
    "operates on a small de-identified dataset of population-level expression "
    "medians. Please rephrase without any patient data, names, dates, or IDs."
)


# Order matters: more specific categories first so they take precedence.
CLINICAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "diagnosis",
        re.compile(
            r"\bdoes\b.*\bpatient\b.*\bhave\b"
            r"|\bdiagnos(?:e|is)\b.*\bpatient\b",
            re.IGNORECASE,
        ),
    ),
    (
        "risk_stratification",
        re.compile(
            r"\b(?:which|what|who)\b[^?\.]*\bpatient(?:s)?\b[^?\.]*\b(?:high|low|moderate)?\s*risk\b"
            r"|\bstratif(?:y|ication)\b[^?\.]*\b(?:risk|patient)",
            re.IGNORECASE,
        ),
    ),
    (
        "prognosis",
        re.compile(
            r"\b(?:predict|forecast|estimat\w*)\b[^?\.]*\b(?:mortality|readmission|survival|outcome|progression|recurrence)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "therapeutic_decision_support",
        re.compile(
            r"\bwho\b[^?\.]*\b(?:needs?|requires?)\b[^?\.]*\btreatment\b"
            r"|\bwhich\s+patient(?:s)?\b[^?\.]*\btreatment\b",
            re.IGNORECASE,
        ),
    ),
    (
        "treatment_selection",
        re.compile(
            r"\b(?:which|what)\s+(?:drug|treatment|therapy|medication)\b[^?\.]*\b(?:for|to|should|use)\b"
            r"|\brank\b[^?\.]*\b(?:drugs?|treatments?|therapies?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clinical_decision_support",
        re.compile(
            r"\b(?:recommend|suggest)\b[^?\.]*\b(?:next\s+(?:action|step)|clinical\s+action)\b"
            r"|\bwhat\s+should\s+(?:we|i|the\s+doctor|the\s+clinician)\s+do\b",
            re.IGNORECASE,
        ),
    ),
    (
        "patient_interpretation",
        re.compile(
            r"\binterpret\b[^?\.]*\b(?:this|these|the)\s+(?:lab|result|record|chart|scan|test|biopsy)\b"
            r"|\bwhat\s+does\s+(?:this|these)\s+(?:lab|result|record|chart|scan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "triage",
        re.compile(
            r"\b(?:prioriti[sz]e|triage)\b[^?\.]*\bpatient(?:s)?\b"
            r"|\brank\b[^?\.]*\bpatient(?:s)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "biomarker_validation",
        re.compile(
            r"\bis\s+\w+\s+a\s+validated\b"
            r"|\bvalidated\s+(?:biomarker|target|marker)\s+for\b",
            re.IGNORECASE,
        ),
    ),
    (
        "off_label_inference",
        re.compile(
            r"\b(?:does|do|will|would)\b[^?\.]*\bexpression\b[^?\.]*\b(?:mean|means|indicate|indicates|suggest|suggests|imply|implies|predict|predicts|work|works|effective)\b"
            r"|\b(?:does|do)\b[^?\.]*\b(?:this|the)\s+(?:value|median)\b[^?\.]*\b(?:mean|indicate|suggest|imply)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class ClinicalMatch:
    matched: bool
    category: str | None = None


def detect_clinical_question(text: str) -> ClinicalMatch:
    if not isinstance(text, str) or not text:
        return ClinicalMatch(matched=False)
    for category, pat in CLINICAL_PATTERNS:
        if pat.search(text):
            return ClinicalMatch(matched=True, category=category)
    return ClinicalMatch(matched=False)


_REFUSAL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "diagnosis": (
        "diagnosis or interpretation of whether a patient has a condition",
        "What genes are associated with breast cancer in this dataset?",
    ),
    "risk_stratification": (
        "patient risk stratification",
        "What are the top 5 genes by expression in pancreatic cancer?",
    ),
    "prognosis": (
        "prognosis or prediction of patient outcomes",
        "Compare the gene sets for lung and prostate.",
    ),
    "therapeutic_decision_support": (
        "therapeutic decision support",
        "List the cancer indications covered in the dataset.",
    ),
    "treatment_selection": (
        "treatment selection or drug ranking",
        "Show median expression values for the genes in breast cancer.",
    ),
    "clinical_decision_support": (
        "clinical decision support or next-action recommendations",
        "What are the top genes by expression in glioblastoma?",
    ),
    "patient_interpretation": (
        "patient-specific interpretation of clinical results",
        "Plot the expression values for genes in lung.",
    ),
    "triage": (
        "patient triage or prioritisation",
        "Compare breast and gastric cancer gene sets.",
    ),
    "biomarker_validation": (
        "biomarker validation claims",
        "Show the median expression for BRCA1 and BRCA2 in breast cancer.",
    ),
    "off_label_inference": (
        "inferences about drug response beyond the dataset",
        "What is the median expression of KRAS in lung cancer?",
    ),
}


def refusal_for(category: str) -> str:
    desc, redirect = _REFUSAL_DESCRIPTIONS.get(
        category,
        (
            "clinical or patient-specific reasoning",
            "List the cancer indications in the dataset.",
        ),
    )
    return (
        f"This system doesn't provide {desc}. It only summarises "
        f"population-level expression medians from a small research dataset. "
        f"For example, you can ask: '{redirect}'"
    )
