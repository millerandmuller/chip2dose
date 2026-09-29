"""Clinical ground truth lookup: FDA DILIrank 2.0."""

from __future__ import annotations

from dataclasses import dataclass

from . import load

NOT_FOUND = "Not in DILIrank 2.0 - no clinical label available for this compound."


@dataclass(frozen=True)
class ClinicalLabel:
    found: bool
    compound: str
    label: str
    severity: int | None
    text: str


def dilirank_label(name: str) -> ClinicalLabel:
    """Free-text drug name -> one of the four DILIrank 2.0 classes, or an honest not-found."""
    key = load.normalize_name(name)
    table = load.dilirank()
    hit = table[table["key"] == key]
    if hit.empty or not key:
        return ClinicalLabel(False, name, "", None, NOT_FOUND)
    row = hit.iloc[0]
    severity = int(row["severity"]) if str(row["severity"]).isdigit() else None
    section = str(row["LabelSection"])
    if section.strip().lower() == "no match":
        detail = f"severity class {row['severity']}; no matching liver text in the drug label, as published in DILIrank"
    else:
        detail = f"severity class {row['severity']}, {section}"
    text = f"DILIrank 2.0: {row['label']} ({detail})"
    return ClinicalLabel(True, row["compound"], row["label"], severity, text)
