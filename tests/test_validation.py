"""The validation cannot leak: split fixed in advance, groups never cross folds."""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from src import validate


def _table_and_split():
    table = validate.benchmark_table()
    return table, validate.load_verified_split(table)


def test_split_matches_preregistration():
    table, frame = _table_and_split()
    doc = json.loads(validate.PREREG.read_text())
    assert doc["n_drugs"] == len(table)
    assert doc["split_sha256"] == validate.sha256_text(validate.split_csv_text(frame))


def test_no_group_is_split_across_folds():
    _, frame = _table_and_split()
    for column in [c for c in frame.columns if c.startswith("repeat_")]:
        assert (frame.groupby("group")[column].nunique() == 1).all()


def test_matched_pairs_share_a_group():
    table, frame = _table_and_split()
    group = dict(zip(table["identity"], frame["group"]))
    for a, b in [("troglitazone", "pioglitazone"), ("tolcapone", "entacapone"), ("trovafloxacin", "levofloxacin")]:
        assert group[a] == group[b]


def test_one_row_per_molecule():
    table, _ = _table_and_split()
    assert table["identity"].is_unique


def test_potency_alone_separates_worse_than_the_margin():
    # Reproduces the published finding that in-vitro values alone separate DILI classes poorly.
    table, _ = _table_and_split()
    y = table["y_wide"].to_numpy()
    potency = roc_auc_score(y, -table["log_pod"])
    margin = roc_auc_score(y, -table["log_margin_total"])
    assert potency < margin


def test_rule_of_thumb_is_a_binary_flag_on_dose_and_logp():
    table, _ = _table_and_split()
    expected = ((table["dose_mg"] >= 100) & (table["logp"] >= 3)).astype(int)
    assert np.array_equal(table["rule_of_thumb"], expected)


def test_changed_analysis_plan_is_refused(tmp_path, monkeypatch):
    doc = json.loads(validate.PREREG.read_text())
    doc["primary_comparison"] = doc["primary_comparison"][::-1]
    fake = tmp_path / "preregistration.json"
    fake.write_text(json.dumps(doc))
    monkeypatch.setattr(validate, "PREREG", fake)
    table = validate.benchmark_table()
    try:
        validate.load_verified_split(table)
    except SystemExit as exc:
        assert "primary_comparison" in str(exc)
    else:
        raise AssertionError("a changed plan must be refused")


def test_code_frozen_at_preregistration_is_unchanged():
    frozen = json.loads(validate.CODE_FREEZE.read_text())
    assert frozen["commit"] == "a76e7e9"
    assert frozen["fingerprints"] == validate.code_fingerprints()


def test_changed_frozen_code_is_refused(monkeypatch):
    real = validate.code_fingerprints()
    monkeypatch.setattr(validate, "code_fingerprints", lambda: {**real, "src/validate.py:BOOTSTRAP_SEED": "changed"})
    try:
        validate.load_verified_split(validate.benchmark_table())
    except SystemExit as exc:
        assert "BOOTSTRAP_SEED" in str(exc)
    else:
        raise AssertionError("changed frozen code must be refused")
