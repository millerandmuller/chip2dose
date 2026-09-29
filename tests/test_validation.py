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


def _fingerprints_of_modified_copy(tmp_path, monkeypatch, rel, old, new):
    import shutil
    from src import config
    for sub in ("src",):
        shutil.copytree(config.ROOT / sub, tmp_path / sub)
    target = tmp_path / rel
    text = target.read_text()
    if old == new:  # append mode
        target.write_text(text + new)
    else:
        assert old in text
        target.write_text(text.replace(old, new))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return validate.code_fingerprints()


def test_freeze_catches_a_changed_feature_construction(tmp_path, monkeypatch):
    frozen = json.loads(validate.CODE_FREEZE.read_text())["fingerprints"]
    prints = _fingerprints_of_modified_copy(tmp_path, monkeypatch, "src/validate.py",
                                            "clip(lower=1e-4, upper=1.0)", "clip(lower=1e-1, upper=1.0)")
    assert prints["src/validate.py:benchmark_table"] != frozen["src/validate.py:benchmark_table"]


def test_freeze_catches_a_nested_redefinition(tmp_path, monkeypatch):
    frozen = json.loads(validate.CODE_FREEZE.read_text())["fingerprints"]
    redefinition = "\nif True:\n    def ci(values):\n        return (0.0, 1.0)\n"
    prints = _fingerprints_of_modified_copy(tmp_path, monkeypatch, "src/validate.py", redefinition, redefinition)
    assert prints["src/validate.py:ci"] != frozen["src/validate.py:ci"]


# Post-hoc range check (2026-09-29, run once). Literals on purpose: these are the numbers the report cites,
# so a data or code change must turn the suite red rather than move a published number quietly.
def test_range_check_counts_and_spans_are_pinned():
    table = validate.benchmark_table()
    spans = validate.log_spans(table)
    assert round(spans["log_pod"]["span_orders"], 2) == 5.22
    assert round(spans["log_cmax"]["span_orders"], 2) == 7.07
    keep = validate.range_matched_mask(table)
    assert int(keep.sum()) == 195
    tails = {(t["tail"], t["label"]): t["n"] for t in validate.removed_tails(table, keep)}
    assert tails == {("Cmax below the lowest POD", "concern"): 7, ("Cmax below the lowest POD", "no concern"): 18}


def test_no_pod_in_the_benchmark_source_is_censored():
    """Why the uncensored-only restriction was a no-op: the source file writes no POD as a bound."""
    assert validate.censored_pod_entries() == 0


def test_range_check_is_labelled_post_hoc_and_never_primary():
    result = validate.evaluate_range_matched()
    assert "post-hoc" in result["status"] and "not pre-registered" in result["status"]
    rows = result["arms"] + result["comparisons"]
    assert all(r.get("arm", r.get("comparison")).startswith(validate.RANGE_MATCHED_PREFIX) for r in rows)
    assert all("primary" not in r for r in result["comparisons"])
    (comparison,) = result["comparisons"]
    assert (result["n"], result["n_positive"], result["n_negative"]) == (195, 165, 30)
    assert round(comparison["delta_auc"], 3) == 0.162
    assert (round(comparison["ci_low"], 3), round(comparison["ci_high"], 3)) == (0.058, 0.275)
