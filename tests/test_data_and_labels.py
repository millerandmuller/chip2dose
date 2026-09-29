"""Inputs are read correctly: checksums, cross-checks, clinical labels, the neural AED file."""

import json

from src import config, labels, load, neural, pod

MANIFEST = config.DATA / "sources.json"


def test_every_source_is_pinned_and_licensed():
    sources = json.loads(MANIFEST.read_text())["sources"]
    assert len(sources) == 7
    for s in sources:
        assert s["sha256"] and len(s["sha256"]) == 64
        assert s["license"] and s["access_date"] and s["urls"]


def test_published_values_rederive_exactly():
    exact = {
        "Geci lowestPOD = min(PODValues)",
        "Geci Cmax = curated value, else median of sources",
        "EPA MEA min.ec50 = min over entries of the same chemical",
        "Liver-Chip 'both donors' MOS = combination rule",
    }
    checks = {c["check"]: c for c in pod.all_checks()}
    for name in exact:
        assert checks[name]["n_mismatch"] == 0, checks[name]["mismatches"][:3]


def test_dosing_mismatches_are_reported_not_dropped():
    check = pod.check_liverchip_dosing()
    assert check["n_checked"] == 224
    assert check["n_mismatch"] == len(check["mismatches"])
    assert all(m.startswith("Telithromycin") for m in check["mismatches"])


def test_three_named_compounds_rederive():
    table = pod.named_rederivations()
    assert len(table) >= 3 and table["match"].all()


def test_dilirank_label_or_honest_not_found():
    found = labels.dilirank_label("Troglitazone")
    assert found.found and found.label == "vMost-DILI-concern"
    assert labels.dilirank_label("pioglitazone").label == "vLess-DILI-concern"
    missing = labels.dilirank_label("definitely-not-a-drug")
    assert not missing.found and missing.text == labels.NOT_FOUND
    assert set(load.dilirank()["label"].dropna()) == {
        "vMost-DILI-concern", "vLess-DILI-concern", "vNo-DILI-concern", "Ambiguous-DILI-concern"
    }


def test_neural_aed_is_taken_from_the_dataset_not_reestimated():
    file_values = load.mea_ivive().dropna(subset=["aed_mg_kg_day"]).drop_duplicates("name").set_index("name")
    _, margins = neural.margin_table()
    aed_rows = margins[margins["route"] == "AED vs predicted exposure"].set_index("compound")
    assert len(aed_rows) == len(file_values)
    for name, row in aed_rows.iterrows():
        assert row["aed_mg_kg_day"] == file_values.at[name, "aed_mg_kg_day"]


def test_neural_table_covers_every_chemical_and_issues_no_verdict():
    potency, margins = neural.margin_table()
    assert len(potency) == 136 and potency["active"].sum() == 82
    assert (margins["verdict"] == neural.NO_CONVENTION).all()


# The counts Beat 5 speaks aloud and the README prints as a finding. Literals on purpose: the test above
# re-derives them from the same tables `coverage()` counted, so it cannot see input loss (dropping a
# measured row still reconciles). These do. A data-vintage change must turn this red, so that a recorded
# script is revisited rather than silently contradicted.
SPOKEN_NEURAL_COUNTS = {
    "chemicals tested": 136,
    "active": 82,
    "with a human exposure comparator": 21,
    "via predicted exposure": 13,
    "via measured exposure": 9,
    "in both routes": 1,
}
SPOKEN_MEASURED_ROUTE = {
    "17beta-Estradiol", "CP-457920", "Diphenhydramine hydrochloride", "Folic acid", "Lovastatin",
    "Pravastatin sodium", "Reserpine", "Simvastatin", "Tamoxifen",
}


def test_the_neural_counts_the_script_speaks_are_pinned_to_literals():
    potency, margins = neural.margin_table()
    coverage = neural.coverage(potency, margins)
    assert dict(zip(coverage["stage"].str.strip(), coverage["count"])) == SPOKEN_NEURAL_COUNTS
    assert coverage.attrs["check"] == "13 + 9 = 22 route rows for 21 chemicals (1 in both)"


def test_the_measured_exposure_route_is_pinned_to_its_compounds():
    """Mirror of the AED test for the other route: which nine chemicals, not just how many."""
    _, margins = neural.margin_table()
    assert set(margins.loc[margins["route"] == neural.DRUG_ROUTE, "compound"]) == SPOKEN_MEASURED_ROUTE
    both = (set(margins.loc[margins["route"] == neural.DRUG_ROUTE, "compound"])
            & set(margins.loc[margins["route"] == neural.AED_ROUTE, "compound"]))
    assert both == {"Simvastatin"}


def test_neural_coverage_counts_reconcile_with_the_margin_table():
    """The funnel is a measurement: every count is re-derived here from the tables, not asserted.

    Scope: internal consistency only. Input loss is caught by the pinned literals above, not here.
    """
    potency, margins = neural.margin_table()
    coverage = neural.coverage(potency, margins)
    counts = dict(zip(coverage["stage"].str.strip(), coverage["count"]))
    assert counts["chemicals tested"] == len(potency)
    assert counts["active"] == int(potency["active"].sum())
    assert counts["with a human exposure comparator"] == margins["compound"].nunique()
    # the sanity row: per-route counts are not a total, and the table says so
    assert counts["via predicted exposure"] + counts["via measured exposure"] == len(margins)
    assert (counts["via predicted exposure"] + counts["via measured exposure"]
            - counts["in both routes"] == counts["with a human exposure comparator"])
    assert "route rows" in coverage.attrs["check"]


def test_neural_coverage_refuses_counts_that_do_not_reconcile():
    """A comparator for a chemical with no active potency value is a counting error, not a result."""
    import pytest
    potency, margins = neural.margin_table()
    broken = margins.copy()
    broken.loc[broken.index[0], "compound"] = "not-a-tested-chemical"
    with pytest.raises(ValueError, match="comparator without an active potency value"):
        neural.coverage(potency, broken)


def test_no_verdict_is_issued_for_any_neural_compound():
    from src import neural
    _, margins = neural.margin_table()
    assert set(margins["verdict"]) == {neural.NO_CONVENTION}
