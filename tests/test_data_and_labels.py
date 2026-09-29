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
