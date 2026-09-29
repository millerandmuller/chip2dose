"""Pair view, single-compound path and the no-score rule."""

import time

import pandas as pd

from src import compound, config, figures, liver, load

PROBABILITY_WORDS = ("probability", "proba", "score")


def test_troglitazone_is_the_riskier_partner_of_a_published_pair():
    pairs = load.liverchip_pairs().set_index("key")
    assert pairs.at["troglitazone", "partner_key"] == "pioglitazone"  # from Table 1, not our choice
    assert pairs.at["troglitazone", "garside_rank"] < pairs.at["pioglitazone", "garside_rank"]
    margins = liver.margin_table().set_index("key")
    assert margins.at["troglitazone", "margin_free"] < margins.at["pioglitazone", "margin_free"]


def test_pair_view_renders_every_matched_pair_under_two_seconds(tmp_path):
    margins = liver.margin_table()
    matched = {tuple(sorted(p)) for p in zip(margins["key"], margins["partner_key"]) if p[1]}
    assert len(matched) == 7
    for a, b in matched:
        started = time.time()
        path = figures.pair_view(margins, a, b, tmp_path / f"{a}_{b}.png")
        assert time.time() - started < 2.0
        assert path.stat().st_size > 10_000


def test_single_compound_run_is_fast_and_in_real_units():
    started = time.time()
    lines = compound.describe("troglitazone")
    assert time.time() - started < 60
    text = "\n".join(lines)
    assert "uM" in text and "mg" in text and "Band" in text
    assert "DILIrank 2.0" in text


def test_rule_of_thumb_flag_shown_next_to_the_margin():
    text = "\n".join(compound.describe("troglitazone"))
    assert "Dose rule of thumb" in text and "FLAGGED" in text
    assert "not flagged" in "\n".join(compound.describe("pioglitazone"))


def test_no_per_compound_score_in_any_results_table():
    tables = list(config.RESULTS.glob("*.csv"))
    assert tables, "run `make` first"
    for path in tables:
        columns = " ".join(pd.read_csv(path, nrows=1).columns).lower()
        assert not any(word in columns for word in PROBABILITY_WORDS), path.name


def test_censored_comparator_never_counts_as_a_wrong_order():
    pairs = liver.pair_table().set_index("pair")
    row = pairs.loc["Trovafloxacin / Levofloxacin"]
    assert row["margin_fold_is_lower_bound"]
    assert row["margin_orders_like_clinic"] == "inconclusive (comparator censored)"


def test_pair_cli_rejects_unknown_and_labels_unpublished_pairs(capsys):
    import run_demo
    assert run_demo.main(["--pair", "aspirin", "troglitazone"]) == 1
    assert "Not in the Liver-Chip set" in capsys.readouterr().out
    assert run_demo.main(["--pair", "troglitazone", "troglitazone"]) == 1


def test_readout_slot_gives_bands_and_units(capsys):
    import run_demo
    assert run_demo.main(["--readout", "12", "--cmax", "0.8", "--fu-plasma", "0.05", "--fu-medium", "0.7"]) == 0
    out = capsys.readouterr().out
    assert "margin (free)" in out and "Band" in out


def test_neural_chemicals_are_found_by_name():
    text = "\n".join(compound.describe("rotenone"))
    assert "Neural MEA" in text and "AED vs predicted exposure" in text


def test_dose_relation_respects_the_band():
    assert "above the whole band" in figures.dose_relation(16.1, 16.1, 61.7, 600)
    # pioglitazone: 45 mg lies inside 39.6-291, so no "below" claim without the qualifier
    text = figures.dose_relation(90.4, 39.6, 291, 45)
    assert "inside the band" in text and "whole band" not in text
    assert "below the whole band" in figures.dose_relation(90, 60, 200, 10)


def test_dose_view_renders_and_refuses_pairs_without_a_dose(tmp_path):
    margins = liver.margin_table()
    pairs = liver.pair_table(margins)
    path = figures.dose_view(margins, pairs, "troglitazone", "pioglitazone", tmp_path / "d.png")
    assert path.stat().st_size > 10_000
    import pytest
    with pytest.raises(ValueError):
        figures.dose_view(margins, pairs, "ambrisentan", "sitaxsentan", tmp_path / "x.png")
