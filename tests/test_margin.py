"""Margin engine: arithmetic, bands, censoring, thresholds as conventions."""

import numpy as np
import pytest

from src import config, liver, margin, pod

FREE_375 = config.THRESHOLDS["liver_free_375"]
TOTAL_50 = config.THRESHOLDS["liver_total_50"]


def _input(pod_uM=10.0, cmax=1.0, fu_p=0.1, fu_m=0.5, censored=False, pod_range=None):
    lo, hi = pod_range or (pod_uM, pod_uM)
    return margin.MarginInput(
        name="fixture",
        pod=margin.Quantity(pod_uM, lo, hi, "fixture POD"),
        cmax=margin.Quantity.exact(cmax, "fixture Cmax"),
        fu_plasma=margin.Quantity.exact(fu_p, "fixture fu plasma"),
        fu_medium=margin.Quantity.exact(fu_m, "fixture fu medium"),
        pod_censored=censored,
    )


def test_minimum_toxic_concentration_over_cmax_classified_against_50():
    # Lowest toxic in-vitro concentration + clinical Cmax -> MOS-like ratio vs the convention 50.
    result = margin.compute(_input(pod_uM=20.0, cmax=1.0), TOTAL_50)
    assert result.margin_total == pytest.approx(20.0)
    assert result.verdict == "BELOW"
    assert "threshold 50" in result.line
    assert result.threshold.error_rates in result.line  # a convention carries its error rates


def test_total_and_free_reported_separately_free_is_reference():
    result = margin.compute(_input(pod_uM=10.0, cmax=1.0, fu_p=0.1, fu_m=0.5), FREE_375)
    assert result.margin_total == pytest.approx(10.0)
    assert result.margin_free == pytest.approx(10.0 * 0.5 / 0.1)
    row = result.as_row()
    assert {"margin_total", "margin_free", "band_low", "band_high"} <= row.keys()
    assert result.threshold.basis == "free"


def test_every_output_has_a_band_containing_the_point():
    result = margin.compute(_input(pod_uM=10.0, pod_range=(5.0, 40.0)), FREE_375)
    for point, (lo, hi) in [(result.margin_total, result.total_band), (result.margin_free, result.free_band)]:
        assert lo <= point <= hi
        assert lo < hi


def test_band_straddling_threshold_asks_for_more_chips():
    result = margin.compute(_input(pod_uM=375 * 0.1 / 0.5, pod_range=(10.0, 400.0)), FREE_375)
    assert result.verdict == "STRADDLES"
    assert "needs more chips" in result.line


def test_censored_pod_is_carried_through_not_treated_as_a_number():
    result = margin.compute(_input(pod_uM=5.0, censored=True), FREE_375)
    assert result.censored
    assert result.verdict.startswith("INCONCLUSIVE")
    assert ">" in result.line


def test_unknown_fraction_unbound_gives_no_free_margin():
    inp = _input()
    inp.fu_plasma = None
    result = margin.compute(inp, TOTAL_50)
    assert np.isnan(result.margin_free)
    with pytest.raises(ValueError):
        margin.compute(inp, FREE_375)


def test_herg_threshold_30_is_a_tradeoff_with_error_rates():
    # hERG IC50 / free Cmax against 30 (Redfern). Fixture values, not a real compound.
    herg = config.THRESHOLDS["herg_free_30"]
    result = margin.compute(_input(pod_uM=3.0, cmax=0.5, fu_p=0.1, fu_m=1.0), herg)
    assert result.margin_free == pytest.approx(60.0)
    assert result.verdict in {"ABOVE", "STRADDLES"}
    assert "27%" in result.line and "33%" in result.line


def test_liver_chip_margins_reproduce_published_mos():
    # margin_total must equal the published Table 4 MOS-like value for every drug.
    table = liver.margin_table()
    ok = table["published_mos_total"] > 0
    assert np.allclose(table.loc[ok, "margin_total"], table.loc[ok, "published_mos_total"])


def test_both_donor_rule_matches_published_column():
    assert pod.combine_donors([0.03, 0.1], [False, False]) == (0.03, False)
    assert pod.combine_donors([3.0, 10.0], [True, True]) == (10.0, True)
    assert pod.combine_donors([7.0, 15.0], [True, False]) == (15.0, False)
