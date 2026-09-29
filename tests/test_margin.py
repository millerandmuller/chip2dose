"""Margin engine: arithmetic, bands, censoring, thresholds as conventions."""

import numpy as np
import pandas as pd
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


def test_unknown_fraction_unbound_widens_the_band_and_says_so():
    known = margin.compute(_input(), FREE_375)
    inp = _input()
    inp.fu_plasma = None
    result = margin.compute(inp, FREE_375)
    assert np.isnan(result.margin_free)  # no single value is claimed
    assert result.free_band[1] / result.free_band[0] > known.free_band[1] / max(known.free_band[0], 1e-12)
    assert "fraction unbound unknown" in result.line
    assert any("fraction unbound in plasma unknown" in a for a in result.assumptions)


@pytest.mark.parametrize("pod_uM,cmax,fu_p", [
    (0.0, 1.0, 0.1), (float("nan"), 1.0, 0.1), (float("inf"), 1.0, 0.1), (-1.0, 1.0, 0.1),
    (1.0, 0.0, 0.1), (1e300, 1e-300, 0.1), (1.0, 1.0, 5.0),
])
def test_unusable_inputs_give_no_margin_never_a_verdict(pod_uM, cmax, fu_p):
    result = margin.compute(_input(pod_uM=pod_uM, cmax=cmax, fu_p=fu_p), FREE_375)
    assert result.verdict == margin.NO_MARGIN
    assert "no margin computed" in result.line


def test_missing_exposure_gives_no_margin():
    inp = _input()
    inp.cmax = None
    result = margin.compute(inp, TOTAL_50)
    assert result.verdict == margin.NO_MARGIN and "no default is substituted" in result.line


def test_herg_threshold_30_is_a_tradeoff_with_error_rates():
    # hERG IC50 / free Cmax against 30 (Redfern). Fixture values, not a real compound.
    herg = config.THRESHOLDS["herg_free_30"]
    result = margin.compute(_input(pod_uM=3.0, cmax=0.5, fu_p=0.1, fu_m=1.0), herg)
    assert result.margin_free == pytest.approx(60.0)
    assert result.verdict in {"ABOVE", "STRADDLES"}
    assert "27%" in result.line and "33%" in result.line


def test_free_margin_matches_hand_calculation_from_the_source_tables():
    # Troglitazone by hand: Table 4 MOS 0.03; SD1 free/total dosing ratio 0.000657/0.01259; fu plasma 0.0011.
    table = liver.margin_table().set_index("key")
    assert table.at["troglitazone", "margin_free"] == pytest.approx(0.03 * (0.000657 / 0.01259) / 0.0011, rel=0.01)
    # Tolcapone, SD1 row x0.1: free 0.005712 / total 0.10074; fu plasma 0.0012
    assert table.at["tolcapone", "margin_free"] == pytest.approx(0.004 * (0.005712 / 0.10074) / 0.0012, rel=0.01)


def test_same_compound_same_band_everywhere():
    from src import compound
    row = liver.margin_table().set_index("key").loc["troglitazone"]
    assert row["result_line"] in "\n".join(compound.describe("troglitazone"))


def test_chip_drugs_get_an_equivalent_daily_dose_when_a_dose_is_known():
    table = liver.margin_table().set_index("key")
    trog = table.loc["troglitazone"]
    assert trog["clinical_dose_mg"] == 600
    assert trog["equivalent_dose_band_low"] <= trog["equivalent_dose_mg"] <= trog["equivalent_dose_band_high"]
    assert np.isnan(table.at["ambrisentan", "equivalent_dose_mg"])  # censored: no dose claimed


def test_both_donor_rule_matches_published_column():
    assert pod.combine_donors([0.03, 0.1], [False, False]) == (0.03, False)
    assert pod.combine_donors([3.0, 10.0], [True, True]) == (10.0, True)
    assert pod.combine_donors([7.0, 15.0], [True, False]) == (15.0, False)


def test_point_outside_its_own_range_or_missing_medium_fu_gives_no_margin():
    bad = margin.MarginInput("x", margin.Quantity(1e9, 1.0, 2.0, "pod"), margin.Quantity.exact(1.0, "cmax"),
                             margin.Quantity.exact(0.1, "fu"), margin.Quantity.exact(1.0, "m"))
    assert margin.compute(bad, FREE_375).verdict == margin.NO_MARGIN
    missing = _input()
    missing.fu_medium = None
    assert margin.compute(missing, FREE_375).verdict == margin.NO_MARGIN


def test_dose_relation_wording():
    assert "1.4x more than the point estimate" in margin.dose_relation(10, 5, 20, 14)
    assert "about the chip-derived dose" in margin.dose_relation(10, 5, 20, 10)
    assert "no comparison possible" in margin.dose_relation(10, 5, 20, 0)


def test_subnormal_margin_earns_no_verdict_and_no_equivalent_dose():
    inp = _input(pod_uM=1e-300, cmax=1e10)
    inp.dose_mg = 100.0
    result = margin.compute(inp, TOTAL_50)
    assert result.verdict == margin.NO_MARGIN
    assert f"smaller than {config.MARGIN_FLOOR:g}" in result.line
    assert result.equivalent_dose_mg is None and result.equivalent_dose_band is None
    assert "threshold" not in result.line and "0 mg" not in result.line


def test_margin_floor_leaves_every_real_margin_untouched():
    """The floor must not clip a real result: the smallest margin in any generated table is ~1.8e-4."""
    smallest = min(pd.read_csv(config.RESULTS / "liver_margin_table.csv")["margin_total_band_low"].min(),
                   pd.read_csv(config.RESULTS / "neural_margin_table.csv")["band_low"].min())
    assert smallest > config.MARGIN_FLOOR * 1e6


def test_out_of_range_value_is_not_reported_as_missing():
    inp = _input()
    inp.pod = margin.Quantity.exact(float("inf"), "user POD")
    line = margin.compute(inp, TOTAL_50).line
    assert "not a finite number" in line and "missing" not in line
    inp.pod = None
    assert "no point of departure" in margin.compute(inp, TOTAL_50).line


def test_unmeasured_fraction_unbound_gives_a_band_but_no_free_verdict():
    inp = _input()
    inp.fu_plasma = None
    result = margin.compute(inp, FREE_375)
    assert result.verdict == margin.NO_VERDICT
    assert "No verdict against the convention threshold 375" in result.line
    assert "needs a measured fraction unbound" in result.line
    # the band is still reported, and no threshold claim is attached to it
    assert "Band:" in result.line
    for claim in ("below the convention", "above the convention", "straddles", "needs more chips"):
        assert claim not in result.line
    # a measured fraction unbound still earns a verdict
    assert margin.compute(_input(), FREE_375).verdict in {"BELOW", "ABOVE", "STRADDLES"}


def test_total_basis_verdict_survives_an_unmeasured_fraction_unbound():
    """The equivalent-dose path builds a total-basis input with no fu; it must still be classified."""
    inp = _input()
    inp.fu_plasma = None
    assert margin.compute(inp, TOTAL_50).verdict == "BELOW"
