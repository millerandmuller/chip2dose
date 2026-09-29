"""Liver-Chip application: margins for the 27 drugs of Ewart et al. (2022) and their matched pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, load, margin, pod

REFERENCE_THRESHOLD = config.THRESHOLDS["liver_free_375"]  # free concentration is the referenced basis
TOTAL_THRESHOLD = config.THRESHOLDS["liver_total_50"]

# Table 4 prints the smallest values with three decimals (e.g. 0.001); a printed '0' is read as
# below 0.0005, i.e. half the smallest printed step.
BELOW_PRECISION_MOS = 0.0005


def _geci_row(key: str) -> pd.Series | None:
    geci = load.geci_benchmark().drop_duplicates("key").set_index("key")
    return geci.loc[key] if key in geci.index else None


def build_input(row: pd.Series) -> margin.MarginInput:
    """One Liver-Chip drug -> MarginInput with every range sourced or named."""
    assumptions: list[str] = []
    cmax_point = row["cmax_total_uM"]
    geci = _geci_row(row["key"])

    # Point of departure: the two-donor range where both donors showed toxicity.
    if row["below_precision"]:
        high = BELOW_PRECISION_MOS * cmax_point
        pod_q = margin.Quantity(high, high / config.DEFAULT_FOLD_UNCERTAINTY, high, "Table 4 printed 0")
        assumptions.append("MOS printed as 0 in Table 4: read as < 0.0005, 3-fold range below that")
    elif row["pod_low_uM"] < row["pod_high_uM"]:
        pod_q = margin.Quantity(row["pod_uM"], row["pod_low_uM"], row["pod_high_uM"], "two-donor range, Table 4")
    else:
        pod_q, note = margin.point_with_default(row["pod_uM"], "chip POD (one donor informative)")
        assumptions.append(note)

    # Exposure: the value the chip was dosed against, ranged by every independent report of it.
    cmax_values = [cmax_point] + (list(geci["cmax_values_uM"]) if geci is not None else [])
    cmax_q, note = margin.range_or_default(cmax_values, cmax_point, "total Cmax")
    if note:
        cmax_q = margin.Quantity.exact(cmax_point, "total Cmax, SD1 (single report)")
        assumptions.append("total Cmax: single report (SD1), treated as exact")

    fu_values = [row["fu_plasma"]]
    if geci is not None:
        fu_values += [geci["fu_opera"], geci["fu_admetlab"], geci["fu_simplus"]]
    fu_q, note = margin.range_or_default(fu_values, row["fu_plasma"], "fraction unbound in plasma", cap=1.0)
    if note:
        assumptions.append(note)

    fu_medium = margin.Quantity.exact(row["fu_medium"], "fraction unbound in chip medium, SD1")
    assumptions.append("fraction unbound in chip medium (2% FBS) as extrapolated by the study authors, treated as exact")
    if row["exposure_assumed_from_partner"]:
        assumptions.append("Cmax and fraction unbound assumed equal to the partner drug by the study authors")

    return margin.MarginInput(
        name=row["drug"],
        pod=pod_q,
        cmax=cmax_q,
        fu_plasma=fu_q,
        fu_medium=fu_medium,
        pod_censored=bool(row["censored"]),
        dose_mg=None,
        assumptions=assumptions,
    )


def equivalent_dose(inp: margin.MarginInput, key: str) -> dict:
    """Daily dose at which total Cmax would reach the chip's toxic concentration.

    Uses a dose and the Cmax reported at that dose from the same source (Geci et al.), so dose and
    exposure belong together; linear pharmacokinetics is assumed. Drugs without such a pair get none.
    """
    geci = _geci_row(key)
    empty = {"clinical_dose_mg": np.nan, "equivalent_dose_mg": np.nan,
             "equivalent_dose_band_low": np.nan, "equivalent_dose_band_high": np.nan, "dose_source": ""}
    if geci is None or not np.isfinite(geci["dose_mg"]) or not np.isfinite(geci["cmax_uM"]) or inp.pod_censored:
        return empty
    cmax_q, _ = margin.range_or_default(list(geci["cmax_values_uM"]), geci["cmax_uM"], "Cmax at clinical dose")
    paired = margin.MarginInput(name=inp.name, pod=inp.pod, cmax=cmax_q, fu_plasma=None,
                                fu_medium=inp.fu_medium, dose_mg=geci["dose_mg"])
    result = margin.compute(paired, TOTAL_THRESHOLD)
    if result.equivalent_dose_mg is None:
        return empty
    return {
        "clinical_dose_mg": geci["dose_mg"],
        "equivalent_dose_mg": result.equivalent_dose_mg,
        "equivalent_dose_band_low": result.equivalent_dose_band[0],
        "equivalent_dose_band_high": result.equivalent_dose_band[1],
        "dose_source": "dose and dose-matched Cmax from Geci et al. 2026; linear PK assumed",
    }


def margin_table() -> pd.DataFrame:
    pods = pod.liverchip_pods()
    rows = []
    for _, row in pods.iterrows():
        inp = build_input(row)
        # A fresh, fixed-seed generator per call: the same compound gets the same band everywhere.
        free = margin.compute(inp, REFERENCE_THRESHOLD)
        total = margin.compute(inp, TOTAL_THRESHOLD)
        out = free.as_row()
        out.update(
            {
                "key": row["key"],
                "partner_key": row["partner_key"],
                "garside_rank": row["garside_rank"],
                "pod_uM": row["pod_uM"],
                "pod_low_uM": row["pod_low_uM"],
                "pod_high_uM": row["pod_high_uM"],
                "cmax_total_uM": row["cmax_total_uM"],
                "fu_plasma": row["fu_plasma"],
                "fu_plasma_range": f"{inp.fu_plasma.low:.3g}-{inp.fu_plasma.high:.3g}",
                "published_mos_total": row["mos_point"],
                "verdict_total_50": total.verdict,
                "result_line_total": total.line,
            }
        )
        out.update(equivalent_dose(inp, row["key"]))
        rows.append(out)
    return pd.DataFrame(rows)


def pair_table(margins: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every matched pair from Table 1: does potency separate them, does the margin, and does the
    order agree with the clinic (Garside rank 1 = most severe)?"""
    margins = margins if margins is not None else margin_table()
    by_key = margins.set_index("key")
    seen, rows = set(), []
    for key, row in by_key.iterrows():
        partner = row["partner_key"]
        if not partner or partner not in by_key.index or frozenset((key, partner)) in seen:
            continue
        seen.add(frozenset((key, partner)))
        a, b = row, by_key.loc[partner]
        worse, better = (a, b) if a["garside_rank"] < b["garside_rank"] else (b, a)
        # A censored POD is a lower bound; a fold against it is not a measured difference.
        both_observed = not (worse["censored"] or better["censored"])
        potency_fold = better["pod_uM"] / worse["pod_uM"] if both_observed else np.nan
        margin_fold = better["margin_free"] / worse["margin_free"]
        rows.append(
            {
                "pair": f"{worse['compound']} / {better['compound']}",
                "clinically_worse": worse["compound"],
                "garside_worse": int(worse["garside_rank"]),
                "comparator": better["compound"],
                "garside_comparator": int(better["garside_rank"]),
                "pod_worse_uM": worse["pod_uM"],
                "pod_comparator_uM": better["pod_uM"],
                "comparator_pod_censored": bool(better["censored"]),
                "potency_fold": potency_fold,
                "cmax_worse_uM": worse["cmax_total_uM"],
                "cmax_comparator_uM": better["cmax_total_uM"],
                "margin_free_worse": worse["margin_free"],
                "margin_free_comparator": better["margin_free"],
                "margin_fold": margin_fold,
                "margin_fold_is_lower_bound": bool(better["censored"]),
                "verdict_worse": worse["verdict"],
                "verdict_comparator": better["verdict"],
                # With a censored comparator only one direction can be shown: a lower bound above the
                # worse drug's margin confirms the order; a lower bound below it proves nothing.
                "margin_orders_like_clinic": (
                    True if worse["margin_free"] < better["margin_free"]
                    else ("inconclusive (comparator censored)" if better["censored"] else False)
                ),
                "opposite_verdicts": worse["verdict"] == "BELOW" and better["verdict"].startswith("ABOVE"),
                # Exposure adds information beyond potency when the margin separates the pair by
                # more than potency alone does.
                "exposure_gain_fold": margin_fold / potency_fold if both_observed else np.nan,
                "both_potencies_observed": both_observed,
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["both_potencies_observed", "margin_fold"], ascending=False).reset_index(drop=True)
