"""Point-of-departure (POD) ingestion and cross-checks.

A POD here is the lowest concentration at which a system shows toxicity:
  * Liver-Chip: minimum toxic concentration = published MOS-like value x total Cmax (Table 4 x SD1)
  * Geci et al.: lowest in-vitro POD across 17 hepatotoxicity datasets
  * EPA MEA: lowest EC50 across 17 network-formation parameters

The cross-checks re-derive published numbers from the underlying values, so a parsing or unit
error shows up as a mismatch instead of a silently wrong margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import load

REL_TOL = 1e-6


# --------------------------------------------------------------------------- Liver-Chip PODs

def combine_donors(values: list[float], censored: list[bool]) -> tuple[float, bool]:
    """The rule behind the published 'both donors' column: the lowest observed toxic value wins;
    if no donor showed toxicity, the highest tested bound is reported as a lower bound."""
    observed = [v for v, c in zip(values, censored) if not c and np.isfinite(v)]
    if observed:
        return min(observed), False
    bounds = [v for v, c in zip(values, censored) if c and np.isfinite(v)]
    return (max(bounds), True) if bounds else (float("nan"), False)


def liverchip_pods() -> pd.DataFrame:
    """One row per Liver-Chip drug with POD point, donor range and censoring, in uM (total)."""
    info = load.liverchip_drug_info().set_index("key")
    mos = load.liverchip_mos().set_index("key")
    pairs = load.liverchip_pairs().set_index("key")

    rows = []
    for key in pairs.index:
        m, i, p = mos.loc[key], info.loc[key], pairs.loc[key]
        donor_values = [m["mos_d1"], m["mos_d2"]]
        donor_censored = [bool(m["mos_d1_censored"]), bool(m["mos_d2_censored"])]
        mos_point, censored = combine_donors(donor_values, donor_censored)
        observed = [v for v, c in zip(donor_values, donor_censored) if not c and np.isfinite(v)]
        n_donors = sum(np.isfinite(v) for v in donor_values)
        cmax = i["cmax_total_uM"]
        rows.append(
            {
                "key": key,
                "drug": p["drug"],
                "partner_key": p["partner_key"],
                "garside_rank": int(p["garside_rank"]),
                "cmax_total_uM": cmax,
                "fu_plasma": i["fu_plasma"],
                "fu_medium": i["fu_medium"],
                "mos_point": mos_point,
                "mos_low": min(observed) if observed else mos_point,
                "mos_high": max(observed) if observed else mos_point,
                "censored": censored,
                "n_donors": int(n_donors),
                "pod_uM": mos_point * cmax,
                "pod_low_uM": (min(observed) if observed else mos_point) * cmax,
                "pod_high_uM": (max(observed) if observed else mos_point) * cmax,
                # Table 4 prints lomitapide as '0': below the table's precision, not zero toxicity.
                "below_precision": bool(mos_point == 0),
                "exposure_assumed_from_partner": bool(i["assumed_from_partner"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- cross-checks

def _check(name: str, published: pd.Series, derived: pd.Series, labels: pd.Series, tol: float, note: str) -> dict:
    ok = np.isclose(derived.astype(float), published.astype(float), rtol=tol, atol=0) | (
        published.isna() & derived.isna()
    )
    mismatches = [
        f"{label}: published {pub:.6g}, re-derived {der:.6g}"
        for label, pub, der, good in zip(labels, published, derived, ok)
        if not good
    ]
    return {
        "check": name,
        "n_checked": int(len(published)),
        "n_mismatch": len(mismatches),
        "tolerance": f"relative {tol:g}",
        "note": note,
        "mismatches": mismatches,
    }


def check_geci_lowest_pod() -> dict:
    g = load.geci_benchmark()
    has = g["pod_values_uM"].map(len) > 0
    g = g[has]
    derived = g["pod_values_uM"].map(min)
    return _check(
        "Geci lowestPOD = min(PODValues)", g["lowest_pod_uM"], derived, g["name"], REL_TOL,
        "lowest in-vitro POD re-derived from the per-dataset values",
    )


def integrated_cmax(values: list[float], sources: list[str]) -> float:
    """How Geci et al. fill 'medianCmax': the manually curated Cmax at the stated clinical dose
    (source 'Manual') when present, otherwise the median across literature sources."""
    if "Manual" in sources and len(sources) == len(values):
        return values[sources.index("Manual")]
    return float(np.median(values))


def check_geci_median_cmax() -> dict:
    g = load.geci_benchmark()
    g = g[g["cmax_values_uM"].map(len) > 0]
    derived = pd.Series([integrated_cmax(v, s) for v, s in zip(g["cmax_values_uM"], g["cmax_sources"])], index=g.index)
    return _check(
        "Geci Cmax = curated value, else median of sources", g["cmax_uM"], derived, g["name"], REL_TOL,
        "the column is named medianCmax but holds the curated dose-matched value when one exists",
    )


def check_mea_min_ec50() -> dict:
    m = load.mea_potency()
    derived = m.groupby("DTXSID")["entry_min_ec50_uM"].transform("min")
    return _check(
        "EPA MEA min.ec50 = min over entries of the same chemical", m["min_ec50_uM"], derived, m["name"],
        REL_TOL, "per-chemical minimum EC50 across duplicate test entries",
    )


def check_liverchip_dosing() -> dict:
    """SD1 against the dosing rule stated in the Methods: the free medium concentration is a
    multiple of the free plasma Cmax, i.e. chip_free = multiplier x Cmax_total x fu_plasma."""
    info = load.liverchip_drug_info()
    sd1 = pd.read_excel(load.config.LIVERCHIP_SD1, header=None).iloc[4:, [1, 6, 8]]
    sd1.columns = ["drug", "mult", "chip_free"]
    sd1["drug"] = sd1["drug"].ffill().map(load.normalize_name)
    sd1 = sd1.dropna(subset=["mult", "chip_free"])
    sd1 = sd1[pd.to_numeric(sd1["chip_free"], errors="coerce").notna()]
    by_key = info.set_index("key")
    labels, published, derived = [], [], []
    for _, row in sd1.iterrows():
        drug = by_key.loc[row["drug"]]
        labels.append(f"{drug['drug']} x{float(row['mult']):g}")
        published.append(float(row["chip_free"]))
        derived.append(float(row["mult"]) * drug["cmax_total_uM"] * drug["fu_plasma"])
    return _check(
        "Liver-Chip free dose = multiplier x Cmax x fu_plasma", pd.Series(published), pd.Series(derived),
        pd.Series(labels), 0.05,
        "dosing rule from the Methods; the source footnotes a dosing calculation error for troglitazone",
    )


def check_liverchip_both_donors() -> dict:
    mos = load.liverchip_mos()
    mos = mos[mos["mos_both"].notna()]
    derived, flags_ok = [], []
    for _, row in mos.iterrows():
        value, censored = combine_donors(
            [row["mos_d1"], row["mos_d2"]], [row["mos_d1_censored"], row["mos_d2_censored"]]
        )
        derived.append(value)
        flags_ok.append(censored == row["mos_both_censored"])
    result = _check(
        "Liver-Chip 'both donors' MOS = combination rule", mos["mos_both"].reset_index(drop=True),
        pd.Series(derived), mos["drug"].reset_index(drop=True), REL_TOL,
        "lowest observed toxic value wins; otherwise the highest tested bound",
    )
    bad_flags = [d for d, ok in zip(mos["drug"], flags_ok) if not ok]
    result["n_mismatch"] += len(bad_flags)
    result["mismatches"] += [f"{d}: censoring flag differs" for d in bad_flags]
    return result


def check_cross_source_cmax(max_fold: float = 3.0) -> dict:
    """Independent sources for the same drug's Cmax should agree within a stated fold."""
    chip = load.liverchip_drug_info().set_index("key")
    geci = load.geci_benchmark().drop_duplicates("key").set_index("key")
    shared = sorted(set(chip.index) & set(geci.index))
    mismatches = []
    for key in shared:
        a, b = chip.at[key, "cmax_total_uM"], geci.at[key, "cmax_uM"]
        fold = max(a, b) / min(a, b)
        if fold > max_fold:
            mismatches.append(f"{key}: Liver-Chip SD1 {a:.4g} uM vs Geci median {b:.4g} uM ({fold:.1f}-fold)")
    return {
        "check": "Cmax agreement, Liver-Chip SD1 vs Geci et al.",
        "n_checked": len(shared),
        "n_mismatch": len(mismatches),
        "tolerance": f"within {max_fold:g}-fold",
        "note": "two independent literature compilations of total plasma Cmax",
        "mismatches": mismatches,
    }


def named_rederivations(names: tuple[str, ...] = ("troglitazone", "ketoconazole", "diclofenac")) -> pd.DataFrame:
    """Three named compounds whose published lowest POD is re-derived from the source values."""
    g = load.geci_benchmark().drop_duplicates("key").set_index("key")
    rows = []
    for key in names:
        row = g.loc[key]
        values, sources = row["pod_values_uM"], row["pod_sources"]
        idx = int(np.argmin(values))
        rows.append(
            {
                "compound": key,
                "published_lowest_pod_uM": row["lowest_pod_uM"],
                "rederived_uM": values[idx],
                "from_dataset": sources[idx] if idx < len(sources) else "",
                "n_source_values": len(values),
                "match": bool(np.isclose(values[idx], row["lowest_pod_uM"], rtol=REL_TOL)),
            }
        )
    return pd.DataFrame(rows)


def all_checks() -> list[dict]:
    return [
        check_geci_lowest_pod(),
        check_geci_median_cmax(),
        check_mea_min_ec50(),
        check_liverchip_dosing(),
        check_liverchip_both_donors(),
        check_cross_source_cmax(),
    ]
