"""Neural-chip application: the same margin method on the US EPA microelectrode-array (MEA)
network-formation data (Shafer et al. 2019).

Two exposure routes, used only where the data carry them:
  * AED route: the dataset's own administered equivalent dose (mg/kg/day, EPA reverse dosimetry)
    against the ExpoCast median exposure prediction (mg/kg/day) in the same file.
  * Drug route: for pharmaceuticals, the lowest network EC50 against their clinical total Cmax
    (Geci et al. or Ewart et al.), total and free.
No published convention threshold exists for MEA network endpoints, so no verdict is issued.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, load, margin

NO_CONVENTION = "no published convention threshold for MEA network-formation endpoints"


def potency_table() -> pd.DataFrame:
    """One row per chemical (DTXSID): lowest EC50 across the 17 network parameters and entries."""
    m = load.mea_potency().copy()
    m["key"] = m["name"].map(load.normalize_name)
    m["key_common"] = m["common_name"].map(load.normalize_name)
    m["key_entry"] = m["entry_name"].map(load.normalize_name)
    first = m.sort_values("entry_min_ec50_uM").drop_duplicates("DTXSID")
    return pd.DataFrame(
        {
            "dtxsid": first["DTXSID"],
            "casrn": first["casrn"],
            "compound": first["name"],
            "keys": [
                {k for k in keys if k}
                for keys in zip(first["key"], first["key_common"], first["key_entry"])
            ],
            "min_network_ec50_uM": first["min_ec50_uM"],
            "active": first["min_ec50_uM"].notna(),
            "min_cytotoxicity_uM": first["min_cyto_uM"],
        }
    ).reset_index(drop=True)


def aed_route() -> pd.DataFrame:
    """AED (dataset value, not re-estimated) against predicted population exposure."""
    iv = load.mea_ivive().dropna(subset=["aed_mg_kg_day"]).drop_duplicates("name")
    rows = []
    for _, r in iv.iterrows():
        aed, note_a = margin.point_with_default(r["aed_mg_kg_day"], "AED (EPA, from min EC50)")
        exposure, note_e = margin.point_with_default(r["expocast_mg_kg_day"], "ExpoCast median exposure")
        rng = np.random.default_rng(config.MC_SEED)  # per compound, so results do not depend on row order
        ratio = aed.sample(rng, config.MC_SAMPLES) / exposure.sample(rng, config.MC_SAMPLES)
        point = r["aed_mg_kg_day"] / r["expocast_mg_kg_day"]
        band = margin.band_of(ratio, point)
        rows.append(
            {
                "compound": r["name"],
                "route": "AED vs predicted exposure",
                "potency": f"AED {r['aed_mg_kg_day']:.3g} mg/kg/day",
                "exposure": f"ExpoCast {r['expocast_mg_kg_day']:.3g} mg/kg/day",
                "aed_mg_kg_day": r["aed_mg_kg_day"],
                "expocast_mg_kg_day": r["expocast_mg_kg_day"],
                "invivo_lowest_pod_mg_kg": r["invivo_min_pod_mg_kg"],
                "margin": point,
                "band_low": band[0],
                "band_high": band[1],
                "margin_basis": "AED / exposure (both mg/kg/day)",
                "verdict": NO_CONVENTION,
                "n_assumptions": 2,
                "assumptions": f"{note_a} | {note_e}",
            }
        )
    return pd.DataFrame(rows)


def _cmax_lookup() -> dict[str, dict]:
    """Clinical Cmax for pharmaceuticals: Geci et al. first (ranged by all reports), else SD1."""
    out: dict[str, dict] = {}
    for _, r in load.liverchip_drug_info().iterrows():
        out[r["key"]] = {"cmax": r["cmax_total_uM"], "values": [r["cmax_total_uM"]], "fu": [r["fu_plasma"]],
                         "source": "Ewart et al. SD1"}
    for _, r in load.geci_benchmark().drop_duplicates("key").iterrows():
        out[r["key"]] = {"cmax": r["cmax_uM"], "values": list(r["cmax_values_uM"]) or [r["cmax_uM"]],
                         "fu": [r["fu_opera"], r["fu_admetlab"], r["fu_simplus"]], "source": "Geci et al."}
    return out


def drug_route(potency: pd.DataFrame) -> pd.DataFrame:
    lookup = _cmax_lookup()
    rows = []
    for _, r in potency[potency["active"]].iterrows():
        match = next((k for k in sorted(r["keys"]) if k in lookup), None)
        if match is None:
            continue
        info = lookup[match]
        assumptions: list[str] = []
        pod, note = margin.point_with_default(r["min_network_ec50_uM"], "network EC50")
        assumptions.append(note)
        cmax, note = margin.range_or_default(info["values"], info["cmax"], f"total Cmax ({info['source']})")
        if note:
            assumptions.append(note)
        fu_point = float(np.nanmedian(info["fu"]))
        fu, note = margin.range_or_default(info["fu"], fu_point, "fraction unbound in plasma", cap=1.0)
        if note:
            assumptions.append(note)
        assumptions.append("fraction unbound in MEA medium taken as 1 (nominal = free); serum content not modelled")
        inp = margin.MarginInput(
            name=r["compound"], pod=pod, cmax=cmax, fu_plasma=fu,
            fu_medium=margin.Quantity.exact(1.0, "MEA medium, nominal"), assumptions=assumptions,
        )
        p = margin.propagate(inp, np.random.default_rng(config.MC_SEED))
        rows.append(
            {
                "compound": r["compound"],
                "route": "network EC50 vs clinical Cmax",
                "potency": f"EC50 {r['min_network_ec50_uM']:.3g} uM",
                "exposure": f"Cmax {info['cmax']:.3g} uM ({info['source']})",
                "margin": p.margin_free,
                "band_low": p.free_band[0],
                "band_high": p.free_band[1],
                "margin_total": p.margin_total,
                "margin_total_band_low": p.total_band[0],
                "margin_total_band_high": p.total_band[1],
                "margin_basis": "EC50 / free Cmax",
                "verdict": NO_CONVENTION,
                "n_assumptions": len(assumptions),
                "assumptions": " | ".join(assumptions),
            }
        )
    return pd.DataFrame(rows)


AED_ROUTE = "AED vs predicted exposure"
DRUG_ROUTE = "network EC50 vs clinical Cmax"


def coverage(potency: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """How far the pipeline gets on this dataset, counted rather than described: tested -> active ->
    has a public human exposure comparator, split by which kind of comparator.

    The two routes are counted separately because they measure different things, and a chemical can
    appear in both. The last row reconciles the per-route counts with the number of chemicals, so a
    reader never has to guess whether 13 + 9 is a total.
    """
    aed = set(margins.loc[margins["route"] == AED_ROUTE, "compound"])
    drug = set(margins.loc[margins["route"] == DRUG_ROUTE, "compound"])
    both = sorted(aed & drug)
    active = set(potency.loc[potency["active"], "compound"])
    if not (aed | drug) <= active:
        raise ValueError(f"comparator without an active potency value: {sorted((aed | drug) - active)}")
    rows = [
        ("chemicals tested", len(potency), "", "EPA MEA network-formation set (Shafer et al. 2019)"),
        ("active", int(potency["active"].sum()), "", "at least one network-formation EC50"),
        ("with a human exposure comparator", len(aed | drug), "",
         "the only chemicals a margin can be computed for"),
        ("  via predicted exposure", len(aed), "EPA ExpoCast prediction", "AED / predicted population exposure"),
        ("  via measured exposure", len(drug), "measured clinical Cmax", "network EC50 / free clinical Cmax"),
        ("  in both routes", len(both), "both", ", ".join(both) or "none"),
    ]
    frame = pd.DataFrame(rows, columns=["stage", "count", "comparator", "detail"])
    # Internal consistency only, and deliberately not called a completeness check: every number here is
    # counted from `margins`, so a row missing from the input reconciles just as well as one that is
    # present. The counts the video speaks are pinned to literals in tests/test_data_and_labels.py.
    reconciles = len(aed) + len(drug) - len(both) == len(aed | drug) == margins["compound"].nunique()
    if not (reconciles and len(margins) == len(aed) + len(drug)):
        raise ValueError("neural coverage counts are not self-consistent with the margin table")
    frame.attrs["check"] = (f"{len(aed)} + {len(drug)} = {len(margins)} route rows for "
                            f"{len(aed | drug)} chemicals ({len(both)} in both)")
    return frame


def margin_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(potency for all chemicals, margins where an exposure comparator exists)."""
    potency = potency_table()
    margins = pd.concat([aed_route(), drug_route(potency)], ignore_index=True)
    potency_out = potency.drop(columns=["keys"]).assign(
        has_margin=potency["compound"].isin(set(margins["compound"]))
    )
    return potency_out, margins.sort_values("margin").reset_index(drop=True)
