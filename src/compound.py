"""Single-compound run: everything Chip2Dose can say about one drug, in plain sentences."""

from __future__ import annotations

import numpy as np

from . import config, labels, liver, load, margin, neural, pod

NO_EXPOSURE = "No clinical exposure (Cmax) available for this compound - no margin is computed."
NO_LITERATURE_CONVENTION = (
    "No convention threshold exists for literature in-vitro PODs; see the validation for how this margin "
    "separates clinical outcomes across the benchmark."
)


def _rule_of_thumb_line(dose_mg: float, logp: float) -> str:
    flagged = dose_mg >= config.RULE_OF_THUMB_DOSE_MG and logp >= config.RULE_OF_THUMB_LOGP
    state = "FLAGGED" if flagged else "not flagged"
    return (f"Dose rule of thumb (>= {config.RULE_OF_THUMB_DOSE_MG:g} mg/day and logP >= {config.RULE_OF_THUMB_LOGP:g}): "
            f"{state} (daily dose {dose_mg:g} mg, logP {logp:.2f}). Shown as a comparison baseline, not a verdict.")


def _chip_pod_text(row) -> str:
    """The chip's point of departure in words: a value, a lower bound, or below the table's precision."""
    if row["below_precision"]:
        return (f"lowest toxic concentration below the source table's precision (printed as 0; "
                f"< {liver.BELOW_PRECISION_MOS * row['cmax_total_uM']:.2g} uM total)")
    prefix = ">" if row["censored"] else ""
    return (f"lowest toxic concentration {prefix}{row['pod_uM']:.3g} uM total "
            f"(two-donor range {row['pod_low_uM']:.3g}-{row['pod_high_uM']:.3g} uM)")


CHIP_SOURCE = "Ewart et al. 2022"
LITERATURE_SOURCE = "Geci et al. 2026"


def _printed(value: float) -> str:
    """The value as this report prints it. Two numbers that print alike need no reconciling."""
    return f"{value:.3g}"


def _cmax_note(chip: dict, lit_cmax: float, lit_dose_mg: float) -> list[str]:
    """A2: the same quantity, total plasma Cmax, measured twice in two datasets. Printed only when
    the two values do not print alike, so no boilerplate appears where there is nothing to reconcile."""
    chip_cmax = chip["cmax_total_uM"]
    if not (np.isfinite(chip_cmax) and np.isfinite(lit_cmax)):
        return []
    if _printed(chip_cmax) == _printed(lit_cmax):
        return []
    dose = f" at {margin.fmt_dose(lit_dose_mg)} mg" if np.isfinite(lit_dose_mg) and lit_dose_mg > 0 else ""
    return [f"  Two published measurements of one quantity, total plasma Cmax: {_printed(chip_cmax)} uM in the chip "
            f"block ({CHIP_SOURCE}) and {_printed(lit_cmax)} uM{dose} here ({LITERATURE_SOURCE}). Each block uses its "
            f"own source; neither is adjusted to the other."]


def _dose_note(chip: dict, lit_position: str | None, lit_pod_uM: float, lit_pod_source: str) -> list[str]:
    """A4: why the two 'patients take Nx ...' sentences can point different ways. Printed only where
    the prescribed dose falls differently against the two bands."""
    chip_position = chip.get("dose_position")
    if chip_position is None or lit_position is None or chip_position == lit_position:
        return []
    side = {"above the whole band": "above", "below the whole band": "below", "inside the band": "inside"}
    return [f"  The two dose comparisons answer different questions: the chip's point of departure is "
            f"{chip['pod_uM']:.3g} uM ({CHIP_SOURCE}), the literature value is the lowest of the published assays, "
            f"{lit_pod_uM:.3g} uM from {lit_pod_source}. The prescribed dose sits {side[chip_position]} the "
            f"chip-derived band and {side[lit_position]} the assay-derived band; different assay, different point of "
            f"departure, so the two are not expected to agree."]


def _chip_lines(key: str) -> tuple[list[str], dict]:
    """The chip block, and the facts the literature block needs to bridge to it."""
    pods = pod.liverchip_pods().set_index("key")
    row = pods.loc[key].copy()
    row["key"] = key
    inp = liver.build_input(row)
    free = margin.compute(inp, liver.REFERENCE_THRESHOLD)
    total = margin.compute(inp, liver.TOTAL_THRESHOLD)
    lines = [
        "Liver-Chip (Ewart et al. 2022): " + _chip_pod_text(row) + f"; patient Cmax {row['cmax_total_uM']:.3g} uM.",
        "  " + free.line,
        "  " + total.line,
    ]
    facts = {"cmax_total_uM": row["cmax_total_uM"], "pod_uM": row["pod_uM"], "dose_position": None}
    dose = liver.equivalent_dose(inp, key)
    if np.isfinite(dose["equivalent_dose_mg"]):
        facts["dose_position"] = margin.dose_position(
            dose["equivalent_dose_mg"], dose["equivalent_dose_band_low"],
            dose["equivalent_dose_band_high"], dose["clinical_dose_mg"])
        relation = margin.dose_relation(dose["equivalent_dose_mg"], dose["equivalent_dose_band_low"],
                                        dose["equivalent_dose_band_high"], dose["clinical_dose_mg"])
        lines.append(
            f"  Chip-derived daily dose: the chip's toxic concentration is reached at "
            f"{margin.fmt_dose(dose['equivalent_dose_mg'])} mg/day (band {margin.fmt_dose(dose['equivalent_dose_band_low'])}-"
            f"{margin.fmt_dose(dose['equivalent_dose_band_high'])}) against {margin.fmt_dose(dose['clinical_dose_mg'])} mg "
            f"prescribed: {relation}. ({dose['dose_source']})"
        )
    elif row["censored"]:
        lines.append("  No chip-derived daily dose: no toxicity was seen up to the highest tested concentration.")
    else:
        lines.append("  No chip-derived daily dose: no clinical dose with a Cmax measured at that dose is available "
                     "for this drug in our inputs.")
    lines += [f"  assumption: {a}" for a in free.assumptions]
    return lines, facts


def _literature_lines(key: str, chip: dict | None = None) -> list[str]:
    """The literature block. `chip` carries the chip block's facts, so the cross-source notes can be
    printed where the two blocks actually contradict each other rather than in a header above both."""
    geci = load.geci_benchmark()
    rows = geci[geci["key"] == key]
    if rows.empty:
        return []
    row = rows.iloc[0]
    if not np.isfinite(row["cmax_uM"]) or not np.isfinite(row["lowest_pod_uM"]):
        return [NO_EXPOSURE]
    pod_q, pod_note = margin.point_with_default(row["lowest_pod_uM"], "lowest literature POD")
    cmax_q, cmax_note = margin.range_or_default(list(row["cmax_values_uM"]), row["cmax_uM"], "total Cmax")
    fu_values = [row["fu_opera"], row["fu_admetlab"], row["fu_simplus"]]
    fu_q, fu_note = margin.range_or_default(fu_values, float(np.nanmedian(fu_values)), "fraction unbound", cap=1.0)
    notes = [n for n in (pod_note, cmax_note, fu_note) if n]
    notes.append("literature assays: fraction unbound in medium taken as 1 (nominal = free)")
    inp = margin.MarginInput(
        name=row["name"], pod=pod_q, cmax=cmax_q, fu_plasma=fu_q,
        fu_medium=margin.Quantity.exact(1.0, "nominal"), dose_mg=row["dose_mg"], assumptions=notes,
    )
    p = margin.propagate(inp, np.random.default_rng(config.MC_SEED))
    idx = int(np.argmin(row["pod_values_uM"])) if row["pod_values_uM"] else 0
    source = row["pod_sources"][idx] if idx < len(row["pod_sources"]) else "n/a"
    equivalent = row["dose_mg"] * p.margin_total
    band_low, band_high = row["dose_mg"] * p.total_band[0], row["dose_mg"] * p.total_band[1]
    lines = [
        "Literature benchmark (different data, not the chip: lowest POD across published 2D in-vitro assays, "
        "Geci et al. 2026). Its numbers differ from the chip's and are shown for comparison.",
        f"  lowest in-vitro POD {row['lowest_pod_uM']:.3g} uM (from {source}); "
        f"Cmax {row['cmax_uM']:.3g} uM at {row['dose_mg']:g} mg.",
    ]
    if chip:
        lines += _cmax_note(chip, row["cmax_uM"], row["dose_mg"])
    lines.append(
        f"  Margin (total) {margin.fmt_ratio(p.margin_total)}, band {margin.fmt_ratio(p.total_band[0])} to "
        f"{margin.fmt_ratio(p.total_band[1])}; margin (free) {margin.fmt_ratio(p.margin_free)}, band "
        f"{margin.fmt_ratio(p.free_band[0])} to {margin.fmt_ratio(p.free_band[1])}."
    )
    if row["iv_only"]:
        lines.append("  Given intravenously only: no oral equivalent daily dose and no oral dose rule of thumb.")
    else:
        lines.append(
            f"  Assay-derived daily dose (Cmax would reach this POD): {margin.fmt_dose(equivalent)} mg, band "
            f"{margin.fmt_dose(band_low)}-{margin.fmt_dose(band_high)} mg, "
            f"against {margin.fmt_dose(row['dose_mg'])} mg prescribed: "
            f"{margin.dose_relation(equivalent, band_low, band_high, row['dose_mg'], label='assay-derived dose')} "
            "(linear-PK assumption)."
        )
        if chip:
            lines += _dose_note(chip, margin.dose_position(equivalent, band_low, band_high, row["dose_mg"]),
                                row["lowest_pod_uM"], source)
    lines.append("  " + NO_LITERATURE_CONVENTION)
    if not row["iv_only"]:
        lines.append("  " + _rule_of_thumb_line(row["dose_mg"], row["logp"]))
    return lines + [f"  assumption: {n}" for n in notes]


def _neural_lines(key: str) -> list[str]:
    potency, margins = neural.margin_table()
    potency_rows = neural.potency_table()
    hit = potency_rows[potency_rows["keys"].map(lambda keys: key in keys)]
    if hit.empty:
        return []
    row = hit.iloc[0]
    if not row["active"]:
        return [f"Neural MEA network formation (Shafer et al. 2019): {row['compound']} inactive at every tested concentration."]
    lines = [f"Neural MEA network formation (Shafer et al. 2019): lowest network EC50 {row['min_network_ec50_uM']:.3g} uM."]
    for _, m in margins[margins["compound"] == row["compound"]].iterrows():
        lines.append(f"  {m['route']}: margin {margin.fmt_ratio(m['margin'])} ({m['margin_basis']}), band "
                     f"{margin.fmt_ratio(m['band_low'])} to {margin.fmt_ratio(m['band_high'])}; {m['verdict']}.")
    if len(lines) == 1:
        lines.append("  No exposure comparator in the data for this chemical - no margin is computed.")
    return lines


def describe(name: str) -> list[str]:
    key = load.normalize_name(name)
    lines = [f"== {name} =="]
    chip_keys = set(pod.liverchip_pods()["key"])
    chip_facts = None
    if key in chip_keys:
        chip_lines, chip_facts = _chip_lines(key)
        lines += chip_lines
    literature = _literature_lines(key, chip_facts)
    lines += literature
    neural_lines = _neural_lines(key)
    lines += neural_lines
    if key not in chip_keys and not literature and not neural_lines:
        lines.append("Not in the Liver-Chip set, the literature benchmark or the neural dataset - no point of departure available.")
    lines.append(labels.dilirank_label(name).text)
    return lines
