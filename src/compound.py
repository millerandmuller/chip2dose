"""Single-compound run: everything Chip2Dose can say about one drug, in plain sentences."""

from __future__ import annotations

import numpy as np

from . import config, labels, liver, load, margin, pod

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


def _chip_lines(key: str) -> list[str]:
    pods = pod.liverchip_pods().set_index("key")
    row = pods.loc[key].copy()
    row["key"] = key
    inp = liver.build_input(row)
    rng = np.random.default_rng(config.MC_SEED)
    free = margin.compute(inp, liver.REFERENCE_THRESHOLD, rng)
    total = margin.compute(inp, liver.TOTAL_THRESHOLD, rng)
    lines = [
        "Liver-Chip (Ewart et al. 2022): "
        f"lowest toxic concentration {'>' if row['censored'] else ''}{row['pod_uM']:.3g} uM total "
        f"(two-donor range {row['pod_low_uM']:.3g}-{row['pod_high_uM']:.3g} uM); patient Cmax {row['cmax_total_uM']:.3g} uM.",
        "  " + free.line,
        "  " + total.line,
    ]
    lines += [f"  assumption: {a}" for a in free.assumptions]
    return lines


def _literature_lines(key: str) -> list[str]:
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
    return [
        f"Literature benchmark (Geci et al. 2026): lowest in-vitro POD {row['lowest_pod_uM']:.3g} uM (from {source}); "
        f"Cmax {row['cmax_uM']:.3g} uM at {row['dose_mg']:g} mg.",
        f"  Margin (total) {margin.fmt_ratio(p.margin_total)}, band {margin.fmt_ratio(p.total_band[0])} to "
        f"{margin.fmt_ratio(p.total_band[1])}; margin (free) {margin.fmt_ratio(p.margin_free)}, band "
        f"{margin.fmt_ratio(p.free_band[0])} to {margin.fmt_ratio(p.free_band[1])}.",
        f"  Equivalent daily dose (Cmax would reach the POD): {equivalent:,.3g} mg, band "
        f"{row['dose_mg'] * p.total_band[0]:,.3g}-{row['dose_mg'] * p.total_band[1]:,.3g} mg, "
        f"against a prescribed {row['dose_mg']:g} mg (linear-PK assumption).",
        "  " + NO_LITERATURE_CONVENTION,
        "  " + _rule_of_thumb_line(row["dose_mg"], row["logp"]),
    ] + [f"  assumption: {n}" for n in notes]


def describe(name: str) -> list[str]:
    key = load.normalize_name(name)
    lines = [f"== {name} =="]
    chip_keys = set(pod.liverchip_pods()["key"])
    if key in chip_keys:
        lines += _chip_lines(key)
    literature = _literature_lines(key)
    lines += literature
    if key not in chip_keys and not literature:
        lines.append("Not in the Liver-Chip set or the literature benchmark - no point of departure available.")
    lines.append(labels.dilirank_label(name).text)
    return lines
