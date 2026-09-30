"""Every figure in the report, generated from results tables. Never edited by hand.

Palette: Okabe-Ito (colour-blind safe). Size: 16 x 9 in at 120 dpi = 1920 x 1080 px, legible in video.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import config, labels, margin, output  # noqa: E402

VERMILLION, BLUE, GREEN, ORANGE, SKY, GREY, BLACK = (
    "#D55E00", "#0072B2", "#009E73", "#E69F00", "#56B4E9", "#7F7F7F", "#000000",
)
VIDEO = dict(figsize=(16, 9), dpi=120)
# The Kaggle Writeup editor asks for 560 x 280 ("Dimensions for the image (560 x 280)", read off the
# editor 2026-09-29, m0_findings.md 6b). Produced at exactly that size so no crop step can move it.
CARD = dict(figsize=(5.6, 2.8), dpi=100)

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 16, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def _save(fig: plt.Figure, path: Path, **savefig: object) -> Path:
    # The temporary name carries no image extension, so the format is stated rather than inferred.
    # `savefig` overrides the rcParams for one figure; the card passes the full canvas as bbox_inches
    # because the "tight" default would crop it away from the exact size the editor asks for.
    # (bbox_inches=None does NOT disable it - matplotlib reads savefig.bbox from rcParams instead.)
    output.atomic_write(path, lambda tmp: fig.savefig(tmp, format=path.suffix.lstrip("."), **savefig))
    plt.close(fig)
    return path


def _log_axis(ax: plt.Axes, values: list[float], pad: float = 4.0) -> None:
    """Log x-axis padded around the data so labels never touch the frame; plain-number ticks."""
    finite = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    ax.set_xscale("log")
    ax.set_xlim(min(finite) / pad, max(finite) * pad)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _tick(v)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _tick(value: float) -> str:
    """Axis ticks: plain numbers up to 1,000, compact powers of ten above (they collide otherwise)."""
    if value >= 1e4:
        return f"1e{int(round(np.log10(value)))}"
    return _fmt(value)


def _fmt(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.3g}"
    return f"{value:.2g}"


# --------------------------------------------------------------------------- F5 pair view

def pair_view(margins: pd.DataFrame, key_a: str, key_b: str, path: Path) -> Path:
    """Potency | patient exposure | margin, for one matched pair, with the clinic's answer."""
    by_key = margins.set_index("key")
    rows = [by_key.loc[key_a], by_key.loc[key_b]]
    rows.sort(key=lambda r: r["garside_rank"])  # clinically worse first
    colours = [VERMILLION, BLUE]
    ypos = [1, 0]

    fig, axes = plt.subplots(1, 3, **VIDEO, gridspec_kw={"wspace": 0.45})
    ax_pot, ax_exp, ax_mar = axes

    for row, colour, y in zip(rows, colours, ypos):
        censored = bool(row["censored"])
        lo, hi = row["pod_low_uM"], row["pod_high_uM"]
        ax_pot.plot([lo, hi], [y, y], color=colour, lw=6, alpha=0.35, solid_capstyle="round")
        ax_pot.plot(row["pod_uM"], y, "o", color=colour, ms=16)
        label = f"{'>' if censored else ''}{_fmt(row['pod_uM'])} uM"
        if censored:
            ax_pot.annotate("", xy=(row["pod_uM"] * 6, y), xytext=(row["pod_uM"], y),
                            arrowprops=dict(arrowstyle="->", color=colour, lw=3))
            label += "\n(no toxicity seen up to here)"
        ax_pot.annotate(label, (row["pod_uM"], y), textcoords="offset points", xytext=(0, 18),
                        ha="center", fontsize=14, color=colour)

        ax_exp.plot(row["cmax_total_uM"], y, "D", color=colour, ms=14)
        ax_exp.annotate(f"{_fmt(row['cmax_total_uM'])} uM", (row["cmax_total_uM"], y), textcoords="offset points",
                        xytext=(0, 18), ha="center", fontsize=14, color=colour)

        ax_mar.plot([row["band_low"], row["band_high"]], [y, y], color=colour, lw=10, alpha=0.35,
                    solid_capstyle="butt")
        ax_mar.plot(row["margin_free"], y, "o", color=colour, ms=16)
        ax_mar.annotate(f"{'>' if censored else ''}{_fmt(row['margin_free'])}x", (row["margin_free"], y),
                        textcoords="offset points", xytext=(0, 18), ha="center", fontsize=15, color=colour)

    threshold = rows[0]["threshold"]
    ax_mar.axvline(threshold, color=BLACK, ls="--", lw=1.5)
    ax_mar.annotate(f"convention\n{threshold:g}", (threshold, 1.45), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=12)

    names = []
    for row in rows:
        clinical = labels.dilirank_label(row["compound"])
        clinic = f"Garside rank {row['garside_rank']}"
        if clinical.found:
            clinic += f"\n{clinical.label}"
        names.append(f"{row['compound']}\n{clinic}")
    _log_axis(ax_pot, [v for r in rows for v in (r["pod_low_uM"], r["pod_high_uM"], r["pod_uM"] * (6 if r["censored"] else 1))], pad=10.0)
    _log_axis(ax_exp, [r["cmax_total_uM"] for r in rows])
    _log_axis(ax_mar, [v for r in rows for v in (r["band_low"], r["band_high"])] + [threshold])
    for ax in axes:
        ax.set_ylim(-0.6, 1.7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names if ax is ax_pot else ["", ""])
        ax.grid(axis="x", alpha=0.25)
    for tick, colour in zip(ax_pot.get_yticklabels(), colours):
        tick.set_color(colour)

    ax_pot.set_title("Chip potency\nlowest toxic concentration")
    ax_pot.set_xlabel("uM, total (bar = two-donor range)")
    ax_exp.set_title("Patient exposure\nclinical Cmax")
    ax_exp.set_xlabel("uM, total plasma")
    ax_mar.set_title("Margin\nchip free conc. / patient free Cmax")
    ax_mar.set_xlabel("x (bar = 5-95% band)")

    matched = rows[0]["partner_key"] == rows[1].name or rows[1]["partner_key"] == rows[0].name
    kind = "a published structurally matched pair" if matched else "NOT a published matched pair"
    fig.suptitle(f"{rows[0]['compound']} vs {rows[1]['compound']}: {kind}", fontsize=22, y=1.02)
    fig.text(0.01, -0.06,
             "Sources: chip MOS-like values and pairs, Ewart et al. Commun Med 2022 (Tables 1, 4; Suppl. Data 1); "
             "clinical labels, FDA DILIrank 2.0. Garside rank 1 = most severe clinical liver injury. "
             f"Threshold {threshold:g}: {config.THRESHOLDS['liver_free_375'].error_rates}.",
             fontsize=11, color=GREY, wrap=True)
    return _save(fig, path)


def dose_view(margins: pd.DataFrame, pairs: pd.DataFrame, key_a: str, key_b: str, path: Path) -> Path:
    """The output in real units: the daily dose at which the chip's toxic concentration is reached,
    next to the dose patients take. Illustrates what the tool returns; it is not the comparative
    evidence (that is the benchmark), and the footer says so."""
    by_key = margins.set_index("key")
    rows = sorted([by_key.loc[key_a], by_key.loc[key_b]], key=lambda r: r["garside_rank"])
    for row in rows:
        if not np.isfinite(row["equivalent_dose_mg"]):
            raise ValueError(f"{row['compound']}: no dose-matched Cmax, no equivalent daily dose to show")
    colours, ypos = [VERMILLION, BLUE], [1, 0]

    fig, ax = plt.subplots(**VIDEO)
    values = []
    for row, colour, y in zip(rows, colours, ypos):
        lo, hi = row["equivalent_dose_band_low"], row["equivalent_dose_band_high"]
        dose, prescribed = row["equivalent_dose_mg"], row["clinical_dose_mg"]
        values += [lo, hi, prescribed]
        ax.plot([lo, hi], [y, y], color=colour, lw=14, alpha=0.3, solid_capstyle="butt")
        ax.plot(dose, y, "o", color=colour, ms=18)
        ax.annotate(f"chip toxic concentration reached at ~{_fmt(dose)} mg/day\n(band {_fmt(lo)}-{_fmt(hi)})", (dose, y),
                    textcoords="offset points", xytext=(0, 26), ha="center", fontsize=15, color=colour)
        ax.plot(prescribed, y, "D", color=BLACK, ms=16)
        ax.annotate(f"patients: {_fmt(prescribed)} mg/day", (prescribed, y), textcoords="offset points",
                    xytext=(0, -38), ha="center", fontsize=15)
        ax.annotate(margin.dose_relation(dose, lo, hi, prescribed), (0.99, y + 0.46), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=15, color=colour, weight="bold")

    names = []
    for row in rows:
        clinical = labels.dilirank_label(row["compound"])
        status = "withdrawn" if clinical.found and "withdrawn" in clinical.text.lower() else "not withdrawn"
        names.append(f"{row['compound']}\n{status}; Garside rank {row['garside_rank']}")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=16)
    for tick, colour in zip(ax.get_yticklabels(), colours):
        tick.set_color(colour)
    _log_axis(ax, values, pad=3.0)
    ax.set_ylim(-0.7, 1.8)
    ax.set_xlabel("mg per day (log scale)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(f"{rows[0]['compound']} and {rows[1]['compound']}: the chip result as a daily dose", fontsize=22)

    pair = pairs[pairs["clinically_worse"].eq(rows[0]["compound"]) & pairs["comparator"].eq(rows[1]["compound"])]
    potency_note = (f"Chip potency alone already ranks {rows[0]['compound']} as more toxic "
                    f"({pair['potency_fold'].iloc[0]:.0f}-fold). " if not pair.empty and
                    np.isfinite(pair["potency_fold"].iloc[0]) else "")
    fig.text(0.01, -0.08,
             f"{potency_note}This pair illustrates what the tool returns; the comparison with potency alone is the "
             "220-drug benchmark. Chip-derived dose = clinical dose x (chip toxic concentration / Cmax at that dose), "
             "assuming linear pharmacokinetics; dose and dose-matched Cmax from Geci et al. 2026; chip data Ewart et al. 2022; "
             "withdrawal status from FDA DILIrank 2.0.",
             fontsize=11, color=GREY, wrap=True)
    return _save(fig, path)


def card_image(margins: pd.DataFrame, key_a: str, key_b: str, path: Path,
               aspect: dict | None = None) -> Path:
    """The submission card: the same hero-pair numbers as `dose_view`, composed for a small card.

    Not a shrunk figure. It is read at roughly 300 px wide in a gallery grid, so everything that only
    works at full size is gone - no axis, no ticks, no grid, no footnote - and what remains is the two
    compounds, the chip-derived daily dose with its band, and the dose patients took. `aspect` defaults
    to `CARD` (560 x 280, the size the Kaggle editor asks for); pass another dict to retarget it."""
    by_key = margins.set_index("key")
    rows = sorted([by_key.loc[key_a], by_key.loc[key_b]], key=lambda r: r["garside_rank"])
    for row in rows:
        if not np.isfinite(row["equivalent_dose_mg"]):
            raise ValueError(f"{row['compound']}: no dose-matched Cmax, no equivalent daily dose to show")

    fig, ax = plt.subplots(**(aspect or CARD))
    fig.subplots_adjust(left=0.26, right=0.98, top=0.70, bottom=0.08)
    values = []
    for row, colour, y in zip(rows, [VERMILLION, BLUE], [1, 0]):
        lo, hi = row["equivalent_dose_band_low"], row["equivalent_dose_band_high"]
        dose, prescribed = row["equivalent_dose_mg"], row["clinical_dose_mg"]
        values += [lo, hi, prescribed]
        ax.plot([lo, hi], [y, y], color=colour, lw=9, alpha=0.3, solid_capstyle="butt")
        ax.plot(dose, y, "o", color=colour, ms=9)
        ax.annotate(f"chip: {_fmt(dose)} mg/day ({_fmt(lo)}-{_fmt(hi)})", (dose, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=11, color=colour)
        ax.plot(prescribed, y, "D", color=BLACK, ms=8)
        ax.annotate(f"patients: {_fmt(prescribed)}", (prescribed, y), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=11)

    ax.set_yticks([1, 0])
    ax.set_yticklabels([r["compound"] for r in rows], fontsize=12)
    for tick, colour in zip(ax.get_yticklabels(), [VERMILLION, BLUE]):
        tick.set_color(colour)
    ax.tick_params(axis="y", length=0)
    _log_axis(ax, values, pad=5.0)
    ax.set_ylim(-0.75, 1.75)
    # Everything a full-size figure can afford and a 300 px card cannot. A log scale keeps drawing
    # minor ticks after set_xticks([]), so the locator is cleared too, not just the labels.
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="x", which="both", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    # Two lines, not one: at this width a single line of the sentence runs off the canvas.
    fig.text(0.02, 0.93, "An organ-chip gives a concentration.", fontsize=12.5, weight="bold", va="top")
    fig.text(0.02, 0.80, "A patient gets a dose.", fontsize=12.5, weight="bold", va="top")
    return _save(fig, path, bbox_inches=fig.bbox_inches)


def liver_overview(margins: pd.DataFrame, path: Path) -> Path:
    """All 27 Liver-Chip drugs: free margin with band against the convention threshold."""
    m = margins.sort_values("margin_free").reset_index(drop=True)
    cmap = {1: VERMILLION, 2: ORANGE, 3: SKY, 4: BLUE, 5: GREEN}
    fig, ax = plt.subplots(**VIDEO)
    for i, row in m.iterrows():
        colour = cmap[int(row["garside_rank"])]
        ax.plot([row["band_low"], row["band_high"]], [i, i], color=colour, lw=6, alpha=0.4)
        marker = ">" if row["censored"] else "o"
        ax.plot(row["margin_free"], i, marker, color=colour, ms=10)
    ax.axvline(m["threshold"].iloc[0], color=BLACK, ls="--", lw=1.5)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m["compound"], fontsize=11)
    _log_axis(ax, list(m["band_low"]) + list(m["band_high"]) + [m["threshold"].iloc[0]], pad=2.0)
    ax.set_xlabel("Margin (free): chip free concentration / patient free Cmax   (bar = 5-95% band; > = lower bound)")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, ms=10, label=f"Garside rank {r}") for r, c in cmap.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              title="clinical severity\n(1 = worst)")
    ax.set_title(f"27 Liver-Chip drugs against the free-margin convention of {m['threshold'].iloc[0]:g}")
    return _save(fig, path)


# --------------------------------------------------------------------------- validation

# (arm name as the validation reports it, short label for the legend, colour, line style). The short
# label keeps the legend inside the axes: below the axes it cost a quarter of the frame, and this
# figure is read at video resolution. The rule-of-thumb definition moves to the footnote.
ROC_ARMS = [
    ("potency alone", "potency alone", GREY, "-"),
    ("rule of thumb (dose >= 100 mg, logP >= 3)", "dose rule of thumb", ORANGE, ":"),
    ("margin alone (total)", "margin alone (total)", SKY, "--"),
    ("learned, potency-only features (LR)", "learned: potency only (LR)", BLACK, "-."),
    ("learned, exposure-aware features (LR)", "learned: exposure-aware (LR)", VERMILLION, "-"),
]


EXPLORATORY_ROC_ARM = ("exploratory: total Cmax alone", "exploratory: Cmax alone", GREY, (0, (1, 1)))


def roc_figure(result: dict, exploratory: dict, path: Path) -> Path:
    """Pre-registered arms plus one exploratory curve (exposure alone, no chip data), labelled as such."""
    y = result["_y"]
    scores = {**result["_scores"], **exploratory["_scores"]}
    aucs = {a["arm"]: a for a in result["arms"] + exploratory["arms"]}
    fig, ax = plt.subplots(figsize=(11, 9), dpi=120)
    for name, short, colour, style in ROC_ARMS + [EXPLORATORY_ROC_ARM]:
        fpr, tpr, _ = roc_curve(y, scores[name])
        a = aucs[name]
        ax.plot(fpr, tpr, color=colour, ls=style, lw=3 if not name.startswith("exploratory") else 2,
                label=f"{short}: AUC {a['auc']:.2f} [{a['ci_low']:.2f}-{a['ci_high']:.2f}]")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, alpha=0.5)
    ax.set_xlabel("False-positive rate (drugs without clinical DILI concern flagged)")
    ax.set_ylabel("True-positive rate (drugs with DILI concern flagged)")
    # "Held-out" would be false for the score arms on this plot: a fixed ratio has nothing to fit, so the
    # pre-registered split fixes the set and the bootstrap groups, not the point estimate. Name the set.
    ax.set_title(f"Pre-registered evaluation set, grouped by matched pair and structure (n = {result['n']}; "
                 f"{result['n_positive']} with concern, {result['n_negative']} without)", fontsize=15)
    # Lower right: empty by construction in a ROC plot, so the legend costs no curve area.
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor=GREY, fontsize=13,
              borderpad=0.7, labelspacing=0.5, handlelength=2.6)
    ax.set_aspect("equal")
    # Two footnote lines, not three: a third needs a bigger bottom margin, which costs ~8% of the plot
    # area that round 4 bought back for video legibility.
    fig.text(0.01, 0.005,
             f"Dose rule of thumb: flagged when the daily dose is >= {config.RULE_OF_THUMB_DOSE_MG:g} mg and "
             f"logP >= {config.RULE_OF_THUMB_LOGP:g}. LR = logistic regression. Brackets are 95% CIs (group bootstrap)."
             "\nPublished comparator: Geci et al. 2026 report 90% for this class definition, retrospectively on all "
             "241 drugs, with no interval. Their ratio is parameter-free, and so are the margin arms here.",
             fontsize=11, color=GREY, linespacing=1.4)
    return _save(fig, path)


def paired_difference_figure(result: dict, exploratory: dict, path: Path,
                             range_matched: dict | None = None) -> Path:
    """Forest plot: pre-registered paired AUC differences, then the exploratory ones, separated.

    `range_matched` appends the post-hoc range check below a second divider. It is the only row on this
    figure with a different denominator, so its label states both n values rather than relying on the
    reader to notice, and its marker is an open square instead of a filled circle for the same reason."""
    rows = [(c["comparison"], c["delta_auc"], c["ci_low"], c["ci_high"], "pre-registered", c["primary"])
            for c in result["comparisons"]]
    rows += [(c["comparison"], c["delta_auc"], c["ci_low"], c["ci_high"], "exploratory", False)
             for c in exploratory["comparisons"]]
    if range_matched is not None:
        c = range_matched["comparisons"][0]
        label = (f"post-hoc: same primary pair refit on\n  the range-matched subset "
                 f"(n = {range_matched['n']}, not {result['n']})")
        rows.append((label, c["delta_auc"], c["ci_low"], c["ci_high"], "post-hoc", False))
    fig, ax = plt.subplots(**VIDEO)
    n = len(rows)
    for i, (name, d, lo, hi, kind, primary) in enumerate(rows):
        y = n - 1 - i
        colour = VERMILLION if primary else (BLUE if kind == "pre-registered" else GREY)
        post_hoc = kind == "post-hoc"
        ax.plot([lo, hi], [y, y], color=colour, lw=5, alpha=0.6, ls=":" if post_hoc else "-")
        ax.plot(d, y, "s" if post_hoc else "o", color=colour, ms=11 if post_hoc else (14 if primary else 10),
                markerfacecolor="none" if post_hoc else colour, markeredgewidth=2.5 if post_hoc else 1.0)
        ax.annotate(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", (hi, y), textcoords="offset points", xytext=(10, -5),
                    fontsize=13, color=colour)
    ax.axvline(0, color=BLACK, lw=1.5)
    n_pre = len(result["comparisons"])
    ax.axhline(n - n_pre - 0.5, color=GREY, lw=1, ls=":")
    if range_matched is not None:
        # Dashed where the first divider is dotted: the two separate different things and must not
        # read as one kind of break. The post-hoc row is appended last, so it sits at y = 0.
        ax.axhline(0.5, color=GREY, lw=1.5, ls="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0].replace(" minus ", "\n  minus ") for r in rows][::-1], fontsize=11)
    # Secondary rows compare parameter-free score arms, so "held-out folds" would not be true of every row;
    # what every row does share is that both arms are scored on the same drugs in the same bootstrap draw.
    ax.set_xlabel("Paired difference in ROC AUC, both arms scored on the same drugs (95% CI, group bootstrap)")
    # "(top row)", not a colour name: the bar is vermilion and reads orange at 1080p, and a position
    # survives a palette change that a colour word does not.
    ax.set_title("Primary comparison (top row) and secondary comparisons were pre-registered;\n"
                 "grey rows were added after the first evaluation and are exploratory", fontsize=16)
    ax.set_xlim(min(-0.05, min(r[2] for r in rows) - 0.02), max(r[3] for r in rows) + 0.25)
    return _save(fig, path)


# --------------------------------------------------------------------------- F6 neural

def neural_coverage_figure(coverage: pd.DataFrame, path: Path) -> Path:
    """Where the pipeline stops on this dataset, and why: not at the chip, at the exposure half.

    Three funnel bars for the counted stages, then the two comparator routes as separate bars,
    because they measure different things and one chemical appears in both.
    """
    funnel = coverage.iloc[:3]
    routes = coverage.iloc[3:5]
    total = int(funnel["count"].iloc[0])
    fig, (ax, ax_r) = plt.subplots(1, 2, **VIDEO, gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.05})

    colours = [SKY, BLUE, VERMILLION]
    for i, (_, row) in enumerate(funnel.iterrows()):
        y = len(funnel) - 1 - i
        ax.barh(y, row["count"], color=colours[i], height=0.5)
        ax.annotate(f"{row['count']}", (row["count"], y), textcoords="offset points", xytext=(12, 0),
                    va="center", fontsize=22, weight="bold", color=colours[i])
    ax.set_yticks(range(len(funnel)))
    # The qualifier rides with its stage, so no floating text can collide with a bar or the axis.
    ax.set_yticklabels([f"{r['stage'].strip()}\n{r['detail']}" for _, r in funnel.iloc[::-1].iterrows()], fontsize=14)
    ax.set_ylim(-0.6, len(funnel) - 0.4)
    ax.set_xlim(0, total * 1.2)
    ax.set_xlabel("chemicals", fontsize=14)
    ax.set_title("The pipeline runs out of exposure data, not chip data", fontsize=20, pad=18)

    route_colours = [GREEN, BLUE]
    for i, (_, row) in enumerate(routes.iterrows()):
        y = len(routes) - 1 - i
        ax_r.barh(y, row["count"], color=route_colours[i], height=0.4)
        ax_r.annotate(f"{row['count']}", (row["count"], y), textcoords="offset points", xytext=(10, 0),
                      va="center", fontsize=20, weight="bold", color=route_colours[i])
    ax_r.set_yticks(range(len(routes)))
    ax_r.set_yticklabels([c.replace(" ", "\n", 1) for c in routes["comparator"]][::-1], fontsize=14)
    ax_r.tick_params(axis="y", pad=8)
    ax_r.set_ylim(-0.7, len(routes) - 0.3)
    ax_r.set_xlim(0, max(routes["count"]) * 1.3)
    ax_r.set_xlabel("chemicals", fontsize=14)
    ax_r.set_title("...and the comparator\nthose few have", fontsize=17, pad=18)
    for axis in (ax, ax_r):  # counts of chemicals: whole numbers only
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.text(0.01, -0.02,
             f"Counted from the generated tables, not asserted: {coverage.attrs['check']}. The two routes measure "
             "different things (a predicted population exposure and a measured clinical Cmax) and are never pooled "
             "into one number. No published convention threshold exists for this endpoint, so no verdict is issued.",
             fontsize=12, color=GREY, wrap=True)
    return _save(fig, path)


def neural_figure(neural_margins: pd.DataFrame, n_chemicals: int, n_active: int, path: Path) -> Path:
    """Two panels, because the two routes measure different things and must not share an axis."""
    routes = [
        ("network EC50 vs clinical Cmax", BLUE, "EC50 / free clinical Cmax (x)", "Pharmaceuticals: chip vs patient exposure"),
        ("AED vs predicted exposure", GREEN, "AED / ExpoCast predicted exposure (x)", "Environmental chemicals: chip-derived dose vs population exposure"),
    ]
    fig, axes = plt.subplots(1, 2, **VIDEO, gridspec_kw={"wspace": 0.55})
    for ax, (route, colour, xlabel, title) in zip(axes, routes):
        m = neural_margins[neural_margins["route"] == route].sort_values("margin").reset_index(drop=True)
        for i, row in m.iterrows():
            ax.plot([row["band_low"], row["band_high"]], [i, i], color=colour, lw=6, alpha=0.4)
            ax.plot(row["margin"], i, "o", color=colour, ms=10)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m["compound"], fontsize=12)
        _log_axis(ax, list(m["band_low"]) + list(m["band_high"]), pad=2.0)
        ax.set_xlabel(xlabel + "   bar = 5-95% band", fontsize=13)
        ax.set_title(f"{title}\n({len(m)} compounds)", fontsize=15)
    n_unique = neural_margins["compound"].nunique()
    both = sorted(set(neural_margins.loc[neural_margins["route"] == routes[0][0], "compound"])
                  & set(neural_margins.loc[neural_margins["route"] == routes[1][0], "compound"]))
    if both:
        fig.text(0.01, -0.02, f"{', '.join(both)} appears in both panels (both exposure routes exist), so the panels "
                 f"hold {len(neural_margins)} points for {n_unique} chemicals.", fontsize=12, color=GREY)
    fig.suptitle(f"Neural network-formation chip (EPA MEA): {n_chemicals} chemicals tested, {n_active} active, "
                 f"{n_unique} with an exposure comparator.\nNo published threshold exists for this endpoint; no verdicts.",
                 fontsize=17, y=1.03)
    return _save(fig, path)
