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

from . import config, labels  # noqa: E402

VERMILLION, BLUE, GREEN, ORANGE, SKY, GREY, BLACK = (
    "#D55E00", "#0072B2", "#009E73", "#E69F00", "#56B4E9", "#7F7F7F", "#000000",
)
VIDEO = dict(figsize=(16, 9), dpi=120)

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 16, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _log_axis(ax: plt.Axes, values: list[float], pad: float = 4.0) -> None:
    """Log x-axis padded around the data so labels never touch the frame; plain-number ticks."""
    finite = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    ax.set_xscale("log")
    ax.set_xlim(min(finite) / pad, max(finite) * pad)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


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
    ax_mar.annotate(f"convention: {threshold:g}", (threshold, 1.45), ha="center", fontsize=12)

    names = []
    for row in rows:
        clinical = labels.dilirank_label(row["compound"])
        clinic = f"Garside rank {row['garside_rank']}"
        if clinical.found:
            clinic += f"\n{clinical.label}"
        names.append(f"{row['compound']}\n{clinic}")
    _log_axis(ax_pot, [v for r in rows for v in (r["pod_low_uM"], r["pod_high_uM"], r["pod_uM"] * (6 if r["censored"] else 1))])
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

    fig.suptitle(f"{rows[0]['compound']} vs {rows[1]['compound']}: a structurally matched pair", fontsize=22, y=1.02)
    fig.text(0.01, -0.06,
             "Sources: chip MOS-like values and pairs, Ewart et al. Commun Med 2022 (Tables 1, 4; Suppl. Data 1); "
             "clinical labels, FDA DILIrank 2.0. Garside rank 1 = most severe clinical liver injury. "
             f"Threshold {threshold:g}: {config.THRESHOLDS['liver_free_375'].error_rates}.",
             fontsize=11, color=GREY, wrap=True)
    return _save(fig, path)


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
    ax.legend(handles=handles, loc="lower right", frameon=False, title="clinical severity (1 = worst)")
    ax.set_title(f"27 Liver-Chip drugs against the free-margin convention of {m['threshold'].iloc[0]:g}")
    return _save(fig, path)


# --------------------------------------------------------------------------- validation

ROC_ARMS = [
    ("potency alone", GREY, "-"),
    ("rule of thumb (dose >= 100 mg, logP >= 3)", ORANGE, ":"),
    ("margin alone (total)", SKY, "--"),
    ("learned, potency-only features (LR)", BLACK, "-."),
    ("learned, exposure-aware features (LR)", VERMILLION, "-"),
]


EXPLORATORY_ROC_ARM = ("exploratory: total Cmax alone", GREY, (0, (1, 1)))


def roc_figure(result: dict, exploratory: dict, path: Path) -> Path:
    """Pre-registered arms plus one exploratory curve (exposure alone, no chip data), labelled as such."""
    y = result["_y"]
    scores = {**result["_scores"], **exploratory["_scores"]}
    aucs = {a["arm"]: a for a in result["arms"] + exploratory["arms"]}
    fig, ax = plt.subplots(figsize=(11, 9), dpi=120)
    for name, colour, style in ROC_ARMS + [EXPLORATORY_ROC_ARM]:
        fpr, tpr, _ = roc_curve(y, scores[name])
        a = aucs[name]
        ax.plot(fpr, tpr, color=colour, ls=style, lw=3 if not name.startswith("exploratory") else 2,
                label=f"{name}: AUC {a['auc']:.2f} [{a['ci_low']:.2f}-{a['ci_high']:.2f}]")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, alpha=0.5)
    ax.set_xlabel("False-positive rate (drugs without clinical DILI concern flagged)")
    ax.set_ylabel("True-positive rate (drugs with DILI concern flagged)")
    ax.set_title(f"Held-out drugs, grouped by matched pair and structure (n = {result['n']}; "
                 f"{result['n_positive']} with concern, {result['n_negative']} without)", fontsize=15)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    ax.set_aspect("equal")
    return _save(fig, path)


def paired_difference_figure(result: dict, exploratory: dict, path: Path) -> Path:
    """Forest plot: pre-registered paired AUC differences, then the exploratory ones, separated."""
    rows = [(c["comparison"], c["delta_auc"], c["ci_low"], c["ci_high"], "pre-registered", c["primary"])
            for c in result["comparisons"]]
    rows += [(c["comparison"], c["delta_auc"], c["ci_low"], c["ci_high"], "exploratory", False)
             for c in exploratory["comparisons"]]
    fig, ax = plt.subplots(**VIDEO)
    n = len(rows)
    for i, (name, d, lo, hi, kind, primary) in enumerate(rows):
        y = n - 1 - i
        colour = VERMILLION if primary else (BLUE if kind == "pre-registered" else GREY)
        ax.plot([lo, hi], [y, y], color=colour, lw=5, alpha=0.6)
        ax.plot(d, y, "o", color=colour, ms=14 if primary else 10)
        ax.annotate(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", (hi, y), textcoords="offset points", xytext=(10, -5),
                    fontsize=13, color=colour)
    ax.axvline(0, color=BLACK, lw=1.5)
    n_pre = len(result["comparisons"])
    ax.axhline(n - n_pre - 0.5, color=GREY, lw=1, ls=":")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0].replace(" minus ", "\n  minus ") for r in rows][::-1], fontsize=11)
    ax.set_xlabel("Paired difference in ROC AUC on identical held-out folds (95% CI, group bootstrap)")
    ax.set_title("Primary comparison (red) and secondary comparisons were pre-registered;\n"
                 "grey rows were added after the first evaluation and are exploratory", fontsize=16)
    ax.set_xlim(min(-0.05, min(r[2] for r in rows) - 0.02), max(r[3] for r in rows) + 0.25)
    return _save(fig, path)


# --------------------------------------------------------------------------- F6 neural

def neural_figure(neural_margins: pd.DataFrame, n_chemicals: int, n_active: int, path: Path) -> Path:
    m = neural_margins.sort_values("margin").reset_index(drop=True)
    colours = {"AED vs predicted exposure": GREEN, "network EC50 vs clinical Cmax": BLUE}
    fig, ax = plt.subplots(**VIDEO)
    for i, row in m.iterrows():
        colour = colours[row["route"]]
        ax.plot([row["band_low"], row["band_high"]], [i, i], color=colour, lw=6, alpha=0.4)
        ax.plot(row["margin"], i, "o", color=colour, ms=10)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m["compound"], fontsize=11)
    _log_axis(ax, list(m["band_low"]) + list(m["band_high"]), pad=2.0)
    ax.set_xlabel("Margin (bar = 5-95% band). AED route: AED / predicted exposure; drug route: EC50 / free Cmax")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, ms=10, label=k) for k, c in colours.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_title(f"Neural network-formation chip (EPA MEA): {len(m)} margins where exposure data exist\n"
                 f"({n_chemicals} chemicals tested, {n_active} active; no published threshold for this endpoint)",
                 fontsize=16)
    return _save(fig, path)
