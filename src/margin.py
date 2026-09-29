"""Margin engine: in-vitro point of departure -> margin against real human exposure, with a band.

    margin_total = POD / Cmax_total
    margin_free  = (POD x fu_medium) / (Cmax_total x fu_plasma)
    equivalent daily dose = clinical dose x margin_total     (only under a linear-PK assumption)

Every input is a range. Where a source gives a range (two donors, several Cmax reports, several
fraction-unbound predictors) that range is used; where it gives a single value, a stated default
uncertainty is applied and counted in n_assumptions. The band is the 5th-95th percentile of a
Monte-Carlo sample, widened where needed to contain the reported point value: the published
point is often the conservative end of a range (e.g. the lower of two donors), and a band that
excludes the number printed next to it would read as an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .config import Threshold


@dataclass(frozen=True)
class Quantity:
    """A positive quantity known to lie in [low, high]; sampled log-uniformly."""

    point: float
    low: float
    high: float
    source: str

    @classmethod
    def exact(cls, value: float, source: str) -> "Quantity":
        return cls(value, value, value, source)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if not (self.low > 0 and self.high > 0):
            raise ValueError(f"non-positive range for {self.source}: {self.low}..{self.high}")
        if self.low == self.high:
            return np.full(n, self.point)
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), n))


def point_with_default(value: float, source: str, cap: float | None = None) -> tuple[Quantity, str]:
    """A single reported value widened by the stated default fold; returns the assumption text."""
    fold = config.DEFAULT_FOLD_UNCERTAINTY
    high = value * fold if cap is None else min(value * fold, cap)
    note = f"{source}: single value, {fold:g}-fold uncertainty assumed"
    return Quantity(value, value / fold, high, f"{source} (+/-{fold:g}-fold assumed)"), note


def range_or_default(values: list[float], point: float, source: str, cap: float | None = None) -> tuple[Quantity, str | None]:
    """Use the spread of several reported values; fall back to the default fold for one value."""
    finite = sorted(v for v in values if v is not None and np.isfinite(v) and v > 0)
    if len(finite) >= 2 and finite[0] < finite[-1]:
        return Quantity(point, finite[0], finite[-1], f"{source} (range of {len(finite)} values)"), None
    return point_with_default(point, source, cap)


@dataclass
class MarginInput:
    name: str
    pod: Quantity                     # uM, lowest toxic concentration in the test system
    cmax: Quantity                    # uM, total plasma Cmax
    fu_plasma: Quantity | None        # None = unknown: no free margin is computed
    fu_medium: Quantity               # free fraction in the test medium
    pod_censored: bool = False        # True = no toxicity up to pod; the margin is a lower bound
    dose_mg: float | None = None      # clinical daily dose the Cmax belongs to
    assumptions: list[str] = field(default_factory=list)


@dataclass
class MarginResult:
    name: str
    margin_total: float
    margin_free: float
    total_band: tuple[float, float]
    free_band: tuple[float, float]
    censored: bool
    threshold: Threshold
    verdict: str
    line: str
    equivalent_dose_mg: float | None
    equivalent_dose_band: tuple[float, float] | None
    dose_mg: float | None
    n_assumptions: int
    assumptions: list[str]

    def as_row(self) -> dict:
        return {
            "compound": self.name,
            "margin_total": self.margin_total,
            "margin_total_band_low": self.total_band[0],
            "margin_total_band_high": self.total_band[1],
            "margin_free": self.margin_free,
            "band_low": self.free_band[0],
            "band_high": self.free_band[1],
            "censored": self.censored,
            "threshold": self.threshold.value,
            "threshold_basis": self.threshold.basis,
            "threshold_source": self.threshold.source,
            "threshold_error_rates": self.threshold.error_rates,
            "verdict": self.verdict,
            "clinical_dose_mg": self.dose_mg,
            "equivalent_dose_mg": self.equivalent_dose_mg,
            "equivalent_dose_band_low": None if self.equivalent_dose_band is None else self.equivalent_dose_band[0],
            "equivalent_dose_band_high": None if self.equivalent_dose_band is None else self.equivalent_dose_band[1],
            "n_assumptions": self.n_assumptions,
            "assumptions": " | ".join(self.assumptions),
            "result_line": self.line,
        }


def band_of(samples: np.ndarray, point: float) -> tuple[float, float]:
    lo_pct, hi_pct = config.BAND_PERCENTILES
    low, high = float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct))
    return (min(low, point), max(high, point))


def classify(band: tuple[float, float], threshold: float, censored: bool) -> str:
    """Compare a band, not a point, with the threshold. Below = margin smaller than the convention."""
    low, high = band
    if censored:
        return "ABOVE (lower bound)" if low > threshold else "INCONCLUSIVE (censored below threshold)"
    if high < threshold:
        return "BELOW"
    if low > threshold:
        return "ABOVE"
    return "STRADDLES"


def fmt_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value >= 100:
        return f"{value:,.0f}x"
    if value >= 1:
        return f"{value:.3g}x"
    return f"{value:.2g}x"


def result_line(name: str, basis: str, margin: float, band: tuple[float, float], threshold: Threshold,
                verdict: str, censored: bool) -> str:
    """The sentence a safety scientist reads. Plain, quantitative, no score."""
    prefix = ">" if censored else ""
    head = f"{name}: margin ({basis}) {prefix}{fmt_ratio(margin)}"
    band_text = f"Band: {prefix}{fmt_ratio(band[0])} to {prefix}{fmt_ratio(band[1])}."
    conv = f"convention threshold {threshold.value:g} ({threshold.error_rates})"
    if verdict == "BELOW":
        return f"{head} - below the {conv}. {band_text}"
    if verdict.startswith("ABOVE"):
        return f"{head} - above the {conv}. {band_text}"
    if verdict == "STRADDLES":
        return f"{head}. {band_text} Band straddles the threshold of {threshold.value:g}. This compound needs more chips before a decision."
    return (
        f"{head}. No toxicity observed up to the highest tested concentration, which does not reach the "
        f"threshold of {threshold.value:g}. Test higher before drawing a conclusion. {band_text}"
    )


def compute(inp: MarginInput, threshold: Threshold, rng: np.random.Generator | None = None,
            n: int = config.MC_SAMPLES) -> MarginResult:
    rng = rng if rng is not None else np.random.default_rng(config.MC_SEED)

    pod = inp.pod.sample(rng, n)
    cmax = inp.cmax.sample(rng, n)
    total = pod / cmax
    margin_total = inp.pod.point / inp.cmax.point
    total_band = band_of(total, margin_total)

    if inp.fu_plasma is None:
        margin_free, free_band = float("nan"), (float("nan"), float("nan"))
    else:
        free = total * inp.fu_medium.sample(rng, n) / inp.fu_plasma.sample(rng, n)
        margin_free = margin_total * inp.fu_medium.point / inp.fu_plasma.point
        free_band = band_of(free, margin_free)

    if threshold.basis == "free":
        if inp.fu_plasma is None:
            raise ValueError(f"{inp.name}: free-basis threshold requested but fraction unbound is unknown")
        ref_margin, ref_band = margin_free, free_band
    else:
        ref_margin, ref_band = margin_total, total_band

    verdict = classify(ref_band, threshold.value, inp.pod_censored)
    line = result_line(inp.name, threshold.basis, ref_margin, ref_band, threshold, verdict, inp.pod_censored)

    equivalent, equivalent_band = None, None
    if inp.dose_mg is not None and np.isfinite(inp.dose_mg):
        # Linear PK: the daily dose at which total Cmax would reach the chip's toxic concentration.
        equivalent = inp.dose_mg * margin_total
        equivalent_band = (inp.dose_mg * total_band[0], inp.dose_mg * total_band[1])

    assumptions = list(inp.assumptions)
    if equivalent is not None:
        assumptions.append("equivalent daily dose assumes linear pharmacokinetics (Cmax proportional to dose)")
    return MarginResult(
        name=inp.name,
        margin_total=margin_total,
        margin_free=margin_free,
        total_band=total_band,
        free_band=free_band,
        censored=inp.pod_censored,
        threshold=threshold,
        verdict=verdict,
        line=line,
        equivalent_dose_mg=equivalent,
        equivalent_dose_band=equivalent_band,
        dose_mg=inp.dose_mg,
        n_assumptions=len(assumptions),
        assumptions=assumptions,
    )
