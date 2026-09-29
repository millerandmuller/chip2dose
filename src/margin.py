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
    pod: Quantity | None              # uM, lowest toxic concentration in the test system
    cmax: Quantity | None             # uM, total plasma Cmax (None = unknown: no margin)
    fu_plasma: Quantity | None        # None = unknown: free band widened to the full plausible range
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


# Fraction unbound in plasma spans roughly 0.001 (highly bound) to 1 (unbound) across drugs.
# When it is unknown, the free margin is not a single value: the band covers that whole range.
UNKNOWN_FU_RANGE = (1e-3, 1.0)
NO_MARGIN = "NO MARGIN"


def band_of(samples: np.ndarray, point: float | None = None) -> tuple[float, float]:
    """5th-95th percentile, widened to contain the reported point when there is one."""
    lo_pct, hi_pct = config.BAND_PERCENTILES
    low, high = float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct))
    if point is None or not np.isfinite(point):
        return (low, high)
    return (min(low, point), max(high, point))


def _bad(value: float | None) -> bool:
    return value is None or not np.isfinite(value) or value <= 0


def invalid_reason(inp: MarginInput) -> str | None:
    """Why no margin can be computed, in the words a scientist would use; None if inputs are usable."""
    if inp.fu_medium is None:
        return "fraction unbound in the test medium is missing (use 1 for nominal = free)"
    if inp.cmax is None or _bad(inp.cmax.point) or _bad(inp.cmax.low) or _bad(inp.cmax.high):
        return "no usable clinical exposure (Cmax); no default is substituted"
    if inp.pod is None or _bad(inp.pod.point) or _bad(inp.pod.low) or _bad(inp.pod.high):
        return "no usable point of departure (missing, zero, negative or non-finite)"
    for label, q in (("plasma", inp.fu_plasma), ("medium", inp.fu_medium)):
        if q is not None and (_bad(q.low) or _bad(q.high) or q.high > 1.0 or q.low > q.high):
            return f"fraction unbound in {label} must lie in (0, 1]; got {q.low:g}-{q.high:g}"
    quantities = [inp.pod, inp.cmax, inp.fu_medium] + ([inp.fu_plasma] if inp.fu_plasma is not None else [])
    for q in quantities:
        # Relative tolerance: the same published value can be parsed from two cells with different last digits.
        if _bad(q.point) or q.low > q.high or not (q.low * (1 - 1e-9) <= q.point <= q.high * (1 + 1e-9)):
            return f"{q.source}: the value {q.point:g} must be positive and lie inside its range {q.low:g}-{q.high:g}"
    return None


def classify(band: tuple[float, float], threshold: float, censored: bool) -> str:
    """Compare a band, not a point, with the threshold. Below = margin smaller than the convention."""
    low, high = band
    if not (np.isfinite(low) and np.isfinite(high)):
        return NO_MARGIN  # a numerical failure must never read as a decision
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
    if value >= 1e6:
        return f"{value:.2g}x"
    if value >= 100:
        return f"{value:,.0f}x"
    if value >= 1:
        return f"{value:.3g}x"
    return f"{value:.2g}x"


def result_line(name: str, basis: str, margin: float, band: tuple[float, float], threshold: Threshold,
                verdict: str, censored: bool) -> str:
    """The sentence a safety scientist reads. Plain, quantitative, no score."""
    prefix = ">" if censored else ""
    value = f"{prefix}{fmt_ratio(margin)}" if np.isfinite(margin) else "not a single value (fraction unbound unknown)"
    head = f"{name}: margin ({basis}) {value}"
    band_text = f"Band: {prefix}{fmt_ratio(band[0])} to {prefix}{fmt_ratio(band[1])}."
    conv = f"convention threshold {threshold.value:g} ({threshold.error_rates})"
    if verdict == NO_MARGIN:
        return f"{name}: no margin computed - the calculation did not produce a finite band."
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


@dataclass
class Propagated:
    margin_total: float
    total_band: tuple[float, float]
    margin_free: float
    free_band: tuple[float, float]


def propagate(inp: MarginInput, rng: np.random.Generator, n: int = config.MC_SAMPLES) -> Propagated:
    """Margins and their bands, with no judgement attached. Unknown fu widens the free band to
    the full plausible range and leaves the free point undefined."""
    total = inp.pod.sample(rng, n) / inp.cmax.sample(rng, n)
    margin_total = inp.pod.point / inp.cmax.point
    total_band = band_of(total, margin_total)
    if inp.fu_plasma is None:
        fu = np.exp(rng.uniform(np.log(UNKNOWN_FU_RANGE[0]), np.log(UNKNOWN_FU_RANGE[1]), n))
        free = total * inp.fu_medium.sample(rng, n) / fu
        return Propagated(margin_total, total_band, float("nan"), band_of(free))
    free = total * inp.fu_medium.sample(rng, n) / inp.fu_plasma.sample(rng, n)
    margin_free = margin_total * inp.fu_medium.point / inp.fu_plasma.point
    return Propagated(margin_total, total_band, margin_free, band_of(free, margin_free))


def _no_margin(inp: MarginInput, threshold: Threshold, reason: str) -> MarginResult:
    nan = float("nan")
    return MarginResult(
        name=inp.name, margin_total=nan, margin_free=nan, total_band=(nan, nan), free_band=(nan, nan),
        censored=inp.pod_censored, threshold=threshold, verdict=NO_MARGIN,
        line=f"{inp.name}: no margin computed - {reason}.",
        equivalent_dose_mg=None, equivalent_dose_band=None, dose_mg=None,
        n_assumptions=len(inp.assumptions), assumptions=list(inp.assumptions),
    )


def compute(inp: MarginInput, threshold: Threshold, rng: np.random.Generator | None = None,
            n: int = config.MC_SAMPLES) -> MarginResult:
    """Margins, bands and a verdict against a stated convention threshold."""
    reason = invalid_reason(inp)
    if reason:
        return _no_margin(inp, threshold, reason)
    rng = rng if rng is not None else np.random.default_rng(config.MC_SEED)
    p = propagate(inp, rng, n)

    assumptions = list(inp.assumptions)
    if threshold.basis == "free":
        ref_margin, ref_band = p.margin_free, p.free_band
        if inp.fu_plasma is None:
            assumptions.append(
                f"fraction unbound in plasma unknown: free band spans fu {UNKNOWN_FU_RANGE[0]:g}-{UNKNOWN_FU_RANGE[1]:g}"
            )
    else:
        ref_margin, ref_band = p.margin_total, p.total_band

    verdict = classify(ref_band, threshold.value, inp.pod_censored)
    line = result_line(inp.name, threshold.basis, ref_margin, ref_band, threshold, verdict, inp.pod_censored)

    equivalent, equivalent_band = None, None
    dose = inp.dose_mg if inp.dose_mg is not None and np.isfinite(inp.dose_mg) and inp.dose_mg > 0 else None
    if dose is not None and verdict != NO_MARGIN:
        # Linear PK: the daily dose at which total Cmax would reach the chip's toxic concentration.
        equivalent = dose * p.margin_total
        equivalent_band = (dose * p.total_band[0], dose * p.total_band[1])
        assumptions.append("equivalent daily dose assumes linear pharmacokinetics (Cmax proportional to dose)")
    return MarginResult(
        name=inp.name,
        margin_total=p.margin_total,
        margin_free=p.margin_free,
        total_band=p.total_band,
        free_band=p.free_band,
        censored=inp.pod_censored,
        threshold=threshold,
        verdict=verdict,
        line=line,
        equivalent_dose_mg=equivalent,
        equivalent_dose_band=equivalent_band,
        dose_mg=dose,
        n_assumptions=len(assumptions),
        assumptions=assumptions,
    )


def _times(ratio: float) -> str:
    return f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"


def dose_relation(dose: float, low: float, high: float, prescribed: float) -> str:
    """Prescribed dose against the chip-derived dose AND its band; the band decides the wording."""
    if not all(np.isfinite(v) and v > 0 for v in (dose, low, high, prescribed)):
        return "no comparison possible (missing or non-positive dose)"
    if prescribed > high:
        return f"patients take {_times(prescribed / dose)} the chip-derived dose (above the whole band)"
    if prescribed < low:
        return f"patients take {_times(dose / prescribed)} less than the chip-derived dose (below the whole band)"
    if np.isclose(prescribed, dose, rtol=0.05):
        return "patients take about the chip-derived dose (inside the band)"
    side = "less than" if prescribed < dose else "more than"
    ratio = dose / prescribed if prescribed < dose else prescribed / dose
    return f"patients take {_times(ratio)} {side} the point estimate, but inside the band"
