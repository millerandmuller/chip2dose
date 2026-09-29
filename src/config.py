"""Constants with their sources. Nothing numeric lives anywhere else without a citation here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
RESULTS = ROOT / "results"
VALIDATION = ROOT / "validation"

LIVERCHIP_SD1 = RAW / "liverchip_supplementary_data_1.xlsx"
LIVERCHIP_ARTICLE = RAW / "liverchip_article_PMC9727064.xml"
DILIRANK = RAW / "dilirank_2.0.xlsx"
EPA_MEA = RAW / "epa_mea_toxcast_aeds_25Oct2018.xlsx"
GECI_LITERATURE = EXTERNAL / "geci_drug_literature_data.xlsx"
GECI_PROPERTIES = EXTERNAL / "geci_predicted_compound_properties.xlsx"

EWART_2022 = "Ewart et al., Commun Med 2:154 (2022)"
REDFERN_VIA_FDA = "Redfern et al. (2003) threshold as analysed in PMC7166077"


@dataclass(frozen=True)
class Threshold:
    """A margin threshold is a convention with error rates, never a law."""

    name: str
    value: float
    basis: str  # "total" or "free"
    source: str
    error_rates: str


THRESHOLDS = {
    "liver_total_50": Threshold(
        name="Liver MOS-like, total concentration",
        value=50.0,
        basis="total",
        source=f"{EWART_2022}, Table 5 (threshold 50, as used for 3D hepatic spheroids)",
        error_rates="Liver-Chip, two donors: sensitivity 80% [54-93%], specificity 100% (27 drugs)",
    ),
    "liver_free_375": Threshold(
        name="Liver MOS-like, free (protein-binding corrected)",
        value=375.0,
        basis="free",
        source=f"{EWART_2022}, Table 6 (threshold 375, chosen to maximise sensitivity at 100% specificity)",
        error_rates="Liver-Chip, two donors: sensitivity 87% [62-96%], specificity 100% (27 drugs)",
    ),
    "herg_free_30": Threshold(
        name="hERG IC50 / free Cmax",
        value=30.0,
        basis="free",
        source=REDFERN_VIA_FDA,
        error_rates="false-negative rate 27%, false-positive rate 33%",
    ),
}

# Monte-Carlo settings for uncertainty bands.
MC_SAMPLES = 20_000
MC_SEED = 20261010
BAND_PERCENTILES = (5.0, 95.0)

# Smallest margin this tool will report. Below it a ratio is no longer a toxicological statement,
# and the arithmetic itself degenerates: products with a daily dose reach the subnormal range, where
# precision is lost and an "equivalent dose" would print as a number without meaning. The smallest
# margin in any real result here is 1.8e-4 (liver total band low), eight orders of magnitude above.
MARGIN_FLOOR = 1e-12

# Stated default uncertainty where a source gives only a point value.
# Assumption: a single reported potency or fraction unbound is uncertain by a factor of 3
# (log-uniform over [x/3, 3x]). Every use is counted in n_assumptions and named in the output.
DEFAULT_FOLD_UNCERTAINTY = 3.0

# The dose-based rule of thumb (lipophilic drugs at > 100 mg/day carry higher DILI risk).
# Assumption: "lipophilic" is operationalised as logP >= 3, the cut-off of the rule-of-two
# literature; the source sentence itself does not give a number.
RULE_OF_THUMB_DOSE_MG = 100.0
RULE_OF_THUMB_LOGP = 3.0
