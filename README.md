# Chip2Dose

**An organ-chip tells you a concentration. A patient gets a dose. Chip2Dose is the step between them.**

Chip2Dose turns an organ-on-a-chip readout (the lowest concentration at which the chip shows toxicity) into a
**margin against the drug's real human exposure, with an uncertainty band**, and tests on published clinical
outcomes whether that exposure-aware view predicts drug-induced liver injury (DILI) better than chip potency alone.
The same method is applied to neural network-formation data from microelectrode arrays.

The output is a ratio against patient exposure or a daily dose in mg, never a 0-1 toxicity score.

## Reproduce everything

```bash
git clone https://github.com/millerandmuller/chip2dose && cd chip2dose
make            # creates .venv (Python 3.11), downloads + verifies data, writes results/ (~2 min on a laptop CPU)
make test       # 27 tests
```

No GPU, no paid service, no API key. If `python3.11` is not on your PATH: `make PYTHON=python3`.

Single compound (under a second):

```bash
.venv/bin/python run_demo.py --compound troglitazone
.venv/bin/python run_demo.py --pair clozapine olanzapine
```

## What comes out

| File | Content |
|---|---|
| `results/summary.md` | every headline number, generated |
| `results/pair_view.png`, `results/pairs/` | potency, patient exposure and margin for each of the 7 matched toxic/non-toxic pairs |
| `results/liver_margin_table.csv`, `liver_margins.png` | 27 Liver-Chip drugs: total and free margin, band, verdict against the published convention, assumptions |
| `results/roc.png`, `paired_difference.png`, `validation_*.csv` | the pre-registered validation on 220 drugs |
| `results/failure_cases.csv` | drugs the model gets wrong, in real units |
| `results/neural_margin_table.csv`, `neural_margins.png` | the neural application |
| `results/crosschecks.csv` | published numbers re-derived from their inputs, with every mismatch listed |

## Method

1. **Point of departure (POD).** Liver-Chip: minimum toxic concentration = published MOS-like value x total Cmax
   (Ewart et al. 2022, Table 4 and Supplementary Data 1), per donor. Literature benchmark: lowest in-vitro POD across
   17 hepatotoxicity datasets (Geci et al. 2026). Neural: lowest EC50 across 17 network-formation parameters (Shafer et al. 2019).
2. **Margin.** `margin_total = POD / Cmax_total`; `margin_free = POD x fu_medium / (Cmax_total x fu_plasma)`. The free
   margin is the reference, because unbound drug drives the effect. The Liver-Chip was dosed so that the free medium
   concentration is a multiple of free plasma Cmax; we verified that rule against Supplementary Data 1 (216/224 dosing
   rows reproduce within 5 %).
3. **Uncertainty.** Every input is a range: two donors, several published Cmax values, fraction unbound from the study plus
   three predictors. Where a source gives one value, a 3-fold range is assumed and counted in `n_assumptions`. Bands are
   the 5th-95th percentile of 20,000 Monte-Carlo draws.
4. **Verdict.** Against a *convention* with its error rates printed next to it: free margin 375 (Liver-Chip, sensitivity
   87 %, specificity 100 %) or total margin 50 (sensitivity 80 %, specificity 100 %). A band that straddles the threshold
   is reported as such. A POD where no toxicity was seen is a lower bound and is never treated as a number.

## Validation (pre-registered)

`validation/preregistration.json` and `validation/split.csv` were committed **before the first evaluation run**
(see the git history). The code refuses to evaluate if the split no longer matches that hash.

- 220 oral drugs with a lowest in-vitro POD, a clinical Cmax and a DILIrank label (172 with DILI concern, 48 without).
- Folds are grouped: matched toxic/non-toxic pairs, the same molecule, and structurally similar drugs
  (Morgan Tanimoto >= 0.4) never sit on both sides of a split. 20 repeats of 5-fold CV.
- Primary comparison: logistic regression on exposure-aware features vs the same model on potency-only features,
  as a paired difference in AUC with a group-bootstrap CI.

| Arm (primary endpoint) | AUC [95% CI] |
|---|---|
| potency alone | 0.65 [0.55, 0.74] |
| dose rule of thumb (>= 100 mg/day, logP >= 3) | 0.64 [0.57, 0.70] |
| margin alone, total | 0.89 [0.83, 0.94] |
| learned, potency-only features | 0.64 [0.55, 0.73] |
| **learned, exposure-aware features** | **0.89 [0.83, 0.94]** |
| **paired difference (primary)** | **+0.24 [+0.15, +0.34]** |

**What this does and does not show.** An exploratory analysis added after the first run (labelled as such everywhere)
shows that **exposure alone, without any chip data, reaches AUC 0.85** (total Cmax). The potency-exposure margin adds a
small increment over exposure alone: +0.04 [+0.00, +0.09] (total) and +0.08 [+0.03, +0.13] (free); on the narrower
endpoint the total-margin increment's CI includes zero. In plain words: in-vitro potency means little until it is set
against exposure, and most of the signal is the exposure.

## Findings we report against ourselves

- **The hero pair is not "identical potency".** Troglitazone (withdrawn) and pioglitazone (still prescribed) are a
  published matched pair, and the free margin orders them like the clinic (1.4x vs 94x). But their chip potencies
  already differ 46-fold, and both fall below the convention threshold. No matched pair in the Liver-Chip set shows
  near-identical potency: the non-toxic partners are mostly censored (no toxicity up to the highest tested concentration).
- **Trovafloxacin / levofloxacin:** chip potency orders the pair correctly; adding exposure erases the separation.
- **Pioglitazone on the literature benchmark:** its lowest POD is a potent BSEP IC50 (0.3 uM), so the margin ranks it
  riskier than troglitazone, the opposite of the clinic.
- **Neural application:** of 136 chemicals (82 active), 21 have an exposure comparator in the data; the table reports
  margins only for those. No published threshold exists for this endpoint, so no verdict is issued.
- Source issues found and reported, not corrected: telithromycin's low-dose rows in Supplementary Data 1 do not follow
  the stated dosing rule; olanzapine's Cmax in Supplementary Data 1 (0.00009 uM) has no second source in our inputs.

## Data

Seven inputs, each with URL, license, access date and SHA-256 in `data/sources.json` and `data/provenance.csv`.
Public-domain and CC-BY files are cached in `data/raw/`. The Geci et al. repository carries no license file, so its two
spreadsheets are **not redistributed**: `make data` fetches them at a pinned commit and verifies the checksum.

- Ewart L, et al. Performance assessment and economic analysis of a human Liver-Chip for predictive toxicology. *Commun Med* 2, 154 (2022). CC-BY 4.0.
- Geci R, Sayin AZ, Schaller S, Kuepfer L. Integration of in vitro and in silico approaches enables prediction of drug-induced liver injury. *Arch Toxicol* 100(5):2029-2046 (2026).
- Shafer TJ, et al. Evaluation of chemical effects on network formation in cortical neurons grown on microelectrode arrays. *Toxicol Sci* 169(2):436-455 (2019). Data: doi:10.23719/1503191 (US EPA, public domain).
- U.S. FDA. DILIrank 2.0 (public domain).

## Limitations

- Margins support a decision; they do not make one. No clinical or regulatory validity is claimed.
- Linear pharmacokinetics is assumed wherever a margin is turned into an equivalent daily dose.
- Thresholds are conventions from one 27-drug study; their error rates are small-sample estimates.
- Reverse dosimetry (httk) is not re-run; neural AEDs are the dataset's own.

## AI use

Code in this repository was written with the help of an AI coding assistant. No language model is in the numeric path:
every number comes from the deterministic pipeline above and the published inputs.

## License

MIT (code). Data files keep their own licenses, listed in `data/provenance.csv`.
