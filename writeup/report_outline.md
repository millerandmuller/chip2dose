# Chip2Dose — technical report outline

Target: 15–20 pages. Sections marked *(outline)* carry their intent and their source of numbers, not yet their
prose. Sections written out below are final text, and every number in them is taken from `results/summary.md`
or from the cited paper; none is typed by hand from memory.

| # | Section | Pages | Status |
|---|---|---|---|
| 1 | Problem: a chip gives a concentration, a patient gets a dose | 1.5 | *(outline)* |
| 2 | Prior work and what is new here | 1.5 | **written** |
| 3 | Data: three published sets, provenance and licenses | 2 | *(outline)* |
| 4 | Method: point of departure → dose → margin → uncertainty | 3 | *(outline)* |
| 5 | Method: the learned model, and why it is the AI method here | 1.5 | *(outline)* |
| 6 | Experiments: the pre-registered plan and the split | 1.5 | *(outline)* |
| 7 | Results | 3 | partly **written** (7.2) |
| 8 | Failures, limitations and bias | 2 | *(outline)* |
| 9 | Reproduction | 0.5 | *(outline)* |
| 10 | AI-use disclosure, team composition, future plans | 1 | *(outline)* |

---

## 1. Problem *(outline)*

Opens on the gap the tool fills: an organ-chip experiment returns a concentration, a prescription is a dose,
and the step between them is done by hand or not at all. Names the audience (a safety scientist who must
defend a dose decision) and states the one-sentence contribution. Sources: `README.md` opening, Section 4 of
the analysis plan.

---

## 2. Prior work and what is new here

The benchmark of 220 drugs used throughout this report is derived from the integrated dataset published by
Geci et al. (2026, *Arch Toxicol* 100(5):2029–2046, doi:10.1007/s00204-026-04305-2). That paper, not this
work, is the source of the central observation that in-vitro potency carries little information about
clinical liver injury until it is set against human exposure. The authors report that "all collected in
vitro toxicity values of drugs showed little ability for distinguishing DILI concern classes on their own",
and that "the ratio of in vivo Cmax to lowest functional in vitro toxicity of each compound" separates
clinical DILI classes with ROC AUC up to 96 %. Both statements come from a retrospective analysis of the
full set of 241 drugs, reported as point estimates without confidence intervals, without a pre-specified
analysis set, and without structure-based grouping of the drugs. This is not a criticism of that work:
a retrospective analysis is the right way to establish that a relationship exists, and it is what the paper
set out to do. (The abstract additionally reports a predicted-Cmax PBK arm reaching "up to 91 %
prospectively", which is a separate claim from the ratio compared here.)

What this work adds on that benchmark is a pre-registered evaluation of the same relationship, out-of-fold
estimation for the arms that are actually fitted, and the dose layer that sits on top of both. Three things
are new here. First, the comparison is pre-registered: the analysis
plan and the fold assignment were committed to `validation/preregistration.json` and `validation/split.csv`
before the first evaluation was run, and the code refuses to evaluate if either has since changed
(`validation/code_freeze.json` additionally fingerprints the 36 functions and constants that execute the
plan). Second, the folds are grouped, so that matched toxic/non-toxic pairs, the same molecule at different
doses, and structurally similar drugs (Morgan radius 2, 2048-bit, Tanimoto ≥ 0.4) never sit on both sides of
a split; on the primary endpoint the 220 drugs collapse into 177 such groups. Third, every arm is reported
with a confidence interval, and the comparison between arms is a paired difference on identical folds rather
than a contrast of separately estimated AUCs. Beyond the benchmark, the margin is expressed as an
equivalent daily dose in mg rather than a ratio, and the identical pipeline is applied to a second organ
(neural network formation on microelectrode arrays).

### 2.1 The published ratio on a pre-registered set, with an interval

The paper reports two AUC figures for its ratio, one per class definition. Those two definitions are
reproduced here class for class, which makes each figure directly comparable with one of our endpoints:

| Class definition (the paper's wording) | Geci et al., retrospective, 241 drugs | This work, pre-registered set, grouped CI |
|---|---|---|
| "No- from Most-DILI and Clinical Development Failure drugs" | 96 % (point estimate, no CI) | **0.938 [0.884, 0.980]** (n = 152, 132 groups) |
| "No- from Less-, Most-DILI and Clinical Development Failures" | 90 % (point estimate, no CI) | **0.889 [0.829, 0.940]** (n = 220, 177 groups) |

The section states plainly what kind of agreement this is, because the estimator on both sides is the same:
the paper's ratio has no fitted parameters, and neither does the `margin alone` arm reported here
(`src/validate.py`, `kind="score"`). The ratio is not re-estimated on held-out drugs, because there is
nothing in it to estimate, and the pre-registered split therefore cannot move the point estimate — it fixes
which drugs are in the set and it defines the groups the bootstrap resamples. That identity of estimator is
what makes the two columns comparable at all, and it is the reason the agreement is worth reporting: a fixed
ratio carried from the paper's 241 drugs to a set and a plan committed in advance, now with an interval
around it and with a paired comparison against potency alone that the retrospective analysis does not
report. Held-out estimation in this work belongs to the learned arms of Section 7.2, where the
pre-registered primary comparison (+0.243 [+0.152, +0.340]) sits between two fitted models.

Both of our confidence intervals contain the published point estimate. The two comparisons are not like for
like, and the report states the three reasons in this same paragraph: our analysis covers 220 of the 241
drugs (oral only, one row per molecule, non-ambiguous label);
the published values carry no interval, so "contains their point estimate" is a one-sided statement about
our uncertainty and not about theirs; and the two ratios are not built from the same assays, because the
paper's ratio uses functional toxicity only ("Functional toxicity refers to all in vitro toxicity data
except BSEP inhibition") while the lowest point of departure used here is the minimum over all available
assays, BSEP inhibition included. A BSEP-excluded variant of the margin is named in Section 8 as future
work; it is not reported here, because adding an arm after the first evaluation would break the
pre-registration this comparison rests on.

One further reading of the same table is worth stating, because a reviewer who has read only the abstract
will arrive with the figure 96 % in mind: setting that 96 % against our 0.889 would suggest that a grouped,
pre-registered evaluation cost about 0.07. It is an artefact of comparing two different class definitions.
The matched comparisons are the two rows above.

What does not survive as a novelty claim is the direction of the finding itself, and the report says so
where the result is first presented (Section 7.2): the ordering of the arms, and the size of the paired
advantage over potency alone (+0.243 [+0.152, +0.340] on the primary endpoint), are a confirmation of
Geci et al.'s observation rather than a discovery.

### 2.2 What the benchmark's point of departure is made of

Every AUC in this report is computed on Geci et al.'s `lowestPOD` column, so the report states what that column
contains before it states any result. It is the lowest in-vitro value across the seventeen published hepatotoxicity
datasets the authors integrated, and across the 220 drugs analysed here those values carry 23 distinct assay labels.
Counted from the shipped `pod_sources` column rather than from the documentation: a BSEP inhibition IC50 is the lowest
point of departure for 70 of the 220 drugs; THLE and HepG2 cytotoxicity in cell-line monolayers appear among the
points of departure of 117 and 113 drugs; Cell Painting cytotoxicity of 79; and the remainder are per-publication
lowest-POD aggregates together with ToxCast HepG2 high-content imaging, mitochondrial inhibition and uncoupling
readouts, and a second transporter assay (MRP2). **None of the 23 labels is an organ-chip measurement, and the most
frequent single label is a transporter-inhibition assay.**

This is a property of where clinical ground truth exists, not an oversight in the benchmark. A DILIrank outcome is
available for a drug because that drug reached patients, and a drug that reached patients was characterised in the
conventional assays available at the time; no public dataset pairs organ-chip measurements for 220 drugs with clinical
liver outcomes. The consequence is stated here and again in Section 8: the benchmark is where the thesis is tested and
the chip is where the method is applied (27 Liver-Chip drugs in Section 7.1, 136 neural chemicals in Section 7.4), so
**the step from "exposure-aware features separate clinical outcomes better than potency-only features on published
in-vitro PODs" to "the same holds for a chip-derived POD" is an assumption this report states, not a result it
presents.** How far apart the two worlds sit is itself measured in Section 7.4: for 115 of the 136 chemicals the neural
chip has measured, no public human exposure value exists to set a chip concentration against.

---

## 3. Data *(outline)*

Seven inputs with URL, license, access date and SHA-256, from `data/provenance.csv` and `data/sources.json`.
Names the three non-redistributed files and why. Records the cross-check table (`results/crosschecks.csv`),
including the 8 of 224 dosing rows and the 3 of 21 Cmax comparisons that do not reconcile, before any result
is presented.

## 4. Method: concentration to dose *(outline)*

The four steps from `README.md` "Method", at report length: point of departure per donor; total and free
margin; Monte-Carlo propagation of every input range; the equivalent daily dose under a stated linear-PK
assumption. States the convention thresholds as conventions with their published error rates, never as
limits. Sources: `src/pod.py`, `src/margin.py`, `src/config.py`.

## 5. Method: the learned model *(outline)*

Why the AI method is a small, auditable classifier and not an image model: the question is whether adding
exposure features to potency features changes the separation of clinical outcomes, which needs a model whose
coefficients can be read (`results/model_coefficients.csv`). Logistic regression and gradient boosting, the
two feature sets, and the rule that classifier probabilities never leave the validation figures. Names what the
potency feature is measured in, pointing at Section 2.2: the published in-vitro PODs of 23 assay readouts, none of
them an organ chip, so "potency" in every arm of this report means published laboratory potency.

## 6. Experiments *(outline)*

The pre-registered plan verbatim from `validation/preregistration.json`: question, inclusion rule, arms,
features, primary and secondary comparisons, the three outcome cases fixed in advance, the statistic, the
grouping and the split hash. Then what the freeze does and does not cover, from `README.md`.

## 7. Results *(outline, except 7.2)*

- **7.1** Liver-Chip: 27 drugs, the pair views, the hero pair as a daily dose (`results/dose_view.png`).
- **7.2** The benchmark: arms table, ROC (`results/roc.png`), paired differences
  (`results/paired_difference.png`), and the published-ratio comparison from Section 2.1.
- **7.3** The exploratory exposure-alone analysis, labelled as added after the first run.
- **7.4** The second organ, and the measurement of the field's bottleneck: 136 chemicals tested, 82
  active, 21 with any public human exposure comparator (13 predicted by the EPA, 9 measured clinical
  Cmax, simvastatin in both — 22 route rows for 21 chemicals). Figure: `results/neural_coverage.png`,
  table: `results/neural_coverage.csv`, both generated. The point for the reader: for 61 of the 82
  active chemicals — and for 115 of all 136 the chip has measured — the step from concentration to dose
  cannot be taken from any exposure source in these inputs, and the missing half is the published
  exposure value, not the chip.

## 8. Failures, limitations and bias *(outline)*

**First: part of the headline is range, and we measured how much** (exploratory, post-hoc, run once on
2026-09-29 after the pre-registered results were read; `results/exploratory_range_matched.csv`). Geci et al.
warn that Cmax looks predictive in their integrated dataset partly because of a range artefact: in-vitro
potency is compressed by test-concentration ranges while Cmax spreads wider. On our 220 drugs the lowest POD
spans 5.22 orders of magnitude and total Cmax 7.07 (5th-95th percentile: 2.44 against 4.45). Their "4 against 7"
describes their dataset. Ours has the same shape, less extreme. Restricting to the 195 drugs whose Cmax lies
inside the observed POD range, and refitting both primary arms inside that subset on the pre-registered folds,
the primary difference falls from +0.243 [+0.152, +0.340] to **+0.162 [+0.058, +0.275]**. It shrinks by about a
third, and its interval still excludes zero (exposure-aware LR 0.839 [0.742, 0.917], potency-only LR 0.677
[0.562, 0.774]; 165 with concern, 30 without). Which drugs the restriction removes decides how to read this.
All 25 lie below the lowest POD, none above the highest: 18 without DILI concern and 7 with concern. On the
full-set out-of-fold scores, at the Youden cut used for `failure_cases.csv` (chosen in-sample, so indicative
only), the exposure-aware model calls all 18 correctly and misses 6 of the 7. The restriction therefore mostly
takes away drugs the exposure-aware arm gets right, and so it penalises that arm. It is not a flattering subset.
Two limits on the check: the boundary is the observed POD range, which is derived rather than chosen but is
still one of several possible definitions; and a direct test of truncation was not possible. No POD in the
benchmark source is written as a bound (`censored_pod_entries() == 0`), because Geci et al. kept only compounds
with a reported potency value, so restricting to uncensored PODs keeps all 220 drugs. That is a fact about how
the benchmark was built. It means drugs that were inactive at every tested concentration are absent by
construction, and potency's weakness here is partly a property of the benchmark and not only of potency.

**Second: what the chip adds over exposure alone, assembled in one place** (the question the Oct 20–30 defense will
ask, and the one a problem owner asks first). Each concentration basis is its own baseline, and the margin's increment
is small over both. On the primary endpoint: total Cmax alone reaches AUC 0.845 and the margin on total concentrations
reaches 0.889, an increment of +0.044 [+0.003, +0.090]; free Cmax alone reaches 0.745 and the margin on free
concentrations reaches 0.824, an increment of +0.078 [+0.026, +0.132]. The section states both baselines explicitly
because the larger increment is the one over the weaker baseline: the free margin arm sits *below* the 0.845 that total
Cmax alone reaches, so adding +0.078 to 0.845 produces a number this report does not contain. On the narrower endpoint
the total increment's interval includes zero (+0.035 [−0.006, +0.083]). That is the size of the
effect on this benchmark, and the report does not claim it settles the question. What it does not measure is the setting
the tool is built for. A drug can be in this benchmark only if it reached patients, and a drug that reached patients has
a measured clinical Cmax by construction — so "exposure alone" is available for all 220 of these drugs and for none of
the compounds a chip lab tests before any human dose exists. That asymmetry is measured rather than argued: of the 136
chemicals on the neural chip, 21 have any public human exposure value, 13 of them through a predicted one; for 61 of the
82 active chemicals — 115 of all 136 — no margin is computed at all, because the missing half is the exposure and not the
chip (`results/neural_coverage.csv`, Section 7.4). The benchmark is therefore the right place to test whether
exposure-aware features separate outcomes better than potency alone, and the wrong place to estimate what a chip
contributes where it is actually used; that estimate needs a dataset that does not yet exist publicly, which is the same
absence recorded in Section 2.2.

Then leads with the findings that work against us, from `README.md` "Findings we report against ourselves":
mifepristone and the other cases in `results/failure_cases.csv`; trovafloxacin/levofloxacin as inconclusive;
pioglitazone's BSEP-driven literature POD ranking it against the clinic; the assay-count confound, computed as its own
exploratory post-hoc arm (AUC 0.655 [0.566, 0.741] on the primary endpoint, 0.723 [0.634, 0.806] on the stricter one,
against potency alone's 0.647 and 0.664 — so on the stricter endpoint the assay count beats potency alone);
the 47 cross-group pairs between Tanimoto 0.3 and 0.4. Then the standing limitations: no clinical or
regulatory validity, linear PK, thresholds from one 27-drug study, EPA exposure predictions behind 13 of the
21 neural comparators, and the practitioner-flagged gaps (free-concentration basis, benchmark-concentration
tooling, no Python port of httk). Names the BSEP-excluded margin variant and external validation on a second
DILI dataset as future work rather than omissions.

## 9. Reproduction *(outline)*

`git clone` → `make` → every figure in this report, with runtime, the one network dependency, and `make test`.

## 10. Disclosure, team, future plans *(outline)*

AI-use disclosure as in `README.md` (no language model in the numeric path). Team composition and the two
disciplines. Future plans, stated explicitly for the review round: a raw concentration-response fitter for a
lab's own uploads, reverse dosimetry via httk, the BSEP-excluded margin arm, and external validation.
