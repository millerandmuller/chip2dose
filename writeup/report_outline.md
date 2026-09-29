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
full set of 241 drugs, reported as point estimates without confidence intervals; the paper describes no
cross-validation, no held-out test set and no structure-based grouping of the drugs. This is not a criticism
of that work: an in-sample analysis is the right way to establish that a relationship exists, and it is what
the paper set out to do.

What this work adds on that benchmark is an out-of-sample test of the same relationship, and the dose layer
that sits on top of it. Three things are new here. First, the comparison is pre-registered: the analysis
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

### 2.1 In-sample and out-of-sample on the same benchmark

The paper reports two AUC figures for its ratio, one per class definition. Those two definitions are
reproduced here class for class, which makes each figure directly comparable with one of our endpoints:

| Class definition (the paper's wording) | Geci et al., retrospective, 241 drugs | This work, grouped out-of-sample split |
|---|---|---|
| "No- from Most-DILI and Clinical Development Failure drugs" | 96 % (point estimate, no CI) | **0.938 [0.884, 0.980]** (n = 152, 132 groups) |
| "No- from Less-, Most-DILI and Clinical Development Failures" | 90 % (point estimate, no CI) | **0.889 [0.829, 0.940]** (n = 220, 177 groups) |

Both of our confidence intervals contain the published point estimate. The headline finding of this
comparison is therefore that the published in-sample result holds up when the ratio is re-estimated on
held-out drugs under structure-grouped folds — it does not depend on the drugs the relationship was found
in. The two comparisons are not like for like, and the report states the three reasons in this same
paragraph: our analysis covers 220 of the 241 drugs (oral only, one row per molecule, non-ambiguous label);
the published values carry no interval, so "contains their point estimate" is a one-sided statement about
our uncertainty and not about theirs; and the two ratios are not built from the same assays, because the
paper's ratio uses functional toxicity only ("Functional toxicity refers to all in vitro toxicity data
except BSEP inhibition") while the lowest point of departure used here is the minimum over all available
assays, BSEP inhibition included. A BSEP-excluded variant of the margin is named in Section 8 as future
work; it is not reported here, because adding an arm after the first evaluation would break the
pre-registration this comparison rests on.

One further reading of the same table is worth stating, because a reviewer who has read only the abstract
will arrive with the figure 96 % in mind: setting that 96 % against our 0.889 would suggest a drop of about
0.07 from out-of-sample validation. It is an artefact of comparing two different class definitions. The
matched comparisons are the two rows above.

What does not survive as a novelty claim is the direction of the finding itself, and the report says so
where the result is first presented (Section 7.2): the ordering of the arms, and the size of the paired
advantage over potency alone (+0.243 [+0.152, +0.340] on the primary endpoint), are a confirmation of
Geci et al.'s observation rather than a discovery.

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
two feature sets, and the rule that classifier probabilities never leave the validation figures.

## 6. Experiments *(outline)*

The pre-registered plan verbatim from `validation/preregistration.json`: question, inclusion rule, arms,
features, primary and secondary comparisons, the three outcome cases fixed in advance, the statistic, the
grouping and the split hash. Then what the freeze does and does not cover, from `README.md`.

## 7. Results *(outline, except 7.2)*

- **7.1** Liver-Chip: 27 drugs, the pair views, the hero pair as a daily dose (`results/dose_view.png`).
- **7.2** The benchmark: arms table, ROC (`results/roc.png`), paired differences
  (`results/paired_difference.png`), and the in-sample/out-of-sample comparison from Section 2.1.
- **7.3** The exploratory exposure-alone analysis, labelled as added after the first run.
- **7.4** The second organ, and the measurement of the field's bottleneck: 136 chemicals tested, 82
  active, 21 with any public human exposure comparator (13 predicted by the EPA, 9 measured clinical
  Cmax, simvastatin in both — 22 route rows for 21 chemicals). Figure: `results/neural_coverage.png`,
  table: `results/neural_coverage.csv`, both generated. The point for the reader: for roughly three
  quarters of the chemicals this chip has already measured, nobody can take the step from
  concentration to dose, and the missing half is the published exposure value, not the chip.

## 8. Failures, limitations and bias *(outline)*

Leads with the findings that work against us, from `README.md` "Findings we report against ourselves":
mifepristone and the other cases in `results/failure_cases.csv`; trovafloxacin/levofloxacin as inconclusive;
pioglitazone's BSEP-driven literature POD ranking it against the clinic; the assay-count confound (AUC 0.66);
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
