"""The two documents a reviewer reads: README.md and writeup/report_outline.md.

Until this module existed, no test in the suite opened a single prose file, while
`test_data_and_labels.py`'s own docstring claimed "the README, Beat 4 and the report outline each state
its composition". The data half of that claim was airtight and the prose half had nothing, so four
corruptions shipped green: a decider count replaced by that assay's presence count, "14 further labels"
moved to 15 with "5 of the 23" moved to 4, deletion of every concentration-basis clause, and 0.845
rounded to 0.85 in both copies of the increment paragraph. The 15-20 page report is written from these
two files, so a number that drifts here reaches the jury with the evidence underneath it still correct.

Every literal below is built from something else that is already checked - the counts the data side pins
in `test_data_and_labels.py`, or `results/validation.json` exactly as the run generated it - rather than
restated here. A data-vintage change therefore turns this red and the prose gets revisited, instead of
the prose silently contradicting the repo it ships in.
"""

import json
import re
import subprocess
import sys

from src import config

from tests.test_data_and_labels import (
    BENCHMARK_ASSAY_DECIDES,
    BENCHMARK_ASSAY_LABELS,
    BENCHMARK_ASSAY_PRESENCE,
    BENCHMARK_N,
    LABELS_THAT_DECIDE_AT_LEAST_ONE,
)

README = "README.md"
OUTLINE = "writeup/report_outline.md"

# The three figures the prose derives from the pinned decider counts rather than stating independently.
# Written as expressions, not numbers, so the arithmetic is visible and moves with the data.
NAMED_DECIDERS = sum(BENCHMARK_ASSAY_DECIDES.values())
DRUGS_ON_FURTHER_LABELS = BENCHMARK_N - NAMED_DECIDERS
FURTHER_LABELS = LABELS_THAT_DECIDE_AT_LEAST_ONE - len(BENCHMARK_ASSAY_DECIDES)
LABELS_THAT_DECIDE_NOTHING = len(BENCHMARK_ASSAY_LABELS) - LABELS_THAT_DECIDE_AT_LEAST_ONE

# Every number a claim-bearing clause states, keyed to the clause, plus the assay each count belongs to.
# Two assertions per clause, because they catch different mutations: the phrases pin which assay each
# count is attributed to (swapping Cell Painting's 27 for its presence count 79 is the corruption that
# shipped), and the number list pins that the clause states nothing else and nothing more (moving "14
# further labels" to 15 is not visible to the phrases).
CLAIMS = (
    {
        "file": README,
        "opens": "Produced the lowest POD:",
        "closes": "Merely present among",
        "phrases": (
            "a BSEP inhibition IC50 for {medianBSEPIC50} of the {N} drugs",
            "Cell Painting cytotoxicity for {CellPaintCytoToxPOD}",
            "HepG2 cytotoxicity for {Aleo2019CytoToxHPG2}",
            "THLE cytotoxicity for {Aleo2019CytoToxTHLE}",
            "the remaining {remaining} drugs are spread over {further} further labels",
            "{decide_nothing} of the {labels} labels never produce any drug's lowest POD",
        ),
        "numbers": ("{medianBSEPIC50}", "{N}", "{CellPaintCytoToxPOD}", "{Aleo2019CytoToxHPG2}",
                    "{Aleo2019CytoToxTHLE}", "{remaining}", "{further}", "{decide_nothing}", "{labels}"),
    },
    {
        "file": README,
        "opens": "Merely present among a drug's points of departure:",
        "closes": "None of the {labels} labels",
        "phrases": (
            "BSEP for {present_medianBSEPIC50} drugs",
            "THLE for {present_Aleo2019CytoToxTHLE}",
            "HepG2 for {present_Aleo2019CytoToxHPG2}",
            "Cell Painting for {present_CellPaintCytoToxPOD}",
        ),
        "numbers": ("{present_medianBSEPIC50}", "{present_Aleo2019CytoToxTHLE}",
                    "{present_Aleo2019CytoToxHPG2}", "{present_CellPaintCytoToxPOD}"),
    },
    {
        # The outline states the same composition in a shorter form, and mixes one decider count with
        # three presence counts in a single sentence - which is exactly why the phrases are keyed to the
        # assay: the two questions are one comma apart here.
        "file": OUTLINE,
        "opens": "Counted from the shipped",
        "closes": "None of the {labels} labels",
        "phrases": (
            "a BSEP inhibition IC50 is the lowest point of departure for {medianBSEPIC50} of the {N} drugs",
            "appear among the points of departure of {present_Aleo2019CytoToxTHLE} and "
            "{present_Aleo2019CytoToxHPG2} drugs",
            "Cell Painting cytotoxicity of {present_CellPaintCytoToxPOD}",
        ),
        "numbers": ("{medianBSEPIC50}", "{N}", "{present_Aleo2019CytoToxTHLE}",
                    "{present_Aleo2019CytoToxHPG2}", "{present_CellPaintCytoToxPOD}"),
    },
)

# The concentration-basis clauses the README carries, one per surface it describes. Deleting any one of
# them un-labels a converted number on the page a screening reviewer reads first, and the fourth is the
# method step that labels every dose printed below it, so the deletion costs two doses rather than one.
README_BASIS_CLAUSES = (
    "the output in real units for troglitazone / pioglitazone, **on total concentration**",
    "the hero pair's safety margins **on free concentration**",
    "equivalent daily dose (on total concentration) where a dose-matched Cmax exists",
    "The conversion is **on total concentration** by construction",
)

# A number shaped like a result: an AUC stated after "AUC"/"reaches", or a difference with its interval.
# Anchored on those words rather than on the digits alone so a Tanimoto cutoff or a percentage is not
# mistaken for a model score.
PROSE_AUC = re.compile(r"(?:AUC|reaches|at) (\d\.\d+)")
PROSE_INCREMENT = re.compile(r"([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]")
# AUC-shaped numbers the prose states that `results/validation.json` does not contain, and where each one
# comes from. Declared rather than skipped silently, so the law below is a contract in both directions: a
# new score in the prose is either an arm the run generated or is named here, and an entry that leaves the
# prose fails too. Empty on purpose: writing this dict is what surfaced the one entry it used to hold - a
# stated `AUC 0.66` for the assay-count confound that no script in the repo computed, and that passed a
# naive version of the law only because the narrow endpoint's potency arm is 0.663662. The arm now exists
# (`validate.EXPLORATORY_ARMS`), so the number is generated and the exemption is gone rather than kept.
UNGENERATED_PROSE_SCORES: dict[str, str] = {}
# The confound arm is the one the two documents state against themselves, so it is pinned to its own arm
# by name rather than left to the "some generated value matches" law - which a collision can satisfy.
ASSAY_COUNT_ARM = "exploratory, post-hoc: number of assays alone"
POTENCY_ARM = "potency alone"


def _flat(name: str) -> str:
    """The file as one line. Both documents are hard-wrapped, so a sentence that states three numbers is
    split across lines on disk and has to be rejoined before any clause can be found in it."""
    return " ".join((config.ROOT / name).read_text().split())


def _counts() -> dict:
    """Every count the prose may state, under the name the clause templates use."""
    values = {"N": BENCHMARK_N, "labels": len(BENCHMARK_ASSAY_LABELS),
              "remaining": DRUGS_ON_FURTHER_LABELS, "further": FURTHER_LABELS,
              "decide_nothing": LABELS_THAT_DECIDE_NOTHING}
    # Decider counts under the assay's own name, presence counts under `present_<assay>` - the prose
    # states both about the same four assays, and the whole point is that they cannot be interchanged.
    values |= BENCHMARK_ASSAY_DECIDES
    values |= {f"present_{assay}": n for assay, n in BENCHMARK_ASSAY_PRESENCE.items()}
    return values


def _clause(text: str, opens: str, closes: str, where: str) -> str:
    """The span between two anchors, each of which must occur exactly once before the clause is read.
    Asserted rather than assumed: `text.index` would silently return the first of several matches, and a
    clause found in the wrong place is a guard pointing at prose nobody is checking."""
    for anchor in (opens, closes):
        assert text.count(anchor) == 1, f"{where}: {text.count(anchor)} occurrences of {anchor!r}"
    start = text.index(opens)
    return text[start:text.index(closes, start)]


def _numbers(clause: str) -> list[str]:
    """Every standalone number in a clause. Digits glued to letters are not numbers here: 'HepG2' and
    'MRP2' are assay names, and reading a 2 out of either would make the list unassertable."""
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?!\d)", clause)


def test_the_composition_counts_the_prose_states_are_the_ones_the_data_side_pins():
    """The assay composition of the 220-drug benchmark is the claim the A1 disclosure rests on, and both
    judge-facing documents state it in prose. These counts were published wrong once already - presence
    counted where the sentence claimed the minimum - so each is compared against the literal the data
    side asserts against the live frame, by assay and then as a complete list."""
    values = _counts()
    for claim in CLAIMS:
        text, where = _flat(claim["file"]), f"{claim['file']} ({claim['opens']})"
        clause = _clause(text, claim["opens"].format(**values), claim["closes"].format(**values), where)
        for phrase in claim["phrases"]:
            assert phrase.format(**values) in clause, (where, phrase.format(**values), clause)
        expected = [template.format(**values) for template in claim["numbers"]]
        assert _numbers(clause) == expected, (where, _numbers(clause), expected, clause)


def test_the_increment_paragraph_states_the_values_the_run_generated_in_every_copy():
    """The paragraph that concedes the exploratory finding - exposure alone already reaches AUC 0.845,
    and the margin's increment over it is small - is the most load-bearing honesty claim in the writeup,
    and the README states it twice. Built from `validation.json` rather than from literals so a changed
    arm turns this red, and counted rather than merely found, because rounding one copy of a duplicated
    paragraph and not the other is the drift a containment check cannot see."""
    wide = json.loads((config.RESULTS / "validation.json").read_text())["wide"]
    arms = {a["arm"]: a for block in wide.values() if isinstance(block, dict)
            for a in block.get("arms", [])}
    comparisons = {c["comparison"]: c for block in wide.values() if isinstance(block, dict)
                   for c in block.get("comparisons", [])}

    def increment(name):
        c = comparisons[name]
        return f"{c['delta_auc']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"

    sentence = (
        f"{arms['exploratory: total Cmax alone']['auc']:.3f} and the margin on total concentrations "
        f"reaches {arms['margin alone (total)']['auc']:.3f}, an increment of "
        f"{increment('margin alone (total) minus exploratory: total Cmax alone')}; free Cmax alone "
        f"reaches {arms['exploratory: free Cmax alone']['auc']:.3f} and the margin on free "
        f"concentrations reaches {arms['margin alone (free)']['auc']:.3f}, an increment of "
        f"{increment('margin alone (free) minus exploratory: free Cmax alone')}.")
    assert _flat(README).count(sentence) == 2, sentence
    assert _flat(OUTLINE).count(sentence) == 1, sentence


def _arm(endpoint: str, name: str) -> dict:
    """One arm of one endpoint, out of `validation.json` as the run wrote it."""
    blocks = json.loads((config.RESULTS / "validation.json").read_text())[endpoint]
    arms = {a["arm"]: a for block in blocks.values() if isinstance(block, dict)
            for a in block.get("arms", [])}
    return arms[name]


def test_the_assay_count_confound_is_stated_from_its_own_arm_on_both_endpoints():
    """The confound the two documents report against themselves: the lowest POD is a minimum over however
    many assays a drug was run in, so a drug tested more often has more chances at a low one. Both files
    stated a value for it while nothing in the repo computed it - it passed the general law below only
    because the narrow endpoint's potency arm rounds to the same two decimals, which is a collision and
    not a source. Pinned here to the arm by name, with its interval and against the potency arm it is
    compared with, on both endpoints, because the comparison is the claim."""
    for endpoint in ("wide", "narrow"):
        arm, potency = _arm(endpoint, ASSAY_COUNT_ARM), _arm(endpoint, POTENCY_ARM)
        stated = (f"{arm['auc']:.3f} [{arm['ci_low']:.3f}, {arm['ci_high']:.3f}]", f"{potency['auc']:.3f}")
        for document in (README, OUTLINE):
            for value in stated:
                assert value in _flat(document), (endpoint, document, value)
    # And the label travels with the number, in both documents: the arm names itself post-hoc wherever it
    # is printed, and a reader must not meet the figure without that word.
    assert "post-hoc" in ASSAY_COUNT_ARM
    for document in (README, OUTLINE):
        assert "post-hoc" in _flat(document), document


def _generated_scores() -> tuple[set[float], set[tuple[float, float, float]]]:
    """Every arm AUC and every difference-with-interval the run wrote, across both endpoints and all
    three blocks (pre-registered, exploratory, and the post-hoc range-matched run)."""
    validation = json.loads((config.RESULTS / "validation.json").read_text())
    aucs, deltas = set(), set()
    for endpoint in validation.values():
        for block in endpoint.values():
            for arm in block.get("arms", []):
                aucs.add(arm["auc"])
            for comparison in block.get("comparisons", []):
                deltas.add((comparison["delta_auc"], comparison["ci_low"], comparison["ci_high"]))
    return aucs, deltas


def _prints_as(value: float, printed: str) -> bool:
    """Whether a generated value rounds to what the prose printed, at the prose's own precision, so a
    paragraph is free to quote +0.24 where the table says +0.243 without that counting as a drift."""
    decimals = len(printed.partition(".")[2])
    return f"{value:.{decimals}f}" == printed.lstrip("+")


def test_every_score_the_prose_states_is_one_the_run_generated():
    """The general law behind the two tests above: a reviewer reading either document must not find a
    model score the repo cannot produce. Pinning the sentences that matter catches a changed value in
    those sentences; this catches a value invented in a sentence nobody pinned, which is how a number
    enters prose in the first place. Exceptions are declared, not skipped."""
    aucs, deltas = _generated_scores()
    assert aucs and deltas, "run `make` first"
    for name in (README, OUTLINE):
        text = _flat(name)
        printed = set(PROSE_AUC.findall(text))
        assert set(UNGENERATED_PROSE_SCORES) <= printed, (name, sorted(printed))
        for score in sorted(printed - set(UNGENERATED_PROSE_SCORES)):
            assert any(_prints_as(v, score) for v in aucs), (name, score, sorted(aucs))
        for triple in sorted(set(PROSE_INCREMENT.findall(text))):
            assert any(all(_prints_as(v, p) for v, p in zip(generated, triple)) for generated in deltas), \
                (name, triple, sorted(deltas))


def test_the_readme_states_the_number_of_tests_the_suite_actually_collects():
    """`make test  # N tests` sits on the first page, inside the command block a reviewer copies, and it
    went stale in five consecutive rounds (75 → 82 → 84 → 86 → 87 → 92) because every round that adds a
    test has no reason to look at it. Collected in a subprocess rather than from this run's session, so
    the number is the whole suite's and not whatever `-k` selected."""
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                               cwd=config.ROOT, capture_output=True, text=True)
    found = re.search(r"(\d+) tests? collected", collected.stdout)
    assert found, collected.stdout[-400:]
    # Both sides as short strings, so a failure prints the two numbers rather than the whole README.
    stated = re.search(r"make test\s+# (\d+) tests", _flat(README))
    assert stated is not None, "README.md no longer states a test count next to `make test`"
    assert stated.group(1) == found.group(1), (stated.group(1), found.group(1))


def test_the_readme_names_the_concentration_basis_on_every_surface_it_describes():
    """The basis clauses added in round 6 are the labels that keep four converted numbers from being read
    on the wrong basis, and the last of them labels every dose the method section prints below it.
    Deleting all four left the suite green, so they are asserted where they were written."""
    text = _flat(README)
    for clause in README_BASIS_CLAUSES:
        assert clause in text, clause


# The two claims the confound paragraph turns on. Neither is a value out of `validation.json`; each is a
# statement ABOUT it, so the number law above cannot see either one - inverting "contains" to "neither
# contains", or "was not computed" to "was computed", left the whole suite green in both documents. They
# are pinned the way the increment paragraph is, in both directions at once: the relation is recomputed
# from the run first, so the sentence cannot quietly become false when the data move, and then the
# sentence is required verbatim, so it cannot quietly stop saying what was computed. Identical wording in
# both files on purpose - one literal, two documents, no room for them to drift apart.
CONFOUND_CLAIM_NO_COMPARISON = ("no paired difference between these two arms was pre-registered or computed")
CONFOUND_CLAIM_CONTAINMENT = ("on both endpoints each arm's interval contains the other arm's point estimate")
CONFOUND_CLAIM_CONCLUSION = ("nothing here shows either arm separating the outcome better than the other")


def _contains_each_others_point_estimate(endpoint: str) -> tuple[bool, bool]:
    """Whether each of the two arms' intervals covers the other's point estimate, both directions."""
    arm, potency = _arm(endpoint, ASSAY_COUNT_ARM), _arm(endpoint, POTENCY_ARM)
    return (arm["ci_low"] <= potency["auc"] <= arm["ci_high"],
            potency["ci_low"] <= arm["auc"] <= potency["ci_high"])


def _comparisons() -> list[str]:
    """Every paired comparison the run computed, across both endpoints and all of their blocks."""
    validation = json.loads((config.RESULTS / "validation.json").read_text())
    return [comparison["comparison"]
            for endpoint in validation.values()
            for block in endpoint.values() if isinstance(block, dict)
            for comparison in block.get("comparisons", [])]


def test_the_confound_paragraph_claims_only_what_the_run_supports():
    """The paragraph says the assay count and potency alone are two levels and not a comparison. That
    rests on two facts about the run, and a reader has no way to check either, so they are checked here.

    Both halves matter and they fail differently. If the data move so that an interval no longer covers
    the other arm's point estimate, the first assertion fires and the prose is wrong and must be
    rewritten. If someone edits the prose to assert the opposite of what the run supports, the second
    fires and the prose is wrong and the run is right. Guarding only one of the two leaves the other
    free, which is how this paragraph came to say the assay count "beats" potency alone with no paired
    difference anywhere in the repository."""
    for endpoint in ("wide", "narrow"):
        covers_potency, covered_by_potency = _contains_each_others_point_estimate(endpoint)
        assert covers_potency and covered_by_potency, (endpoint, covers_potency, covered_by_potency)

    both = [c for c in _comparisons() if ASSAY_COUNT_ARM in c and POTENCY_ARM in c]
    assert both == [], both
    # Cardinality on the side that could pass vacuously: an empty comparison list would satisfy the line
    # above while meaning the run computed nothing at all.
    assert len(_comparisons()) >= 15, len(_comparisons())

    for document in (README, OUTLINE):
        text = _flat(document)
        for claim in (CONFOUND_CLAIM_NO_COMPARISON, CONFOUND_CLAIM_CONTAINMENT):
            assert text.count(claim) == 1, (document, claim, text.count(claim))
    assert _flat(README).count(CONFOUND_CLAIM_CONCLUSION) == 1, CONFOUND_CLAIM_CONCLUSION
