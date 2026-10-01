"""Pair view, single-compound path and the no-score rule."""

import pathlib
import re
import time

import pandas as pd

from src import compound, config, figures, liver, load, margin, neural, pod

PROBABILITY_WORDS = ("probability", "proba", "score")


def test_troglitazone_is_the_riskier_partner_of_a_published_pair():
    pairs = load.liverchip_pairs().set_index("key")
    assert pairs.at["troglitazone", "partner_key"] == "pioglitazone"  # from Table 1, not our choice
    assert pairs.at["troglitazone", "garside_rank"] < pairs.at["pioglitazone", "garside_rank"]
    margins = liver.margin_table().set_index("key")
    assert margins.at["troglitazone", "margin_free"] < margins.at["pioglitazone", "margin_free"]


def test_pair_view_renders_every_matched_pair_under_two_seconds(tmp_path):
    margins = liver.margin_table()
    matched = {tuple(sorted(p)) for p in zip(margins["key"], margins["partner_key"]) if p[1]}
    assert len(matched) == 7
    for a, b in matched:
        started = time.time()
        path = figures.pair_view(margins, a, b, tmp_path / f"{a}_{b}.png")
        assert time.time() - started < 2.0
        assert path.stat().st_size > 10_000


def test_single_compound_run_is_fast_and_in_real_units():
    started = time.time()
    lines = compound.describe("troglitazone")
    assert time.time() - started < 60
    text = "\n".join(lines)
    assert "uM" in text and "mg" in text and "Band" in text
    assert "DILIrank 2.0" in text


def test_rule_of_thumb_flag_shown_next_to_the_margin():
    text = "\n".join(compound.describe("troglitazone"))
    assert "Dose rule of thumb" in text and "FLAGGED" in text
    assert "not flagged" in "\n".join(compound.describe("pioglitazone"))


def test_no_per_compound_score_in_any_results_table():
    tables = list(config.RESULTS.glob("*.csv"))
    assert tables, "run `make` first"
    for path in tables:
        columns = " ".join(pd.read_csv(path, nrows=1).columns).lower()
        assert not any(word in columns for word in PROBABILITY_WORDS), path.name


def test_censored_comparator_never_counts_as_a_wrong_order():
    pairs = liver.pair_table().set_index("pair")
    row = pairs.loc["Trovafloxacin / Levofloxacin"]
    assert row["margin_fold_is_lower_bound"]
    assert row["margin_orders_like_clinic"] == "inconclusive (comparator censored)"


def test_pair_cli_rejects_unknown_and_labels_unpublished_pairs(capsys):
    import run_demo
    assert run_demo.main(["--pair", "aspirin", "troglitazone"]) == 1
    assert "Not in the Liver-Chip set" in capsys.readouterr().out
    assert run_demo.main(["--pair", "troglitazone", "troglitazone"]) == 1


def test_readout_slot_gives_bands_and_units(capsys):
    import run_demo
    assert run_demo.main(["--readout", "12", "--cmax", "0.8", "--fu-plasma", "0.05", "--fu-medium", "0.7"]) == 0
    out = capsys.readouterr().out
    assert "margin (free)" in out and "Band" in out


# The arguments typed on camera in the walkthrough, and every line the audience reads back.
#
# Pinned by EQUALITY over the whole output, not by containment on a phrase, for three reasons.
# (1) These values are spoken aloud, and a recording cannot be re-run the way code can, so a
# silent move here ships a video asserting a number the repo no longer produces. (2) Containment
# cannot see a new sibling line appearing in the middle of the frame, which has bitten us twice
# before in this same output. (3) The arguments only this entry point supplies
# -- --fu-plasma and --fu-medium, which every other path reads from the published study instead
# of from the user -- are covered by nothing else: the hero pair pins the shared margin engine
# through results/summary.md, so a change to the engine goes red there, but dropping either of
# these two on the way in moves the first line from 100x to 2x with the rest of the suite green.
#
# The spoken figures live on line 3: 500 mg, band 186 to 1,340, against 200 prescribed. Lines 1
# and 2 are on screen beside them and each names its own basis, which is the constraint the beat
# works under. Line 6 is the organ caveat that bounds what the walkthrough may claim about other
# tissues, so it is pinned here rather than assumed. If a deliberate change makes this test red,
# the walkthrough script has to change with it -- that is the point, not a nuisance.
WALKTHROUGH_ARGV = ["--readout", "5", "--cmax", "2", "--fu-plasma", "0.02",
                    "--fu-medium", "0.8", "--dose-mg", "200", "--name", "Your compound"]
WALKTHROUGH_OUTPUT = (
    "Your compound: margin (free) 100x - below the convention threshold 375 (Liver-Chip, two donors: "
    "sensitivity 87% [62-96%], specificity 100% (27 drugs)). Band: 37.2x to 268x.",
    "Your compound: margin (total) 2.5x - below the convention threshold 50 (Liver-Chip, two donors: "
    "sensitivity 80% [54-93%], specificity 100% (27 drugs)). Band: 0.93x to 6.7x.",
    "Chip-derived daily dose (on total concentration): 500 mg (band 186 to 1,340 mg) against 200 mg, "
    "linear PK assumed.",
    "  assumption: chip POD: single value, 3-fold uncertainty assumed",
    "  assumption: equivalent daily dose assumes linear pharmacokinetics (Cmax proportional to dose)",
    "Thresholds are Liver-Chip conventions (Ewart et al. 2022); for another organ, read the margin, "
    "not the verdict.",
)


def test_the_walkthrough_command_returns_exactly_the_lines_it_is_shown_returning(capsys):
    """A reader's own chip value in, a dose out, pinned line for line.

    Checked in two steps so a failure says which kind it is: the three figures that are read
    aloud first, then the whole output by equality. Without the first step a reformatted line
    and a moved number fail identically, and the one that matters is the number."""
    import run_demo
    assert run_demo.main(WALKTHROUGH_ARGV) == 0
    lines = capsys.readouterr().out.splitlines()

    # Anchored on the line's own prefix, not on "daily dose": that phrase also occurs in the
    # linear-PK assumption line below it, so the loose selector picks two lines and would have
    # gone on to check the figures against whichever came first. Measured on this output:
    # "daily dose" matches 2, "Chip-derived daily dose" matches 1.
    dose_line = [line for line in lines if line.startswith("Chip-derived daily dose")]
    assert len(dose_line) == 1, lines          # cardinality first: an empty selection proves nothing
    for figure in ("500 mg", "band 186 to 1,340 mg", "against 200 mg"):
        assert figure in dose_line[0], (figure, dose_line[0])

    assert tuple(lines) == WALKTHROUGH_OUTPUT


def test_neural_chemicals_are_found_by_name():
    text = "\n".join(compound.describe("rotenone"))
    assert "Neural MEA" in text and "AED vs predicted exposure" in text


def test_dose_relation_respects_the_band():
    assert "above the whole band" in margin.dose_relation(16.1, 16.1, 61.7, 600)
    # pioglitazone: 45 mg lies inside 39.6-291, so no "below" claim without the qualifier
    text = margin.dose_relation(90.4, 39.6, 291, 45)
    assert "inside the band" in text and "whole band" not in text
    assert "below the whole band" in margin.dose_relation(90, 60, 200, 10)


def test_dose_view_renders_and_refuses_pairs_without_a_dose(tmp_path):
    margins = liver.margin_table()
    pairs = liver.pair_table(margins)
    path = figures.dose_view(margins, pairs, "troglitazone", "pioglitazone", tmp_path / "d.png")
    assert path.stat().st_size > 10_000
    import pytest
    with pytest.raises(ValueError):
        figures.dose_view(margins, pairs, "ambrisentan", "sitaxsentan", tmp_path / "x.png")


def test_compound_run_shows_the_same_chip_dose_as_the_dose_view():
    table = liver.margin_table().set_index("key")
    for key in ("troglitazone", "pioglitazone"):
        text = "\n".join(compound.describe(key))
        assert f"reached at {table.at[key, 'equivalent_dose_mg']:,.3g} mg/day" in text
        assert "different data, not the chip" in text


def test_empty_or_combined_cli_modes_are_refused(capsys):
    import run_demo
    import pytest
    assert run_demo.main(["--compound", ""]) == 1
    with pytest.raises(SystemExit):
        run_demo.main(["--compound", "x", "--pair", "a", "b"])


def test_censored_readout_dose_is_a_lower_bound(capsys):
    import run_demo
    run_demo.main(["--readout", "12", "--cmax", "0.8", "--dose-mg", "100", "--censored"])
    assert "Chip-derived daily dose (on total concentration): >" in capsys.readouterr().out


def test_two_cmax_values_for_one_drug_are_named_with_their_sources():
    text = "\n".join(compound.describe("troglitazone"))
    assert "Two published measurements of one quantity, total plasma Cmax: 6.08 uM" in text
    assert "6.79 uM at 600 mg" in text
    assert "Ewart et al. 2022" in text and "Geci et al. 2026" in text


def test_no_cmax_note_when_the_two_sources_print_the_same_value():
    """No drug in the current inputs agrees to three significant figures (closest: clozapine, 1.06-fold),
    so the agreeing branch is exercised directly."""
    chip = {"cmax_total_uM": 6.08, "pod_uM": 0.182, "dose_position": "above the whole band"}
    assert compound._cmax_note(chip, 6.0799, 600) == []
    assert compound._cmax_note(chip, 6.79, 600) != []


def test_contradicting_dose_sentences_carry_the_reason_between_them():
    for drug in ("troglitazone", "pioglitazone"):
        lines = compound.describe(drug)
        assay = next(i for i, l in enumerate(lines) if "Assay-derived daily dose" in l)
        bridge = lines[assay + 1]
        assert "different assay, different point of departure" in bridge
        assert "chip-derived band" in bridge and "assay-derived band" in bridge


def test_no_dose_note_when_both_bands_place_the_prescribed_dose_alike():
    chip = {"cmax_total_uM": 3.0, "pod_uM": 8.4, "dose_position": "above the whole band"}
    assert compound._dose_note(chip, "above the whole band", 0.3, "medianBSEPIC50") == []
    assert compound._dose_note(chip, "inside the band", 0.3, "medianBSEPIC50") != []
    assert compound._dose_note({**chip, "dose_position": None}, "inside the band", 0.3, "x") == []


def test_compound_without_a_literature_block_carries_no_cross_source_notes():
    text = "\n".join(compound.describe("olanzapine"))
    assert "different data, not the chip" not in text
    assert "Two published measurements" not in text and "answer different questions" not in text


def test_assay_route_is_never_called_chip_data():
    for drug in ("pioglitazone", "acetaminophen", "troglitazone"):
        for line in compound.describe(drug):
            if "Assay-derived" in line:
                assert "chip-derived" not in line


def test_doses_print_as_plain_numbers():
    assert margin.fmt_dose(4177.4) == "4,177" and margin.fmt_dose(16.107) == "16.1"
    text = "\n".join(compound.describe("acetaminophen"))
    assert "e+0" not in text


def test_readout_flags_without_readout_are_refused(capsys):
    import run_demo
    assert run_demo.main(["--compound", "troglitazone", "--cmax", "3"]) == 1
    assert "only apply with --readout" in capsys.readouterr().out


def test_numerical_underflow_gives_no_margin():
    q = margin.Quantity.exact
    inp = margin.MarginInput("u", q(1e-320, "p"), q(1e300, "c"), q(0.1, "f"), q(1.0, "m"), dose_mg=5)
    result = margin.compute(inp, config.THRESHOLDS["liver_total_50"])
    assert result.verdict == margin.NO_MARGIN and result.equivalent_dose_mg is None


def test_atomic_write_keeps_the_previous_result_when_a_write_is_interrupted(tmp_path):
    import pytest
    from src import output
    target = tmp_path / "summary.md"
    target.write_text("previous complete result\n")

    def half_then_interrupt(tmp):
        tmp.write_text("half a resu")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        output.atomic_write(target, half_then_interrupt)
    assert target.read_text() == "previous complete result\n"
    assert [p.name for p in tmp_path.iterdir()] == ["summary.md"]  # no .part file left behind


def test_atomic_write_leaves_nothing_when_a_first_write_is_interrupted(tmp_path):
    import pytest
    from src import output
    target = tmp_path / "sub" / "figure.png"

    def die(tmp):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        output.atomic_write(target, die)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_atomic_write_actually_writes_when_it_is_not_interrupted(tmp_path):
    from src import output
    target = tmp_path / "deep" / "table.csv"
    output.atomic_write_text(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"
    assert [p.name for p in target.parent.iterdir()] == ["table.csv"]


def test_every_result_writer_goes_through_the_atomic_helper(tmp_path, monkeypatch):
    """Wiring, not source text: each writer is executed and must pass through the helper."""
    import run_demo
    from src import output
    seen = []
    real = output.atomic_write

    def spy(path, write):
        seen.append(pathlib.Path(path).name)
        return real(path, write)

    monkeypatch.setattr(output, "atomic_write", spy)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    run_demo.write_csv(pd.DataFrame({"a": [1]}), "probe.csv")
    output.atomic_write_text(tmp_path / "probe.md", "text")
    figures.pair_view(liver.margin_table(), "troglitazone", "pioglitazone", tmp_path / "probe.png")
    assert seen == ["probe.csv", "probe.md", "probe.png"]


def _png_size(path):
    """Width and height out of the IHDR chunk, so the check needs no image library."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def test_the_run_writes_the_submission_card_at_the_size_the_editor_requires(tmp_path, monkeypatch):
    """The card blocks Submit, so the run has to produce it, and at 560x280: the Kaggle Writeup editor
    states those dimensions (read off the editor 2026-09-29). The literal is the point of the test -
    a tight bounding box or a changed figsize would still write a valid PNG of the wrong size."""
    import run_demo
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    run_demo.run_liver()
    card = tmp_path / "card.png"
    assert card.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert _png_size(card) == (560, 280)


def test_the_run_also_writes_the_card_at_video_resolution(tmp_path, monkeypatch):
    """The submission video holds this card full-frame for 25 s of its opening demonstration, so it is
    needed at 1080p rather than as a 3.4x upscale of the 560 px card. Pinned the same way the card is,
    and for the same reason: the literal is the test. 1920 x 960 keeps the card's 2:1 aspect, so the
    two renderings are the same composition at two sizes and neither needs a crop."""
    import run_demo
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    run_demo.run_liver()
    card_video = tmp_path / "card_video.png"
    assert card_video.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert _png_size(card_video) == (1920, 960)
    small, large = _png_size(tmp_path / "card.png"), _png_size(card_video)
    assert small[0] / small[1] == large[0] / large[1], "the two cards must share one aspect ratio"


def test_the_card_caption_stays_inside_both_renderings(tmp_path):
    """Glyph advances are hinted to whole pixels, so a caption tuned to the edge of the 560 px card
    runs past it at 1920 px and loses its last word - in the one shot of the video that cannot be cut.
    This pins the fit itself rather than the font size, so a longer caption fails here instead of
    silently clipping in a video frame."""
    import matplotlib.image

    margins = liver.margin_table()
    for name, aspect in (("card.png", figures.CARD), ("card_video.png", figures.CARD_VIDEO)):
        path = figures.card_image(margins, "troglitazone", "pioglitazone", tmp_path / name,
                                  aspect=aspect)
        # Second-to-last column, not the last: the saved canvas can carry a single-pixel edge
        # artefact that says nothing about the text. Anything darker than white here is ink.
        column = matplotlib.image.imread(path)[:, -2, :3]
        assert (column > 0.99).all(), f"{name}: ink in the last pixel column - a text is clipped"


def test_the_card_refuses_a_pair_whose_margin_is_only_a_lower_bound(tmp_path):
    """The card carries the margin and has no room for a footnote, so it must never show a censored
    POD's margin, which is a lower bound and not a measured value. Ambrisentan's POD is censored."""
    import pytest
    with pytest.raises(ValueError):
        figures.card_image(liver.margin_table(), "ambrisentan", "sitaxsentan", tmp_path / "x.png")


# The hero pair's free-basis margins and their fold: spoken in demo Beat 3, printed on card.png and
# on pair_view.png. Literals, because the basis is the whole point of the claim - the same pair on
# total concentration separates 93-fold, and the equivalent-dose comparison changes direction with
# the basis - so a silent switch of basis or data vintage has to turn this red, not move the number.
HERO_MARGINS_FREE = {"troglitazone": 1.42309, "pioglitazone": 94.0772}
HERO_MARGIN_FOLD_FREE = 66
HERO_CONVENTION_THRESHOLD = 375.0
HERO_COMPOUNDS_ON_SCREEN = ("Troglitazone", "Pioglitazone")
# Everything the y-tick labels state after the compound name, per figure, in FULL. Two literals rather
# than one because the two figures legitimately draw different suffixes, and full strings rather than a
# parsed first clause because the parse is where the blindness goes: splitting on ";" to reach
# "withdrawn" also throws away the published severity rank drawn beside it, so a false clause appended
# after the semicolon renders on the figure while an equality check reports exact agreement. The ranks
# are Garside's published ordering, cross-checked against pair_table.csv (garside_worse / comparator).
WITHDRAWAL_ON_SCREEN = {"Troglitazone": "withdrawn", "Pioglitazone": "not withdrawn"}
DOSE_VIEW_STATUS_ON_SCREEN = {"Troglitazone": "withdrawn; Garside rank 1",
                              "Pioglitazone": "not withdrawn; Garside rank 3"}
# The basis clause each dose-bearing surface of dose_view.png must carry, keyed by the surface of the
# axis that draws the doses. Deleting any one of these leaves the others intact, so each needs its own
# assertion - and each is asserted on that axis rather than across `fig.axes`, because a clause on an
# invisible pad axis satisfies a figure-wide check while the data axis's own label stays bare.
DOSE_VIEW_BASIS = {
    "title": "on total concentration",
    "xlabel": "conversion on total plasma concentration",
}
DOSE_VIEW_FOOTNOTE_BASIS = ("TOTAL Cmax", "The basis matters", "basis-dependent")
# results/summary.md is what README.md sends a reviewer to as "every headline number, generated", and it
# composes the total-basis dose as a child bullet of the free-basis margin line, so the only basis word
# within a reader's reach used to be the wrong one. The rule is per line, not per file: a file-wide
# "contains a basis word" check is already satisfied by the validation section's "total Cmax alone" arms.
SUMMARY_DOSE_LINE = re.compile(r"\d\s*mg/day")
SUMMARY_DOSE_BASIS = ("on total concentration", "on free concentration")


def test_the_hero_pairs_free_margins_and_the_fold_the_demo_speaks_are_pinned():
    """Beat 3 says '1.4x', '94x', '66-fold apart' and 'below the convention of 375'. A recorded video
    cannot be re-run, so each of those is compared with a literal here rather than re-derived from the
    table that produced it. The verdicts are pinned too: the claim is that the convention puts BOTH on
    the same side while the distance from it separates them, which is false if either goes ABOVE."""
    import numpy as np

    table = liver.margin_table().set_index("key")
    for key, expected in HERO_MARGINS_FREE.items():
        row = table.loc[key]
        assert np.isclose(row["margin_free"], expected, rtol=1e-5), (key, row["margin_free"])
        assert not bool(row["censored"]), key
        assert row["verdict"] == "BELOW", (key, row["verdict"])
        assert row["threshold"] == HERO_CONVENTION_THRESHOLD, key
    a, b = (table.loc[k]["margin_free"] for k in HERO_MARGINS_FREE)
    assert round(max(a, b) / min(a, b)) == HERO_MARGIN_FOLD_FREE
    # The bands do not overlap, which is what "decisive" means here and what the card shows.
    tro, pio = table.loc["troglitazone"], table.loc["pioglitazone"]
    assert tro["band_high"] < pio["band_low"], (tro["band_high"], pio["band_low"])
    # The ordering survives the other basis; only the dose comparison changes direction with it.
    assert tro["margin_total"] < pio["margin_total"]


def _rendered_axis_parts(ax) -> dict:
    """Every string ONE axis draws. Per axis, not per surface kind across `fig.axes`: a clause collected
    from every axis proves "this figure names its basis somewhere", where the claim is "the axis that
    draws the doses names its basis" - and those differ by an 0.005 x 0.005 `set_axis_off()` pad axis."""
    return {
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "texts": [t.get_text() for t in ax.texts],
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
    }


def _rendered_figure_parts(monkeypatch, render) -> dict:
    """Every string a figure actually draws: figure-level texts, plus one entry per axis. The figure is
    held open (`_save` closes it, so the close is what we intercept) instead of being read back off the
    PNG, so the check is on the rendered artists rather than on the source that composed them.

    Shared rather than card-specific because the edit that named the concentration basis changed two
    figures, and a rendered-text guard pointed at one of them is aimed at the review instead of at the
    code. Keyed by axis rather than by surface kind because an assertion can only be trusted to be about
    a surface if the surface it names is the one a reader looks at."""
    import matplotlib.pyplot as plt
    real_close, held = plt.close, []
    monkeypatch.setattr(plt, "close", held.append)
    render()
    fig = held[-1]
    parts = {
        "figure_texts": [t.get_text() for t in fig.texts],
        "axes": [_rendered_axis_parts(ax) for ax in fig.axes],
    }
    real_close(fig)
    return parts


def _axis_strings(axis: dict) -> list[str]:
    """Every string on one axis, single labels and lists alike, read off the dict rather than from a
    second copy of its keys - so a surface added to `_rendered_axis_parts` is covered without a matching
    edit here, which is how the two would drift apart."""
    flat = []
    for value in axis.values():
        flat.extend([value] if isinstance(value, str) else value)
    return flat


def _flatten(parts: dict) -> str:
    """Every string anywhere on the figure. Only for claims that really are figure-wide - on the card,
    which is one small composition read at 300 px. Anything about a particular axis goes through
    `_claim_axis` instead."""
    strings = list(parts["figure_texts"])
    for axis in parts["axes"]:
        strings += _axis_strings(axis)
    return " | ".join(strings)


def _claim_axis(parts: dict, compounds=HERO_COMPOUNDS_ON_SCREEN) -> dict:
    """The one axis that draws these compounds' rows, found by what it draws rather than by its index,
    so an axis added in front of it does not silently move which surface the assertions are about.
    Exactly one axis must match: two matching axes would make "the axis that draws the doses" ambiguous,
    and zero means the figure stopped drawing the pair."""
    named = [ax for ax in parts["axes"]
             if {label.partition("\n")[0] for label in ax["yticklabels"]} >= set(compounds)]
    assert len(named) == 1, [ax["yticklabels"] for ax in parts["axes"]]
    return named[0]


def _drawn_status_labels(axis: dict) -> dict[str, str]:
    """Everything each y-tick label states after the compound name, in full and per compound, so a
    caller can compare by equality. In full because every transform in front of an equality check is a
    hole behind it: `"withdrawn" in label` is satisfied by `"not withdrawn"`, and splitting on ";" to
    reach the first clause hides both a false clause appended after it and the severity rank the figure
    draws there. If the parse exists because the real label carries extra content, that content is a
    claim too."""
    return {label.partition("\n")[0]: label.partition("\n")[2].strip()
            for label in axis["yticklabels"]}


def _rendered_card_parts(tmp_path, monkeypatch):
    return _rendered_figure_parts(
        monkeypatch,
        lambda: figures.card_image(liver.margin_table(), "troglitazone", "pioglitazone", tmp_path / "card.png"))


def _rendered_dose_view_parts(tmp_path, monkeypatch):
    margins = liver.margin_table()
    return _rendered_figure_parts(
        monkeypatch,
        lambda: figures.dose_view(margins, liver.pair_table(margins),
                                  "troglitazone", "pioglitazone", tmp_path / "dose_view.png"))


def test_the_card_states_the_basis_and_shows_the_free_margins(tmp_path, monkeypatch):
    """P2-1: the card is the only surface with no footnote, and the quantity that used to be on it -
    the equivalent daily dose - changes direction on the other basis. So the card carries the margin,
    and says which basis, on its face. Rendered text, because that is what a reviewer reads."""
    parts = _rendered_card_parts(tmp_path, monkeypatch)
    drawn = _flatten(parts)
    assert "free concentration" in drawn, drawn
    assert f"{HERO_MARGIN_FOLD_FREE}-fold" in drawn, drawn
    assert f"convention {HERO_CONVENTION_THRESHOLD:g}" in drawn, drawn
    # The plotted values are the free margins (1.42x / 94.1x), not the total ones (0.03x / 2.8x).
    assert "margin 1.42x" in drawn and "margin 94.1x" in drawn, drawn
    # By equality on the FULL label, not containment: "withdrawn" in label is also satisfied by
    # "not withdrawn", so the containment form stayed green when the producer was inverted - and this is
    # the factual claim the Oh! moment turns on, spoken aloud in the video and drawn on the first
    # artifact a reviewer sees. The card draws the status alone, with nothing appended.
    assert _drawn_status_labels(_claim_axis(parts)) == WITHDRAWAL_ON_SCREEN, parts["axes"]
    # And the basis-dependent quantity is gone from this surface rather than merely relabelled.
    assert "mg/day" not in drawn, drawn


def test_dose_view_names_its_concentration_basis_on_every_surface(tmp_path, monkeypatch):
    """P3-7: `dose_view.png` shows a dose, and the dose conversion is on TOTAL concentration by
    construction (`margin.compute` multiplies the clinical dose by `margin_total` whatever the threshold's
    basis is). On the free basis each prescribed dose changes which side of its band it falls on, so an
    unlabelled dose here is a different claim from the one the figure makes.

    The sibling card got this guard in the round that added the labels; this figure got the labels and no
    guard, and four separate deletions - title, axis, footnote formula, footnote passage - each left the
    whole suite green. One assertion per surface, because a single flattened check would stay green while
    three of the four are missing - and each surface is taken off the axis that actually draws the doses,
    because keyed by surface KIND is not keyed by surface INSTANCE: collecting titles and axis labels
    across `fig.axes` lets the clause live on an invisible pad axis while the data axis stays bare."""
    parts = _rendered_dose_view_parts(tmp_path, monkeypatch)
    axis = _claim_axis(parts)
    for surface, clause in DOSE_VIEW_BASIS.items():
        assert clause in axis[surface], (surface, axis[surface])
    footnote = " | ".join(parts["figure_texts"])
    for clause in DOSE_VIEW_FOOTNOTE_BASIS:
        assert clause in footnote, (clause, footnote)
    # The same regulatory claim as the card, drawn from the same shared helper, pinned on both surfaces -
    # and here the full label, so the Garside rank this figure draws beside the status is pinned too.
    assert _drawn_status_labels(axis) == DOSE_VIEW_STATUS_ON_SCREEN, axis["yticklabels"]


def _composed_summary() -> str:
    """`results/summary.md` as `run_demo.summary_text` composes it right now, from the live tables plus
    the already-generated validation block. Read from the producer and not only off disk, because a rule
    checked against a committed artifact cannot see a change to the code that writes it - dropping the
    basis in the composer leaves the file on disk untouched and the check green.

    `validation.json` supplies the validation argument, which is the same dict the run serialised, so the
    slow step (grouped cross-validation and 2,000 bootstrap draws) is skipped and this costs ~2 s."""
    import json

    import run_demo
    for required in ("summary.md", "validation.json"):
        assert (config.RESULTS / required).exists(), f"{required} is missing; run `make` first"
    validation = json.loads((config.RESULTS / "validation.json").read_text())
    margins = liver.margin_table()
    potency, neural_margins = neural.margin_table()
    return run_demo.summary_text(pod.all_checks(), margins, liver.pair_table(margins), potency,
                                 neural_margins, neural.coverage(potency, neural_margins), validation)


def test_summary_md_states_no_dose_without_naming_its_basis():
    """P2-1: the file README.md calls "every headline number, generated" composed the total-basis dose as
    a CHILD BULLET of the free-basis margin line and contained no basis word anywhere, so the only basis
    a reader could reach for that number was the wrong one - on both hero compounds, on the quantity
    whose direction changes with the basis, in a file that ships in the public repo.

    A file-level rule rather than the two literals, because the previous round fixed five surfaces that
    PRINT a dose and missed the one that COMPOSES one: any future line here that states a dose has to
    carry its basis on that same line. Per line and not per file for the same reason - a file-wide
    "contains a basis word" check is already green from the validation section's "total Cmax alone" arm
    names, forty lines away from any dose. The compounds are pinned too, so a dropped line fails rather
    than vacuously passing an all-lines-satisfy check over an empty list.

    Checked against the composer, with the committed file asserted to equal it. The file header says "do
    not edit", and this is what makes that a checked claim rather than a request."""
    composed = _composed_summary()
    # The rule first, so a composer that drops the basis fails on the basis and not on the file compare.
    dose_lines = [line for line in composed.splitlines() if SUMMARY_DOSE_LINE.search(line)]
    named = {c for c in HERO_COMPOUNDS_ON_SCREEN for line in dose_lines if line.lstrip("- ").startswith(c)}
    assert named == set(HERO_COMPOUNDS_ON_SCREEN), (named, dose_lines)
    for line in dose_lines:
        assert any(clause in line for clause in SUMMARY_DOSE_BASIS), line
    # Then the artifact, which is what carries the rule into the public repo.
    assert composed == (config.RESULTS / "summary.md").read_text(), \
        "results/summary.md is not what run_demo composes from these inputs; re-run `make`"


def test_roc_legend_labels_stay_short_enough_to_sit_inside_the_axes():
    """The ROC legend sits in the empty lower-right corner so the curve keeps the frame. A longer
    label grows that box into the curves. 28 characters is the longest label measured clear of every
    curve at 1920x1080; the check is the rendered figure, this is the guard against drifting past it."""
    labels = [short for _, short, _, _ in figures.ROC_ARMS + [figures.EXPLORATORY_ROC_ARM]]
    assert max(len(s) for s in labels) <= 28, labels
