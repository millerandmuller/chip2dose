"""Pair view, single-compound path and the no-score rule."""

import pathlib
import time

import pandas as pd

from src import compound, config, figures, liver, load, margin

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
# The regulatory fate every figure must state for this pair, as a literal. The two surfaces that draw it
# share one helper, so they can no longer disagree with each other - which is exactly why the helper's
# output needs pinning here: they can now be wrong together.
WITHDRAWAL_ON_SCREEN = {"Troglitazone": "withdrawn", "Pioglitazone": "not withdrawn"}
# The basis clause each dose-bearing surface of dose_view.png must carry, surface by surface. Deleting any
# one of these from the figure leaves the other three intact, so each needs its own assertion.
DOSE_VIEW_BASIS = {
    "titles": "on total concentration",
    "xlabels": "conversion on total plasma concentration",
}
DOSE_VIEW_FOOTNOTE_BASIS = ("TOTAL Cmax", "The basis matters", "basis-dependent")


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


def _rendered_figure_parts(monkeypatch, render) -> dict[str, list[str]]:
    """Every string a figure actually draws, keyed by the surface that draws it. The figure is held open
    (`_save` closes it, so the close is what we intercept) instead of being read back off the PNG, so the
    check is on the rendered artists rather than on the source that composed them.

    Keyed rather than flattened, and shared rather than card-specific, for one reason each: an assertion
    that names its own surface can only be turned green by that surface, and the edit that named the
    concentration basis changed two figures - a rendered-text guard pointed at one of them is aimed at the
    review instead of at the code."""
    import matplotlib.pyplot as plt
    real_close, held = plt.close, []
    monkeypatch.setattr(plt, "close", held.append)
    render()
    fig = held[-1]
    parts = {
        "figure_texts": [t.get_text() for t in fig.texts],
        "titles": [ax.get_title() for ax in fig.axes],
        "xlabels": [ax.get_xlabel() for ax in fig.axes],
        "ylabels": [ax.get_ylabel() for ax in fig.axes],
        "axes_texts": [t.get_text() for ax in fig.axes for t in ax.texts],
        "yticklabels": [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()],
    }
    real_close(fig)
    return parts


def _flatten(parts: dict[str, list[str]]) -> str:
    return " | ".join(s for surface in parts.values() for s in surface)


def _drawn_withdrawal_status(parts: dict[str, list[str]]) -> dict[str, str]:
    """The regulatory fate each y-tick label actually states, per compound. Returned as an exact string
    so a caller can compare by equality: `"withdrawn" in label` is satisfied by `"not withdrawn"`, which
    is the one mutation a guard on this claim exists to catch."""
    drawn = {}
    for label in parts["yticklabels"]:
        compound, _, rest = label.partition("\n")
        drawn[compound] = rest.split(";")[0].strip()
    return drawn


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
    # By equality, not containment: "withdrawn" in label is also satisfied by "not withdrawn", so the
    # containment form stayed green when the producer was inverted - and this is the factual claim the
    # Oh! moment turns on, spoken aloud in the video and drawn on the first artifact a reviewer sees.
    assert _drawn_withdrawal_status(parts) == WITHDRAWAL_ON_SCREEN, parts["yticklabels"]
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
    three of the four are missing."""
    parts = _rendered_dose_view_parts(tmp_path, monkeypatch)
    for surface, clause in DOSE_VIEW_BASIS.items():
        assert any(clause in s for s in parts[surface]), (surface, parts[surface])
    footnote = " | ".join(parts["figure_texts"])
    for clause in DOSE_VIEW_FOOTNOTE_BASIS:
        assert clause in footnote, (clause, footnote)
    # The same regulatory claim as the card, drawn from the same shared helper, pinned on both surfaces.
    assert _drawn_withdrawal_status(parts) == WITHDRAWAL_ON_SCREEN, parts["yticklabels"]


def test_roc_legend_labels_stay_short_enough_to_sit_inside_the_axes():
    """The ROC legend sits in the empty lower-right corner so the curve keeps the frame. A longer
    label grows that box into the curves. 28 characters is the longest label measured clear of every
    curve at 1920x1080; the check is the rendered figure, this is the guard against drifting past it."""
    labels = [short for _, short, _, _ in figures.ROC_ARMS + [figures.EXPLORATORY_ROC_ARM]]
    assert max(len(s) for s in labels) <= 28, labels
