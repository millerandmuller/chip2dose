"""Does an exposure-aware model predict clinical liver injury better than potency alone?

Pre-registration protocol (enforced in code):
  1. `python -m src.validate --preregister` writes validation/preregistration.json and
     validation/split.csv (the grouped folds). Both are committed before any evaluation runs.
  2. `python -m src.validate` refuses to run if the recomputed split does not hash to the
     pre-registered value, so the folds cannot drift after results have been seen.

Grouping: pair identity first, chemical class second. Two drugs share a group if they are a
matched toxic/non-toxic pair in Ewart et al. Table 1, the same active ingredient, identical
structures, or structurally similar (Morgan radius-2 Tanimoto >= 0.4). Groups are connected
components of those links; a group never spans train and test.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, load

PREREG = config.VALIDATION / "preregistration.json"
SPLIT = config.VALIDATION / "split.csv"
# Added in review rounds 2-3: the pre-registration file describes the plan in words, but the
# plan is executed by code. These fingerprints freeze that code at its state in the
# pre-registration commit a76e7e9 (each item verified byte-identical to that commit when added).
CODE_FREEZE = config.VALIDATION / "code_freeze.json"
FROZEN_CODE = {
    "src/validate.py": [
        "N_FOLDS", "N_REPEATS", "SPLIT_SEED", "BOOTSTRAP", "BOOTSTRAP_SEED", "TANIMOTO_LINK", "LR_C",
        "POSITIVE_WIDE", "POSITIVE_NARROW", "NEGATIVE", "POTENCY_FEATURES", "EXPOSURE_FEATURES",
        "molecule_identity", "benchmark_table", "_union_find", "structural_groups", "make_split", "split_frame",
        "ARMS", "PRIMARY", "SECONDARY", "build_model", "out_of_fold_scores", "group_bootstrap", "ci",
        "outcome_case", "preregistration_document", "evaluate",
    ],
    "src/load.py": ["normalize_name", "SALT_WORDS", "_split_values", "GECI_IV_ONLY", "geci_benchmark", "liverchip_pairs"],
    "src/config.py": ["RULE_OF_THUMB_DOSE_MG", "RULE_OF_THUMB_LOGP"],
}

N_FOLDS = 5
N_REPEATS = 20
SPLIT_SEED = 20260929
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261005
TANIMOTO_LINK = 0.4
LR_C = 1.0

POSITIVE_WIDE = ("Most-DILI-Concern", "Less-DILI-Concern", "Clinical Dev")
POSITIVE_NARROW = ("Most-DILI-Concern", "Clinical Dev")
NEGATIVE = ("No-DILI-Concern",)

POTENCY_FEATURES = ["log_pod", "log_nonbsep_pod", "log_cytotox", "log_bsep_ic50"]
EXPOSURE_FEATURES = POTENCY_FEATURES + [
    "log_cmax", "log_dose", "logp", "log_fu", "log_margin_total", "log_margin_free",
]


# --------------------------------------------------------------------------- data

def molecule_identity(name: str) -> str:
    """'LY2409021 (90 mg)*' and 'Flupirtine_600' -> one molecule regardless of dose."""
    key = load.normalize_name(name)
    key = re.sub(r"_\d+$", "", key)
    return re.sub(r"\s*\(?\d+\s*mg\)?\*?$", "", key).strip()


def benchmark_table() -> pd.DataFrame:
    """One row per molecule with a lowest POD, a Cmax and a non-ambiguous clinical label."""
    g = load.geci_benchmark()
    g = g[~g["iv_only"] & g["lowest_pod_uM"].notna() & g["cmax_uM"].notna() & g["dili_concern"].notna()].copy()
    g["identity"] = g["name"].map(molecule_identity)
    n_before = len(g)
    g = g.drop_duplicates("identity", keep="first")
    g.attrs["n_dose_duplicates_dropped"] = n_before - len(g)
    g = g[g["dili_concern"].isin(POSITIVE_WIDE + NEGATIVE)].copy()

    g["y_wide"] = g["dili_concern"].isin(POSITIVE_WIDE).astype(int)
    g["y_narrow"] = np.where(
        g["dili_concern"].isin(POSITIVE_NARROW), 1, np.where(g["dili_concern"].isin(NEGATIVE), 0, -1)
    )
    fu = g[["fu_opera", "fu_admetlab", "fu_simplus"]].median(axis=1).clip(lower=1e-4, upper=1.0)
    g["fu_median"] = fu
    g["log_pod"] = np.log10(g["lowest_pod_uM"])
    g["log_nonbsep_pod"] = np.log10(g["lowest_nonbsep_pod_uM"])
    g["log_cytotox"] = np.log10(g["lowest_cytotox_uM"])
    g["log_bsep_ic50"] = np.log10(g["bsep_ic50_uM"])
    g["log_cmax"] = np.log10(g["cmax_uM"])
    g["log_dose"] = np.log10(g["dose_mg"])
    g["log_fu"] = np.log10(fu)
    g["log_margin_total"] = g["log_pod"] - g["log_cmax"]
    g["log_margin_free"] = g["log_pod"] - np.log10(g["cmax_uM"] * fu)
    g["rule_of_thumb"] = (
        (g["dose_mg"] >= config.RULE_OF_THUMB_DOSE_MG) & (g["logp"] >= config.RULE_OF_THUMB_LOGP)
    ).astype(int)
    return g.reset_index(drop=True)


# --------------------------------------------------------------------------- grouping

def _union_find(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return [find(i) for i in range(n)]


def structural_groups(table: pd.DataFrame) -> pd.Series:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table["smiles"]]
    n = len(table)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
        edges += [(i, i + 1 + j) for j, s in enumerate(sims) if s >= TANIMOTO_LINK]

    index_of = {k: i for i, k in enumerate(table["identity"])}
    pairs = load.liverchip_pairs()
    for key, partner in zip(pairs["key"], pairs["partner_key"]):
        if key in index_of and partner in index_of:
            edges.append((index_of[key], index_of[partner]))

    roots = _union_find(n, edges)
    # Stable group ids: numbered by first appearance.
    order = {root: gid for gid, root in enumerate(dict.fromkeys(roots))}
    return pd.Series([order[r] for r in roots], index=table.index, name="group")


def make_split(groups: pd.Series) -> pd.DataFrame:
    """Repeated grouped K-fold: each repeat shuffles the groups and deals them to the fold with
    the fewest drugs so far, largest groups first."""
    rng = np.random.default_rng(SPLIT_SEED)
    sizes = groups.value_counts()
    folds = {}
    for r in range(N_REPEATS):
        shuffled = rng.permutation(sizes.index.to_numpy())
        ordered = sorted(shuffled, key=lambda gid: -sizes[gid])  # stable: ties keep shuffled order
        load_per_fold = np.zeros(N_FOLDS, dtype=int)
        assignment = {}
        for gid in ordered:
            fold = int(np.argmin(load_per_fold))
            assignment[gid] = fold
            load_per_fold[fold] += sizes[gid]
        folds[f"repeat_{r:02d}"] = groups.map(assignment).to_numpy()
    return pd.DataFrame(folds, index=groups.index)


def split_frame(table: pd.DataFrame) -> pd.DataFrame:
    groups = structural_groups(table)
    frame = pd.concat([table[["identity"]], groups, make_split(groups)], axis=1)
    return frame


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


# --------------------------------------------------------------------------- arms

@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                       # "score" (no learning) or "model"
    features: tuple[str, ...] = ()
    score_column: str = ""
    sign: float = 1.0               # orient scores so that higher = more concern
    model: str = ""                 # "lr" or "gb"


ARMS = [
    Arm("potency alone", "score", score_column="log_pod", sign=-1.0),
    Arm("rule of thumb (dose >= 100 mg, logP >= 3)", "score", score_column="rule_of_thumb", sign=1.0),
    Arm("margin alone (total)", "score", score_column="log_margin_total", sign=-1.0),
    Arm("margin alone (free)", "score", score_column="log_margin_free", sign=-1.0),
    Arm("learned, potency-only features (LR)", "model", tuple(POTENCY_FEATURES), model="lr"),
    Arm("learned, exposure-aware features (LR)", "model", tuple(EXPOSURE_FEATURES), model="lr"),
    Arm("learned, potency-only features (GB)", "model", tuple(POTENCY_FEATURES), model="gb"),
    Arm("learned, exposure-aware features (GB)", "model", tuple(EXPOSURE_FEATURES), model="gb"),
]
PRIMARY = ("learned, exposure-aware features (LR)", "learned, potency-only features (LR)")
SECONDARY = [
    ("learned, exposure-aware features (GB)", "learned, potency-only features (GB)"),
    ("margin alone (free)", "potency alone"),
    ("margin alone (total)", "potency alone"),
    ("learned, exposure-aware features (LR)", "rule of thumb (dose >= 100 mg, logP >= 3)"),
]


def build_model(kind: str):
    if kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=LR_C, max_iter=5000),
        )
    if kind == "gb":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10, random_state=0
        )
    raise ValueError(kind)


def out_of_fold_scores(table: pd.DataFrame, y: np.ndarray, folds: pd.DataFrame, arm: Arm) -> np.ndarray:
    """Mean over repeats of each drug's held-out score. Score arms need no fitting."""
    if arm.kind == "score":
        return arm.sign * table[arm.score_column].to_numpy(dtype=float)
    X = table[list(arm.features)].to_numpy(dtype=float)
    total = np.zeros(len(table))
    for column in folds.columns:
        fold_ids = folds[column].to_numpy()
        for k in range(N_FOLDS):
            test = fold_ids == k
            model = build_model(arm.model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[~test], y[~test])
            total[test] += model.predict_proba(X[test])[:, 1]
    return total / folds.shape[1]


# --------------------------------------------------------------------------- statistics

def group_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], groups: np.ndarray,
                    comparisons: list[tuple[str, str]]) -> dict:
    """Resample whole groups with replacement; AUCs and paired AUC differences on the same draws."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    aucs = {name: [] for name in scores}
    diffs = {c: [] for c in comparisons}
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        draw = {name: roc_auc_score(yb, s[idx]) for name, s in scores.items()}
        for name, value in draw.items():
            aucs[name].append(value)
        for a, b in comparisons:
            diffs[(a, b)].append(draw[a] - draw[b])
    return {"aucs": aucs, "diffs": diffs}


def ci(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def outcome_case(diff: float, low: float) -> str:
    """The three outcomes fixed in advance for the headline claim."""
    if diff <= 0:
        return "3: paired difference <= 0 - exposure-aware model does not beat potency-only"
    if low > 0:
        return "1: paired difference positive, CI excludes zero"
    return "2: paired difference positive, CI includes zero - direction only, not significant"


# --------------------------------------------------------------------------- pre-registration

def preregistration_document(table: pd.DataFrame, frame: pd.DataFrame) -> dict:
    return {
        "question": "Does an exposure-aware model separate clinical DILI outcomes better than potency alone?",
        "data": "Geci et al. 2026 integrated table (pinned commit, see data/sources.json), oral drugs only",
        "inclusion": "lowest in-vitro POD, total Cmax and a DILI label present; 'Ambiguous' excluded; "
                     "one row per molecule (first row in the authors' file order when a molecule appears at several doses)",
        "n_drugs": int(len(table)),
        "n_positive_primary": int(table["y_wide"].sum()),
        "n_negative_primary": int((table["y_wide"] == 0).sum()),
        "primary_endpoint": "DILIrank binary convention: vMost + vLess + clinical-development DILI failures = 1, vNo = 0",
        "secondary_endpoint": "narrow: vMost + clinical-development failures = 1, vNo = 0 (vLess excluded)",
        "arms": [arm.name for arm in ARMS],
        "features_potency_only": POTENCY_FEATURES,
        "features_exposure_aware": EXPOSURE_FEATURES,
        "primary_comparison": list(PRIMARY),
        "secondary_comparisons": [list(c) for c in SECONDARY],
        "statistic": "paired difference in ROC AUC on identical folds; 95% CI from a group-level bootstrap "
                     f"({BOOTSTRAP} draws) of the repeat-averaged out-of-fold scores",
        "outcome_cases": [
            "1: paired difference positive, CI excludes zero -> claim stands",
            "2: paired difference positive, CI includes zero -> directional only, reported as not significant",
            "3: paired difference <= 0 -> reported as a negative result",
        ],
        "grouping": f"connected components of: matched pairs (Ewart et al. Table 1), same molecule, "
                    f"Morgan r=2 2048-bit Tanimoto >= {TANIMOTO_LINK}",
        "n_groups": int(frame["group"].nunique()),
        "largest_group": int(frame["group"].value_counts().iloc[0]),
        "cv": f"{N_REPEATS} repeats of {N_FOLDS}-fold grouped CV, seed {SPLIT_SEED}",
        "models": {"lr": f"median imputation + missing indicators, standardised, L2 logistic regression C={LR_C}, no tuning",
                   "gb": "HistGradientBoosting, 150 iterations, learning rate 0.05, 8 leaves, min 10 per leaf, no tuning"},
        "reporting_rule": "classifier probabilities appear only inside validation figures; never as per-compound scores",
        "split_sha256": sha256_text(split_csv_text(frame)),
    }


def preregister(force: bool = False) -> dict:
    if PREREG.exists() and not force:
        raise SystemExit(f"{PREREG} exists; pre-registration is written once. Use --force only before any evaluation.")
    table = benchmark_table()
    frame = split_frame(table)
    config.VALIDATION.mkdir(exist_ok=True)
    SPLIT.write_text(split_csv_text(frame), encoding="utf-8")
    doc = preregistration_document(table, frame)
    PREREG.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _defined_name(node) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        return target.id if isinstance(target, ast.Name) else None
    return None


def code_fingerprints() -> dict[str, str]:
    """sha256 of the source text of each frozen constant or function. A frozen name defined more
    than once anywhere in its module (e.g. a later redefinition inside a block) gets a fingerprint
    that cannot match, because the last definition would win at runtime."""
    prints = {}
    for rel, names in FROZEN_CODE.items():
        source = (config.ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        counts: dict[str, int] = {}
        for node in ast.walk(tree):
            name = _defined_name(node)
            if name in names:
                counts[name] = counts.get(name, 0) + 1
        for node in tree.body:
            name = _defined_name(node)
            if name in names:
                digest = sha256_text(ast.get_source_segment(source, node))
                prints[f"{rel}:{name}"] = digest if counts.get(name) == 1 else f"redefined {counts[name]} times"
    return prints


def load_verified_split(table: pd.DataFrame) -> pd.DataFrame:
    """Recompute the split, the analysis plan and the frozen code; all must match the files."""
    for required in (PREREG, SPLIT, CODE_FREEZE):
        if not required.exists():
            raise SystemExit(f"{required.name} is missing; the pre-registered validation cannot run without it")
    frozen = json.loads(CODE_FREEZE.read_text(encoding="utf-8"))["fingerprints"]
    drifted_code = sorted(k for k in frozen.keys() | code_fingerprints().keys()
                          if frozen.get(k) != code_fingerprints().get(k))
    if drifted_code:
        raise SystemExit(f"code frozen at pre-registration has changed: {', '.join(drifted_code)}; refusing to evaluate")
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = split_frame(table)
    text = split_csv_text(frame)
    if sha256_text(text) != doc["split_sha256"] or SPLIT.read_text(encoding="utf-8") != text:
        raise SystemExit("split does not match the pre-registration; refusing to evaluate")
    expected = preregistration_document(table, frame)
    drifted = sorted(k for k in expected.keys() | doc.keys() if expected.get(k) != doc.get(k))
    if drifted:
        raise SystemExit(f"analysis plan differs from the pre-registration in: {', '.join(drifted)}; refusing to evaluate")
    return frame


# --------------------------------------------------------------------------- evaluation

def evaluate(endpoint: str = "wide") -> dict:
    table = benchmark_table()
    frame = load_verified_split(table)
    if endpoint == "narrow":
        keep = table["y_narrow"] >= 0
        table, frame = table[keep].reset_index(drop=True), frame[keep].reset_index(drop=True)
        y = table["y_narrow"].to_numpy()
    else:
        y = table["y_wide"].to_numpy()
    folds = frame[[c for c in frame.columns if c.startswith("repeat_")]]
    groups = frame["group"].to_numpy()

    scores = {arm.name: out_of_fold_scores(table, y, folds, arm) for arm in ARMS}
    comparisons = [PRIMARY] + SECONDARY
    boot = group_bootstrap(y, scores, groups, comparisons)

    arms = []
    for arm in ARMS:
        low, high = ci(boot["aucs"][arm.name])
        arms.append({"arm": arm.name, "auc": roc_auc_score(y, scores[arm.name]), "ci_low": low, "ci_high": high})
    diffs = []
    for a, b in comparisons:
        point = roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b])
        low, high = ci(boot["diffs"][(a, b)])
        diffs.append({
            "comparison": f"{a} minus {b}", "primary": (a, b) == PRIMARY, "delta_auc": point,
            "ci_low": low, "ci_high": high,
            "share_of_draws_leq_0": float(np.mean(np.array(boot["diffs"][(a, b)]) <= 0)),
        })
    primary = diffs[0]
    return {
        "endpoint": endpoint,
        "n": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        "n_groups": int(len(np.unique(groups))),
        "arms": arms,
        "comparisons": diffs,
        "outcome_case": outcome_case(primary["delta_auc"], primary["ci_low"]),
        "_scores": scores, "_y": y, "_table": table,
    }


# Added after the first evaluation run (2026-09-28, local time), NOT pre-registered. Question it answers:
# how much of the exposure-aware advantage comes from exposure alone, without any chip data?
EXPLORATORY_ARMS = [
    Arm("exploratory: total Cmax alone", "score", score_column="log_cmax", sign=1.0),
    Arm("exploratory: daily dose alone", "score", score_column="log_dose", sign=1.0),
    Arm("exploratory: free Cmax alone", "score", score_column="log_free_cmax", sign=1.0),
]
EXPLORATORY_COMPARISONS = [
    ("margin alone (total)", "exploratory: total Cmax alone"),
    ("margin alone (free)", "exploratory: free Cmax alone"),
    ("exploratory: total Cmax alone", "potency alone"),
]


def evaluate_exploratory(endpoint: str = "wide") -> dict:
    """Exposure-only baselines on the same drugs and bootstrap as the pre-registered analysis."""
    table = benchmark_table()
    frame = load_verified_split(table)
    table = table.assign(log_free_cmax=table["log_cmax"] + table["log_fu"])
    if endpoint == "narrow":
        keep = table["y_narrow"] >= 0
        table, frame = table[keep].reset_index(drop=True), frame[keep].reset_index(drop=True)
        y = table["y_narrow"].to_numpy()
    else:
        y = table["y_wide"].to_numpy()
    arms = [a for a in ARMS if a.kind == "score"] + EXPLORATORY_ARMS
    folds = frame[[c for c in frame.columns if c.startswith("repeat_")]]
    scores = {arm.name: out_of_fold_scores(table, y, folds, arm) for arm in arms}
    boot = group_bootstrap(y, scores, frame["group"].to_numpy(), EXPLORATORY_COMPARISONS)
    rows = []
    for a, b in EXPLORATORY_COMPARISONS:
        low, high = ci(boot["diffs"][(a, b)])
        rows.append({
            "comparison": f"{a} minus {b}",
            "delta_auc": roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b]),
            "ci_low": low, "ci_high": high,
            "share_of_draws_leq_0": float(np.mean(np.array(boot["diffs"][(a, b)]) <= 0)),
        })
    aucs = []
    for arm in EXPLORATORY_ARMS:
        low, high = ci(boot["aucs"][arm.name])
        aucs.append({"arm": arm.name, "auc": roc_auc_score(y, scores[arm.name]), "ci_low": low, "ci_high": high})
    return {"endpoint": endpoint, "status": "exploratory, added after the first evaluation run; not pre-registered",
            "arms": aucs, "comparisons": rows, "_scores": scores, "_y": y}


# Added 2026-09-29, after the pre-registered results had been read: exploratory and post-hoc, NOT
# pre-registered, and run once. Question it answers: Geci et al. attribute part of Cmax's predictive power
# to a range artefact in the integrated dataset (in-vitro potency compressed by test-concentration ranges,
# Cmax spread wider). Does the primary difference survive when the drugs whose Cmax lies outside the
# observed POD range are removed? The boundary is the observed POD range itself, not a chosen constant.
RANGE_MATCHED_STATUS = ("exploratory, post-hoc: added 2026-09-29 after the pre-registered results were read; "
                        "not pre-registered; run once")
RANGE_MATCHED_PREFIX = "exploratory, post-hoc (range-matched)"


def log_spans(table: pd.DataFrame) -> dict:
    """Observed spread, in orders of magnitude, of lowest POD and total Cmax (full range and 5th-95th)."""
    spans = {}
    for column in ("log_pod", "log_cmax"):
        values = table[column]
        spans[column] = {"min": float(values.min()), "max": float(values.max()),
                         "span_orders": float(values.max() - values.min()),
                         "span_p5_p95_orders": float(values.quantile(0.95) - values.quantile(0.05))}
    return spans


def range_matched_mask(table: pd.DataFrame) -> pd.Series:
    """True for drugs whose total Cmax lies inside the observed range of lowest POD in the same table."""
    return table["log_cmax"].between(table["log_pod"].min(), table["log_pod"].max())


def removed_tails(table: pd.DataFrame, keep: pd.Series) -> list[dict]:
    """Which side of the POD range each removed drug's Cmax falls on, by clinical label (wide endpoint)."""
    removed = table[~keep]
    side = np.where(removed["log_cmax"] > table["log_pod"].max(), "Cmax above the highest POD",
                    "Cmax below the lowest POD")
    label = np.where(removed["y_wide"] == 1, "concern", "no concern")
    counts = pd.crosstab(side, label)
    return [{"tail": tail, "label": lab, "n": int(counts.loc[tail, lab]) if lab in counts.columns else 0}
            for tail in counts.index for lab in ("concern", "no concern")]


def censored_pod_entries() -> int:
    """Raw PODValues entries in the Geci file written as a bound ('<x' or '>x') rather than a number."""
    raw = pd.read_excel(config.GECI_LITERATURE, usecols=["PODValues"])["PODValues"].dropna().astype(str)
    return int(raw.str.split(";").explode().str.strip().str.match(r"^[<>]").sum())


def evaluate_range_matched() -> dict:
    """The pre-registered primary pair, refit on the range-matched subset (wide endpoint only).

    Same fold assignment restricted to the kept rows, same models, same group bootstrap and seed. The two
    LR arms are refit inside the subset, so no model learns from a drug outside the POD range."""
    table = benchmark_table()
    frame = load_verified_split(table)
    spans = log_spans(table)
    keep = range_matched_mask(table)
    tails = removed_tails(table, keep)
    sub, sub_frame = table[keep].reset_index(drop=True), frame[keep].reset_index(drop=True)
    y = sub["y_wide"].to_numpy()
    folds = sub_frame[[c for c in sub_frame.columns if c.startswith("repeat_")]]
    groups = sub_frame["group"].to_numpy()
    arms = [arm for arm in ARMS if arm.name in PRIMARY]
    scores = {arm.name: out_of_fold_scores(sub, y, folds, arm) for arm in arms}
    boot = group_bootstrap(y, scores, groups, [PRIMARY])
    a, b = PRIMARY
    low, high = ci(boot["diffs"][PRIMARY])
    arm_rows = []
    for arm in arms:
        a_low, a_high = ci(boot["aucs"][arm.name])
        arm_rows.append({"arm": f"{RANGE_MATCHED_PREFIX}: {arm.name}", "auc": roc_auc_score(y, scores[arm.name]),
                         "ci_low": a_low, "ci_high": a_high})
    comparison = {"comparison": f"{RANGE_MATCHED_PREFIX}: {a} minus {b}",
                  "delta_auc": roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b]),
                  "ci_low": low, "ci_high": high,
                  "share_of_draws_leq_0": float(np.mean(np.array(boot["diffs"][PRIMARY]) <= 0))}
    return {"endpoint": "wide", "status": RANGE_MATCHED_STATUS,
            "subset": "total Cmax inside the observed lowest-POD range of the 220-drug set",
            "n": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
            "n_groups": int(len(np.unique(groups))), "n_removed": int((~keep).sum()),
            "removed_tails": tails, "spans_full_set": spans,
            "censored_pod_entries_in_source": censored_pod_entries(),
            "censoring_note": "no lowest POD in the benchmark is a bound: Geci et al. kept only compounds with a "
                              "reported potency value, so a restriction to uncensored PODs keeps every drug and cannot "
                              "test truncation at the top of the test-concentration range",
            "arms": arm_rows, "comparisons": [comparison]}


def coefficients(table: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Standardised LR coefficients of the exposure-aware model fitted on all drugs (interpretation only)."""
    model = build_model("lr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(table[EXPOSURE_FEATURES].to_numpy(dtype=float), y)
    imputer = model.named_steps["simpleimputer"]
    names = list(EXPOSURE_FEATURES) + [f"missing:{EXPOSURE_FEATURES[i]}" for i in imputer.indicator_.features_]
    coef = model.named_steps["logisticregression"].coef_[0]
    frame = pd.DataFrame({"feature": names, "standardised_coefficient": coef})
    return frame.reindex(frame["standardised_coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def failure_cases(result: dict, arm_name: str = PRIMARY[0], n: int = 8) -> pd.DataFrame:
    """Drugs the exposure-aware model gets wrong at the Youden-optimal cut on its own held-out
    scores. Reported with their real-unit inputs only; the model score itself is not reported.
    The cut is chosen in-sample on the same held-out scores, so this error list is slightly
    optimistic; it illustrates failure modes and is not a performance estimate."""
    from sklearn.metrics import roc_curve

    y, s, table = result["_y"], result["_scores"][arm_name], result["_table"]
    fpr, tpr, thr = roc_curve(y, s)
    cut = thr[int(np.argmax(tpr - fpr))]
    predicted = (s >= cut).astype(int)
    wrong = table.assign(predicted_concern=predicted, _s=s)[predicted != y]
    # Most confident errors first: false negatives with the lowest scores, false positives with the highest.
    wrong = wrong.assign(_conf=np.where(wrong["predicted_concern"] == 0, -wrong["_s"], wrong["_s"]))
    wrong = wrong.sort_values("_conf", ascending=False).head(n)
    return pd.DataFrame({
        "drug": wrong["name"],
        "clinical_label": wrong["dili_concern"],
        "model_call": np.where(wrong["predicted_concern"] == 1, "concern", "no concern"),
        "lowest_pod_uM": wrong["lowest_pod_uM"],
        "cmax_uM": wrong["cmax_uM"],
        "margin_total": 10 ** wrong["log_margin_total"],
        "daily_dose_mg": wrong["dose_mg"],
        "logp": wrong["logp"],
    }).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-registered validation")
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.preregister:
        doc = preregister(force=args.force)
        print(json.dumps({k: doc[k] for k in ("n_drugs", "n_positive_primary", "n_negative_primary", "n_groups",
                                              "largest_group", "split_sha256")}, indent=2))
        return 0
    print("run the evaluation through run_demo.py, which also writes figures and tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
