"""Read each raw source into a tidy DataFrame. Column meanings are taken from the source files'
own headers; nothing is renamed without saying what it was."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

SALT_WORDS = (
    "hydrochloride", "sodium salt", "sodium", "maleate", "mesylate", "hcl", "dihydrochloride",
    "potassium", "calcium", "sulfate", "tartrate", "fumarate", "citrate", "succinate", "besylate",
)

# Names that differ between sources for the same active ingredient.
ALIASES = {
    "sitax(s)entan": "sitaxsentan",
    "sitaxentan": "sitaxsentan",
    "beta-estradiol": "estradiol",
    "17beta-estradiol": "estradiol",
    "paracetamol": "acetaminophen",
}


def normalize_name(name: object) -> str:
    """Lower-case active-ingredient key: drops salts, footnote stars and parenthetical notes."""
    if not isinstance(name, str):
        return ""
    text = name.split(";")[0].strip().lower().replace("*", "")
    for alias, target in ALIASES.items():
        if text.startswith(alias):
            return target
    text = text.split("(")[0]
    for salt in SALT_WORDS:
        text = re.sub(rf"\b{re.escape(salt)}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-")
    return ALIASES.get(text, text)


def parse_censored(text: object) -> tuple[float, bool]:
    """'>3' -> (3.0, True); '0.03' -> (0.03, False); '–' or blank -> (nan, False).

    Censored means no toxicity was observed up to that value; it is a lower bound, not a number.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return float("nan"), False
    raw = str(text).strip().replace(",", "").replace(" ", "").replace(" ", "")
    if raw in {"", "–", "-", "—", "NA", "nan"}:
        return float("nan"), False
    censored = raw.startswith(">")
    return float(raw.lstrip(">")), censored


def _parse_micromolar(text: object) -> float:
    """SD1 writes concentrations like '0.79µM' or '*0.64µM'."""
    if isinstance(text, (int, float)):
        return float(text)
    match = re.search(r"([0-9]*\.?[0-9]+(?:[eE]-?[0-9]+)?)", str(text))
    return float(match.group(1)) if match else float("nan")


def _parse_fraction(text: object) -> float:
    """SD1 fraction unbound; one cell reads '0.05 69' (a footnote number fused to the value)."""
    if isinstance(text, (int, float)):
        return float(text)
    match = re.match(r"\s*\*?([0-9]*\.?[0-9]+)", str(text))
    return float(match.group(1)) if match else float("nan")


# --------------------------------------------------------------------------- Liver-Chip

@lru_cache(maxsize=1)
def liverchip_drug_info() -> pd.DataFrame:
    """Supplementary Data 1: one row per drug.

    cmax_total_uM   human Cmax, total plasma concentration
    fu_plasma       expected fraction unbound in plasma
    fu_medium       free / total chip dosing concentration (2% FBS medium), per drug
    multipliers     tested multiples of Cmax
    """
    sheet = pd.read_excel(config.LIVERCHIP_SD1, header=None)
    body = sheet.iloc[4:, [1, 2, 4, 6, 7, 8]].copy()
    body.columns = ["drug", "cmax", "fu", "mult", "chip_total", "chip_free"]
    footnote = body["drug"].astype(str).str.match(r"^\s*(\*|The number|\s)")
    first_footnote = footnote[footnote].index.min()
    if pd.notna(first_footnote):
        body = body.loc[: first_footnote - 1]
    body["drug"] = body["drug"].ffill()
    rows = []
    for drug, group in body.groupby("drug", sort=False):
        total = pd.to_numeric(group["chip_total"], errors="coerce")
        free = pd.to_numeric(group["chip_free"], errors="coerce")
        ratio = (free / total).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "drug": str(drug).replace("*", "").strip(),
                "key": normalize_name(drug),
                "cmax_total_uM": _parse_micromolar(group["cmax"].dropna().iloc[0]),
                "fu_plasma": _parse_fraction(group["fu"].dropna().iloc[0]),
                "fu_medium": float(ratio.median()) if len(ratio) else float("nan"),
                "multipliers": [float(m) for m in pd.to_numeric(group["mult"], errors="coerce").dropna()],
                "chip_total_uM": [float(v) for v in total.dropna()],
                "assumed_from_partner": str(group["cmax"].dropna().iloc[0]).startswith("*"),
            }
        )
    return pd.DataFrame(rows)


def _jats_table(table_id: str) -> list[list[str]]:
    tree = ET.parse(config.LIVERCHIP_ARTICLE)
    for wrap in tree.iter("table-wrap"):
        if wrap.get("id") == table_id:
            rows = []
            for tr in wrap.iter("tr"):
                rows.append(["".join(cell.itertext()).strip() for cell in tr if cell.tag in ("td", "th")])
            return rows
    raise KeyError(f"{table_id} not found in {config.LIVERCHIP_ARTICLE.name}")


@lru_cache(maxsize=1)
def liverchip_pairs() -> pd.DataFrame:
    """Article Table 1: IQ MPS matched structurally related pairs and Garside DILI rank (1 = most severe)."""
    rows = _jats_table("Tab1")
    header, body = rows[0], rows[1:]
    table = pd.DataFrame(body, columns=header)
    table.columns = ["drug", "iq_mps_list", "tested_in_spheroid", "spheroid_false_negative", "garside_rank"]
    table["key"] = table["drug"].map(normalize_name)
    partner = table["iq_mps_list"].str.extract(r"matched with (.+)$")[0].fillna("")
    table["partner_key"] = partner.map(normalize_name)
    table["garside_rank"] = pd.to_numeric(table["garside_rank"])
    # One drug cell is empty in the published XML (olanzapine, set in italics in the print
    # version). Recover it from its partner's row: that row names it as the match.
    for idx in table.index[table["key"] == ""]:
        partner_row = table[table["key"] == table.at[idx, "partner_key"]]
        if len(partner_row) == 1:
            table.at[idx, "key"] = partner_row["partner_key"].iloc[0]
            table.at[idx, "drug"] = table.at[idx, "key"].capitalize()
    return table


@lru_cache(maxsize=1)
def liverchip_mos() -> pd.DataFrame:
    """Article Table 4: chip MOS-like value = minimum toxic concentration / total Cmax, per donor."""
    rows = _jats_table("Tab4")
    table = pd.DataFrame(rows[1:], columns=["drug", "d1", "d2", "both", "spheroid"])
    table["key"] = table["drug"].map(normalize_name)
    for col in ["d1", "d2", "both", "spheroid"]:
        parsed = table[col].map(parse_censored)
        table[f"mos_{col}"] = [value for value, _ in parsed]
        table[f"mos_{col}_censored"] = [flag for _, flag in parsed]
    return table.drop(columns=["d1", "d2", "both", "spheroid"])


# --------------------------------------------------------------------------- DILIrank

@lru_cache(maxsize=1)
def dilirank() -> pd.DataFrame:
    table = pd.read_excel(config.DILIRANK, sheet_name="version 2", header=1)
    table = table.rename(
        columns={"CompoundName": "compound", "vDILI-Concern": "label", "SeverityClass": "severity"}
    )
    table["key"] = table["compound"].map(normalize_name)
    # The file mixes 'vMOST-DILI-concern' and 'vMost-DILI-concern'; one canonical spelling per class.
    canonical = {
        "vmost-dili-concern": "vMost-DILI-concern",
        "vless-dili-concern": "vLess-DILI-concern",
        "vno-dili-concern": "vNo-DILI-concern",
        "ambiguous-dili-concern": "Ambiguous-DILI-concern",
    }
    table["label"] = table["label"].str.strip().str.lower().map(canonical)
    return table[["LTKBID", "compound", "key", "label", "severity", "LabelSection"]]


# --------------------------------------------------------------------------- Geci et al.

def _split_values(text: object) -> list[float]:
    if not isinstance(text, str):
        return []
    values = []
    for part in text.split(";"):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


# Only available intravenously; Geci et al. exclude them from the oral analysis.
GECI_IV_ONLY = ("Dobutamine", "isoproterenol", "Ethacrynic acid", "dacarbazine", "Deferoxamine")


@lru_cache(maxsize=1)
def geci_benchmark() -> pd.DataFrame:
    """Geci et al. integrated table joined to predicted properties, one row per drug entry.

    lowest_pod_uM   lowest in-vitro point of departure across 17 hepatotoxicity datasets (lowestPOD)
    cmax_uM         total plasma Cmax as integrated by the authors (column 'medianCmax'; despite the
                    name it is the curated value at the stated dose when one exists, see pod.py)
    dose_mg         clinical dose that the Cmax refers to (CmaxDoseValue, mg)
    dili_concern    DILIrank concern as integrated by Geci et al. (incl. 'Clinical Dev')
    logp            OPERA-predicted logP
    fu_*            predicted fraction unbound in plasma from three tools
    """
    lit = pd.read_excel(config.GECI_LITERATURE).copy()
    props = pd.read_excel(config.GECI_PROPERTIES)

    # Same disambiguation as the authors' analysis: flupirtine appears at two doses.
    flup = lit["compoundSMILES"] == "CCOC(=O)NC1=C(N=C(C=C1)NCC2=CC=C(C=C2)F)N"
    lit.loc[flup & (lit["CmaxDoseValue"] == 400), "compoundName"] = "Flupirtine_400"
    lit.loc[flup & (lit["CmaxDoseValue"] == 600), "compoundName"] = "Flupirtine_600"
    props.loc[props["compoundID"] == 144, "compoundName"] = "Flupirtine_400"
    props.loc[props["compoundID"] == 230, "compoundName"] = "Flupirtine_600"

    lit["first_name"] = lit["compoundName"].astype(str).str.split(";").str[0]
    props["first_name"] = props["compoundName"].astype(str).str.split(";").str[0]
    keep = [
        "compoundID", "first_name", "compoundSMILES", "OPERA_logP", "OPERA_Fu", "ADMETLab_Fu",
        "SimPlusv12_Fu", "ADMETLab_logP", "MW",
    ]
    merged = lit.merge(props[keep], on=["first_name", "compoundSMILES"], how="inner")

    out = pd.DataFrame(
        {
            "compound_id": merged["compoundID"],
            "name": merged["first_name"],
            "key": merged["first_name"].map(normalize_name),
            "smiles": merged["compoundSMILES"],
            "lowest_pod_uM": merged["lowestPOD"],
            "lowest_nonbsep_pod_uM": merged["lowestnonBSEPPOD"],
            "lowest_cytotox_uM": merged["lowestCytoTox"],
            "bsep_ic50_uM": merged["medianBSEPIC50"],
            "pod_values_uM": merged["PODValues"].map(_split_values),
            "pod_sources": merged["PODValueSources"].fillna("").str.strip(";").str.split(";"),
            "cmax_uM": merged["medianCmax"],
            "cmax_min_uM": merged["minCmax"],
            "cmax_max_uM": merged["maxCmax"],
            "cmax_values_uM": merged["CmaxValues"].map(_split_values),
            "cmax_sources": merged["CmaxValueSources"].fillna("").str.strip(";").str.split(";"),
            "dose_mg": merged["CmaxDoseValue"],
            "dose_unit": merged["CmaxDoseUnit"],
            "dili_concern": merged["DILIConcern"],
            "logp": merged["OPERA_logP"],
            "fu_opera": merged["OPERA_Fu"],
            "fu_admetlab": merged["ADMETLab_Fu"],
            "fu_simplus": merged["SimPlusv12_Fu"],
            "atc": merged.get("Faes2024ATC"),
        }
    )
    iv_only = out["name"].str.contains("|".join(GECI_IV_ONLY), case=False, regex=True)
    out["iv_only"] = iv_only
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- EPA MEA

@lru_cache(maxsize=1)
def mea_potency() -> pd.DataFrame:
    """MEA_data sheet: one row per test entry (146), minimum EC50 across network parameters (uM).

    Entries without an EC50 were inactive at every tested concentration.
    """
    table = pd.read_excel(config.EPA_MEA, sheet_name="MEA_data")
    return table.rename(
        columns={
            "name": "entry_name",
            "PREFERRED_NAME": "name",
            "Common.name/Abbreviation": "common_name",
            "Casrn": "casrn",
            "Min.Parameter.EC50": "entry_min_ec50_uM",
            "min.ec50": "min_ec50_uM",
            "Min.Cytotoxicity": "entry_min_cyto_uM",
            "min.cyto": "min_cyto_uM",
            "Specificity.Score": "specificity_score",
        }
    )


@lru_cache(maxsize=1)
def mea_ivive() -> pd.DataFrame:
    """in.vivo.and.ivive sheet: AEDs (mg/kg/day) for chemicals with in-vivo developmental
    neurotoxicity data, plus ExpoCast median exposure predictions (mg/kg bw/day)."""
    table = pd.read_excel(config.EPA_MEA, sheet_name="in.vivo.and.ivive")
    return table.rename(
        columns={
            "PREFERRED_NAME": "name",
            "EXPOCAST_MEDIAN_EXPOSURE_PREDICTION_MG/KG-BW/DAY": "expocast_mg_kg_day",
            "aed.min.ec50": "aed_mg_kg_day",
            "min.pod": "invivo_min_pod_mg_kg",
        }
    )
