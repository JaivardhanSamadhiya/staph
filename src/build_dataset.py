"""Filter the INPHARED metadata and assemble a balanced labelled corpus.

The INPHARED ``data.tsv.gz`` has no header row; columns follow a fixed layout
documented in the INPHARED README. We use the columns relevant to dataset
construction:

* 0 - GenBank accession
* 3 - phage description (organism)
* 4 - genome length (bp)
* 5 - GC fraction
* 11 - viral family (Caudoviricetes etc.)
* 12 - viral subfamily
* 13 - viral genus
* 14 - lifestyle prediction (Virulent / Temperate / Unknown)
* 15 - completeness flag
* 16 - completeness percentage
* 17 - host genus parsed from organism name
* 18 - host genus parsed from /host or /lab_host GenBank tags

We treat *Staphylococcus* phages as the positive class and assemble a host-
diverse, host-balanced negative class from the remaining genera. We also apply
length and completeness filters so the model is not learning artefacts of
draft assemblies.
"""

from __future__ import annotations

import argparse
import gzip

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger("build_dataset")

# Column indices documented above
COL_ACCESSION = 0
COL_DESCRIPTION = 3
COL_LENGTH = 4
COL_GC = 5
COL_FAMILY = 11
COL_SUBFAMILY = 12
COL_GENUS = 13
COL_LIFESTYLE = 14
COL_COMPLETE_FLAG = 15
COL_COMPLETE_PCT = 16
COL_HOST_TAX = 17
COL_HOST_LAB = 18

COLUMNS = [
    "accession",     # 0
    "release_date",  # 1
    "molecule",      # 2
    "description",   # 3
    "length",        # 4
    "gc",            # 5
    "realm",         # 6
    "kingdom",       # 7
    "phylum",        # 8
    "class",         # 9
    "order",         # 10
    "family",        # 11
    "subfamily",     # 12
    "viral_genus",   # 13
    "lifestyle",     # 14
    "complete",      # 15
    "complete_pct",  # 16
    "host_tax",      # 17
    "host_lab",      # 18
]


def load_metadata(path=config.META_PATH) -> pd.DataFrame:
    """Read the gzipped TSV into a DataFrame with our column names."""
    log.info("Reading INPHARED metadata from %s", path)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(
            fh,
            sep="\t",
            header=None,
            names=COLUMNS,
            dtype=str,
            na_values=[""],
            keep_default_na=False,
        )
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["gc"] = pd.to_numeric(df["gc"], errors="coerce")
    df["complete_pct"] = pd.to_numeric(df["complete_pct"], errors="coerce")
    log.info("Loaded %d phage records", len(df))
    return df


def _quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["accession", "length", "gc"]).copy()
    df = df.drop_duplicates(subset=["accession"], keep="first")
    df = df[df["length"].between(config.MIN_GENOME_LEN, config.MAX_GENOME_LEN)]
    df = df[~df["host_tax"].isin(config.EXCLUDE_HOST_VALUES)]

    # Reject entries where the canonical phage organism name does not begin
    # with the asserted bacterial host genus.  This removes records where the
    # ``host_tax`` column has been populated from a sample-source organism
    # (e.g. lemur or human gut metagenomes) rather than the bacterial host
    # the phage actually infects.
    first_word = df["description"].str.split(" ").str[0].str.lower()
    df = df[first_word == df["host_tax"].str.lower()].copy()

    log.info("Quality filter retained %d / %d records", len(df), n0)
    return df


def _resolve_host(row: pd.Series) -> str:
    """Use the curated host genus parsed from the phage organism name.

    INPHARED also exposes a ``host_lab`` column built from the GenBank
    ``/host`` and ``/lab_host`` qualifiers.  That column is much noisier (it
    sometimes contains the organism the sample was collected *from*, e.g.
    primate genera for phages isolated from gut metagenomes), so we
    deliberately do not fall back to it here.
    """
    h = (row["host_tax"] or "").strip()
    if h and h not in config.EXCLUDE_HOST_VALUES:
        return h
    return ""


def build(seed: int = config.SEED) -> pd.DataFrame:
    """Construct the balanced manifest CSV used for downstream steps."""
    rng = np.random.default_rng(seed)
    df = load_metadata()
    df = _quality_filter(df)
    df["host"] = df.apply(_resolve_host, axis=1)
    df = df[~df["host"].isin(config.EXCLUDE_HOST_VALUES)].copy()

    pos = df[df["host"].str.startswith(config.TARGET_HOST, na=False)].copy()
    pos["label"] = 1
    log.info("Positives (%s phages): %d", config.TARGET_HOST, len(pos))

    neg_pool = df[~df["host"].str.startswith(config.TARGET_HOST, na=False)].copy()
    log.info("Non-%s pool: %d", config.TARGET_HOST, len(neg_pool))

    # Stratified, host-balanced negative sampling. We cap the count per host
    # genus and draw without replacement until the negative count matches the
    # positive count.
    target_neg = len(pos)
    chosen_idx: list[int] = []
    grouped = neg_pool.groupby("host", sort=False)
    # Shuffle group order for fairness across runs.
    group_order = list(grouped.groups.keys())
    rng.shuffle(group_order)

    # First pass: take up to NEG_PER_GENUS_CAP from each genus.
    for host_name in group_order:
        idx = grouped.groups[host_name].to_list()
        rng.shuffle(idx)
        chosen_idx.extend(idx[: config.NEG_PER_GENUS_CAP])
        if len(chosen_idx) >= target_neg * 2:
            break

    chosen_idx = list(dict.fromkeys(chosen_idx))  # de-dupe, preserve order
    rng.shuffle(chosen_idx)
    chosen_idx = chosen_idx[:target_neg]
    neg = neg_pool.loc[chosen_idx].copy()
    neg["label"] = 0

    n_genera = neg["host"].nunique()
    if n_genera < config.MIN_GENERA_PER_NEG:
        raise RuntimeError(
            f"Negative set has only {n_genera} host genera "
            f"(< {config.MIN_GENERA_PER_NEG}); check filters."
        )
    log.info("Negatives: %d phages across %d host genera", len(neg), n_genera)

    keep_cols = [
        "accession", "description", "length", "gc",
        "family", "subfamily", "viral_genus",
        "lifestyle", "host", "label",
    ]
    manifest = pd.concat([pos[keep_cols], neg[keep_cols]], ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out = config.PROC_DIR / "manifest.csv"
    manifest.to_csv(out, index=False)
    log.info("Wrote manifest to %s (%d rows)", out, len(manifest))

    return manifest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    build(seed=_parse_args().seed)
