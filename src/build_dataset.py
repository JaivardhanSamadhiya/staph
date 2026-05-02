"""Filter the INPHARED metadata and assemble a labelled corpus.

Negatives are quota-sampled across *phylogenetic distance tertiles* (NCBI
Taxonomy lineage) relative to genus *Staphylococcus*, so the classifier sees
near-neighbour hosts (e.g. other Firmicutes) as well as more distant phyla.

Host names are resolved against NCBI with a fallback chain (exact → first
token → unresolved); see ``taxonomy.py`` and the strata summary CSV for counts.
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import pandas as pd

from . import config, taxonomy
from .utils import get_logger

log = get_logger("build_dataset")

COLUMNS = [
    "accession", "release_date", "molecule", "description", "length", "gc",
    "realm", "kingdom", "phylum", "class", "order", "family", "subfamily",
    "viral_genus", "lifestyle", "complete", "complete_pct", "host_tax", "host_lab",
]


def load_metadata(path=config.META_PATH) -> pd.DataFrame:
    log.info("Reading INPHARED metadata from %s", path)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(
            fh, sep="\t", header=None, names=COLUMNS, dtype=str,
            na_values=[""], keep_default_na=False,
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

    first_word = df["description"].str.split(" ").str[0].str.lower()
    df = df[first_word == df["host_tax"].str.lower()].copy()

    log.info("Quality filter retained %d / %d records", len(df), n0)
    return df


def _resolve_host(row: pd.Series) -> str:
    h = (row["host_tax"] or "").strip()
    if h and h not in config.EXCLUDE_HOST_VALUES:
        return h
    return ""


def _taxonomy_table_for_genera(uniq_genera: list[str]) -> pd.DataFrame:
    """Resolve every unique negative host genus once; persiste NCBI cache."""
    cache = taxonomy.load_cache()
    rows = []
    distances: list[float] = []
    for g in uniq_genera:
        res = taxonomy.resolve_genus(str(g), cache)
        d = float(res.distance_to_staph)
        if d == d:
            distances.append(d)
        rows.append({
            "host_genus": str(g),
            "tax_id": res.tax_id,
            "resolution_method": res.resolution,
            "scientific_rank": res.scientific_name_rank,
            "distance_to_staphylococcus": d if d == d else np.nan,
            "query_used": res.query_used,
        })
    taxonomy.save_cache(cache)
    genus_df = pd.DataFrame(rows)

    if len(distances) >= 3:
        q1, q2 = np.quantile(distances, (1.0 / 3.0, 2.0 / 3.0))
    else:
        q1, q2 = (float("nan"), float("nan"))

    def label_row(dval: float) -> str:
        return taxonomy.strata_label_from_quantiles(float(dval), float(q1), float(q2))

    genus_df["taxonomy_stratum"] = genus_df["distance_to_staphylococcus"].apply(label_row)
    return genus_df, float(q1), float(q2)


def _merge_tax_columns(neg_pool: pd.DataFrame, genus_df: pd.DataFrame) -> pd.DataFrame:
    lk = genus_df.set_index("host_genus")
    def dist(h: str) -> float:
        return float(lk.loc[str(h), "distance_to_staphylococcus"])
    def stratum(h: str) -> str:
        return str(lk.loc[str(h), "taxonomy_stratum"])
    def resolv(h: str) -> str:
        return str(lk.loc[str(h), "resolution_method"])
    out = neg_pool.copy()
    out["tax_distance"] = out["host"].astype(str).apply(dist)
    out["tax_stratum"] = out["host"].astype(str).apply(stratum)
    out["tax_resolution"] = out["host"].astype(str).apply(resolv)
    return out


def _sample_stratified_negatives(
    neg: pd.DataFrame, target_neg: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    frac = config.NEG_STRATUM_FRACTIONS
    strata = ["near", "mid", "far", "unresolved"]
    raw = [int(round(target_neg * frac[s])) for s in strata]
    diff = target_neg - sum(raw)
    raw[strata.index("far")] += diff

    picked_idx: list[int] = []

    def take_from(stratum_label: str, quota: int) -> None:
        sub = neg.loc[neg["tax_stratum"] == stratum_label]
        if sub.empty:
            log.warning("No negatives in stratum %s", stratum_label)
            return
        grp = sub.groupby("host")
        gens = list(grp.groups.keys())
        rng.shuffle(gens)
        counts = {g: 0 for g in gens}
        chosen = 0
        rounds = 0
        max_rounds = quota * 15 + 400
        picked_set = set(picked_idx)

        while chosen < quota and rounds < max_rounds:
            moved = False
            for g in gens:
                if chosen >= quota:
                    break
                if counts[g] >= config.NEG_PER_GENUS_CAP:
                    continue
                cand = grp.get_group(g)
                avail = [int(i) for i in cand.index.to_list()]
                rng.shuffle(avail)
                for ix in avail:
                    if ix not in picked_set:
                        picked_idx.append(ix)
                        picked_set.add(ix)
                        counts[g] += 1
                        chosen += 1
                        moved = True
                        break
            if not moved:
                break
            rounds += 1

    for s, q in zip(strata, raw):
        take_from(s, q)

    picked_set = set(picked_idx)
    if len(picked_idx) < target_neg:
        leftover = [int(i) for i in neg.index if int(i) not in picked_set]
        rng.shuffle(leftover)
        need = target_neg - len(picked_idx)
        for ix in leftover[:need]:
            picked_idx.append(ix)

    out = neg.loc[picked_idx].drop_duplicates()
    if len(out) > target_neg:
        out = out.iloc[:target_neg]
    elif len(out) < target_neg:
        leftover = [int(i) for i in neg.index if int(i) not in set(out.index)]
        rng.shuffle(leftover)
        spill = neg.loc[leftover[: target_neg - len(out)]]
        out = pd.concat([out, spill])

    out = out.iloc[:target_neg]
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["label"] = 0
    return out


def build(seed: int = config.SEED) -> pd.DataFrame:

    df = load_metadata()
    df = _quality_filter(df)
    df["host"] = df.apply(_resolve_host, axis=1)
    df = df[~df["host"].isin(config.EXCLUDE_HOST_VALUES)].copy()

    cor_pos = df["host"].astype(str).str.startswith(config.TARGET_HOST, na=False).sum()
    corp_frac = float(cor_pos / len(df)) if len(df) else float("nan")
    corpus_stats = {
        "n_quality_filtered_phages": int(len(df)),
        "n_staphylococcus_host": int(cor_pos),
        "empirical_staphylococcus_fraction": corp_frac,
    }
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROC_DIR / "corpus_stats.json", "w") as fh:
        json.dump(corpus_stats, fh, indent=2)

    pos = df[df["host"].astype(str).str.startswith(config.TARGET_HOST, na=False)].copy()
    pos["label"] = 1
    pos["tax_distance"] = 0.0
    pos["tax_stratum"] = "target_genus"
    pos["tax_resolution"] = "positive_genus"

    neg_pool = df[~df["host"].astype(str).str.startswith(config.TARGET_HOST, na=False)].copy()
    neg_pool = neg_pool.reset_index(drop=True)
    log.info("Non-%s pool: %d host genera (%d rows)",
             config.TARGET_HOST, neg_pool["host"].nunique(), len(neg_pool))

    uniq_genera = sorted(neg_pool["host"].astype(str).unique())
    genus_df, q1, q2 = _taxonomy_table_for_genera(uniq_genera)
    genus_df.to_csv(config.PROC_DIR / "taxonomy_genera.csv", index=False)

    unresolved_n = int((genus_df["resolution_method"] == "unresolved").sum())
    strata_counts = genus_df.groupby("taxonomy_stratum").size()

    strata_summary = pd.DataFrame({
        "metric": list(strata_counts.index) + [
            "unique_genera_total", "unique_genera_unresolved",
            "neg_pool_rows", "distance_tertile_q1", "distance_tertile_q2",
            "empirical_staphylococcus_fraction_corpus",
        ],
        "value": list(strata_counts.values) + [
            len(genus_df), unresolved_n, len(neg_pool),
            q1, q2, corp_frac,
        ],
    })
    strata_summary.to_csv(config.PROC_DIR / "taxonomy_strata_summary.csv", index=False)

    neg_pool_t = _merge_tax_columns(neg_pool, genus_df)
    target_neg = len(pos)
    neg = _sample_stratified_negatives(neg_pool_t, target_neg, seed)

    if neg["host"].nunique() < config.MIN_GENERA_PER_NEG:
        raise RuntimeError(
            f"Negative genera diversity too low ({neg['host'].nunique()} < "
            f"{config.MIN_GENERA_PER_NEG}); adjust caps / filters."
        )

    keep_cols = [
        "accession", "description", "length", "gc",
        "family", "subfamily", "viral_genus",
        "lifestyle", "host", "label",
        "tax_distance", "tax_stratum", "tax_resolution",
    ]
    manifest = pd.concat([pos[keep_cols], neg[keep_cols]], ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path = config.PROC_DIR / "manifest.csv"
    manifest.to_csv(out_path, index=False)
    log.info(
        "Wrote manifest to %s (%d rows); unresolved genera=%d; neg strata counts: %s",
        out_path, len(manifest), unresolved_n,
        neg.groupby("tax_stratum").size().to_dict(),
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    build(seed=_parse_args().seed)
