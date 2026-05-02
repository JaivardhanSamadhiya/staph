"""Resolve bacterial host genus names to NCBI Taxonomy IDs and phylogenetic distance.

Supports a reproducible fallback chain for informal host strings commonly found
in INPHARED-derived metadata:

1. Exact string search in ``taxonomy`` followed by lineage fetch,
2. If empty and the first-whitespace fallback token is not a placeholder
   (blocklisted prefixes such as ``Candidatus``, ``uncultured``, …), retry on
   that token alone,
3. Else mark **unresolved** (distance / stratum unavailable).

Responses are cached at ``CACHE_PATH``. **Stale JSON from an older pipeline
revision (e.g. before the placeholder genus blocklist) corrupts phylogenetic
distances silently** — delete ``data/aux/taxonomy_resolve.json`` after such
upgrades before re-running dataset build.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .utils import get_logger

log = get_logger("taxonomy")

# Canonical NCBI Taxonomy genus ID for Staphylococcus (verified periodically).
REFERENCE_STAPHYLOCOCCUS_GENUS_TAXID = 1279

# First-whitespace-token fallback would map e.g. "Candidatus Xyz" → "Candidatus",
# which hits a meaningless placeholder taxon — skip truncation for these tokens.
GENUS_TRUNCATION_FIRST_TOKEN_BLOCKLIST = frozenset({
    "candidatus",
    "uncultured",
    "unclassified",
    "environmental",
})

CACHE_PATH = config.AUX_DIR / "taxonomy_resolve.json"
# IMPORTANT: Changing resolution/blocklist behaviour does **not** invalidate
# cached rows automatically. Delete ``taxonomy_resolve.json`` before the next
# ``build_dataset`` run if caches may contain stale ``Candidatus``/trunc hits.
RATE_LIMIT_SECONDS = float(os.environ.get("NCBI_ENTREZ_PAUSE", "0.35"))


def _entrez_tools() -> dict[str, str]:
    """NCBI Entrez etiquette: tool name + optional email."""
    return {
        "tool": os.environ.get("ENTREZ_TOOL", config.ENTREZ_TOOL),
        "email": os.environ.get("ENTREZ_EMAIL", config.ENTREZ_EMAIL),
    }


def _pause() -> None:
    if RATE_LIMIT_SECONDS > 0:
        time.sleep(RATE_LIMIT_SECONDS)


def esearch_taxonomy(term: str, retmax: int = 20) -> list[str]:
    params = urllib.parse.urlencode({
        **{"db": "taxonomy", "term": term, "retmode": "xml", "retmax": str(retmax)},
        **_entrez_tools(),
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    _pause()
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    ids: list[str] = []
    for id_el in root.findall(".//IdList/Id"):
        if id_el.text:
            ids.append(id_el.text)
    return ids


def efetch_lineage_taxids(tax_id: str) -> list[int]:
    """Return ordered lineage tax IDs from lineage root toward the queried taxon."""
    params = urllib.parse.urlencode({
        "db": "taxonomy",
        "id": tax_id,
        "retmode": "xml",
        **_entrez_tools(),
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    _pause()
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    lineage: list[int] = []
    lineage_ex = root.find(".//LineageEx")
    if lineage_ex is not None:
        for taxon in lineage_ex.findall("Taxon"):
            tid = taxon.find("TaxId")
            if tid is not None and tid.text:
                lineage.append(int(tid.text))
        return lineage

    lineage_text = root.find(".//Taxon/Lineage")
    if lineage_text is not None and lineage_text.text:
        for token in lineage_text.text.strip().split():
            if token.isdigit():
                lineage.append(int(token))
    return lineage


def lineage_distance_suffix(line_a: list[int], line_b: list[int]) -> int:
    """Symmetric tree distance assuming ``LineageEx`` IDs share root prefixes.

    Uses longest common prefix from the root-most side --- NCBI returns lineage
    ordered from ancestral superkingdom/domain toward the query taxon, with the
    query taxon's ID as the deepest entry.
    """
    m = 0
    for a, b in zip(line_a, line_b):
        if a != b:
            break
        m += 1
    # Remaining tail lengths after last shared ancestor contribute to distance.
    return (len(line_a) - m) + (len(line_b) - m)


def load_cache() -> dict[str, dict[str, Any]]:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


@dataclass
class ResolvedGenus:
    original: str
    query_used: str
    resolution: str  # exact_genus_hit | truncated_token | unresolved
    tax_id: int | None
    scientific_name_rank: str | None
    lineage_taxids: list[int] | None
    distance_to_staph: float | None  # nan -> unresolved


def _first_tax_hit_details(xml_bytes: bytes) -> tuple[int, str | None] | None:
    root = ET.fromstring(xml_bytes)
    tid_el = root.find(".//Taxon/TaxId")
    rank_el = root.find(".//Taxon/Rank")
    if tid_el is None or not tid_el.text:
        return None
    tid = int(tid_el.text)
    rank = rank_el.text if rank_el is not None else None
    return tid, rank


def _blocked_truncation_genus_token(token: str) -> bool:
    return token.strip().lower() in GENUS_TRUNCATION_FIRST_TOKEN_BLOCKLIST


def fetch_tax_xml(tax_id: str) -> bytes:
    params = urllib.parse.urlencode({
        "db": "taxonomy",
        "id": tax_id,
        "retmode": "xml",
        **_entrez_tools(),
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    _pause()
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def resolve_genus(
    genus_name: str, cache: dict[str, dict[str, Any]],
) -> ResolvedGenus:
    """Resolve a genus or species-like ``host_tax`` token to lineage + distance."""
    key_full = genus_name.strip()
    canonical = key_full.lower()

    cached = cache.get(canonical)
    if cached:
        lg = cached.get("lineage_taxids")
        dist_raw = cached.get("distance_to_staph")
        return ResolvedGenus(
            original=key_full,
            query_used=str(cached.get("query_used", key_full)),
            resolution=str(cached.get("resolution", "cache")),
            tax_id=int(cached["tax_id"]) if cached.get("tax_id") is not None else None,
            scientific_name_rank=cached.get("rank"),
            lineage_taxids=[int(x) for x in lg] if lg else None,
            distance_to_staph=(
                float(dist_raw) if dist_raw is not None else float("nan")
            ),
        )

    attempted: list[str] = []

    # ---- reference lineage (cached singleton key inside the mutable cache dict).
    ref_key = "__ref_lineage_Staphylococcus_genus"
    ref_lineage_entry = cache.get(ref_key)
    if not ref_lineage_entry:
        lineage_list = efetch_lineage_taxids(str(REFERENCE_STAPHYLOCOCCUS_GENUS_TAXID))
        cache[ref_key] = {
            "tax_id": REFERENCE_STAPHYLOCOCCUS_GENUS_TAXID,
            "lineage_taxids": lineage_list,
        }
        ref_lineage_entry = cache[ref_key]
    lineage_ref = [int(x) for x in ref_lineage_entry["lineage_taxids"]]

    parts = key_full.split()
    if len(parts) == 1 and parts[0] and _blocked_truncation_genus_token(parts[0]):
        cache[canonical] = {
            "tax_id": None,
            "query_used": parts[0],
            "resolution": "unresolved",
            "rank": None,
            "lineage_taxids": None,
            "distance_to_staph": None,
            "notes": "placeholder_host_token_whole_string",
        }
        return ResolvedGenus(
            original=key_full,
            query_used=parts[0],
            resolution="unresolved",
            tax_id=None,
            scientific_name_rank=None,
            lineage_taxids=None,
            distance_to_staph=float("nan"),
        )

    steps: list[tuple[str, str]] = [(key_full, "exact_genus_hit")]
    first_tok = parts[0] if parts else ""
    if (
        len(parts) > 1
        and first_tok
        and not _blocked_truncation_genus_token(first_tok)
    ):
        steps.append((first_tok, "truncated_token"))

    for query_txt, lab in steps:
        term = query_txt.strip()
        if term == "":
            continue
        attempted.append(term)
        ids = esearch_taxonomy(f"{term}[Scientific Name]")
        if not ids:
            ids = esearch_taxonomy(term)
        if not ids:
            continue
        best = sorted(ids, key=lambda x: int(x))[0]
        xml_hit = fetch_tax_xml(best)
        details = _first_tax_hit_details(xml_hit)
        lineage_host = efetch_lineage_taxids(best)
        if not lineage_host:
            continue
        dist = float(lineage_distance_suffix(lineage_ref, lineage_host))
        rn = details[1] if details else None
        cache[canonical] = {
            "tax_id": int(best),
            "query_used": term,
            "resolution": lab,
            "rank": rn,
            "lineage_taxids": lineage_host,
            "distance_to_staph": dist,
        }
        return ResolvedGenus(
            original=key_full,
            query_used=term,
            resolution=lab,
            tax_id=int(best),
            scientific_name_rank=rn,
            lineage_taxids=list(lineage_host),
            distance_to_staph=dist,
        )

    cache[canonical] = {
        "tax_id": None,
        "query_used": attempted[-1] if attempted else "",
        "resolution": "unresolved",
        "rank": None,
        "lineage_taxids": None,
        "distance_to_staph": None,
    }
    return ResolvedGenus(
        original=key_full,
        query_used=(attempted[-1] if attempted else ""),
        resolution="unresolved",
        tax_id=None,
        scientific_name_rank=None,
        lineage_taxids=None,
        distance_to_staph=float("nan"),
    )


def strata_label_from_quantiles(distance: float, q1: float, q2: float) -> str:
    if distance != distance:  # NaN
        return "unresolved"
    if not (q1 == q1 and q2 == q2):
        return "mid"
    if distance <= q1:
        return "near"
    if distance <= q2:
        return "mid"
    return "far"
