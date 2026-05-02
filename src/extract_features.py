"""Compute purely sequence-derived features for every phage in the manifest.

We avoid annotation-dependent features (predicted ORFs, protein domains,
codon usage on coding regions) so that the pipeline only requires the raw
nucleotide sequence and is robust to the lifestyle / annotation status of
each phage.

Features per genome
-------------------
* ``length_log10``                 - log10 of genome length in bp
* ``gc_content``, ``gc_skew``,
  ``at_skew``                      - global compositional features
* canonical k-mer frequencies for
  ``k = 2, 3, 4``                  - 10 + 32 + 136 = 178 features

Canonical k-mers collapse a k-mer and its reverse complement to a single
feature, which is the right normalisation for double-stranded DNA where the
strand orientation of the assembly is arbitrary. Non-ACGT characters
(``N``, ``R``, ``Y``, ...) reset the k-mer window so that ambiguous bases
never contribute to any count.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .utils import get_logger

log = get_logger("extract_features")


# ---------------------------------------------------------------------------
# K-mer machinery
# ---------------------------------------------------------------------------

# Map ACGT -> 0..3, anything else -> -1 (acts as a window reset signal).
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for base, code in zip(b"ACGT", range(4)):
    _BASE_LUT[base] = code
for base, code in zip(b"acgt", range(4)):
    _BASE_LUT[base] = code


def _build_canonical_index(k: int) -> tuple[np.ndarray, int]:
    """Return ``(map, n_canonical)`` such that ``map[kmer_id]`` is the index
    of the canonical (lexicographically smaller of kmer / rc(kmer)) form in
    the compact feature vector.
    """
    n = 4 ** k
    rc = np.empty(n, dtype=np.int64)
    for i in range(n):
        v = i
        rc_i = 0
        for _ in range(k):
            base = v & 0b11
            rc_i = (rc_i << 2) | (3 - base)
            v >>= 2
        rc[i] = rc_i
    canonical_id = np.minimum(np.arange(n), rc)
    unique_canonical = np.unique(canonical_id)
    remap = {old: new for new, old in enumerate(unique_canonical)}
    compact = np.array([remap[c] for c in canonical_id], dtype=np.int64)
    return compact, len(unique_canonical)


# Pre-compute lookup tables once at import time.
_CANONICAL = {k: _build_canonical_index(k) for k in config.KMER_SIZES}


def _kmer_frequencies(seq_codes: np.ndarray, k: int) -> np.ndarray:
    """Compute canonical k-mer frequencies for an integer-encoded sequence."""
    compact_map, n_canonical = _CANONICAL[k]
    n = len(seq_codes)
    if n < k:
        return np.zeros(n_canonical, dtype=np.float32)

    # Build k-mer integer ids using a rolling base-4 representation. Whenever
    # an ambiguous base is encountered (-1) we reset the window so the
    # ambiguous position never contaminates any k-mer.
    counts = np.zeros(n_canonical, dtype=np.int64)
    valid_in_window = 0
    kmer_id = 0
    mask = (1 << (2 * k)) - 1

    for code in seq_codes:
        if code < 0:
            valid_in_window = 0
            kmer_id = 0
            continue
        kmer_id = ((kmer_id << 2) | int(code)) & mask
        valid_in_window += 1
        if valid_in_window >= k:
            counts[compact_map[kmer_id]] += 1

    total = counts.sum()
    if total == 0:
        return np.zeros(n_canonical, dtype=np.float32)
    return (counts / total).astype(np.float32)


# Vectorised k-mer counter that processes one phage in a few milliseconds
# even for genomes hundreds of kb long.
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402


def _kmer_frequencies_fast(seq_codes: np.ndarray, k: int) -> np.ndarray:
    compact_map, n_canonical = _CANONICAL[k]
    n = len(seq_codes)
    if n < k:
        return np.zeros(n_canonical, dtype=np.float32)

    counts = np.zeros(n_canonical, dtype=np.int64)
    valid = seq_codes >= 0

    # Find contiguous runs of valid bases.
    diff = np.diff(np.concatenate(([0], valid.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    weights = (1 << (2 * np.arange(k - 1, -1, -1))).astype(np.int64)
    for start, end in zip(starts, ends):
        if end - start < k:
            continue
        run = seq_codes[start:end].astype(np.int64)
        windows = sliding_window_view(run, k)
        kmer_ids = windows @ weights
        canon = compact_map[kmer_ids]
        np.add.at(counts, canon, 1)

    total = counts.sum()
    if total == 0:
        return np.zeros(n_canonical, dtype=np.float32)
    return (counts / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Compositional features
# ---------------------------------------------------------------------------

def _compositional_features(seq_codes: np.ndarray) -> tuple[float, float, float, int]:
    valid = seq_codes >= 0
    a = int(np.sum(seq_codes == 0))
    c = int(np.sum(seq_codes == 1))
    g = int(np.sum(seq_codes == 2))
    t = int(np.sum(seq_codes == 3))
    n_valid = a + c + g + t
    if n_valid == 0:
        return 0.0, 0.0, 0.0, 0
    gc = (g + c) / n_valid
    gc_skew = (g - c) / (g + c) if (g + c) else 0.0
    at_skew = (a - t) / (a + t) if (a + t) else 0.0
    return float(gc), float(gc_skew), float(at_skew), int(valid.sum())


# ---------------------------------------------------------------------------
# FASTA streaming
# ---------------------------------------------------------------------------

def _iter_fasta(paths: list[Path]) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(accession, encoded_sequence)`` tuples from a list of gzipped
    FASTA files. The sequence is encoded as an int8 array (ACGT -> 0..3,
    others -> -1)."""
    for path in paths:
        log.info("Streaming %s", path.name)
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            current_acc: str | None = None
            chunks: list[bytes] = []
            for line in fh:
                if line.startswith(">"):
                    if current_acc is not None:
                        seq_bytes = b"".join(chunks)
                        codes = _BASE_LUT[np.frombuffer(seq_bytes, dtype=np.uint8)]
                        yield current_acc, codes
                    current_acc = line[1:].strip().split()[0]
                    chunks = []
                else:
                    chunks.append(line.strip().encode("ascii", errors="replace"))
            if current_acc is not None:
                seq_bytes = b"".join(chunks)
                codes = _BASE_LUT[np.frombuffer(seq_bytes, dtype=np.uint8)]
                yield current_acc, codes


# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------

def _feature_names() -> list[str]:
    names = ["length_log10", "gc_content", "gc_skew", "at_skew"]
    bases = "ACGT"
    for k in config.KMER_SIZES:
        compact_map, _ = _CANONICAL[k]
        # Recover one representative k-mer string per canonical class.
        seen: dict[int, str] = {}
        for idx in range(4 ** k):
            cid = int(compact_map[idx])
            if cid in seen:
                continue
            kmer_chars = []
            v = idx
            for _ in range(k):
                kmer_chars.append(bases[v & 0b11])
                v >>= 2
            seen[cid] = "".join(reversed(kmer_chars))
        names.extend([f"kmer{k}_{seen[i]}" for i in range(len(seen))])
    return names


def build(manifest_path: Path | None = None) -> Path:
    manifest_path = manifest_path or (config.PROC_DIR / "manifest.csv")
    manifest = pd.read_csv(manifest_path)
    wanted = set(manifest["accession"].astype(str).tolist())
    log.info("Manifest contains %d unique accessions", len(wanted))

    feature_names = _feature_names()
    n_features = len(feature_names)

    found: dict[str, np.ndarray] = {}
    pbar = tqdm(total=len(wanted), desc="features", leave=False)

    for acc, codes in _iter_fasta(list(config.GENOME_FASTA_PATHS)):
        if acc not in wanted or acc in found:
            continue
        gc, gc_skew, at_skew, n_valid = _compositional_features(codes)
        if n_valid < config.MIN_GENOME_LEN:
            continue
        feats = np.empty(n_features, dtype=np.float32)
        feats[0] = float(np.log10(max(n_valid, 1)))
        feats[1] = gc
        feats[2] = gc_skew
        feats[3] = at_skew
        offset = 4
        for k in config.KMER_SIZES:
            kfreq = _kmer_frequencies_fast(codes, k)
            feats[offset : offset + len(kfreq)] = kfreq
            offset += len(kfreq)
        found[acc] = feats
        pbar.update(1)
        if len(found) == len(wanted):
            break
    pbar.close()

    missing = wanted - set(found)
    if missing:
        log.warning("%d / %d accessions had no FASTA match (dropping)",
                    len(missing), len(wanted))

    keep_mask = manifest["accession"].astype(str).isin(found)
    manifest = manifest.loc[keep_mask].reset_index(drop=True)
    X = np.stack([found[a] for a in manifest["accession"].astype(str).tolist()])
    y = manifest["label"].to_numpy().astype(np.int8)

    out_npz = config.PROC_DIR / "features.npz"
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        feature_names=np.array(feature_names),
        accession=manifest["accession"].astype(str).to_numpy(),
        host=manifest["host"].astype(str).to_numpy(),
    )
    out_manifest = config.PROC_DIR / "manifest_with_features.csv"
    manifest.to_csv(out_manifest, index=False)

    log.info("Features: %d phages x %d features -> %s",
             X.shape[0], X.shape[1], out_npz)
    return out_npz


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    return p.parse_args()


if __name__ == "__main__":
    _parse_args()
    build()
