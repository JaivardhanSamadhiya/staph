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
* canonical k-mer frequencies for ``k = 2, 3, 4`` (178 dimensions)
  plus **dinucleotide odds ratios** ρ (10 dims, aligned with canonical 2-mers)

Non-ACGT characters
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


_CANONICAL = {k: _build_canonical_index(k) for k in config.KMER_SIZES}


# Pre-compute canonical k-mer representative strings once per ``k``.
def _canonical_reps_ordered(k: int) -> list[str]:
    bases = "ACGT"
    compact_map, n_canonical = _CANONICAL[k]
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
    return [seen[i] for i in range(n_canonical)]


_REPS_K2 = _canonical_reps_ordered(2)


def _mononucleotide_probs(seq_codes: np.ndarray) -> np.ndarray:
    valid = seq_codes >= 0
    if not np.any(valid):
        return np.full(4, 0.25, dtype=np.float64)
    cnt = np.array([int(np.sum(seq_codes[valid] == b)) for b in range(4)], dtype=np.float64)
    tot = cnt.sum()
    if tot <= 0:
        return np.full(4, 0.25, dtype=np.float64)
    return cnt / tot


def _dinucleotide_odds_ratios(obs2: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """ρ for each *canonical* dinucleotide (symmetrised under RC).

    ``expected`` pools the forward dinucleotide ``XY`` and its reverse complement
    ``(comp(Y) comp(X))`` under mononucleotide independence.
    """
    out = np.empty(len(_REPS_K2), dtype=np.float32)
    for i, s in enumerate(_REPS_K2):
        xa = "ACGT".index(s[0])
        xb = "ACGT".index(s[1])
        exp_bidir = probs[xa] * probs[xb] + probs[3 - xb] * probs[3 - xa]
        out[i] = float(obs2[i] / (exp_bidir + 1e-12))
    return out


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
    for k in config.KMER_SIZES:
        names.extend([f"kmer{k}_{rep}" for rep in _canonical_reps_ordered(k)])
    names.extend([f"dioratio2_{rep}" for rep in _REPS_K2])
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
        k2freq = None
        for k in config.KMER_SIZES:
            kfreq = _kmer_frequencies_fast(codes, k)
            feats[offset : offset + len(kfreq)] = kfreq
            if k == 2:
                k2freq = kfreq
            offset += len(kfreq)
        probs = _mononucleotide_probs(codes)
        assert k2freq is not None
        dior = _dinucleotide_odds_ratios(k2freq, probs)
        feats[offset : offset + len(dior)] = dior
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

    save_opts: dict = {
        "X": X,
        "y": y,
        "feature_names": np.array(feature_names),
        "accession": manifest["accession"].astype(str).to_numpy(),
        "host": manifest["host"].astype(str).to_numpy(),
    }
    for col in ("tax_distance", "tax_stratum", "tax_resolution"):
        if col in manifest.columns:
            save_opts[col] = manifest[col].to_numpy()

    out_npz = config.PROC_DIR / "features.npz"
    np.savez_compressed(out_npz, **save_opts)
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
