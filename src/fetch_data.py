"""Download INPHARED metadata and bulk genome FASTA from the Millard Lab S3.

INPHARED (Cook et al., 2021) curates monthly snapshots of every phage genome in
GenBank with consolidated host metadata, which is exactly the labelled corpus
we need for supervised host-range learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

from . import config
from .utils import get_logger, human_size

log = get_logger("fetch_data")


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Stream a URL to disk with a progress bar.  Skips if the file exists."""
    if dest.exists() and not force:
        log.info("Cached %s (%s)", dest.name, human_size(dest.stat().st_size))
        return dest

    log.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1 << 20  # 1 MiB
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name, leave=False,
        ) as bar:
            for piece in resp.iter_content(chunk_size=chunk):
                if piece:
                    fh.write(piece)
                    bar.update(len(piece))

    tmp.replace(dest)
    log.info("Saved %s (%s)", dest.name, human_size(dest.stat().st_size))
    return dest


def fetch_metadata(force: bool = False) -> Path:
    """Download the INPHARED metadata TSV (small, ~1 MB)."""
    return _download(config.INPHARED_META_URL, config.META_PATH, force=force)


def fetch_genomes(force: bool = False) -> tuple[Path, Path]:
    """Download both INPHARED genome FASTAs (RefSeq ~110 MB, GenBank ~530 MB).

    INPHARED splits its FASTA distribution into a clean RefSeq subset and the
    remainder of GenBank, and the metadata table contains accessions from
    both. We need both files to look up sequences for any accession.
    """
    refseq = _download(config.INPHARED_REFSEQ_URL, config.REFSEQ_FASTA_PATH, force=force)
    genbank = _download(config.INPHARED_GENBANK_URL, config.GENBANK_FASTA_PATH, force=force)
    return refseq, genbank


def run(force: bool = False) -> None:
    """Fetch every file in order of size so failures fail fast."""
    fetch_metadata(force=force)
    fetch_genomes(force=force)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="Re-download even if cached files are present")
    return p.parse_args()


if __name__ == "__main__":
    run(force=_parse_args().force)
