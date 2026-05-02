"""End-to-end pipeline: from raw INPHARED data to publication-ready figures.

Just run::

    python pipeline.py

to execute every stage. Use ``--only`` / ``--skip`` to run a subset, or
``--force`` to re-run a step whose outputs are already cached.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import build_dataset, evaluate, extract_features, fetch_data, train_models
from src import config
from src.utils import get_logger

log = get_logger("pipeline")

# Ordered list of (name, callable, description) tuples. Each step caches its
# outputs to disk so re-running is cheap.
STEPS = [
    (
        "fetch",
        lambda force: fetch_data.run(force=force),
        "Download INPHARED metadata + genome FASTAs",
    ),
    (
        "dataset",
        lambda force: build_dataset.build(),
        "Build the labelled positive/negative corpus",
    ),
    (
        "features",
        lambda force: extract_features.build(),
        "Extract sequence features from genomes",
    ),
    (
        "train",
        lambda force: train_models.train(),
        "Cluster, split, train and cross-validate models",
    ),
    (
        "evaluate",
        lambda force: evaluate.run(),
        "Generate publication-ready figures and metrics",
    ),
]
STEP_NAMES = [s[0] for s in STEPS]


def _expected_outputs(step: str) -> list[Path]:
    """Files that, if present, mean a step is already done."""
    if step == "fetch":
        return [config.META_PATH, config.REFSEQ_FASTA_PATH, config.GENBANK_FASTA_PATH]
    if step == "dataset":
        return [config.PROC_DIR / "manifest.csv"]
    if step == "features":
        return [config.PROC_DIR / "features.npz",
                config.PROC_DIR / "manifest_with_features.csv"]
    if step == "train":
        return [config.PROC_DIR / "predictions.npz",
                config.METRIC_DIR / "test_metrics.json",
                config.METRIC_DIR / "cv_results.csv"]
    if step == "evaluate":
        return [config.FIG_DIR / "fig_roc_pr.pdf",
                config.METRIC_DIR / "summary_report.md"]
    return []


def _step_done(step: str) -> bool:
    return all(p.exists() for p in _expected_outputs(step))


def run_pipeline(only: list[str] | None = None,
                 skip: list[str] | None = None,
                 force: list[str] | None = None) -> None:
    only = only or []
    skip = skip or []
    force = force or []

    selected = STEP_NAMES if not only else only
    selected = [s for s in selected if s not in skip]

    bad = [s for s in selected if s not in STEP_NAMES]
    if bad:
        raise SystemExit(f"Unknown step(s): {bad}.  Valid: {STEP_NAMES}")

    overall_start = time.time()
    for name, fn, desc in STEPS:
        if name not in selected:
            continue
        force_step = name in force
        if not force_step and _step_done(name):
            log.info("[skip] %-9s — outputs already exist (%s)", name, desc)
            continue
        log.info("[run ] %-9s — %s", name, desc)
        t0 = time.time()
        fn(force=force_step)
        log.info("[done] %-9s — %.1fs", name, time.time() - t0)

    log.info("Total wall-time: %.1fs", time.time() - overall_start)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=STEP_NAMES,
                   help="run only the listed steps (default: all)")
    p.add_argument("--skip", nargs="+", choices=STEP_NAMES, default=[],
                   help="skip the listed steps")
    p.add_argument("--force", nargs="+", choices=STEP_NAMES, default=[],
                   help="re-run the listed steps even if outputs exist")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_pipeline(only=args.only, skip=args.skip, force=args.force)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
