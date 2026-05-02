"""End-to-end pipeline: from raw INPHARED data to publication-ready figures.

Just run::

    python pipeline.py

to execute every stage. Use ``--only`` / ``--skip`` to run a subset, or
``--force`` to re-run a step whose outputs are already cached.

Training accepts ``--bootstrap-draws`` to override the bootstrap resample
count on the cluster-disjoint held-out split (see ``train_models.train``).
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

STEP_ORDER = ["fetch", "dataset", "features", "train", "evaluate"]

STEP_DESC = {
    "fetch": "Download INPHARED metadata + genome FASTAs",
    "dataset": "Build the labelled positive/negative corpus",
    "features": "Extract sequence features from genomes",
    "train": "Cluster, split, train and cross-validate models",
    "evaluate": "Generate publication-ready figures and metrics",
}


def _step_fn(name: str, force: bool, bootstrap_draws: int | None) -> None:
    if name == "fetch":
        fetch_data.run(force=force)
    elif name == "dataset":
        build_dataset.build()
    elif name == "features":
        extract_features.build()
    elif name == "train":
        train_models.train(bootstrap_draws=bootstrap_draws)
    elif name == "evaluate":
        evaluate.run()
    else:
        raise ValueError(f"unknown pipeline step {name}")


def _expected_outputs(step: str) -> list[Path]:
    """Files that, if present, mean a step is already done."""
    if step == "fetch":
        return [config.META_PATH, config.REFSEQ_FASTA_PATH, config.GENBANK_FASTA_PATH]
    if step == "dataset":
        return [
            config.PROC_DIR / "manifest.csv",
            config.PROC_DIR / "taxonomy_strata_summary.csv",
            config.PROC_DIR / "corpus_stats.json",
        ]
    if step == "features":
        return [
            config.PROC_DIR / "features.npz",
            config.PROC_DIR / "manifest_with_features.csv",
        ]
    if step == "train":
        return [
            config.PROC_DIR / "predictions.npz",
            config.METRIC_DIR / "test_metrics.json",
            config.METRIC_DIR / "cv_results.csv",
        ]
    if step == "evaluate":
        return [config.FIG_DIR / "fig_roc_pr.pdf", config.METRIC_DIR / "summary_report.md"]
    return []


def _step_done(step: str) -> bool:
    return all(p.exists() for p in _expected_outputs(step))


def run_pipeline(
    only: list[str] | None = None,
    skip: list[str] | None = None,
    force: list[str] | None = None,
    bootstrap_draws: int | None = None,
) -> None:
    only = only or []
    skip = skip or []
    force = force or []

    selected = STEP_ORDER if not only else only
    selected = [s for s in selected if s not in skip]

    bad = [s for s in selected if s not in STEP_ORDER]
    if bad:
        raise SystemExit(f"Unknown step(s): {bad}.  Valid: {STEP_ORDER}")

    overall_start = time.time()
    for name in selected:
        desc = STEP_DESC[name]
        force_step = name in force
        if not force_step and _step_done(name):
            log.info("[skip] %-9s — outputs already exist (%s)", name, desc)
            continue
        log.info("[run ] %-9s — %s", name, desc)
        t0 = time.time()
        _step_fn(name, force_step, bootstrap_draws)
        log.info("[done] %-9s — %.1fs", name, time.time() - t0)

    log.info("Total wall-time: %.1fs", time.time() - overall_start)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        nargs="+",
        choices=STEP_ORDER,
        help="run only the listed steps (default: all)",
    )
    p.add_argument("--skip", nargs="+", choices=STEP_ORDER, default=[],
                   help="skip the listed steps")
    p.add_argument("--force", nargs="+", choices=STEP_ORDER, default=[],
                   help="re-run the listed steps even if outputs exist")
    p.add_argument(
        "--bootstrap-draws",
        type=int,
        default=None,
        metavar="N",
        help="passed through to ``train``: override cluster-bootstrap draw count",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_pipeline(
            only=list(args.only) if args.only else None,
            skip=args.skip,
            force=args.force,
            bootstrap_draws=args.bootstrap_draws,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
