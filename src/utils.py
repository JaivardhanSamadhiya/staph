"""Shared utilities: logging, file helpers, plotting style."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

from . import config


def get_logger(name: str) -> logging.Logger:
    """Return a stream logger configured once per process."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def save_figure(fig, name: str, formats: Iterable[str] = config.FIG_FORMATS) -> list[Path]:
    """Save a matplotlib figure to every configured format under FIG_DIR."""
    out_paths: list[Path] = []
    for ext in formats:
        path = config.FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
        out_paths.append(path)
    return out_paths


def setup_plot_style() -> None:
    """Apply a clean, journal-friendly matplotlib style."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks", palette="deep")
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": config.FIG_DPI,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,   # editable text in vector output
            "ps.fonttype": 42,
        }
    )


def human_size(n_bytes: int) -> str:
    """Compact human-readable byte string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
