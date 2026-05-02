"""Generate publication-ready figures and a summary metrics report.

The pipeline produces six figures:

1. ``fig_dataset_summary``  — composition of the labelled corpus (host genera,
   genome length and GC distribution).
2. ``fig_roc_pr``           — ROC and Precision-Recall curves on the held-out
   test set comparing the four models.
3. ``fig_confusion``        — confusion matrices for the best model.
4. ``fig_feature_importance`` — top discriminative features from the random
   forest (Gini importances).
5. ``fig_umap``             — 2D UMAP projection of the canonical k-mer space
   colour-coded by host class.
6. ``fig_cv_performance``   — mean +/- std of the cross-validation metrics.

All figures are written as 300 dpi PDFs (vector) and PNGs (raster) to
``results/figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    precision_recall_curve, roc_curve, auc, average_precision_score,
)

from . import config
from .utils import get_logger, save_figure, setup_plot_style

log = get_logger("evaluate")

MODEL_ORDER = ["logreg", "random_forest", "gradient_boosting", "mlp"]
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "Neural Network (MLP)",
}
MODEL_COLORS = {
    "logreg": "#377eb8",
    "random_forest": "#4daf4a",
    "gradient_boosting": "#e41a1c",
    "mlp": "#984ea3",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_predictions() -> dict:
    p = np.load(config.PROC_DIR / "predictions.npz", allow_pickle=True)
    out = {k: p[k] for k in p.files}
    return out


def _load_features() -> dict:
    f = np.load(config.PROC_DIR / "features.npz", allow_pickle=True)
    return {k: f[k] for k in f.files}


def _load_test_metrics() -> dict:
    with open(config.METRIC_DIR / "test_metrics.json") as fh:
        return json.load(fh)


def _load_cv_results() -> pd.DataFrame:
    return pd.read_csv(config.METRIC_DIR / "cv_results.csv")


# ---------------------------------------------------------------------------
# Figure 1 - dataset summary
# ---------------------------------------------------------------------------

def fig_dataset_summary(manifest: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A: top host genera (negatives) + Staphylococcus
    counts = manifest["host"].value_counts()
    top = counts.head(15).copy()
    top["Other genera"] = counts.iloc[15:].sum()
    colors = [
        "#d7191c" if h.startswith(config.TARGET_HOST) else "#4575b4"
        for h in top.index
    ]
    axes[0].barh(top.index[::-1], top.values[::-1],
                 color=colors[::-1], edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("phages in corpus")
    axes[0].set_title(f"A. Host genus composition\n(positive = {config.TARGET_HOST})")

    # Panel B: genome length distribution
    sns.kdeplot(
        data=manifest, x="length", hue="label", common_norm=False,
        fill=True, palette={0: "#4575b4", 1: "#d7191c"},
        alpha=0.45, ax=axes[1], log_scale=True, linewidth=1.2,
    )
    axes[1].set_xlabel("genome length (bp, log scale)")
    axes[1].set_title("B. Genome length distribution")
    axes[1].get_legend().set_title("class")
    for t, lab in zip(axes[1].get_legend().get_texts(),
                      [f"non-{config.TARGET_HOST}", config.TARGET_HOST]):
        t.set_text(lab)

    # Panel C: GC content
    sns.kdeplot(
        data=manifest, x="gc", hue="label", common_norm=False,
        fill=True, palette={0: "#4575b4", 1: "#d7191c"},
        alpha=0.45, ax=axes[2], linewidth=1.2,
    )
    axes[2].set_xlabel("GC content")
    axes[2].set_title("C. GC content distribution")
    axes[2].get_legend().set_title("class")
    for t, lab in zip(axes[2].get_legend().get_texts(),
                      [f"non-{config.TARGET_HOST}", config.TARGET_HOST]):
        t.set_text(lab)

    fig.tight_layout()
    save_figure(fig, "fig_dataset_summary")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 - ROC / PR curves
# ---------------------------------------------------------------------------

def fig_roc_pr(preds: dict, test_metrics: dict) -> None:
    y = preds["y"]
    test_mask = preds["test_mask"]
    y_test = y[test_mask]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for name in MODEL_ORDER:
        proba = preds[f"test_{name}"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(
            fpr, tpr, color=MODEL_COLORS[name], lw=2,
            label=f"{MODEL_LABELS[name]} (AUC = {test_metrics[name]['roc_auc']:.3f})",
        )

        precision, recall, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(
            recall, precision, color=MODEL_COLORS[name], lw=2,
            label=f"{MODEL_LABELS[name]} (AP = {test_metrics[name]['pr_auc']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", alpha=0.7)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                xlim=(0, 1), ylim=(0, 1.02),
                title="A. Receiver-operating characteristic (held-out test)")
    axes[0].legend(loc="lower right")

    base_rate = float(np.mean(y_test))
    axes[1].axhline(base_rate, color="grey", lw=1, linestyle="--", alpha=0.7,
                    label=f"baseline = {base_rate:.2f}")
    axes[1].set(xlabel="recall", ylabel="precision",
                xlim=(0, 1), ylim=(0, 1.02),
                title="B. Precision-Recall curve (held-out test)")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    save_figure(fig, "fig_roc_pr")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 - confusion matrix for the best model
# ---------------------------------------------------------------------------

def fig_confusion(preds: dict, test_metrics: dict) -> str:
    """Plot the confusion matrix for the best test-AUC model. Returns the
    name of the chosen model."""
    best = max(test_metrics, key=lambda m: test_metrics[m]["roc_auc"])
    y = preds["y"]
    test_mask = preds["test_mask"]
    y_test = y[test_mask]
    proba = preds[f"test_{best}"]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[f"non-{config.TARGET_HOST}", config.TARGET_HOST],
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Confusion matrix on held-out test\n({MODEL_LABELS[best]})")
    fig.tight_layout()
    save_figure(fig, "fig_confusion")
    plt.close(fig)
    return best


# ---------------------------------------------------------------------------
# Figure 4 - feature importance from the random forest
# ---------------------------------------------------------------------------

def fig_feature_importance(top_n: int = 25) -> None:
    rf = joblib.load(config.MODEL_DIR / "random_forest.joblib")
    importances = rf.named_steps["clf"].feature_importances_
    feature_names = _load_features()["feature_names"]
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 0.32 * top_n + 0.8))
    palette = sns.color_palette("magma_r", n_colors=top_n)
    ax.barh(
        np.array(feature_names)[order][::-1],
        importances[order][::-1],
        color=palette[::-1],
        edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} discriminative features (Random Forest)")
    fig.tight_layout()
    save_figure(fig, "fig_feature_importance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 - UMAP of the k-mer feature space
# ---------------------------------------------------------------------------

def fig_umap(features: dict) -> None:
    import umap

    X = features["X"]
    feature_names = list(features["feature_names"])
    y = features["y"]
    host = features["host"]

    kmer_mask = np.array([n.startswith("kmer") for n in feature_names])
    Xk = X[:, kmer_mask]

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="cosine",
        random_state=config.SEED, n_jobs=1,
    )
    embedding = reducer.fit_transform(Xk)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(
        embedding[y == 0, 0], embedding[y == 0, 1],
        s=6, c="#4575b4", alpha=0.55, label=f"non-{config.TARGET_HOST}",
        edgecolors="none",
    )
    axes[0].scatter(
        embedding[y == 1, 0], embedding[y == 1, 1],
        s=6, c="#d7191c", alpha=0.7, label=config.TARGET_HOST,
        edgecolors="none",
    )
    axes[0].set(xlabel="UMAP-1", ylabel="UMAP-2",
                title="A. UMAP of k-mer profiles\n(coloured by class)")
    axes[0].legend(markerscale=2.5, loc="best")

    # Right panel: top 8 host genera + others
    counts = pd.Series(host).value_counts()
    top_hosts = counts.head(8).index.tolist()
    color_palette = sns.color_palette("tab10", n_colors=len(top_hosts))
    other_color = "#999999"
    plot_order = top_hosts + ["Other"]
    color_map = {h: color_palette[i] for i, h in enumerate(top_hosts)}
    color_map["Other"] = other_color
    host_label = np.where(np.isin(host, top_hosts), host, "Other")

    for h in plot_order:
        idx = host_label == h
        axes[1].scatter(
            embedding[idx, 0], embedding[idx, 1],
            s=6, c=[color_map[h]], alpha=0.7,
            label=f"{h} ({idx.sum()})", edgecolors="none",
        )
    axes[1].set(xlabel="UMAP-1", ylabel="UMAP-2",
                title="B. UMAP of k-mer profiles\n(coloured by host genus)")
    axes[1].legend(markerscale=2.5, fontsize=8, loc="best",
                   ncol=2, handletextpad=0.4, borderpad=0.2)

    fig.tight_layout()
    save_figure(fig, "fig_umap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 - cross-validation summary
# ---------------------------------------------------------------------------

def fig_cv_performance(cv_df: pd.DataFrame) -> None:
    metrics = ["roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy"]
    pretty = {
        "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
        "f1": "F1", "accuracy": "Accuracy",
        "balanced_accuracy": "Bal. Accuracy",
    }
    cv_df = cv_df[cv_df["metric"].isin(metrics)].copy()
    cv_df["metric_pretty"] = cv_df["metric"].map(pretty)
    cv_df["model_pretty"] = cv_df["model"].map(MODEL_LABELS)
    metric_order = [pretty[m] for m in metrics]
    model_order = [MODEL_LABELS[m] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 4.4))

    n_models = len(model_order)
    bar_width = 0.18
    x = np.arange(len(metric_order))
    for i, model in enumerate(model_order):
        sub = cv_df[cv_df["model_pretty"] == model].set_index("metric_pretty")
        means = sub.loc[metric_order, "mean"].values
        stds = sub.loc[metric_order, "std"].values
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width, yerr=stds,
            color=MODEL_COLORS[MODEL_ORDER[i]],
            edgecolor="black", linewidth=0.4, label=model,
            capsize=2.5,
        )
    ax.set_xticks(x, metric_order)
    ax.set_ylabel("score (mean ± SD across 5 group-stratified folds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-validation performance "
                 "(StratifiedGroupKFold on cluster groups)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_cv_performance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal markdown table renderer (avoids the optional ``tabulate`` dep)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(
            f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool)
            else str(v) for v in row
        ) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def write_summary_report(test_metrics: dict, cv_df: pd.DataFrame,
                         best_model: str) -> Path:
    out = config.METRIC_DIR / "summary_report.md"

    cv_pivot = cv_df.pivot(index="model", columns="metric", values="mean")
    cv_pivot = cv_pivot.loc[MODEL_ORDER]
    cv_pivot.index = [MODEL_LABELS[m] for m in cv_pivot.index]
    cv_pivot.index.name = "model"

    lines = [
        "# Phage host-prediction summary",
        "",
        f"Target host: **{config.TARGET_HOST}**",
        f"Best model on held-out test: **{MODEL_LABELS[best_model]}**",
        "",
        "## Cross-validation (StratifiedGroupKFold, 5-fold, cluster groups)",
        "",
        _df_to_markdown(cv_pivot.round(4)),
        "",
        "## Held-out test (cluster-disjoint, 20% of clusters)",
        "",
    ]

    test_rows = []
    for name in MODEL_ORDER:
        m = test_metrics[name]
        test_rows.append({
            "model": MODEL_LABELS[name],
            "ROC AUC": round(m["roc_auc"], 4),
            "PR AUC": round(m["pr_auc"], 4),
            "F1": round(m["f1"], 4),
            "Accuracy": round(m["accuracy"], 4),
            "Bal. Acc": round(m["balanced_accuracy"], 4),
        })
    lines.append(_df_to_markdown(pd.DataFrame(test_rows), index=False))
    lines.append("")
    lines.append(
        "_Threshold for binarising probabilities = 0.5; "
        "AUC and PR AUC are threshold-independent._"
    )

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run() -> None:
    setup_plot_style()
    manifest = pd.read_csv(config.PROC_DIR / "manifest_with_features.csv")
    preds = _load_predictions()
    features = _load_features()
    test_metrics = _load_test_metrics()
    cv_df = _load_cv_results()

    log.info("Generating dataset summary figure")
    fig_dataset_summary(manifest)

    log.info("Generating ROC / PR figure")
    fig_roc_pr(preds, test_metrics)

    log.info("Generating confusion matrix figure")
    best = fig_confusion(preds, test_metrics)
    log.info("Best model on test (by ROC AUC): %s", best)

    log.info("Generating feature-importance figure")
    fig_feature_importance()

    log.info("Generating UMAP figure")
    fig_umap(features)

    log.info("Generating CV performance figure")
    fig_cv_performance(cv_df)

    report = write_summary_report(test_metrics, cv_df, best)
    log.info("Wrote summary report to %s", report)
    log.info("All figures saved under %s", config.FIG_DIR)


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _parse_args()
    run()
