"""Generate publication-ready figures and a summary metrics report.

The pipeline produces seven figures:

1. ``fig_dataset_summary``  — composition of the labelled corpus (host genera,
   genome length and GC distribution).
2. ``fig_roc_pr``           — ROC and Precision-Recall curves on the held-out
   test set comparing the four models.
3. ``fig_confusion``        — confusion matrices for the primary screening model
   (default threshold vs recall-target threshold, with corpus calibration prevalence).
4. ``fig_feature_importance`` — top discriminative features from the random
   forest (Gini importances).
5. ``fig_umap``             — 2D UMAP projection of the canonical k-mer space
   colour-coded by host class.
6. ``fig_cv_performance``   — mean +/- std of the cross-validation metrics.
7. ``fig_roc_pr_stratum_near`` — primary model ROC / PR restricted to positives
   + **near** phylogenetic strata negatives vs **all** negatives (held-out).

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
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
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
    with open(config.METRIC_DIR / "test_metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


def _metrics_by_model(blob: dict) -> dict:
    """Support both new nested JSON and legacy flat per-model dicts."""
    mm = blob.get("metrics_by_model")
    if mm is not None:
        return mm
    return {k: v for k, v in blob.items() if k in MODEL_ORDER}


def _load_cv_results() -> pd.DataFrame:
    return pd.read_csv(config.METRIC_DIR / "cv_results.csv")


def _strata_near_genera_corpus_count() -> int | None:
    """Unique negative genera in ``near'' tertile (corpus taxonomy table)."""
    path = Path(config.PROC_DIR) / "taxonomy_strata_summary.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        hit = df.loc[df["metric"].astype(str) == "near", "value"]
        if hit.empty:
            return None
        return int(hit.iloc[0])
    except (OSError, ValueError):
        return None


def _load_prevalence_eval() -> dict | None:
    path = Path(config.METRIC_DIR) / "prevalence_eval.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        log.warning("Could not read prevalence_eval.json; skipping prevalence appendix.")
        return None


def _prevalence_metric_cell(row: dict, stem: str) -> str:
    """Scalar from without-replacement rows, else ``mean ± std`` MC aggregates."""
    v = row.get(stem)
    if isinstance(v, (int, float)) and v == v and not stem.endswith("__mean"):
        return f"{float(v):.4f}"
    mn = row.get(f"{stem}__mean")
    sd = row.get(f"{stem}__std")
    if isinstance(mn, (int, float)) and mn == mn:
        out = f"{float(mn):.4f}"
        if isinstance(sd, (int, float)) and sd == sd:
            out = f"{out} ± {float(sd):.4f}"
        return out
    return "—"


def _prevalence_table_markdown(primary: str, prev_blob: dict) -> list[str]:
    per_model = prev_blob.get("per_model") or {}
    prow = per_model.get(primary)
    if not prow:
        return [
            "## Synthetic prevalence (held-out test)",
            "",
            f"_No rows for primary model `{primary}` in prevalence_eval.json._",
            "",
        ]
    tbl_rows = []
    for row in prow:
        pi_t = row.get("prevalence_target")
        samp = row.get("sampling", "?")
        ach = row.get("prevalence_achieved_in_mixture")
        if ach is None or ach != ach:
            ach = row.get("approx_prevalence_in_each_draw")
        ach_s = (
            f"{float(ach):.4f}"
            if isinstance(ach, (int, float)) and ach == ach
            else "—"
        )
        try:
            piv = float(pi_t)
        except (TypeError, ValueError):
            piv = pi_t
        tbl_rows.append({
            "pi_target": piv,
            "mix_pi": ach_s,
            "sampling": str(samp)[:56] + ("…" if len(str(samp)) > 56 else ""),
            "prec @0.5": _prevalence_metric_cell(row, "default_0.5__precision"),
            "prec @ theta": _prevalence_metric_cell(row, "recall_target__precision"),
            "F1 @0.5": _prevalence_metric_cell(row, "default_0.5__f1"),
            "F1 @ theta": _prevalence_metric_cell(row, "recall_target__f1"),
        })
    return [
        "## Synthetic prevalence (held-out test only)",
        "",
        "Cells show **mean ± std** where negative resampling uses Monte Carlo draws **with "
        "replacement** (pool too small for requested π_target). Otherwise values are "
        "from a single without-replacement draw (held-out negatives only).",
        "",
        _df_to_markdown(pd.DataFrame(tbl_rows), index=False),
        "",
    ]


def _load_corpus_calibration_fraction() -> tuple[float | None, str]:
    """Human-readable INPHARED empirical prevalence for figure captions."""
    path = Path(config.PROC_DIR) / "corpus_stats.json"
    if not path.exists():
        return None, "unavailable"
    try:
        with path.open(encoding="utf-8") as fh:
            blob = json.load(fh)
        pi_raw = blob.get("empirical_staphylococcus_fraction")
        if pi_raw is None:
            return None, "unavailable"
        pi = float(pi_raw)
        if not (pi == pi):  # NaN
            return None, "unavailable"
        return pi, f"{pi * 100:.2f}"
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None, "unavailable"


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

def fig_roc_pr(preds: dict, test_blob: dict) -> None:
    y = preds["y"]
    test_mask = preds["test_mask"]
    y_test = y[test_mask]
    mm = _metrics_by_model(test_blob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for name in MODEL_ORDER:
        proba = preds[f"test_{name}"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        block = mm[name]
        sub = block.get("threshold_default_0.5", block)
        roc_auc = float(sub["roc_auc"])
        pr_auc = float(sub["pr_auc"])
        axes[0].plot(
            fpr, tpr, color=MODEL_COLORS[name], lw=2,
            label=f"{MODEL_LABELS[name]} (AUC = {roc_auc:.3f})",
        )

        precision, recall, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(
            recall, precision, color=MODEL_COLORS[name], lw=2,
            label=f"{MODEL_LABELS[name]} (AP = {pr_auc:.3f})",
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


def fig_roc_pr_stratum_near(preds: dict, test_blob: dict) -> None:
    """Primary model: ROC/PR on ALL test rows vs positives + NEAR strata only."""
    mm = _metrics_by_model(test_blob)
    primary = test_blob.get("primary_model")
    if primary is None:
        primary = max(mm, key=lambda m: mm[m]["threshold_default_0.5"]["roc_auc"])

    ts_raw = preds.get("tax_stratum")
    if ts_raw is None:
        log.warning("Missing ``tax_stratum`` in predictions.npz — skip strata ROC/PR.")
        return

    test_mask = preds["test_mask"].astype(bool)
    y_te = preds["y"].astype(np.int8)[test_mask]
    tax_te = np.asarray(ts_raw).astype(str)[test_mask]
    proba = preds[f"test_{primary}"].astype(np.float64)

    n_pos = int((y_te == 1).sum())
    near_neg_n = int(((y_te == 0) & (tax_te == "near")).sum())
    if n_pos == 0 or near_neg_n == 0:
        log.warning(
            "Skipping strata ROC/PR: need positives and negatives in stratum ''near'' "
            "(n_pos=%d, n_near_neg=%d).",
            n_pos,
            near_neg_n,
        )
        return
    if near_neg_n < 5:
        log.warning(
            "Few held-out ''near'' negatives (%d); strata ROC will be coarse.",
            near_neg_n,
        )

    ridx = np.flatnonzero((y_te == 1) | ((y_te == 0) & (tax_te == "near")))
    y_rest = y_te[ridx].astype(np.int8)
    n_tot_test = int(y_te.shape[0])
    n_subset = int(len(ridx))

    roc_full = roc_auc_score(y_te, proba) if np.unique(y_te).size > 1 else float("nan")
    pr_full = average_precision_score(y_te, proba)
    roc_r = roc_auc_score(y_rest, proba[ridx]) if np.unique(y_rest).size > 1 else float("nan")
    pr_r = average_precision_score(y_rest, proba[ridx])

    fp_f, tp_f, _ = roc_curve(y_te, proba)
    fp_r, tp_r, _ = roc_curve(y_rest, proba[ridx])
    pref_f, rec_f, _ = precision_recall_curve(y_te, proba)
    pref_r, rec_r, _ = precision_recall_curve(y_rest, proba[ridx])

    corp_near_genera_n = _strata_near_genera_corpus_count()

    clr = MODEL_COLORS[primary]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))

    n_neg_all = int((y_te == 0).sum())
    axes[0].plot(
        fp_f, tp_f, color=clr, lw=2.2, linestyle="-",
        label=(
            f"All strata (n_tot={n_tot_test}; "
            f"n_pos={n_pos}, n_neg={n_neg_all}); "
            f"ROC AUC={roc_full:.3f}"
        ),
    )
    axes[0].plot(
        fp_r, tp_r, color=clr, lw=2.2, linestyle="--",
        label=(
            f"Pos + ''near'' stratum negatives (N_mix={n_subset}; "
            f"n_pos={n_pos}, n_neg_near={near_neg_n}); ROC AUC={roc_r:.3f}"
        ),
    )
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, alpha=0.7)
    axes[0].set(
        xlabel="false positive rate",
        ylabel="true positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=(
            f"A. ROC — primary ({MODEL_LABELS[primary]})\n"
            "Held-out positives + negatives (full pool vs positives + tertile ''near'' only)"
        ),
    )
    axes[0].legend(loc="lower right", fontsize=6.9)

    axes[1].plot(
        rec_f, pref_f, color=clr, lw=2.2, linestyle="-",
        label=f"All negatives (AP={pr_full:.3f}); n_tot={n_tot_test}",
    )
    axes[1].plot(
        rec_r, pref_r, color=clr, lw=2.2, linestyle="--",
        label=f"''Near'' stratum negatives (AP={pr_r:.3f}); N_mix={n_subset}",
    )
    axes[1].axhline(float(np.mean(y_te)), color="grey", linestyle="--", lw=1, alpha=0.7,
                    label=f"prev. (all) = {float(np.mean(y_te)):.2f}")
    sparse_note = ""
    if near_neg_n < 20:
        sparse_note = (
            f"WARNING: sparse held-out ''near'' negatives (n={near_neg_n} < 20); "
            "dashed traces are exploratory. "
        )
    corp_note = ""
    if corp_near_genera_n is not None:
        corp_note = (
            f"Corpus negatives with ''near'' genus tertile: ~{corp_near_genera_n} unique genera "
            "(taxonomy_strata_summary.csv). "
        )
    axes[1].set(
        xlabel="recall",
        ylabel="precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=(
            "B. Precision-recall\n(full held-out negatives vs positives + ''near'' only)"
            + ("\n" + sparse_note.rstrip() if sparse_note else "")
        ),
    )
    axes[1].legend(loc="upper right", fontsize=6.9)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    footer = (
        f"Dashed ROC/PR positives + ''near'' stratum negatives: N_mix={n_subset} "
        f"(n_pos={n_pos}, n_neg_near={near_neg_n}); solid traces use full "
        f"held-out split n={n_tot_test} (n_pos={n_pos}, n_neg_total={n_neg_all}). "
        f"{corp_note}"
    ).strip()
    fig.text(
        0.5,
        0.015,
        footer,
        ha="center",
        va="bottom",
        fontsize=7.9,
        wrap=False,
    )
    save_figure(fig, "fig_roc_pr_stratum_near")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 - confusion matrix for the best model
# ---------------------------------------------------------------------------

def fig_confusion(preds: dict, test_blob: dict) -> str:
    """Confusion matrices for the **primary** model: default vs recall-target thresholds."""
    mm = _metrics_by_model(test_blob)
    recall_targets = test_blob.get("recall_targets", {})
    primary = test_blob.get("primary_model")
    if primary is None:
        primary = max(mm, key=lambda m: mm[m]["threshold_default_0.5"]["roc_auc"])

    thr = float(recall_targets.get(primary, 0.5))
    y = preds["y"]
    test_mask = preds["test_mask"]
    y_test = y[test_mask]
    proba = preds[f"test_{primary}"]
    pred05 = (proba >= 0.5).astype(int)
    pred_rec = (proba >= thr).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    pi_num, pi_pct_plain = _load_corpus_calibration_fraction()
    if pi_num is None:
        cal_rest = "π_cal unavailable (see corpus_stats.json)"
    else:
        cal_rest = f"π_cal = {pi_pct_plain}% INPHARED empirical prevalence"

    title_left = (
        "θ = 0.5000\n"
        "(default diagnostic binarisation; prevalence not calibrated)"
    )
    title_right = (
        f"θ = {thr:.4f}\n"
        f"(OOF {float(config.TARGET_RECALL_OOF) * 100:g}% recall target; "
        f"{cal_rest})"
    )

    cms = [(title_left, pred05), (title_right, pred_rec)]

    labels = [f"non-{config.TARGET_HOST}", config.TARGET_HOST]
    for ax, (tit, preds_bin) in zip(axes.flat, cms):
        cm = confusion_matrix(y_test, preds_bin)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
        ax.set_title(tit, fontsize=10)

    fig.suptitle(
        f"Held-out test — {MODEL_LABELS[primary]} (primary screening model)",
        y=1.06,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, "fig_confusion")
    plt.close(fig)
    return primary


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

    kmer_mask = np.array([str(n).startswith("kmer") for n in feature_names])
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


def write_summary_report(test_blob: dict, cv_df: pd.DataFrame,
                         primary_model: str) -> Path:
    out = config.METRIC_DIR / "summary_report.md"

    mm = _metrics_by_model(test_blob)
    recall_targets = test_blob.get("recall_targets", {})

    cv_pivot = cv_df.pivot(index="model", columns="metric", values="mean")
    cv_pivot = cv_pivot.loc[MODEL_ORDER]
    cv_pivot.index = [MODEL_LABELS[m] for m in cv_pivot.index]
    cv_pivot.index.name = "model"

    lines = [
        "# Phage host-prediction summary",
        "",
        f"Target host: **{config.TARGET_HOST}**",
        f"Primary model (max CV ROC-AUC): **{MODEL_LABELS[primary_model]}**",
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
        m0 = mm[name]["threshold_default_0.5"]
        rt = float(recall_targets.get(name, 0.5))
        m1 = mm[name][f"threshold_recall_{config.TARGET_RECALL_OOF:.2f}"]
        test_rows.append({
            "model": MODEL_LABELS[name],
            "ROC AUC": round(m0["roc_auc"], 4),
            "PR AUC": round(m0["pr_auc"], 4),
            "F1 @0.5": round(m0["f1"], 4),
            "F1 @θ": round(m1["f1"], 4),
            "recall @θ": round(m1["recall"], 4),
            "θ (OOF target recall)": round(rt, 4),
        })
    lines.append(_df_to_markdown(pd.DataFrame(test_rows), index=False))
    lines.append("")
    lines.append(
        f"_θ is the highest probability threshold on training OOF scores that attains "
        f"recall ≥ {config.TARGET_RECALL_OOF:.2f} on known positives. "
        "ROC AUC and PR AUC are threshold-independent (full test set)._"
    )
    lines.append("")

    prev_blob = _load_prevalence_eval()
    if prev_blob is not None:
        lines.extend(_prevalence_table_markdown(primary_model, prev_blob))

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
    test_blob = _load_test_metrics()
    cv_df = _load_cv_results()

    log.info("Generating dataset summary figure")
    fig_dataset_summary(manifest)

    log.info("Generating ROC / PR figure")
    fig_roc_pr(preds, test_blob)

    log.info("Generating phylogenetic-strata ROC / PR (near vs full negatives)")
    fig_roc_pr_stratum_near(preds, test_blob)

    log.info("Generating confusion matrix figure")
    primary = fig_confusion(preds, test_blob)
    log.info("Primary model: %s", primary)

    log.info("Generating feature-importance figure")
    fig_feature_importance()

    log.info("Generating UMAP figure")
    fig_umap(features)

    log.info("Generating CV performance figure")
    fig_cv_performance(cv_df)

    report = write_summary_report(test_blob, cv_df, primary)
    log.info("Wrote summary report to %s", report)
    log.info("All figures saved under %s", config.FIG_DIR)


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _parse_args()
    run()
