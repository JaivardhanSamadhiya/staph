"""Train and cross-validate phage host-prediction models with leakage control.

Why this is non-trivial
-----------------------
A naive random train/test split on phage genomes leaks information badly:
many INPHARED entries are near-duplicates that come from the same outbreak
isolate, the same SEA-PHAGES cluster, or the same patient. A model can
trivially memorise these and return inflated test metrics.

Mitigation
~~~~~~~~~~
1. Compute pairwise cosine *similarity* between every pair of phages using
   their canonical k-mer feature vectors (already a strong proxy for
   sequence identity). High similarity (>= 1 - ``CLUSTER_DISTANCE``) implies
   the two phages should not be split across train and test.
2. Apply ``AgglomerativeClustering`` with a cosine distance threshold to
   collapse near-duplicates into a single ``cluster_id``.
3. Hold out a stratified set of *clusters* (not rows) as the final test
   set. The remaining clusters are used for ``GroupKFold`` cross-validation
   during model selection.
4. Report two complementary metrics: stratified-by-cluster CV AUC (used for
   model selection) and held-out test AUC (the unbiased estimate).

Models compared
~~~~~~~~~~~~~~~
* Logistic regression (linear baseline, interpretable coefficients)
* Random forest (non-linear, gives feature importances)
* HistGradientBoosting (state-of-the-art tabular boosted trees)
* MLPClassifier (small fully-connected neural network)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import get_logger

log = get_logger("train_models")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(config.PROC_DIR / "features.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int8)
    feature_names = np.array(data["feature_names"])
    accession = np.array(data["accession"])
    host = np.array(data["host"])
    log.info("Loaded %d phages x %d features", *X.shape)
    return X, y, feature_names, accession, host


# ---------------------------------------------------------------------------
# Clustering for leakage control
# ---------------------------------------------------------------------------

def _kmer_subset(X: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    """Return only the k-mer columns for similarity-based clustering."""
    mask = np.array([n.startswith("kmer") for n in feature_names])
    return X[:, mask]


def assign_clusters(X: np.ndarray, feature_names: np.ndarray,
                    distance_threshold: float = config.CLUSTER_DISTANCE) -> np.ndarray:
    """Group near-duplicate phages by cosine distance on k-mer profiles."""
    Xk = _kmer_subset(X, feature_names)
    log.info("Clustering with cosine-distance threshold %.3f over %d k-mer features",
             distance_threshold, Xk.shape[1])
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=config.CLUSTER_LINKAGE,
        distance_threshold=distance_threshold,
    )
    labels = clusterer.fit_predict(Xk)
    n_clusters = labels.max() + 1
    sizes = np.bincount(labels)
    log.info("Found %d clusters (median %d, max %d, singletons %d)",
             n_clusters, int(np.median(sizes)), int(sizes.max()),
             int((sizes == 1).sum()))
    return labels


# ---------------------------------------------------------------------------
# Cluster-aware test split
# ---------------------------------------------------------------------------

def split_by_cluster(y: np.ndarray, clusters: np.ndarray,
                     test_fraction: float = config.TEST_CLUSTER_FRACTION,
                     seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a stratified-by-class subset of *clusters* (not rows).

    For each cluster we take the majority class label as a proxy. This is
    safe because clusters are defined by sequence similarity, and clusters
    spanning both classes are extremely rare (mixed-class clusters are an
    indicator of mislabelling and we treat them as positives if any positive
    is present).
    """
    rng = np.random.default_rng(seed)

    cluster_ids = np.unique(clusters)
    cluster_label = np.zeros(len(cluster_ids), dtype=np.int8)
    for i, cid in enumerate(cluster_ids):
        members = y[clusters == cid]
        cluster_label[i] = int(members.max())  # 1 if any positive

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    train_cluster_idx, test_cluster_idx = next(
        sss.split(np.zeros_like(cluster_label), cluster_label)
    )
    test_clusters = set(cluster_ids[test_cluster_idx])
    test_mask = np.array([c in test_clusters for c in clusters])
    train_mask = ~test_mask
    log.info("Test split: %d phages (in %d clusters), train: %d phages (%d clusters)",
             test_mask.sum(), len(test_cluster_idx),
             train_mask.sum(), len(train_cluster_idx))
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def _model_zoo() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000, random_state=config.SEED,
            )),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                min_samples_leaf=2, n_jobs=-1, random_state=config.SEED,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=None,
                random_state=config.SEED,
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu",
                solver="adam", alpha=1e-3, max_iter=200,
                early_stopping=True, validation_fraction=0.1,
                random_state=config.SEED,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cv_metrics(model: Pipeline, X: np.ndarray, y: np.ndarray,
                groups: np.ndarray, n_splits: int) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        accuracy_score, balanced_accuracy_score,
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=config.SEED)
    scores = {"roc_auc": [], "pr_auc": [], "f1": [],
              "accuracy": [], "balanced_accuracy": []}
    fold_predictions = np.full(len(y), np.nan)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        # Wall-clock fold timing helps surface very slow models.
        t0 = time.time()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        scores["roc_auc"].append(roc_auc_score(y[te], proba))
        scores["pr_auc"].append(average_precision_score(y[te], proba))
        scores["f1"].append(f1_score(y[te], pred))
        scores["accuracy"].append(accuracy_score(y[te], pred))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y[te], pred))
        fold_predictions[te] = proba
        log.info("  fold %d  AUC=%.4f  PR-AUC=%.4f  F1=%.4f  (%.1fs)",
                 fold + 1, scores["roc_auc"][-1], scores["pr_auc"][-1],
                 scores["f1"][-1], time.time() - t0)

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}
    return {"summary": summary, "fold_scores": scores, "oof": fold_predictions}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def train(seed: int = config.SEED) -> dict:
    X, y, feature_names, accession, host = _load_features()

    clusters = assign_clusters(X, feature_names)
    np.savez_compressed(
        config.PROC_DIR / "clusters.npz",
        accession=accession, cluster=clusters,
    )

    train_mask, test_mask = split_by_cluster(y, clusters, seed=seed)
    X_tr, y_tr, g_tr = X[train_mask], y[train_mask], clusters[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    cv_summary: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    oof_predictions: dict[str, np.ndarray] = {}

    for name, model in _model_zoo().items():
        log.info("=== Cross-validating %s ===", name)
        result = _cv_metrics(model, X_tr, y_tr, g_tr, n_splits=config.CV_FOLDS)
        cv_summary[name] = result["summary"]
        oof_predictions[name] = result["oof"]
        # Refit on the full training fold for the held-out test evaluation.
        model.fit(X_tr, y_tr)
        fitted[name] = model
        joblib.dump(model, config.MODEL_DIR / f"{name}.joblib")

    # Test-set evaluation.
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        accuracy_score, balanced_accuracy_score, confusion_matrix,
    )

    test_metrics: dict[str, dict] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, model in fitted.items():
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_te, pred).tolist()
        test_metrics[name] = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "f1": float(f1_score(y_te, pred)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
            "confusion_matrix": cm,
        }
        test_predictions[name] = proba
        log.info("Test %s: AUC=%.4f PR-AUC=%.4f F1=%.4f Acc=%.4f BAcc=%.4f",
                 name,
                 test_metrics[name]["roc_auc"],
                 test_metrics[name]["pr_auc"],
                 test_metrics[name]["f1"],
                 test_metrics[name]["accuracy"],
                 test_metrics[name]["balanced_accuracy"])

    # Persist cross-validation table.
    cv_rows = []
    for name, summary in cv_summary.items():
        for metric, (mean, std) in summary.items():
            cv_rows.append({"model": name, "metric": metric,
                            "mean": mean, "std": std})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(config.METRIC_DIR / "cv_results.csv", index=False)

    with open(config.METRIC_DIR / "test_metrics.json", "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    np.savez_compressed(
        config.PROC_DIR / "predictions.npz",
        accession=accession,
        y=y,
        host=host,
        clusters=clusters,
        train_mask=train_mask,
        test_mask=test_mask,
        feature_names=feature_names,
        **{f"oof_{n}": oof_predictions[n] for n in oof_predictions},
        **{f"test_{n}": test_predictions[n] for n in test_predictions},
    )

    log.info("Saved CV results, test metrics, models, and predictions.")
    return {"cv": cv_summary, "test": test_metrics}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    train(seed=_parse_args().seed)
