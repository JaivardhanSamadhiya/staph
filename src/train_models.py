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
4. Report cluster CV metrics, held-out test metrics, **class-weight-balanced**
   training, an **OOF recall-target** operating threshold, **synthetic
   prevalence** sweeps on the test positive set, and **cluster bootstrap**
   confidence intervals on the held-out split.

Models compared
~~~~~~~~~~~~~~~
* Logistic regression (class-weight balanced)
* Random forest (balanced)
* HistGradientBoosting (balanced class weights where supported by sklearn)
* MLPClassifier (balanced ``sample_weight``)
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import config
from .utils import get_logger

log = get_logger("train_models")


def _fit_pipeline(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    clf = pipe.named_steps.get("clf")
    if isinstance(clf, MLPClassifier):
        sw = compute_sample_weight(class_weight="balanced", y=y)
        pipe.fit(X, y, clf__sample_weight=sw)
    else:
        pipe.fit(X, y)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_features() -> tuple[np.ndarray, ...]:
    data = np.load(config.PROC_DIR / "features.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int8)
    feature_names = np.asarray(data["feature_names"])
    accession = np.asarray(data["accession"])
    host = np.asarray(data["host"])
    n = len(y)
    if "tax_stratum" in data:
        tax_stratum = np.asarray(data["tax_stratum"]).astype(str)
    else:
        tax_stratum = np.array(["missing"] * n, dtype=str)
    if "tax_resolution" in data:
        tax_resolution = np.asarray(data["tax_resolution"]).astype(str)
    else:
        tax_resolution = np.array(["missing"] * n, dtype=str)
    log.info("Loaded %d phages x %d features", *X.shape)
    return X, y, feature_names, accession, host, tax_stratum, tax_resolution


def _corpus_calibration_prevalence() -> float | None:
    p = Path(config.PROC_DIR) / "corpus_stats.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        blob = json.load(fh)
    pi = blob.get("empirical_staphylococcus_fraction")
    if pi is None:
        return None
    pi = float(pi)
    if not (pi == pi):  # NaN guard
        return None
    return max(pi, float(config.CALIBRATION_PREVALENCE_FLOOR))


def _oof_recall_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_recall: float,
) -> float:
    """Highest probability threshold attaining ``target_recall`` on OOF positives.

    Recall is non-increasing as the threshold rises; scanning cut-points in
    ascending order keeps the last threshold that still meets the recall
    target, which is the maximal such threshold on the grid.
    """
    pos = y_true == 1
    if not np.any(pos):
        return 0.5

    s = np.asarray(proba[pos], dtype=np.float64)
    if s.size == 0:
        return 0.5
    smax = float(np.nanmax(s))
    cuts = np.sort(
        np.unique(np.concatenate([np.array([0.0], dtype=np.float64), s, np.array([smax + 1e-12])])),
    )
    best_thr = float(cuts[0])
    for t in cuts:
        rec = float(np.mean(s >= t))
        if rec >= float(target_recall) - 1e-9:
            best_thr = float(t)
        else:
            break
    return float(best_thr)


def _binary_metrics(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None,
                    ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
    if proba is not None and np.unique(y_true).size > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    return out


# ---------------------------------------------------------------------------
# Clustering for leakage control
# ---------------------------------------------------------------------------

def _kmer_subset(X: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    mask = np.array([str(n).startswith("kmer") for n in feature_names])
    return X[:, mask]


def assign_clusters(X: np.ndarray, feature_names: np.ndarray,
                    distance_threshold: float = config.CLUSTER_DISTANCE) -> np.ndarray:
    Xk = _kmer_subset(X, feature_names)
    log.info(
        "Clustering with cosine-distance threshold %.3f over %d k-mer features",
        distance_threshold, Xk.shape[1],
    )
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=config.CLUSTER_LINKAGE,
        distance_threshold=distance_threshold,
    )
    labels = clusterer.fit_predict(Xk)
    n_clusters = int(labels.max() + 1)
    sizes = np.bincount(labels)
    log.info(
        "Found %d clusters (median %d, max %d, singletons %d)",
        n_clusters,
        int(np.median(sizes)),
        int(sizes.max()),
        int((sizes == 1).sum()),
    )
    return labels


# ---------------------------------------------------------------------------
# Cluster-aware test split
# ---------------------------------------------------------------------------

def split_by_cluster(
    y: np.ndarray,
    clusters: np.ndarray,
    test_fraction: float = config.TEST_CLUSTER_FRACTION,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_ids = np.unique(clusters)
    cluster_label = np.zeros(len(cluster_ids), dtype=np.int8)
    for i, cid in enumerate(cluster_ids):
        members = y[clusters == cid]
        cluster_label[i] = int(members.max())

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_ci, test_ci = next(sss.split(np.zeros_like(cluster_label), cluster_label))
    test_clusters = set(cluster_ids[test_ci])
    test_mask = np.array([c in test_clusters for c in clusters])
    train_mask = ~test_mask
    log.info(
        "Test split: %d phages (%d clusters), train: %d phages (%d clusters)",
        int(test_mask.sum()),
        len(test_ci),
        int(train_mask.sum()),
        len(train_ci),
    )
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def _model_zoo(seed: int) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000, random_state=seed, class_weight="balanced",
            )),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced",
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_depth=None,
                random_state=seed,
                class_weight="balanced",
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cv_metrics(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, Any]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    scores: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "accuracy": [],
        "balanced_accuracy": [],
    }
    oof_predictions = np.full(len(y), np.nan)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        t0 = time.time()
        fm = clone(model)
        _fit_pipeline(fm, X[tr], y[tr])
        proba = fm.predict_proba(X[te])[:, 1]
        pred_def = (proba >= 0.5).astype(int)
        # Default-threshold folds (historical parity with sklearn examples).
        scores["roc_auc"].append(roc_auc_score(y[te], proba))
        scores["pr_auc"].append(average_precision_score(y[te], proba))
        scores["f1"].append(f1_score(y[te], pred_def, zero_division=0))
        scores["accuracy"].append(accuracy_score(y[te], pred_def))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y[te], pred_def))
        oof_predictions[te] = proba
        log.info(
            "  fold %d  AUC=%.4f  PR-AUC=%.4f  F1@0.5=%.4f  (%.1fs)",
            fold + 1,
            scores["roc_auc"][-1],
            scores["pr_auc"][-1],
            scores["f1"][-1],
            time.time() - t0,
        )

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}
    return {"summary": summary, "fold_scores": scores, "oof": oof_predictions}


def _oof_train_to_full(oof_tr: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Map training-row OOF probs (same order as ``X[train_mask]``) to full-corpus length."""
    out = np.full(len(train_mask), np.nan, dtype=np.float64)
    out[train_mask] = oof_tr
    return out


def _float_metric_slice(y_m: np.ndarray, p_m: np.ndarray, th: float) -> tuple[dict[str, float], dict[str, float]]:
    """Point metrics at default θ=0.5 and at recall-target θ for one mixture."""
    md = _binary_metrics(y_m, (p_m >= 0.5).astype(int), p_m)
    mr = _binary_metrics(y_m, (p_m >= th).astype(int), p_m)
    return (
        {k: float(v) for k, v in md.items() if isinstance(v, float)},
        {k: float(v) for k, v in mr.items() if isinstance(v, float)},
    )


def _flatten_prevalence_metrics(pref_def: dict[str, float], pref_rec: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update({f"default_0.5__{k}": v for k, v in pref_def.items()})
    out.update({f"recall_target__{k}": v for k, v in pref_rec.items()})
    return out


def _summarise_mc_draws(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys: set[str] = set().union(*(r.keys() for r in rows))
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = np.array([float(r[k]) for r in rows if k in r and r[k] == r[k]],
                        dtype=np.float64)
        if vals.size == 0:
            continue
        out[f"{k}__mean"] = float(np.mean(vals))
        out[f"{k}__std"] = float(np.std(vals, ddof=0))
    return out


def _prevalence_row_for_pi(
    pi: float,
    pos_local: np.ndarray,
    neg_local: np.ndarray,
    *,
    rng_wr: np.random.Generator,
    rng_mc: np.random.Generator,
    proba_te: np.ndarray,
    y_te: np.ndarray,
    recall_target_th: float,
    monte_carlo_draws_when_short: int,
    model_tag: str,
) -> dict[str, Any]:
    """Held-out negatives are drawn **only** from ``neg_local`` (test rows).

    Uses sampling without replacement when the pool suffices; otherwise
    negatives-with-replacement Monte Carlo batches with reported variance.
    """
    np_ = int(len(pos_local))
    nn_pool = int(len(neg_local))
    if np_ <= 0 or not (0.0 < pi < 1.0):
        raise ValueError(f"Invalid prevalence/subset ({pi=} n_pos_test={np_})")

    target_neg_ideal = int(round(float(np_) * (1.0 / float(pi) - 1.0)))
    target_neg_ideal = max(0, target_neg_ideal)

    base: dict[str, Any] = {
        "prevalence_target": float(pi),
        "test_only_positives_used": np_,
        "test_only_negative_pool_available": nn_pool,
        "negative_draws_ideal_for_pi": target_neg_ideal,
    }

    if target_neg_ideal <= nn_pool:
        neg_s = (
            rng_wr.choice(neg_local, size=target_neg_ideal, replace=False)
            if target_neg_ideal
            else np.array([], dtype=np.int64)
        )
        mix = np.concatenate([pos_local, neg_s])
        y_m = y_te[mix]
        p_m = proba_te[mix]
        d5, dr = _float_metric_slice(y_m, p_m, recall_target_th)
        base["sampling"] = "without_replacement_from_test_negatives_only"
        base["monte_carlo_draws_negative_resampling_wr"] = 0
        base["prevalence_achieved_in_mixture"] = float(np_) / float(np_ + int(len(neg_s)))
        base.update(_flatten_prevalence_metrics(d5, dr))
        return base

    log.warning(
        "[%s] π_target=%s needs ~%s held-out negatives but only %s exist — "
        "using %s Monte Carlo draws (**with replacement** from test negatives). "
        "See keys ending in __mean / __std in prevalence_eval.json.",
        model_tag,
        pi,
        target_neg_ideal,
        nn_pool,
        monte_carlo_draws_when_short,
    )
    agg_rows: list[dict[str, float]] = []
    for _ in range(monte_carlo_draws_when_short):
        neg_s = rng_mc.choice(
            neg_local, size=max(1, target_neg_ideal), replace=True,
        )
        mix = np.concatenate([pos_local, neg_s])
        y_m = y_te[mix]
        p_m = proba_te[mix]
        d5, dr = _float_metric_slice(y_m, p_m, recall_target_th)
        agg_rows.append(_flatten_prevalence_metrics(d5, dr))

    base["sampling"] = "monte_carlo_negative_resampling_with_replacement"
    base["monte_carlo_draws_negative_resampling_wr"] = int(monte_carlo_draws_when_short)
    # Expectation prevalence under MC is still ~π_target; mixture size varies trivially due to repeats.
    base["approx_prevalence_in_each_draw"] = float(np_) / float(np_ + max(1, target_neg_ideal))
    base.update(_summarise_mc_draws(agg_rows))
    return base


def _cluster_bootstrap_metrics(
    y: np.ndarray,
    proba_full: np.ndarray,
    clusters: np.ndarray,
    test_mask: np.ndarray,
    threshold: float,
    n_draws: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    te_idx = np.flatnonzero(test_mask)
    cl_te = clusters[te_idx]

    uniq = np.unique(cl_te)
    cl_to_ix: dict[Any, np.ndarray] = {
        int(c): te_idx[np.nonzero(cl_te == c)[0]] for c in uniq.astype(int).tolist()
    }
    roc_list: list[float] = []
    pr_list: list[float] = []
    recall_list: list[float] = []
    precision_list: list[float] = []
    balacc_list: list[float] = []

    for _ in range(n_draws):
        redrawn = rng.choice(np.array(list(cl_to_ix.keys())), size=len(cl_to_ix), replace=True)
        samp = np.concatenate([cl_to_ix[int(c)] for c in redrawn])
        yt = y[samp].astype(np.int8)
        pr = proba_full[samp]

        if len(np.unique(yt)) < 2:
            roc_list.append(float("nan"))
            pr_list.append(float("nan"))
        else:
            roc_list.append(float(roc_auc_score(yt, pr)))
            pr_list.append(float(average_precision_score(yt, pr)))
        preds = (pr >= threshold).astype(np.int8)
        recall_list.append(float(recall_score(yt, preds, zero_division=0)))
        precision_list.append(float(precision_score(yt, preds, zero_division=0)))
        balacc_list.append(float(balanced_accuracy_score(yt, preds)))

    return {
        "roc_auc": roc_list,
        "pr_auc": pr_list,
        "recall_at_target_threshold": recall_list,
        "precision_at_target_threshold": precision_list,
        "balanced_accuracy_at_target_threshold": balacc_list,
    }


def _bootstrap_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(s)),
        float(np.quantile(s, alpha / 2)),
        float(np.quantile(s, 1 - alpha / 2)),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def train(seed: int = config.SEED, bootstrap_draws: int | None = None) -> dict[str, Any]:
    (
        X,
        y,
        feature_names,
        accession,
        host,
        tax_stratum,
        tax_resolution,
    ) = _load_features()

    clusters = assign_clusters(X, feature_names)
    np.savez_compressed(
        config.PROC_DIR / "clusters.npz",
        accession=accession,
        cluster=clusters,
    )

    train_mask, test_mask = split_by_cluster(y, clusters, seed=seed)
    X_tr, y_tr, g_tr = X[train_mask], y[train_mask], clusters[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    bootstrap_n = (
        config.BOOTSTRAP_CLUSTER_DRAWS
        if bootstrap_draws is None
        else int(bootstrap_draws)
    )
    if bootstrap_draws is not None:
        log.info("Bootstrap cluster-draw override: %d", bootstrap_n)

    cv_summary: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    recall_thresholds: dict[str, float] = {}

    zoo = _model_zoo(seed)
    for name, model in zoo.items():
        log.info("=== Cross-validating %s ===", name)
        result = _cv_metrics(model, X_tr, y_tr, g_tr, n_splits=config.CV_FOLDS)
        cv_summary[name] = result["summary"]
        oof = result["oof"]
        oof_predictions[name] = oof
        # ``oof`` is indexed 0..len(X_tr)-1 (CV on the training split only), not global row ids.
        y_oof = y_tr
        p_oof = oof
        recall_thresholds[name] = _oof_recall_threshold(
            y_oof, p_oof, float(config.TARGET_RECALL_OOF),
        )
        log.info(
            "  OOF recall-target (%.2f) threshold ≈ %.4f",
            float(config.TARGET_RECALL_OOF),
            recall_thresholds[name],
        )

        fm = clone(model)
        _fit_pipeline(fm, X_tr, y_tr)
        fitted[name] = fm
        joblib.dump(fm, config.MODEL_DIR / f"{name}.joblib")

    primary = max(
        cv_summary.keys(),
        key=lambda n: cv_summary[n]["summary"]["roc_auc"][0],
    )
    log.info("Primary model (max CV ROC-AUC): %s", primary)

    test_metrics: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    stratum_table: dict[str, Any] = {}

    te_idx = np.flatnonzero(test_mask)
    pos_local = np.flatnonzero(y_te == 1)
    neg_local = np.flatnonzero(y_te == 0)

    rng_prev_wr = np.random.default_rng(seed + 11)

    prevalence_eval: dict[str, Any] = {
        "documentation": (
            "All prevalence mixtures use held-out TEST rows exclusively: "
            "every test positive plus negatives subsampled ONLY from "
            "test-negative genomes (never train). Sampling is "
            "**without replacement** while the requested count fits the pool "
            "(see prevalence_achieved_in_mixture). If the corpus cannot supply "
            "enough negatives for the target prevalence, negatives are sampled "
            "**with replacement** in repeated Monte Carlo draws; metric keys "
            "then arrive as PREFIX__METRIC__mean and __std (see monte_carlo_* "
            "fields on each row)."
        ),
        "prevalence_grid": list(config.PREVALENCE_GRID),
        "monte_carlo_negative_draws_when_pool_short": int(
            config.PREVALENCE_MONTE_CARLO_NEGATIVE_DRAWS,
        ),
        "per_model": {},
    }

    for name, model in fitted.items():
        proba = model.predict_proba(X_te)[:, 1]
        pred_def = (proba >= 0.5).astype(int)
        th = recall_thresholds[name]
        pred_rec = (proba >= th).astype(int)

        test_predictions[name] = proba
        test_metrics[name] = {
            "threshold_default_0.5": _binary_metrics(y_te, pred_def, proba),
            f"threshold_recall_{config.TARGET_RECALL_OOF:.2f}": _binary_metrics(
                y_te, pred_rec, proba,
            ),
            "recall_target_threshold": float(th),
        }

        # Negative-host strata (test subset only).
        neg_mask_te = (y_te == 0)
        if np.any(neg_mask_te):
            preds_n = pred_rec[neg_mask_te]
            strata = tax_stratum[test_mask][neg_mask_te]
            rows = []
            for s in np.unique(strata):
                m = strata == s
                rows.append({
                    "tax_stratum": str(s),
                    "n": int(m.sum()),
                    "true_negative_rate": float((preds_n[m] == 0).mean()),
                })
            stratum_table[name] = rows

        prev_rows = []
        for zi, pi in enumerate(config.PREVALENCE_GRID):
            tag = zlib.crc32(f"{seed}|{name}|{pi}|{zi}".encode()) & 0xFFFFFFFF
            rng_mc_pi = np.random.default_rng(tag)
            prev_rows.append(
                _prevalence_row_for_pi(
                    float(pi),
                    pos_local,
                    neg_local,
                    rng_wr=rng_prev_wr,
                    rng_mc=rng_mc_pi,
                    proba_te=proba,
                    y_te=y_te,
                    recall_target_th=float(th),
                    monte_carlo_draws_when_short=int(
                        config.PREVALENCE_MONTE_CARLO_NEGATIVE_DRAWS,
                    ),
                    model_tag=str(name),
                ),
            )
        prevalence_eval["per_model"][name] = prev_rows

    boot_blob: dict[str, Any] = {}
    if bootstrap_n > 0:
        draws = int(bootstrap_n)
        th_p = recall_thresholds[primary]
        proba_p = test_predictions[primary]
        proba_full = np.full(len(y), np.nan, dtype=np.float64)
        proba_full[te_idx] = proba_p

        samp = _cluster_bootstrap_metrics(
            y=y,
            proba_full=proba_full,
            clusters=clusters,
            test_mask=test_mask,
            threshold=float(th_p),
            n_draws=draws,
            seed=seed + 913,
        )
        boot_summ: dict[str, Any] = {}
        for k, vals in samp.items():
            mn, lo, hi = _bootstrap_ci(np.asarray(vals, dtype=np.float64))
            boot_summ[k] = {"mean": mn, "ci95_low": lo, "ci95_high": hi, "draws": draws}
        boot_blob = {
            "primary_model": primary,
            "recall_target_threshold": float(th_p),
            "metrics": boot_summ,
        }

    corpus_pi = _corpus_calibration_prevalence()
    corpus_note = corpus_pi if corpus_pi is not None else "corpus_stats.json missing"

    cv_rows = []
    for name, summary in cv_summary.items():
        for metric, (mean, std) in summary.items():
            cv_rows.append({"model": name, "metric": metric, "mean": mean, "std": std})
    pd.DataFrame(cv_rows).to_csv(config.METRIC_DIR / "cv_results.csv", index=False)

    with open(config.METRIC_DIR / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "primary_model": primary,
                "recall_targets": recall_thresholds,
                "metrics_by_model": test_metrics,
                "negative_strata_test_recall_negative_class": stratum_table,
            },
            fh,
            indent=2,
        )

    with open(config.METRIC_DIR / "prevalence_eval.json", "w", encoding="utf-8") as fh:
        json.dump(prevalence_eval, fh, indent=2)

    if boot_blob:
        with open(config.METRIC_DIR / "bootstrap_primary.json", "w", encoding="utf-8") as fh:
            json.dump(boot_blob, fh, indent=2)

    with open(config.METRIC_DIR / "training_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({
            "primary_model": primary,
            "seed": seed,
            "cv_folds": config.CV_FOLDS,
            "target_recall_oob": float(config.TARGET_RECALL_OOF),
            "prevalence_grid": list(config.PREVALENCE_GRID),
            "corpus_calibration_prevalence_estimate": corpus_note,
            "bootstrap_cluster_draws": bootstrap_n,
        }, fh, indent=2)

    recall_models = np.array(list(recall_thresholds.keys()), dtype=object)
    recall_vals = np.array([recall_thresholds[str(k)] for k in recall_models], dtype=np.float64)

    np.savez_compressed(
        config.PROC_DIR / "predictions.npz",
        accession=accession,
        y=y,
        host=host,
        clusters=clusters,
        train_mask=train_mask,
        test_mask=test_mask,
        tax_stratum=tax_stratum,
        tax_resolution=tax_resolution,
        feature_names=np.asarray(feature_names),
        recall_target_models=recall_models,
        recall_target_values=recall_vals,
        **{f"oof_{n}": _oof_train_to_full(oof_predictions[n], train_mask)
           for n in oof_predictions},
        **{f"test_{n}": test_predictions[n] for n in test_predictions},
    )

    log.info("Saved CV results, test metrics, models, and predictions.")
    return {"cv": cv_summary, "test": test_metrics, "primary": primary}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument(
        "--bootstrap-draws",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override bootstrap resample count on held-out clusters (default: "
            f"{config.BOOTSTRAP_CLUSTER_DRAWS}); use 200–500 locally, "
            f"{config.BOOTSTRAP_CLUSTER_DRAWS}+ for manuscript runs."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(seed=args.seed, bootstrap_draws=args.bootstrap_draws)
