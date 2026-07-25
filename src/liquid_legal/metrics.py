"""Evaluation metrics (dependency-free)."""

from __future__ import annotations

import math

import numpy as np


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC AUC. Returns NaN if single-class."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    pos = y_true >= 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def mae_days(pred_log: np.ndarray, true_log: np.ndarray) -> float:
    """MAE in days between log1p-encoded remaining-time predictions."""
    pred = np.expm1(np.asarray(pred_log, dtype=np.float64))
    true = np.expm1(np.asarray(true_log, dtype=np.float64))
    return float(np.mean(np.abs(pred - true)))


def mae_log_dollars(pred_log: np.ndarray, true_log: np.ndarray) -> float:
    """MAE on the log1p-dollar scale for recovery predictions."""
    return float(
        np.mean(np.abs(np.asarray(pred_log, dtype=np.float64) - np.asarray(true_log, dtype=np.float64)))
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    val = auc_score(y_true, y_score)
    return val if not math.isnan(val) else float("nan")
