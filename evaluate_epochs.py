#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score

OUT_BASE = Path("output")
EPOCH_DIR_FMT = "curriculum_epoch{epoch}"
CSV_NAME = "stablediffusion3-sd3-v3.csv"
METRIC_DIR = Path("metric_output")
EPOCHS = list(range(0, 11))  # 0..10 inclusive

# Column fallbacks
POSSIBLE_LABELS = ["class", "label", "target", "y"]
POSSIBLE_IMAGE = ["image", "path", "filepath"]
SCORE_COL = "spai"  # column created by spai infer

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns {candidates} found. Got: {list(df.columns)}")

def _subset_mask(df: pd.DataFrame, y_col: str, img_col: str, which: str) -> np.ndarray:
    """Return boolean mask for the requested subset."""
    y = df[y_col].astype(int).values
    img = df[img_col].astype(str).str.lower().fillna("")
    is_real = (y == 0)
    is_pos = (y == 1)
    is_matched = img.str.contains("matched", na=False).values

    if which == "real+matched":
        return is_real | (is_pos & is_matched)
    elif which == "real+synthetic":
        return is_real | (is_pos & (~is_matched))
    else:
        raise ValueError(which)

def _best_threshold_for_accuracy(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Sweep thresholds in [0,1] to maximize accuracy.
    Returns (best_acc, best_thr, tpr_at_best, tnr_at_best).
    """
    # if scores are all same, any threshold; handle gracefully
    if np.all(scores == scores[0]):
        thr = 0.5
        y_pred = (scores >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return acc, thr, tpr, tnr

    # Use a dense grid for robustness (fast enough) – avoids O(N) unique thresholds degenerate cases
    grid = np.linspace(0.0, 1.0, 1001)
    best_acc = -1.0
    best_thr = 0.5
    best_tpr = 0.0
    best_tnr = 0.0

    for thr in grid:
        pred = (scores >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
            best_tpr = tpr
            best_tnr = tnr

    return best_acc, float(best_thr), float(best_tpr), float(best_tnr)

def _metrics_for_subset(df: pd.DataFrame, y_col: str, img_col: str, subset_name: str) -> Dict[str, float]:
    mask = _subset_mask(df, y_col, img_col, subset_name)
    sub = df.loc[mask].copy()
    if sub.empty:
        # Return NaNs if subset has no rows
        return {
            "n": 0,
            "acc@0.5": np.nan,
            "oracle_acc": np.nan,
            "oracle_thr": np.nan,
            "ap": np.nan,
            "tpr@oracle": np.nan,
            "tnr@oracle": np.nan,
        }

    y = sub[y_col].astype(int).values
    s = sub[SCORE_COL].astype(float).values

    # acc at 0.5
    y_pred_05 = (s >= 0.5).astype(int)
    acc05 = accuracy_score(y, y_pred_05)

    # oracle acc (best threshold)
    best_acc, best_thr, tpr, tnr = _best_threshold_for_accuracy(y, s)

    # AP
    # Guard: if subset has only one class, AP is undefined; sklearn returns 1.0 if all true are 1?
    # We'll handle class-imbalance by returning NaN if only one class exists
    if len(np.unique(y)) == 1:
        ap = np.nan
    else:
        ap = average_precision_score(y, s)

    return {
        "n": int(len(sub)),
        "acc@0.5": float(acc05),
        "oracle_acc": float(best_acc),
        "oracle_thr": float(best_thr),
        "ap": float(ap) if ap == ap else np.nan,
        "tpr@oracle": float(tpr),
        "tnr@oracle": float(tnr),
    }

def main():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in EPOCHS:
        csv_path = OUT_BASE / EPOCH_DIR_FMT.format(epoch=i) / CSV_NAME
        if not csv_path.exists():
            print(f"[warn] missing CSV for epoch {i}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # --- Column detection
        if SCORE_COL not in df.columns:
            raise KeyError(f"Score column '{SCORE_COL}' not found in {csv_path}")
        y_col = _find_col(df, POSSIBLE_LABELS)
        img_col = _find_col(df, POSSIBLE_IMAGE)

        # --- Keep split=val if present
        if "split" in df.columns:
            df = df[df["split"] == "val"].copy()

        # --- Coerce types
        df[y_col] = df[y_col].astype(int)
        df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
        df = df.dropna(subset=[SCORE_COL])

        # --- Compute metrics for both subsets
        for subset in ("real+matched", "real+synthetic"):
            m = _metrics_for_subset(df, y_col, img_col, subset)
            row = {"epoch": i, "subset": subset, **m}
            rows.append(row)

    if not rows:
        print("[error] No metrics computed (no CSVs found or empty data).")
        return

    results = pd.DataFrame(rows).sort_values(["subset", "epoch"])
    results_path = METRIC_DIR / "curriculum_metrics.csv"
    results.to_csv(results_path, index=False)
    print(f"[info] wrote {results_path}")

    # --------- PLOTS ----------
    def plot_metric(metric: str, ylabel: str, fname: str, ylim: Tuple[float,float] | None = None):
        plt.figure()
        for subset in ("real+matched", "real+synthetic"):
            d = results[results["subset"] == subset]
            plt.plot(d["epoch"], d[metric], marker="o", label=subset)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        outp = METRIC_DIR / fname
        plt.savefig(outp, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[info] saved {outp}")

    plot_metric("acc@0.5", "Accuracy @ 0.5", "acc_at_0p5.png", ylim=(0,1))
    plot_metric("oracle_acc", "Oracle Accuracy", "oracle_acc.png", ylim=(0,1))
    plot_metric("oracle_thr", "Best Threshold", "oracle_threshold.png", ylim=(0,1))
    plot_metric("ap", "Average Precision", "ap.png", ylim=(0,1))
    plot_metric("tpr@oracle", "TPR @ Best Thr", "tpr_at_oracle.png", ylim=(0,1))
    plot_metric("tnr@oracle", "TNR @ Best Thr", "tnr_at_oracle.png", ylim=(0,1))

if __name__ == "__main__":
    main()
