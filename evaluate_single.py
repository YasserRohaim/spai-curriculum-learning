#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

INPUT_DEFAULT = "output/stablediffusion3-sd3-v3.csv"
METRIC_DIR = Path("metric_output2")

# Column fallbacks (adjust if your CSV uses different names)
POSSIBLE_LABELS = ["class", "label", "target", "y"]
POSSIBLE_IMAGE = ["image", "path", "filepath"]
SCORE_COL = "spai"  # column populated by your infer step

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

def _subset_mask(df: pd.DataFrame, y_col: str, img_col: str, which: str) -> np.ndarray:
    if which == "overall":
        return np.ones(len(df), dtype=bool)

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

def _best_threshold_for_accuracy(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    if len(y_true) == 0:
        return np.nan, np.nan
    if np.all(scores == scores[0]):
        thr = 0.5
        acc = accuracy_score(y_true, (scores >= thr).astype(int))
        return float(acc), float(thr)
    grid = np.linspace(0.0, 1.0, 1001)
    best_acc, best_thr = -1.0, 0.5
    for thr in grid:
        acc = accuracy_score(y_true, (scores >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return float(best_acc), float(best_thr)

def _tpr_tnr_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Tuple[float, float]:
    if len(y_true) == 0:
        return np.nan, np.nan
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(tpr), float(tnr)

def _safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, scores))

def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))

def _metrics_for_subset(df: pd.DataFrame, y_col: str, img_col: str, subset_name: str) -> Dict[str, float]:
    mask = _subset_mask(df, y_col, img_col, subset_name)
    sub = df.loc[mask].copy()
    if sub.empty:
        return {
            "n": 0,
            "acc@0.5": np.nan,
            "tpr@0.5": np.nan,
            "tnr@0.5": np.nan,
            "ap": np.nan,
            "auc": np.nan,
            "oracle_acc": np.nan,
            "oracle_thr": np.nan,
        }
    y = sub[y_col].astype(int).values
    s = sub[SCORE_COL].astype(float).values

    acc05 = accuracy_score(y, (s >= 0.5).astype(int))
    tpr05, tnr05 = _tpr_tnr_at_threshold(y, s, 0.5)
    ap = _safe_ap(y, s)
    auc = _safe_auc(y, s)
    best_acc, best_thr = _best_threshold_for_accuracy(y, s)

    return {
        "n": int(len(sub)),
        "acc@0.5": float(acc05),
        "tpr@0.5": float(tpr05) if tpr05 == tpr05 else np.nan,
        "tnr@0.5": float(tnr05) if tnr05 == tnr05 else np.nan,
        "ap": float(ap) if ap == ap else np.nan,
        "auc": float(auc) if auc == auc else np.nan,
        "oracle_acc": float(best_acc),
        "oracle_thr": float(best_thr),
    }

def plot_metrics(df_metrics: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    subsets = list(df_metrics["subset"].unique())

    def _bar(metric: str, ylabel: str, fname: str, ylim=None):
        plt.figure()
        vals = [df_metrics.loc[df_metrics["subset"] == s, metric].values[0] for s in subsets]
        xpos = np.arange(len(subsets))
        plt.bar(xpos, vals)
        plt.xticks(xpos, subsets, rotation=15)
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.grid(axis="y", alpha=0.3)
        outp = out_dir / fname
        plt.savefig(outp, bbox_inches="tight", dpi=150)
        plt.close()

    _bar("acc@0.5", "Accuracy @ 0.5", "acc_at_0p5.png", ylim=(0,1))
    _bar("oracle_acc", "Oracle Accuracy", "oracle_acc.png", ylim=(0,1))
    _bar("oracle_thr", "Best Threshold", "oracle_threshold.png")
    _bar("ap", "Average Precision", "ap.png", ylim=(0,1))
    _bar("auc", "AUC", "auc.png", ylim=(0,1))
    _bar("tpr@0.5", "TPR @ 0.5", "tpr_at_0p5.png", ylim=(0,1))
    _bar("tnr@0.5", "TNR @ 0.5", "tnr_at_0p5.png", ylim=(0,1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=INPUT_DEFAULT, help="Path to the single output CSV to evaluate")
    ap.add_argument("--out", default=str(METRIC_DIR), help="Directory to write metric CSV/plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if SCORE_COL not in df.columns:
        raise KeyError(f"Score column '{SCORE_COL}' not found in {csv_path}. Have: {list(df.columns)}")
    y_col = _find_col(df, POSSIBLE_LABELS)
    img_col = _find_col(df, POSSIBLE_IMAGE)

    # keep only split=val if present
    if "split" in df.columns:
        df = df[df["split"] == "val"].copy()

    df[y_col] = df[y_col].astype(int)
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df = df.dropna(subset=[SCORE_COL])

    rows = []
    for subset in ("overall", "real+matched", "real+synthetic"):
        m = _metrics_for_subset(df, y_col, img_col, subset)
        rows.append({"subset": subset, **m})

    metrics_df = pd.DataFrame(rows)
    out_csv = out_dir / "metrics_single_file.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"[info] wrote {out_csv}")

    plot_metrics(metrics_df, out_dir)

if __name__ == "__main__":
    main()
