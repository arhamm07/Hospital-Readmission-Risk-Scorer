import sys
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

import mlflow

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/evaluate.log")


# METRIC HELPERS

def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Expected Calibration Error — weighted avg of |accuracy - confidence|."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_pred)) * abs(
            y_true[mask].mean() - y_pred[mask].mean()
        )
    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Maximum Calibration Error — worst-case bin miscalibration."""
    bins  = np.linspace(0, 1, n_bins + 1)
    errors = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        errors.append(
            abs(y_true[mask].mean() - y_pred[mask].mean())
        )
    return float(max(errors)) if errors else 0.0


def compute_ks_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    KS Statistic — max separation between CDF of positives vs negatives.
    Industry standard in credit / clinical risk scoring.
    """
    pos_scores = np.sort(y_pred[y_true == 1])
    neg_scores = np.sort(y_pred[y_true == 0])

    thresholds = np.linspace(0, 1, 200)
    ks = 0.0
    for t in thresholds:
        tpr = (pos_scores >= t).mean()
        fpr = (neg_scores >= t).mean()
        ks  = max(ks, abs(tpr - fpr))
    return float(ks)


def compute_gini(auc_roc: float) -> float:
    """Gini = 2 × AUC - 1.  Credit industry convention."""
    return float(2 * auc_roc - 1)


def compute_clinical_utility(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    flag_pct:  float = 0.20,
) -> dict[str, float]:
    """
    Simulates flagging the top flag_pct % of patients as high-risk.
    Returns PPV, NPV, sensitivity, specificity, NNT.
    """
    threshold = float(np.percentile(y_pred, 100 * (1 - flag_pct)))
    preds     = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    nnt         = 1 / ppv if ppv > 0 else float("inf")

    return {
        "threshold":   threshold,
        "flag_pct":    flag_pct,
        "ppv":         float(ppv),
        "npv":         float(npv),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "nnt":         float(nnt),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    For every threshold t in [0, 1]:
    compute precision, recall, F1, cost at each point.
    Returns a DataFrame for plotting.
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    records    = []

    for t in thresholds:
        preds = (y_pred >= t).astype(int)
        if preds.sum() == 0:
            continue
        records.append({
            "threshold":   t,
            "precision":   precision_score(y_true, preds, zero_division=0),
            "recall":      recall_score(y_true, preds, zero_division=0),
            "f1":          f1_score(y_true, preds, zero_division=0),
            "flagged_pct": preds.mean(),
        })

    return pd.DataFrame(records)


def compute_risk_tier_qa(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    tiers:   dict[str, float],
) -> pd.DataFrame:
    """
    Checks that readmission rate increases monotonically
    across Low → Medium → High → Critical tiers.
    """
    def assign_tier(score: float) -> str:
        if score < tiers["p25"]:   return "Low"
        elif score < tiers["p50"]: return "Medium"
        elif score < tiers["p75"]: return "High"
        else:                      return "Critical"

    tier_labels = np.array([assign_tier(s) for s in y_pred])
    order       = ["Low", "Medium", "High", "Critical"]

    records = []
    for tier in order:
        mask = tier_labels == tier
        if mask.sum() == 0:
            continue
        records.append({
            "tier":             tier,
            "n":                int(mask.sum()),
            "pct_of_total":     float(mask.mean()),
            "readmission_rate": float(y_true[mask].mean()),
            "mean_score":       float(y_pred[mask].mean()),
        })

    df = pd.DataFrame(records)

    # Monotonicity check
    rates = df["readmission_rate"].values
    is_monotone = all(
        rates[i] <= rates[i + 1] for i in range(len(rates) - 1)
    )
    logger.info(
        f"Risk tier monotonicity: {'PASS' if is_monotone else 'FAIL'}"
    )
    df["monotone"] = is_monotone
    return df


def compute_subgroup_auc(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    subgroups: pd.Series,
    group_name: str,
) -> pd.DataFrame:
    """
    Computes AUC-ROC per subgroup value.
    Used for: age_risk_bucket, gender, diag_1_category.
    """
    records = []
    for group_val in subgroups.unique():
        mask = subgroups == group_val
        if mask.sum() < 30 or y_true[mask].sum() < 5:
            continue   # not enough samples for reliable AUC
        try:
            auc = roc_auc_score(y_true[mask], y_pred[mask])
        except Exception:
            auc = float("nan")
        records.append({
            "group":      str(group_val),
            "n":          int(mask.sum()),
            "pos_rate":   float(y_true[mask].mean()),
            "auc_roc":    float(auc),
        })

    df = pd.DataFrame(records).sort_values("auc_roc", ascending=False)
    logger.info(f"Subgroup AUC ({group_name}):\n{df.to_string(index=False)}")
    return df


# PLOT HELPERS
def plot_reliability_diagram(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    ece:       float,
    mce:       float,
    save_path: str,
    title:     str = "Reliability Diagram",
):
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[0].plot(mean_pred, frac_pos, "o-", color="#2196F3",
                 lw=2, ms=8, label=f"Model (ECE={ece:.4f})")

    # Shade the gap
    axes[0].fill_between(
        mean_pred, mean_pred, frac_pos,
        alpha=0.15, color="#FF5722", label="Calibration gap"
    )
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title(f"{title}\nECE={ece:.4f}  MCE={mce:.4f}")
    axes[0].legend()

    # Confidence histogram
    axes[1].hist(
        y_pred[y_true == 0], bins=40, alpha=0.6,
        color="#4CAF50", label="Negative class"
    )
    axes[1].hist(
        y_pred[y_true == 1], bins=40, alpha=0.6,
        color="#FF5722", label="Positive class"
    )
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score Distribution by Class")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {save_path}")


def plot_threshold_sweep(sweep_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"],
             label="Precision", color="#2196F3", lw=2)
    plt.plot(sweep_df["threshold"], sweep_df["recall"],
             label="Recall",    color="#FF5722", lw=2)
    plt.plot(sweep_df["threshold"], sweep_df["f1"],
             label="F1 Score",  color="#4CAF50", lw=2)
    plt.plot(sweep_df["threshold"], sweep_df["flagged_pct"],
             label="% Flagged", color="#9C27B0", lw=2, linestyle="--")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Decision Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {save_path}")


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred_bin: np.ndarray,
    threshold:  float,
    save_path:  str,
):
    cm = confusion_matrix(y_true, y_pred_bin)
    labels = ["No Readmission", "Readmitted <30d"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"Pred: {l}" for l in labels],
        yticklabels=[f"True: {l}" for l in labels],
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix  (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {save_path}")


def plot_risk_tier_qa(tier_df: pd.DataFrame, save_path: str):
    colors = {
        "Low": "#4CAF50", "Medium": "#FFC107",
        "High": "#FF9800", "Critical": "#F44336"
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Readmission rate per tier
    bars = axes[0].bar(
        tier_df["tier"],
        tier_df["readmission_rate"],
        color=[colors.get(t, "#999") for t in tier_df["tier"]]
    )
    axes[0].set_ylabel("Actual Readmission Rate")
    axes[0].set_title("Readmission Rate by Risk Tier")
    for bar, row in zip(bars, tier_df.itertuples()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={row.n}",
            ha="center", fontsize=9
        )

    # Patient volume per tier
    axes[1].bar(
        tier_df["tier"],
        tier_df["pct_of_total"] * 100,
        color=[colors.get(t, "#999") for t in tier_df["tier"]]
    )
    axes[1].set_ylabel("% of Total Patients")
    axes[1].set_title("Patient Volume per Risk Tier")

    plt.suptitle(
        "Risk Stratification QA"
        + ("  ✅ Monotone" if tier_df["monotone"].all() else "  ❌ Non-monotone"),
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {save_path}")


def plot_subgroup_auc(
    subgroup_df: pd.DataFrame,
    group_name:  str,
    save_path:   str,
):
    plt.figure(figsize=(10, 5))
    bars = plt.barh(
        subgroup_df["group"],
        subgroup_df["auc_roc"],
        color="#2196F3", alpha=0.85
    )
    plt.axvline(0.5, color="red", linestyle="--", label="Random (0.5)")
    plt.xlabel("AUC-ROC")
    plt.title(f"Subgroup AUC-ROC by {group_name}")
    plt.legend()

    for bar, row in zip(bars, subgroup_df.itertuples()):
        plt.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"n={row.n}",
            va="center", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {save_path}")


def plot_score_decile_analysis(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: str,
):
    """
    Divides patients into 10 score deciles.
    Plots actual readmission rate per decile — should be monotone.
    """
    df = pd.DataFrame({"score": y_pred, "readmitted": y_true})
    df["decile"] = pd.qcut(
        df["score"], q=10, labels=False, duplicates="drop"
    )

    decile_summary = (
        df.groupby("decile")
        .agg(
            n=("readmitted", "count"),
            readmission_rate=("readmitted", "mean"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        decile_summary["decile"] + 1,
        decile_summary["readmission_rate"],
        color="#FF5722", alpha=0.85, edgecolor="white"
    )
    plt.plot(
        decile_summary["decile"] + 1,
        decile_summary["readmission_rate"],
        "ko-", lw=1.5
    )
    plt.xlabel("Score Decile (1 = lowest risk, 10 = highest risk)")
    plt.ylabel("Actual Readmission Rate")
    plt.title("Readmission Rate by Score Decile\n(Should increase monotonically)")
    plt.xticks(range(1, len(decile_summary) + 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {save_path}")


# MAIN EVALUATOR CLASS
class ClinicalEvaluator:
    """
    Full evaluation suite for the readmission risk model.

    Usage:
        evaluator = ClinicalEvaluator(cfg)
        results   = evaluator.evaluate(
            y_true, y_pred_probs,
            test_df=test_df,          # for subgroup analysis
            risk_tiers=tier_dict,     # from train.py
            split_name="test",
            log_to_mlflow=True,
        )
        evaluator.save_metrics(results)
    """

    def __init__(self, cfg):
        self.cfg      = cfg
        self.reports  = Path("reports")
        self.reports.mkdir(exist_ok=True)

    # Core evaluation

    def evaluate(
        self,
        y_true:          np.ndarray,
        y_pred:          np.ndarray,
        test_df:         pd.DataFrame | None = None,
        risk_tiers:      dict | None         = None,
        split_name:      str                 = "test",
        log_to_mlflow:   bool                = True,
    ) -> dict[str, Any]:
        """
        Runs all evaluation sections.
        Returns a flat metrics dict.
        """
        logger.info(
            f"\n{'='*55}\n"
            f"  CLINICAL EVALUATION — {split_name.upper()} SET\n"
            f"{'='*55}"
        )

        metrics: dict[str, Any] = {"split": split_name}

        # A. Standard ML metrics
        metrics.update(
            self._section_standard_ml(y_true, y_pred, split_name)
        )

        # B. Calibration metrics
        metrics.update(
            self._section_calibration(y_true, y_pred, split_name)
        )

        # C. Threshold sweep
        metrics.update(
            self._section_threshold_sweep(y_true, y_pred, split_name)
        )

        # D. Clinical utility
        metrics.update(
            self._section_clinical_utility(y_true, y_pred, split_name)
        )

        # E. Risk tier QA
        if risk_tiers:
            metrics.update(
                self._section_risk_tiers(
                    y_true, y_pred, risk_tiers, split_name
                )
            )

        # F. Subgroup analysis
        if test_df is not None:
            metrics.update(
                self._section_subgroups(y_true, y_pred, test_df, split_name)
            )

        # G. Score decile analysis
        decile_path = str(self.reports / f"eval_decile_{split_name}.png")
        plot_score_decile_analysis(y_true, y_pred, decile_path)

        # Log to MLflow
        if log_to_mlflow:
            self._log_to_mlflow(metrics, split_name)

        self._print_summary(metrics, split_name)
        return metrics

    # Section A: Standard ML

    def _section_standard_ml(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
    ) -> dict:
        auc_roc  = roc_auc_score(y_true, y_pred)
        auc_pr   = average_precision_score(y_true, y_pred)
        brier    = brier_score_loss(y_true, y_pred)
        logloss  = log_loss(y_true, y_pred)
        ks_stat  = compute_ks_statistic(y_true, y_pred)
        gini     = compute_gini(auc_roc)

        logger.info(
            f"[Standard ML]\n"
            f"  AUC-ROC  : {auc_roc:.4f}\n"
            f"  AUC-PR   : {auc_pr:.4f}\n"
            f"  Brier    : {brier:.4f}\n"
            f"  Log Loss : {logloss:.4f}\n"
            f"  KS Stat  : {ks_stat:.4f}\n"
            f"  Gini     : {gini:.4f}"
        )

        return {
            f"{split_name}_auc_roc":  auc_roc,
            f"{split_name}_auc_pr":   auc_pr,
            f"{split_name}_brier":    brier,
            f"{split_name}_log_loss": logloss,
            f"{split_name}_ks_stat":  ks_stat,
            f"{split_name}_gini":     gini,
        }

    # Section B: Calibration

    def _section_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
    ) -> dict:
        ece = compute_ece(y_true, y_pred, self.cfg.calibration.n_bins)
        mce = compute_mce(y_true, y_pred, self.cfg.calibration.n_bins)

        logger.info(
            f"[Calibration]\n"
            f"  ECE : {ece:.4f}\n"
            f"  MCE : {mce:.4f}"
        )

        # Reliability diagram
        rel_path = str(self.reports / f"eval_reliability_{split_name}.png")
        plot_reliability_diagram(y_true, y_pred, ece, mce, rel_path)

        return {
            f"{split_name}_ece": ece,
            f"{split_name}_mce": mce,
        }

    # Section C: Threshold sweep

    def _section_threshold_sweep(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
    ) -> dict:
        sweep_df = compute_threshold_sweep(y_true, y_pred)

        # Best F1 threshold
        best_f1_row = sweep_df.loc[sweep_df["f1"].idxmax()]
        best_f1     = float(best_f1_row["f1"])
        best_t_f1   = float(best_f1_row["threshold"])

        logger.info(
            f"[Threshold Sweep]\n"
            f"  Best F1 : {best_f1:.4f}  at threshold={best_t_f1:.3f}"
        )

        sweep_path = str(self.reports / f"eval_threshold_sweep_{split_name}.png")
        plot_threshold_sweep(sweep_df, sweep_path)

        # Confusion matrix at best-F1 threshold
        preds_best = (y_pred >= best_t_f1).astype(int)
        cm_path    = str(
            self.reports / f"eval_confusion_matrix_{split_name}.png"
        )
        plot_confusion_matrix(y_true, preds_best, best_t_f1, cm_path)

        return {
            f"{split_name}_best_f1":           best_f1,
            f"{split_name}_best_f1_threshold": best_t_f1,
        }

    # Section D: Clinical utility

    def _section_clinical_utility(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
    ) -> dict:
        flag_pct  = 1 - (self.cfg.risk_tiers.clinical_flag_percentile / 100)
        util      = compute_clinical_utility(y_true, y_pred, flag_pct)

        # Business impact estimation
        # Assume: intervention prevents 30% of flagged readmissions
        # Average readmission cost: $15,000
        intervention_efficacy  = 0.30
        avg_readmission_cost   = 15_000
        avoided_readmissions   = util["tp"] * intervention_efficacy
        estimated_savings      = avoided_readmissions * avg_readmission_cost

        logger.info(
            f"[Clinical Utility — top {flag_pct*100:.0f}% flagged]\n"
            f"  PPV (precision) : {util['ppv']:.3f}\n"
            f"  NPV             : {util['npv']:.3f}\n"
            f"  Sensitivity     : {util['sensitivity']:.3f}\n"
            f"  Specificity     : {util['specificity']:.3f}\n"
            f"  NNT             : {util['nnt']:.1f}\n"
            f"  TP / FP / FN    : {util['tp']} / {util['fp']} / {util['fn']}\n"
            f"  Est. savings    : ${estimated_savings:,.0f}"
        )

        return {
            f"{split_name}_ppv":               util["ppv"],
            f"{split_name}_npv":               util["npv"],
            f"{split_name}_sensitivity":       util["sensitivity"],
            f"{split_name}_specificity":       util["specificity"],
            f"{split_name}_nnt":               util["nnt"],
            f"{split_name}_clinical_threshold":util["threshold"],
            f"{split_name}_est_savings_usd":   estimated_savings,
        }

    # Section E: Risk tier QA

    def _section_risk_tiers(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        risk_tiers: dict,
        split_name: str,
    ) -> dict:
        tier_df = compute_risk_tier_qa(y_true, y_pred, risk_tiers)

        tier_path = str(self.reports / f"eval_risk_tiers_{split_name}.png")
        plot_risk_tier_qa(tier_df, tier_path)

        # Monotonicity pass/fail
        is_monotone = bool(tier_df["monotone"].all())

        logger.info(
            f"[Risk Tiers]\n"
            f"{tier_df[['tier','n','readmission_rate']].to_string(index=False)}\n"
            f"  Monotone: {is_monotone}"
        )

        return {
            f"{split_name}_tier_monotone": int(is_monotone),
        }

    # Section F: Subgroup analysis

    def _section_subgroups(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        test_df:    pd.DataFrame,
        split_name: str,
    ) -> dict:
        subgroup_metrics: dict[str, Any] = {}

        # Define which columns to check + their labels
        subgroup_cols = {
            "age_risk_bucket":    "Age Risk Bucket",
            "gender":             "Gender",
            "diag_1_category":    "Primary Diagnosis",
        }

        for col, label in subgroup_cols.items():
            if col not in test_df.columns:
                logger.warning(f"Subgroup column '{col}' not in test_df — skipping")
                continue

            sg_df = compute_subgroup_auc(
                y_true,
                y_pred,
                test_df[col].reset_index(drop=True),
                label,
            )

            if sg_df.empty:
                continue

            sg_path = str(
                self.reports / f"eval_subgroup_{col}_{split_name}.png"
            )
            plot_subgroup_auc(sg_df, label, sg_path)

            # Log min AUC (fairness proxy — no subgroup should be far below overall)
            subgroup_metrics[f"{split_name}_subgroup_{col}_min_auc"] = float(
                sg_df["auc_roc"].min()
            )
            subgroup_metrics[f"{split_name}_subgroup_{col}_max_gap"] = float(
                sg_df["auc_roc"].max() - sg_df["auc_roc"].min()
            )

        return subgroup_metrics

    # Log to MLflow

    def _log_to_mlflow(
        self,
        metrics: dict[str, Any],
        split_name: str,
    ):
        """Logs all scalar metrics + all eval plots as artifacts."""
        # Scalar metrics
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "split"
        }
        mlflow.log_metrics(scalar_metrics)

        # All generated eval plots
        for img_path in self.reports.glob(f"eval_*_{split_name}.png"):
            mlflow.log_artifact(str(img_path))

        logger.info(
            f"Logged {len(scalar_metrics)} metrics + "
            f"plots to MLflow for split='{split_name}'"
        )

    # Save metrics

    def save_metrics(
        self,
        metrics: dict[str, Any],
        dvc_path:      str = "reports/metrics.json",
        extended_path: str = "reports/eval_metrics.json",
    ):
        """
        Saves two JSON files:
          metrics.json      → DVC metrics (only scalar floats)
          eval_metrics.json → full metrics dict including subgroups
        """
        Path(dvc_path).parent.mkdir(exist_ok=True)

        # DVC metrics — only scalars
        dvc_metrics = {
            k: round(v, 6)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "split"
        }

        with open(dvc_path, "w") as f:
            json.dump(dvc_metrics, f, indent=2)
        logger.info(f"DVC metrics saved → {dvc_path}")

        # Extended metrics — everything
        with open(extended_path, "w") as f:
            json.dump(
                {k: (round(v, 6) if isinstance(v, float) else v)
                 for k, v in metrics.items()},
                f, indent=2
            )
        logger.info(f"Extended metrics saved → {extended_path}")

        # Log both to MLflow as artifacts
        try:
            mlflow.log_artifact(dvc_path)
            mlflow.log_artifact(extended_path)
        except Exception:
            pass   # OK if called outside an active MLflow run

    # Print summary

    def _print_summary(
        self,
        metrics: dict[str, Any],
        split_name: str,
    ):
        p = split_name
        logger.info(
            f"\n{'='*55}\n"
            f"  EVALUATION SUMMARY — {split_name.upper()}\n"
            f"{'='*55}\n"
            f"  AUC-ROC       : {metrics.get(f'{p}_auc_roc', 'N/A')}\n"
            f"  AUC-PR        : {metrics.get(f'{p}_auc_pr', 'N/A')}\n"
            f"  KS Statistic  : {metrics.get(f'{p}_ks_stat', 'N/A')}\n"
            f"  Gini          : {metrics.get(f'{p}_gini', 'N/A')}\n"
            f"  ECE           : {metrics.get(f'{p}_ece', 'N/A')}\n"
            f"  MCE           : {metrics.get(f'{p}_mce', 'N/A')}\n"
            f"  Best F1       : {metrics.get(f'{p}_best_f1', 'N/A')}\n"
            f"  PPV (top 20%) : {metrics.get(f'{p}_ppv', 'N/A')}\n"
            f"  NNT           : {metrics.get(f'{p}_nnt', 'N/A')}\n"
            f"  Est. Savings  : ${metrics.get(f'{p}_est_savings_usd', 0):,.0f}\n"
            f"{'='*55}"
        )


# STANDALONE ENTRY POINT
def main():
    """
    Standalone runner — loads saved model + test split,
    runs full evaluation, saves metrics.
    Useful for re-evaluating without retraining.
    """
    import pickle
    from src.features.selector import FeatureSelector

    cfg = load_config()

    mlflow.set_tracking_uri(cfg.project.mlflow_tracking_url)
    mlflow.set_experiment(cfg.project.mlflow_experiment)

    #  Load test data
    proc     = Path(cfg.data.processed_dir)
    TARGET   = cfg.data.target_binary_col
    test_df  = pd.read_parquet(proc / "test.parquet")

    selected_features = FeatureSelector.load(
        str(proc / "selected_features.json")
    )
    selected_features = [
        f for f in selected_features if f in test_df.columns
    ]

    X_test = test_df[selected_features].values
    y_test = test_df[TARGET].values

    # Load model
    cal_path = Path("models/calibrated_model.pkl")
    if not cal_path.exists():
        raise FileNotFoundError(
            "Calibrated model not found. Run train.py first."
        )

    with open(cal_path, "rb") as f:
        calibrated_model = pickle.load(f)

    y_pred = calibrated_model.predict_proba(X_test)[:, 1]

    # Load risk tiers
    tiers_path = Path("models/risk_tier_thresholds.json")
    risk_tiers = None
    if tiers_path.exists():
        with open(tiers_path) as f:
            risk_tiers = json.load(f)

    # Run evaluation
    with mlflow.start_run(run_name="evaluate_standalone"):
        evaluator = ClinicalEvaluator(cfg)
        metrics   = evaluator.evaluate(
            y_true        = y_test,
            y_pred        = y_pred,
            test_df       = test_df,
            risk_tiers    = risk_tiers,
            split_name    = "test",
            log_to_mlflow = True,
        )
        evaluator.save_metrics(metrics)

    logger.info("Standalone evaluation complete")


if __name__ == "__main__":
    main()