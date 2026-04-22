import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow import MlflowClient

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.xgb_model import XGBModel
from src.features.selector import FeatureSelector

logger = get_logger(__name__, log_file="logs/training.log")


# HELPERS
def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (mask.sum() / len(y_pred)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_risk_tiers(
    scores: np.ndarray,
    y_true: np.ndarray,
    cfg,
) -> dict:
    """
    Defines Low / Medium / High / Critical risk tiers from
    the positive-class score distribution.
    Returns thresholds + per-tier readmission rates.
    """
    pos_scores = scores[y_true == 1]
    p25 = float(np.percentile(pos_scores, 25))
    p50 = float(np.percentile(pos_scores, 50))
    p75 = float(np.percentile(pos_scores, 75))

    def tier(s):
        if s < p25:   return "Low"
        elif s < p50: return "Medium"
        elif s < p75: return "High"
        else:         return "Critical"

    tiers = pd.Series(scores).apply(tier)
    summary = (
        pd.DataFrame({"tier": tiers.values, "readmitted": y_true})
        .groupby("tier")["readmitted"]
        .agg(["count", "mean"])
        .reindex(["Low", "Medium", "High", "Critical"])
    )

    return {
        "p25": p25, "p50": p50, "p75": p75,
        "tier_summary": summary,
    }


# PLOT HELPERS

def plot_optuna_convergence(study, save_path: str):
    values = [t.value for t in study.trials if t.value is not None]
    plt.figure(figsize=(10, 4))
    plt.plot(values, alpha=0.5, label="Trial AUC")
    plt.plot(
        pd.Series(values).cummax(),
        color="red", linewidth=2, label="Best so far"
    )
    plt.xlabel("Trial")
    plt.ylabel("Validation AUC-ROC")
    plt.title("Optuna Optimisation Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    save_path: str
):
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, raw_probs, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, cal_probs, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].plot(mean_pred_raw, frac_pos_raw, "s-",
                 color="#FF5722", label="XGBoost raw")
    axes[0].plot(mean_pred_cal, frac_pos_cal, "o-",
                 color="#4CAF50", label="Isotonic calibrated")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()

    axes[1].hist(raw_probs[y_true == 0], bins=40, alpha=0.6,
                 label="Negative class", color="#2196F3")
    axes[1].hist(raw_probs[y_true == 1], bins=40, alpha=0.6,
                 label="Positive class", color="#FF5722")
    axes[1].set_title("Score Distribution")
    axes[1].set_xlabel("Predicted Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr(
    y_true: np.ndarray,
    probs: np.ndarray,
    save_path: str
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_roc = roc_auc_score(y_true, probs)
    axes[0].plot(fpr, tpr, color="#2196F3", lw=2,
                 label=f"XGBoost (AUC={auc_roc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Test Set")
    axes[0].legend()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, probs)
    auc_pr = average_precision_score(y_true, probs)
    baseline = y_true.mean()
    axes[1].plot(recall, precision, color="#FF5722", lw=2,
                 label=f"XGBoost (AP={auc_pr:.3f})")
    axes[1].axhline(baseline, linestyle="--", color="gray",
                    label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — Test Set")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_tiers(tier_info: dict, overall_rate: float, save_path: str):
    summary = tier_info["tier_summary"]
    colors  = {
        "Low": "#4CAF50",
        "Medium": "#FFC107",
        "High": "#FF9800",
        "Critical": "#F44336"
    }
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        summary.index,
        summary["mean"],
        color=[colors[t] for t in summary.index]
    )
    plt.axhline(overall_rate, linestyle="--", color="black",
                label="Overall rate")
    plt.ylabel("Actual Readmission Rate")
    plt.title("Readmission Rate by Risk Tier")
    plt.legend()
    for bar, (_, row) in zip(bars, summary.iterrows()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={int(row['count'])}",
            ha="center", fontsize=10
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# MAIN TRAINING PIPELINE
def run_training_pipeline(cfg):

    # 0. MLflow setup
    mlflow.set_tracking_uri(cfg.project.mlflow_tracking_url)
    mlflow.set_experiment(cfg.project.mlflow_experiment)
    logger.info(
        f"MLflow tracking URI : {cfg.project.mlflow_tracking_url}\n"
        f"Experiment          : {cfg.project.mlflow_experiment}"
    )

    # 1. Load processed data
    proc = Path(cfg.data.processed_dir)
    TARGET = cfg.data.target_binary_col

    logger.info("Loading processed splits ...")
    train_df = pd.read_parquet(proc / "train.parquet")
    val_df   = pd.read_parquet(proc / "val.parquet")
    test_df  = pd.read_parquet(proc / "test.parquet")

    # 2. Load selected features
    selected_features = FeatureSelector.load(
        str(proc / "selected_features.json")
    )
    # Keep only features that exist in the splits
    selected_features = [
        f for f in selected_features if f in train_df.columns
    ]
    logger.info(f"Using {len(selected_features)} selected features")

    X_train = train_df[selected_features].values
    y_train = train_df[TARGET].values
    X_val   = val_df[selected_features].values
    y_val   = val_df[TARGET].values
    X_test  = test_df[selected_features].values
    y_test  = test_df[TARGET].values

    # 3. Create reports dir
    Path("reports").mkdir(exist_ok=True)

    # PARENT MLFLOW RUN
    with mlflow.start_run(run_name="xgboost_readmission_full") as parent_run:

        run_id = parent_run.info.run_id
        logger.info(f"MLflow parent run_id: {run_id}")

        # ── Log dataset metadata ──────────────────────────────
        mlflow.log_params({
            "dataset":            "diabetes_130_uci",
            "n_train":            len(X_train),
            "n_val":              len(X_val),
            "n_test":             len(X_test),
            "n_features":         len(selected_features),
            "pos_rate_train":     round(y_train.mean(), 4),
            "pos_rate_val":       round(y_val.mean(), 4),
            "pos_rate_test":      round(y_test.mean(), 4),
            "feature_selection":  "mi_lasso_rf_union",
            "calibration_method": cfg.model.calibration_method,
        })

        # 4. Initialise model
        model = XGBModel(cfg)

        # 5. Hyperparameter tuning (child runs inside)
        logger.info("Starting hyperparameter tuning ...")
        best_params = model.tune(
            X_train, y_train,
            X_val,   y_val,
            mlflow_parent_run_id=run_id,
        )

        # Log best HPO result to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("hpo_best_val_auc", model.study.best_value)

        # Save + log convergence plot
        conv_path = "reports/optuna_convergence.png"
        plot_optuna_convergence(model.study, conv_path)
        mlflow.log_artifact(conv_path)

        # 6. Train final model
        logger.info("Training final model with best params ...")
        model.fit(X_train, y_train, X_val, y_val)

        # Raw (uncalibrated) val predictions for comparison
        raw_val_probs = model.base_model.predict_proba(X_val)[:, 1]
        ece_raw = compute_ece(y_val, raw_val_probs)

        mlflow.log_metrics({
            "val_auc_roc_raw": roc_auc_score(y_val, raw_val_probs),
            "val_auc_pr_raw":  average_precision_score(y_val, raw_val_probs),
            "val_ece_raw":     ece_raw,
        })

        # 7. Calibrate
        logger.info("Calibrating model ...")
        model.calibrate(X_val, y_val)

        cal_val_probs = model.predict_probabilities(X_val)
        ece_cal = compute_ece(y_val, cal_val_probs)

        mlflow.log_metrics({
            "val_auc_roc_cal": roc_auc_score(y_val, cal_val_probs),
            "val_auc_pr_cal":  average_precision_score(y_val, cal_val_probs),
            "val_ece_cal":     ece_cal,
        })
        logger.info(
            f"ECE before calibration : {ece_raw:.4f}\n"
            f"ECE after  calibration : {ece_cal:.4f}"
        )

        # Calibration curve plot
        cal_path = "reports/calibration_curve.png"
        plot_calibration_curve(y_val, raw_val_probs, cal_val_probs, cal_path)
        mlflow.log_artifact(cal_path)

        # 8. Test set evaluation
        logger.info("Evaluating on test set ...")
        test_probs = model.predict_probabilities(X_test)

        test_auc_roc = roc_auc_score(y_test, test_probs)
        test_auc_pr  = average_precision_score(y_test, test_probs)
        test_ece     = compute_ece(y_test, test_probs)

        # Clinical utility — top 20% flagged
        threshold_20 = float(np.percentile(test_probs, 80))
        clinical_preds = (test_probs >= threshold_20).astype(int)
        tp = int(((clinical_preds == 1) & (y_test == 1)).sum())
        fp = int(((clinical_preds == 1) & (y_test == 0)).sum())
        fn = int(((clinical_preds == 0) & (y_test == 1)).sum())
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        nnt = 1 / ppv if ppv > 0 else float("inf")

        mlflow.log_metrics({
            "test_auc_roc":      test_auc_roc,
            "test_auc_pr":       test_auc_pr,
            "test_ece":          test_ece,
            "clinical_ppv":      ppv,
            "clinical_nnt":      nnt,
            "clinical_threshold": threshold_20,
        })

        logger.info(
            f"\n{'='*50}\n"
            f"TEST SET RESULTS\n"
            f"{'='*50}\n"
            f"AUC-ROC  : {test_auc_roc:.4f}\n"
            f"AUC-PR   : {test_auc_pr:.4f}\n"
            f"ECE      : {test_ece:.4f}\n"
            f"PPV(20%) : {ppv:.3f}\n"
            f"NNT      : {nnt:.1f}\n"
            f"{'='*50}"
        )

        # ROC + PR curves
        roc_path = "reports/roc_pr_curves.png"
        plot_roc_pr(y_test, test_probs, roc_path)
        mlflow.log_artifact(roc_path)

        # 9. Risk tier analysis
        tier_info = compute_risk_tiers(test_probs, y_test, cfg)
        tier_thresholds = {
            "p25": tier_info["p25"],
            "p50": tier_info["p50"],
            "p75": tier_info["p75"],
        }

        mlflow.log_params({
            "risk_tier_p25": round(tier_info["p25"], 4),
            "risk_tier_p50": round(tier_info["p50"], 4),
            "risk_tier_p75": round(tier_info["p75"], 4),
        })

        tier_path = "reports/risk_tier_analysis.png"
        plot_risk_tiers(tier_info, y_test.mean(), tier_path)
        mlflow.log_artifact(tier_path)
        
        #  10. Full clinical evaluation
        from src.training.evaluate import ClinicalEvaluator

        evaluator = ClinicalEvaluator(cfg)

        # Val evaluation
        evaluator.evaluate(
            y_true        = y_val,
            y_pred        = cal_val_probs,
            test_df       = val_df,
            risk_tiers    = tier_thresholds,
            split_name    = "val",
            log_to_mlflow = True,
        )

        # Test evaluation (primary)
        test_metrics = evaluator.evaluate(
            y_true        = y_test,
            y_pred        = test_probs,
            test_df       = test_df,
            risk_tiers    = tier_thresholds,
            split_name    = "test",
            log_to_mlflow = True,
        )

        # Save DVC metrics file
        evaluator.save_metrics(test_metrics)

        # 11. Feature importance
        fi_df = model.get_feature_importance(selected_features)
        fi_path = "reports/feature_importance.png"

        import seaborn as sns
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=fi_df.head(20),
            x="importance", y="feature",
            color="#2196F3"
        )
        plt.title("Top 20 Feature Importances — XGBoost")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(fi_path)

        # 12. Save model to MLflow + registry
        # Log calibrated sklearn model
        mlflow.sklearn.log_model(
            model.calibrated_model,
            name="calibrated_model",
            registered_model_name="hospital-readmission-risk",
        )

        # Also log raw XGBoost (for SHAP, etc.)
        mlflow.xgboost.log_model(
            model.base_model,
            name="xgb_base_model",
        )

        # Save locally (DVC will track this)
        model.save("models")

        # 13. Save tier thresholds for serving
        tier_thresholds = {
            "p25": tier_info["p25"],
            "p50": tier_info["p50"],
            "p75": tier_info["p75"],
            "clinical_threshold_20pct": threshold_20,
        }
        thresholds_path = "models/risk_tier_thresholds.json"
        with open(thresholds_path, "w") as f:
            json.dump(tier_thresholds, f, indent=2)
        mlflow.log_artifact(thresholds_path)

        # 13. Promote to Production in registry
        client = MlflowClient()
        latest = client.get_latest_versions(
            "hospital-readmission-risk", stages=["None"]
        )
        if latest:
            version = latest[-1].version
            client.transition_model_version_stage(
                name="hospital-readmission-risk",
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                f"Model v{version} promoted to Production "
                f"in MLflow registry"
            )

        logger.info(f"Training pipeline complete — run_id: {run_id}")
        return model, run_id


def main():
    cfg = load_config()
    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()