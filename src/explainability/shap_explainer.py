import sys 
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__, log_file= 'logs/shap_explainer.log')


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for XGBoost.

    Methods:
        fit(X_background)         — creates the TreeExplainer
        explain_global(X)         — summary + beeswarm plots
        explain_local(x)          — waterfall + ranked dict for one patient
        save_shap_values(X, path) — saves shap matrix to parquet
    """
    
    def __init__(self,
                 model,
                 feature_names: list[str],
                 cfg,):
        
        self.model = model
        self.feature_names = feature_names
        self.cfg = cfg
        self.explainer = None
        self.expected_value: float = 0.0
        
        self.reports = Path('reports')
        self.reports.mkdir(exist_ok= True)
        
        
    def fit(self,
            X_background: np.ndarray) -> 'SHAPExplainer':
        """
        Creates shap.Explainer using the generic Explainer API.
        Uses a lambda wrapper around predict_proba to avoid XGBoost 3.x compatibility issues.
        X_background is used for the expected value baseline.
        Pass X_train (or a sample of it).
        """
        
        logger.info(
            f"Creating SHAP Explainer — "
            f"background shape: {X_background.shape}"
        )
        
        # Use shap.Explainer with lambda to avoid XGBoost 3.x base_score parsing issues
        self.explainer = shap.Explainer(
            lambda x: self.model.predict_proba(x)[:, 1],
            X_background,
            feature_names=self.feature_names,
        )
        
        # Compute expected value as the average prediction on background data
        # (PermutationExplainer doesn't have expected_value attribute)
        self.expected_value = float(
            self.model.predict_proba(X_background)[:, 1].mean()
        )
        
        logger.info(
            f"SHAP Explainer created — "
            f"expected_value (base rate): {self.expected_value:.4f}"
        )
        
        return self
    
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        sample_size: int | None = 500,) -> np.ndarray:
        """
        Computes SHAP values for X.
        Samples to sample_size rows if X is large
        (global analysis doesn't need all rows).
        """
        
        if self.explainer is None:
            raise ValueError("Call fit() before compute_shap_values()")
        
        
        if sample_size and len(X) > sample_size:
            rng = np.random.default_rng(self.cfg.project.random_seed)
            idx = rng.choice(len(X), size= sample_size, replace= False)
            X_in = X[idx]
            logger.info(
                f"Computing SHAP values — "
                f"input shape: {X.shape} — "
                f"sampled {sample_size} rows"
            )
            
        else:
            X_in = X
            
            
            
        shap_values = self.explainer.shap_values(X_in)
        logger.info(
            f"SHAP values computed — "
            f"output shape: {shap_values.shape}"
        )
        
        return shap_values, X_in
    
    
    
    def explain_global(
        self,
        X: np.ndarray,
        save_prefix: str = "shap",
    ) -> dict[str, str]:
        """
        Generates:
          1. Bar plot  — mean |SHAP| per feature
          2. Beeswarm  — distribution of SHAP values per feature

        Returns dict of {plot_name: file_path}.
        """
        shap_values, X_sample = self.compute_shap_values(X)
        saved_paths: dict[str, str] = {}

        # 1. Bar plot
        bar_path = str(self.reports / f"{save_prefix}_summary_bar.png")
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title("Global SHAP Feature Importance (mean |SHAP|)")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths["bar"] = bar_path
        logger.info(f"SHAP bar plot saved → {bar_path}")

        # 2. Beeswarm
        bee_path = str(self.reports / f"{save_prefix}_beeswarm.png")
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Beeswarm — Feature Impact Distribution")
        plt.tight_layout()
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths["beeswarm"] = bee_path
        logger.info(f"SHAP beeswarm saved → {bee_path}")

        # ── 3. Feature importance DataFrame ──────────────────
        mean_abs = np.abs(shap_values).mean(axis=0)
        self.global_importance = pd.DataFrame({
            "feature":      self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        logger.info(
            f"Top 5 global SHAP features:\n"
            f"{self.global_importance.head(5).to_string(index=False)}"
        )

        return saved_paths
    
    
    
    def explain_local(
        self,
        x_single: np.ndarray,
        patient_id: str = "patient",
        save_plot:  bool = True,
    ) -> dict[str, Any]:
        """
        Explains a single patient prediction.

        Returns:
            {
              "patient_id":     str,
              "base_value":     float,    ← model's average prediction
              "predicted_risk": float,    ← this patient's score
              "top_factors":    [         ← top K features sorted by |SHAP|
                {
                  "feature":   str,
                  "value":     float,     ← actual feature value
                  "shap_impact": float,   ← SHAP contribution
                  "direction": str        ← "increases_risk" / "decreases_risk"
                }, ...
              ]
            }
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain_local()")

        x = x_single.reshape(1, -1) if x_single.ndim == 1 else x_single

        shap_vals  = self.explainer.shap_values(x)[0]
        pred_score = float(
            self.model.predict_proba(x)[0, 1]
        )

        top_k = self.cfg.serving.shap_top_k

        factors = []
        for feat, val, sv in zip(
            self.feature_names,
            x[0],
            shap_vals
        ):
            factors.append({
                "feature":     feat,
                "value":       round(float(val), 4),
                "shap_impact": round(float(sv), 4),
                "direction":   (
                    "increases_risk" if sv > 0 else "decreases_risk"
                ),
            })

        factors = sorted(
            factors,
            key=lambda d: abs(d["shap_impact"]),
            reverse=True
        )[:top_k]

        explanation = {
            "patient_id":     patient_id,
            "base_value":     round(self.expected_value, 4),
            "predicted_risk": round(pred_score, 4),
            "top_factors":    factors,
        }

        # Waterfall plot
        if save_plot:
            wf_path = str(
                self.reports / f"shap_waterfall_{patient_id}.png"
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=self.expected_value,
                    data=x[0],
                    feature_names=self.feature_names,
                ),
                show=False,
                max_display=15,
            )
            plt.title(
                f"SHAP Explanation — {patient_id} "
                f"(risk score: {pred_score:.3f})"
            )
            plt.tight_layout()
            plt.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.close()
            explanation["waterfall_plot"] = wf_path
            logger.info(f"Waterfall plot saved → {wf_path}")

        return explanation
    
    
    def save_shap_values(
        self,
        X: np.ndarray,
        save_path: str = "data/processed/shap_values.parquet",
    ):
        """Saves full SHAP value matrix for downstream analysis."""
        shap_values, X_sample = self.compute_shap_values(X, sample_size=None)

        df = pd.DataFrame(
            shap_values,
            columns=[f"shap_{f}" for f in self.feature_names]
        )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info(f"SHAP values saved → {save_path}  shape={df.shape}")

        
        