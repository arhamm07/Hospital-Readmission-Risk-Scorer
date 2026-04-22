import sys
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__, log_file= 'logs/feature_selection.log')


class FeatureSelector:
    """
    Runs 3 feature selection methods and returns a consensus feature list.

    Methods:
      1. Mutual Information     — filter method, model-agnostic
      2. LASSO (L1 Logistic)    — embedded, finds linearly relevant features
      3. Random Forest Importance — embedded, captures non-linear importance

    Consensus: union of top-K features across all methods.
    Saved artifacts:
      - reports/feature_importance_mi.png
      - reports/feature_importance_lasso.png
      - reports/feature_importance_rf.png
      - reports/feature_consensus_heatmap.png
      - data/processed/selected_features.json
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k = cfg.features.selection_top_k
        
        # Results populated after fit()
        self.mi_scores:     pd.DataFrame | None = None
        self.lasso_scores:  pd.DataFrame | None = None
        self.rf_scores:     pd.DataFrame | None = None
        self.selected_features: list[str] = []
        self.consensus_df:  pd.DataFrame | None = None
        
    # method 1 : Mutual Information
    def _run_mutual_info(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         feature_names: list[str]) -> pd.DataFrame:
        logger.info("Running Mutual Information feature selection...")
        
        mi_scores = mutual_info_classif(X_train, y_train, random_state = self.cfg.project.random_seed)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores
         }).sort_values(by='mi_score', ascending=False).reset_index(drop= True)
        
        logger.info(f"Top {self.top_k} features by MI:\n{df.head(self.top_k)}")
        
        return df
    
    
    # method 2 : LASSO (L1 Logistic Regression)
    
    def _run_lasso(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   feature_names: list[str]) -> pd.DataFrame:
        
        logger.info("Running LASSO (L1 Logistic Regression) feature selection...")
        
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X_train)
        
        lasso = LogisticRegression(penalty= 'l1',
                                   solver= 'liblinear',
                                   C= self.cfg.features.lasso_C,
                                   max_iter= 500,
                                   random_state= self.cfg.project.random_seed)
        lasso.fit(X_scaled, y_train)
        
        
        df = pd.DataFrame({
            'feature': feature_names,
            'lasso_coef': np.abs(lasso.coef_[0])
         }).sort_values(by='lasso_coef', ascending=False).reset_index(drop= True)
        
        n_nonzero = (lasso.coef_[0] != 0).sum()
        logger.info(f"LASSO selected {n_nonzero} features with non-zero coefficients.")
        logger.info(f"Top {self.top_k} features by LASSO:\n{df.head(self.top_k)}")
        
        return df
    
    
    # method 3 : Random Forest Importance
    def _run_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str]) -> pd.DataFrame:
        logger.info("Running Random Forest importance ...")

        rf = RandomForestClassifier(
            n_estimators=self.cfg.feature_selection.rf_n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.cfg.project.random_seed
        )
        rf.fit(X_train, y_train)

        df = pd.DataFrame({
            "feature": feature_names,
            "rf_importance": rf.feature_importances_
        }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

        logger.info(''f"Top {self.top_k} features by Random Forest:\n{df.head(self.top_k)}")
        return df

    # Consensus

    def _build_consensus(self) -> list[str]:
        """
        Normalises each method's score to [0, 1] then averages.
        Returns union of top-K from each method.
        """
        top_mi    = set(self.mi_scores.head(self.top_k)["feature"])
        top_lasso = set(self.lasso_scores.head(self.top_k)["feature"])
        top_rf    = set(self.rf_scores.head(self.top_k)["feature"])

        in_all_three  = top_mi & top_lasso & top_rf
        in_at_least_2 = (
            (top_mi & top_lasso) |
            (top_mi & top_rf)    |
            (top_lasso & top_rf)
        )
        union = top_mi | top_lasso | top_rf

        logger.info(f"In all 3 methods:    {len(in_all_three)}")
        logger.info(f"In at least 2:       {len(in_at_least_2)}")
        logger.info(f"Union (final set):   {len(union)}")

        # Build normalised consensus DataFrame
        all_features = list(union)
        records = []
        for feat in all_features:
            mi_rank    = self.mi_scores[
                self.mi_scores["feature"] == feat
            ].index[0] if feat in self.mi_scores["feature"].values else len(all_features)
            lasso_rank = self.lasso_scores[
                self.lasso_scores["feature"] == feat
            ].index[0] if feat in self.lasso_scores["feature"].values else len(all_features)
            rf_rank    = self.rf_scores[
                self.rf_scores["feature"] == feat
            ].index[0] if feat in self.rf_scores["feature"].values else len(all_features)

            records.append({
                "feature":    feat,
                "mi_rank":    mi_rank,
                "lasso_rank": lasso_rank,
                "rf_rank":    rf_rank,
                "mean_rank":  np.mean([mi_rank, lasso_rank, rf_rank]),
                "in_all_3":   feat in in_all_three,
                "in_2plus":   feat in in_at_least_2,
            })

        self.consensus_df = pd.DataFrame(records).sort_values("mean_rank")
        return list(union)

    # Plotting

    def _plot_importance(
        self,
        df: pd.DataFrame,
        score_col: str,
        title: str,
        save_path: str,
        color: str = "steelblue"
    ):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df.head(25),
            x=score_col,
            y="feature",
            color=color
        )
        plt.title(title)
        plt.xlabel(score_col)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved → {save_path}")

    def _plot_consensus_heatmap(self, save_path: str):
        """
        Heatmap: features (rows) × methods (cols), colour = normalised rank.
        Dark = high rank (important). Light = low rank (less important).
        """
        if self.consensus_df is None:
            return

        n = len(self.consensus_df)
        heat_df = self.consensus_df.set_index("feature")[
            ["mi_rank", "lasso_rank", "rf_rank"]
        ].copy()

        # Normalise: lower rank = higher score
        for col in heat_df.columns:
            heat_df[col] = 1 - (heat_df[col] / n)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, max(6, len(heat_df) * 0.3)))
        sns.heatmap(
            heat_df,
            cmap="YlOrRd",
            linewidths=0.3,
            annot=False,
            cbar_kws={"label": "Normalised Rank (higher = more important)"}
        )
        plt.title("Feature Consensus Heatmap\n(Features × Selection Methods)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Consensus heatmap saved → {save_path}")

    # Main entry point

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str]
    ) -> list[str]:
        """
        Runs all 3 selection methods and returns the final
        consensus feature list.
        """
        logger.info("=" * 50)
        logger.info("Starting Feature Selection")
        logger.info("=" * 50)

        # Run methods
        self.mi_scores    = self._run_mutual_info(X_train, y_train, feature_names)
        self.lasso_scores = self._run_lasso(X_train, y_train, feature_names)
        self.rf_scores    = self._run_random_forest(X_train, y_train, feature_names)

        # Build consensus
        self.selected_features = self._build_consensus()

        # Plots
        self._plot_importance(
            self.mi_scores, "mi_score",
            "Top 25 Features — Mutual Information",
            "reports/feature_importance_mi.png",
            color="#2196F3"
        )
        self._plot_importance(
            self.lasso_scores, "lasso_coef",
            "Top 25 Features — LASSO |Coefficients|",
            "reports/feature_importance_lasso.png",
            color="#FF5722"
        )
        self._plot_importance(
            self.rf_scores, "rf_importance",
            "Top 25 Features — Random Forest Importance",
            "reports/feature_importance_rf.png",
            color="#4CAF50"
        )
        self._plot_consensus_heatmap("reports/feature_consensus_heatmap.png")

        logger.info(
            f"Feature selection complete — "
            f"{len(self.selected_features)} features selected"
        )
        return self.selected_features

    def save(self, save_path: str = "data/processed/selected_features.json"):
        """Persists the selected feature list to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "selected_features": self.selected_features,
            "n_features": len(self.selected_features),
            "top_k_per_method": self.top_k,
            "consensus_top20": (
                self.consensus_df.head(20)["feature"].tolist()
                if self.consensus_df is not None else []
            )
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Selected features saved → {save_path}")

    @staticmethod
    def load(load_path: str = "data/processed/selected_features.json") -> list[str]:
        """Loads the saved feature list from disk."""
        with open(load_path) as f:
            payload = json.load(f)
        logger.info(
            f"Loaded {payload['n_features']} selected features from {load_path}"
        )
        return payload["selected_features"]


def main():
    cfg = load_config()

    # Load processed train split
    train_path = Path(cfg.data.processed_dir) / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed train not found at {train_path}. "
            f"Run feature engineering first."
        )

    train_df = pd.read_parquet(train_path)
    TARGET   = cfg.data.target_binary_col

    feature_names = [c for c in train_df.columns if c != TARGET]
    X_train = train_df[feature_names].values
    y_train = train_df[TARGET].values

    logger.info(
        f"Loaded train — {len(X_train):,} samples × "
        f"{len(feature_names)} features"
    )

    # Run selector
    selector = FeatureSelector(cfg)
    selected = selector.fit(X_train, y_train, feature_names)
    selector.save()

    logger.info(f"\nFinal selected features ({len(selected)}):")
    for f in selected:
        logger.info(f"  • {f}")


if __name__ == "__main__":
    main()