import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.calibration import CalibratedClassifierCV


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file= 'logs/model.log')


class XGBModel:
    """
    XGBoost model wrapper for 30-day readmission prediction.

    Workflow:
        model = XGBoostReadmissionModel(cfg)
        best_params = model.tune(X_train, y_train, X_val, y_val, mlflow_run)
        model.fit(X_train, y_train, X_val, y_val)
        model.calibrate(X_val, y_val)
        probs = model.predict_proba(X_test)
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.study:             optuna.Study | None = None
        self.best_params:       dict[str, Any]      = {}
        self.base_model:        xgb.XGBClassifier | None = None
        self.calibrated_model:  CalibratedClassifierCV | None = None
        self._scale_pos_weight: float = 1.0
        
    #  Internal helpers      
    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """Balances the positive / negative class ratio for XGBoost."""
        
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        ratio = neg / pos
        
        logger.info(f"Computed scale_pos_weight: {ratio:.4f} (neg: {neg}, pos: {pos})")
        
        return float(ratio)
    
    def _build_xgb_params(self, trial_params: dict | None = None) -> dict:
        """
        Merges config defaults with trial-suggested params.
        trial_params (from Optuna) override config defaults.
        """
        
        defaults = dict(self.cfg.model.default_params)
        defaults['scale_pos_weight'] = self._scale_pos_weight
        
        
        if trial_params:
            defaults.update(trial_params)
            
        return defaults
    
    
    # Optuna objective
    def _make_objective(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        mlflow_parent_run_id: str | None = None):
        """
        Returns a closure that Optuna calls for each trial.
        Each trial is logged as an MLflow child run.
        """
        
        import mlflow
        from sklearn.metrics import roc_auc_score
        
        space = self.cfg.model.optuna.search_space
        
        
        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "n_estimators":      trial.suggest_int(
                    "n_estimators",
                    *space.n_estimators
                ),
                "learning_rate":     trial.suggest_float(
                    "learning_rate",
                    *space.learning_rate,
                    log=True
                ),
                "max_depth":         trial.suggest_int(
                    "max_depth",
                    *space.max_depth
                ),
                "subsample":         trial.suggest_float(
                    "subsample",
                    *space.subsample
                ),
                "colsample_bytree":  trial.suggest_float(
                    "colsample_bytree",
                    *space.colsample_bytree
                ),
                "min_child_weight":  trial.suggest_int(
                    "min_child_weight",
                    *space.min_child_weight
                ),
                "gamma":             trial.suggest_float(
                    "gamma",
                    *space.gamma
                ),
                "reg_alpha":         trial.suggest_float(
                    "reg_alpha",
                    *space.reg_alpha,
                    log=True
                ),
                "reg_lambda":        trial.suggest_float(
                    "reg_lambda",
                    *space.reg_lambda,
                    log=True
                ),
            }

            params = self._build_xgb_params(trial_params)

            # ── MLflow child run per trial ────────────────────
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True
            ):
                mlflow.log_params(trial_params)

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                val_probs = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, val_probs)

                mlflow.log_metric("val_auc_roc", auc)

            return auc

        return objective
    
    
    def tune(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             mlflow_parent_run_id: str | None = None) -> dict:
        """
        Runs Optuna TPE study.
        Returns best hyperparameters dict.
        """
        
        self._scale_pos_weight = self._compute_scale_pos_weight(y_train)
        
        optuna_cfg = self.cfg.model.optuna
        
        # Persist study across runs (SQLite)
        storage_dir = Path('data/hpo_history')
        storage_dir.mkdir(parents= True, exist_ok= True)
        storage = f'sqlite:///{storage_dir}/optuna_studies.db'
        
        
        self.study = optuna.create_study(
            study_name= optuna_cfg.study_name,
            direction= optuna_cfg.direction,
            storage= storage,
            load_if_exists= True,
            sampler= optuna.samplers.TPESampler(
                n_startup_trials= 10,
                seed= self.cfg.project.random_seed,
            ),
            pruner= optuna.pruners.MedianPruner(
                n_startup_trials= 5,
                n_warmup_steps= 10,
            ),
        )
        
        logger.info(f'Starting optuna study {optuna_cfg.study_name} with {optuna_cfg.n_trials} trials...')
        
        
        self.study.optimize(
            self._make_objective(X_train, y_train, X_val, y_val, mlflow_parent_run_id),
            n_trials=optuna_cfg.n_trials,
            show_progress_bar=True,
        )
        
        self.best_params = self.study.best_params.copy()
        
        logger.info(f'Best trial: {self.study.best_trial.number} with value: {self.study.best_value:.4f}'
                    f'Best hyperparameters: {self.best_params}')
        
        
        return self.best_params
    
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> 'XGBModel':
        """
        Trains the final XGBoost model using best_params.
        Call tune() first or set best_params manually.
        """
        
        if not self.best_params:
            logger.warning('No best_params found. Using default config parameters.')
            
        self._scale_pos_weight = self._compute_scale_pos_weight(y_train)
        params = self._build_xgb_params(self.best_params)
        
        
        logger.info(f'Training XGBoost with params: {params}')
        
        
        self.base_model = xgb.XGBClassifier(**params)
        self.base_model.fit(
            X_train, y_train,
            eval_set= [(X_val, y_val)],
            verbose= 100,
        )
        
        logger.info('Model training completed.')
        
        return self
    
    
    def calibrate(self,
                  X_cal: np.ndarray,
                  y_cal: np.ndarray) -> 'XGBModel':        
        """
        Wraps base_model in CalibratedClassifierCV (prefit).
        Uses the calibration set (val split) — never the train set.
        """
        
        if self.base_model is None:
            raise ValueError("Base model not trained. Call fit() before calibrate().")
        
        method = self.cfg.calibration.method
        logger.info(f'Calibrating model using method: {method}')
        
        self.calibrated_model = CalibratedClassifierCV(
            estimator= self.base_model, 
            method= method,
            cv= 'prefit'
        )
        
        self.calibrated_model.fit(X_cal, y_cal)
        
        logger.info('Model calibration completed.')
        
        return self
    
    
    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated probabilities if calibrate() was called,
        else raw XGBoost probabilities.
        """
        
        model = self.calibrated_model or self.base_model
        if model is None:
            raise ValueError("Model not trained. Call fit() before predict_probabilities().")
        
        return model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
    

    def get_feature_importance(
        self,
        feature_names: list[str]
    ) -> "pd.DataFrame":
        import pandas as pd
        if self.base_model is None:
            raise RuntimeError("Model not fitted yet.")
        return (
            pd.DataFrame({
                "feature":    feature_names,
                "importance": self.base_model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


    def save(self, model_dir: str = "models"):
        """Saves base_model as XGBoost binary and calibrated_model as pickle."""
        import pickle
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.base_model:
            xgb_path = path / "xgb_base.ubj"
            self.base_model.save_model(str(xgb_path))
            logger.info(f"Base model saved → {xgb_path}")

        if self.calibrated_model:
            cal_path = path / "calibrated_model.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump(self.calibrated_model, f)
            logger.info(f"Calibrated model saved → {cal_path}")

    @classmethod
    def load(cls, model_dir: str = "models", cfg=None) -> "XGBModel":
        """Loads a previously saved model."""
        import pickle
        if cfg is None:
            cfg = load_config()

        instance = cls(cfg)
        path = Path(model_dir)

        xgb_path = path / "xgb_base.ubj"
        if xgb_path.exists():
            instance.base_model = xgb.XGBClassifier()
            instance.base_model.load_model(str(xgb_path))
            logger.info(f"Base model loaded ← {xgb_path}")

        cal_path = path / "calibrated_model.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                instance.calibrated_model = pickle.load(f)
            logger.info(f"Calibrated model loaded ← {cal_path}")

        return instance