import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__, log_file='logs/feature_engineering.log')


# Raw data cleaner

class DataCleaner:
    """
    Handles all cleaning steps that must happen BEFORE
    any feature engineering:
      - Replace '?' with NaN
      - Drop irrelevant / high-missing / leaky columns
      - Encode binary target
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f'starting cleaning data with shape {df.shape}')
        
        df = df.copy()
        
        # 1. Replace '?' sentinel with NaN
        df = df.replace('?', np.nan)
        logger.info('replaced "?" with NaN')
        
        # 2. Binary target
        df[self.cfg.data.target_binary_col] = (
            df[self.cfg.data.target_col] == self.cfg.data.positive_class
            ).astype(int)
        logger.info(f'encoded binary target column "{self.cfg.data.target_binary_col}"'
                    f' Positive rate: {df[self.cfg.data.target_binary_col].mean():.4f}')
        
        # 3. Drop configured columns
        drop_cols = [c for c in self.cfg.data.drop_columns if c in df.columns]
        df = df.drop(columns=drop_cols)
        logger.info(f'dropped columns: {drop_cols}')
        
        # 4. Drop high-missing columns
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[
            missing_pct > self.cfg.data.high_missing_threshold
        ].index.tolist()
        if high_missing:
            df = df.drop(columns=high_missing)
            logger.info(
                f"Dropped high-missing (>{self.cfg.data.high_missing_threshold}%): "
                f"{high_missing}"
            )
        
        # 5. Near-zero-variance: drop cols with only 1 unique value
        nzv = [c for c in df.columns if df[c].nunique() <= 1]
        if nzv:
            df = df.drop(columns=nzv)
            logger.warning(f"Dropped near-zero-variance: {nzv}")

        logger.info(f"Output shape: {df.shape}")
            
        return df
    
    
class FeatureEngineer:
    """
    Builds 50+ clinical features from the cleaned DataFrame.

    Feature groups:
      A. Admission features     — prior visits, admission source
      B. Clinical features      — diagnoses, procedures, medications
      C. Lab features           — abnormal labs, A1C flags
      D. Demographic features   — age buckets, insurance
      E. Medication features    — active meds, insulin changes
      F. Interaction features   — cross-feature combinations
      G. Diagnosis category     — ICD-10 mapped categories
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.age_order = list(self.cfg.features.age_buckets)
        self.age_map   = {bracket: i for i, bracket in enumerate(self.age_order)}
        self.med_cols_used: list[str] = []   # populated in fit_transform

    # A: Admission Features

    def _build_admission_features(self, df: pd.DataFrame) -> pd.DataFrame:
        visit_cols = [
            c for c in self.cfg.features.visit_columns
            if c in df.columns
        ]

        # Total prior visits across all channels
        df["total_prior_visits"] = df[visit_cols].sum(axis=1)

        # High-utiliser flag (top quartile)
        q75 = df["total_prior_visits"].quantile(
            self.cfg.features.high_utilizer_quantile
        )
        df["high_utilizer"] = (df["total_prior_visits"] > q75).astype(int)

        # Binary flags per visit type
        for col in visit_cols:
            df[f"has_{col}"] = (df[col] > 0).astype(int)

        logger.info(
            f"Admission features built "
            f"(visit_cols={visit_cols})"
        )
        return df

    # B: Clinical Features

    def _build_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Complexity score — weighted sum of procedures + diagnoses + meds
        complexity_parts = []
        for col in ["num_procedures", "number_diagnoses", "num_medications"]:
            if col in df.columns:
                complexity_parts.append(df[col].fillna(0))

        if complexity_parts:
            df["clinical_complexity"] = sum(complexity_parts)

        # Time in hospital × number of diagnoses
        if "time_in_hospital" in df.columns and "number_diagnoses" in df.columns:
            df["days_per_diagnosis"] = (
                df["time_in_hospital"] / (df["number_diagnoses"] + 1)
            )

        # Procedure intensity: procedures per day
        if "time_in_hospital" in df.columns and "num_procedures" in df.columns:
            df["procedure_intensity"] = (
                df["num_procedures"] / (df["time_in_hospital"] + 1)
            )

        # Lab intensity: lab procedures per day
        if "time_in_hospital" in df.columns and "num_lab_procedures" in df.columns:
            df["lab_intensity"] = (
                df["num_lab_procedures"] / (df["time_in_hospital"] + 1)
            )

        logger.info("Clinical features built")
        return df

    # C: Lab Features

    def _build_lab_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        max_glu_serum and A1Cresult are the two key lab columns
        in the diabetes dataset.
        """
        # A1C result flags
        if "A1Cresult" in df.columns:
            df["a1c_tested"]   = (df["A1Cresult"] != "None").astype(int)
            df["a1c_high"]     = (df["A1Cresult"].isin([">7", ">8"])).astype(int)
            df["a1c_very_high"]= (df["A1Cresult"] == ">8").astype(int)

        # Glucose serum flags
        if "max_glu_serum" in df.columns:
            df["glucose_tested"] = (df["max_glu_serum"] != "None").astype(int)
            df["glucose_high"]   = (
                df["max_glu_serum"].isin([">200", ">300"])
            ).astype(int)

        # Both tests done — high clinical engagement signal
        if "a1c_tested" in df.columns and "glucose_tested" in df.columns:
            df["both_labs_done"] = (
                df["a1c_tested"] & df["glucose_tested"]
            ).astype(int)

        logger.info("Lab features built")
        return df

    # D: Demographic Features

    def _build_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Numeric age (ordinal position)
        if "age" in df.columns:
            df["age_numeric"] = df["age"].map(self.age_map).fillna(0)

            # Clinical risk buckets
            def age_to_bucket(bracket):
                if pd.isna(bracket):
                    return "Unknown"
                idx = self.age_map.get(bracket, 0)
                if idx <= 2:
                    return "Young"
                elif idx <= 5:
                    return "Middle"
                else:
                    return "Elderly"

            df["age_risk_bucket"] = df["age"].apply(age_to_bucket)
            df = df.drop(columns=["age"])   # replaced by numeric + bucket

        # Gender encode
        if "gender" in df.columns:
            df["gender"] = (
                df["gender"]
                .map({"Male": 0, "Female": 1})
                .fillna(-1)
                .astype(int)
            )

        logger.info("Demographic features built")
        return df

    # E: Medication Features

    def _build_medication_features(self, df: pd.DataFrame) -> pd.DataFrame:
        med_cols_cfg = list(self.cfg.features.medication_columns)
        self.med_cols_used = [c for c in med_cols_cfg if c in df.columns]

        def med_active(val):
            if pd.isna(val):
                return 0
            return 0 if val == "No" else 1

        # Count of medications changed/active
        df["n_active_medications"] = (
            df[self.med_cols_used]
            .applymap(med_active)
            .sum(axis=1)
        )

        # Insulin-specific signals (strongest predictor in this dataset)
        if "insulin" in df.columns:
            df["any_insulin"]       = df["insulin"].isin(
                ["Up", "Down", "Steady"]
            ).astype(int)
            df["insulin_increased"] = (df["insulin"] == "Up").astype(int)
            df["insulin_decreased"] = (df["insulin"] == "Down").astype(int)

        # Overall medication change flag
        if "change" in df.columns:
            df["any_med_change"] = (df["change"] == "Ch").astype(int)

        # diabetesMed prescribed
        if "diabetesMed" in df.columns:
            df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

        logger.info(
            f"Medication features built "
            f"({len(self.med_cols_used)} med columns)"
        )
        return df

    # F: Interaction Features

    def _build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clinical risk interactions
        if "age_numeric" in df.columns and "number_diagnoses" in df.columns:
            df["age_x_ndiagnoses"] = (
                df["age_numeric"] * df["number_diagnoses"]
            )

        if "n_active_medications" in df.columns and "number_diagnoses" in df.columns:
            df["n_meds_x_ndiagnoses"] = (
                df["n_active_medications"] * df["number_diagnoses"]
            )

        if "number_inpatient" in df.columns and "number_emergency" in df.columns:
            df["inpatient_x_emergency"] = (
                df["number_inpatient"] * df["number_emergency"]
            )

        # High-risk combination: elderly + high complexity
        if "age_numeric" in df.columns and "clinical_complexity" in df.columns:
            elderly_flag    = (df["age_numeric"] >= 6).astype(int)
            high_complexity = (
                df["clinical_complexity"]
                > df["clinical_complexity"].median()
            ).astype(int)
            df["elderly_high_complexity"] = elderly_flag * high_complexity

        logger.info("Interaction features built")
        return df

    # G: Diagnosis Category

    @staticmethod
    def _icd_to_category(code) -> str:
        if pd.isna(code) or str(code) in ["?", "E", "V", "nan"]:
            return "Other"
        try:
            code_str = str(code).strip()
            if code_str.startswith("V") or code_str.startswith("E"):
                return "External"
            num = float(code_str.split(".")[0])
            if 1 <= num <= 139:
                return "Infectious"
            elif 140 <= num <= 239:
                return "Neoplasms"
            elif 240 <= num <= 279:
                return "Endocrine"
            elif 280 <= num <= 289:
                return "Blood"
            elif 290 <= num <= 319:
                return "Mental"
            elif 320 <= num <= 389:
                return "Nervous"
            elif 390 <= num <= 459:
                return "Circulatory"
            elif 460 <= num <= 519:
                return "Respiratory"
            elif 520 <= num <= 579: return "Digestive"
            elif 580 <= num <= 629: return "Genitourinary"
            elif 630 <= num <= 679: return "Pregnancy"
            elif 680 <= num <= 709: return "Skin"
            elif 710 <= num <= 739: return "Musculoskeletal"
            elif 740 <= num <= 759: return "Congenital"
            elif 780 <= num <= 799: return "Symptoms"
            elif 800 <= num <= 999: return "Injury"
            else: return "Other"
        except Exception:
            return "Other"

    def _build_diagnosis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for diag_col in ["diag_1", "diag_2", "diag_3"]:
            if diag_col in df.columns:
                df[f"{diag_col}_category"] = df[diag_col].apply(
                    self._icd_to_category
                )

        # Primary diagnosis is circulatory or respiratory — higher risk
        if "diag_1_category" in df.columns:
            high_risk_diag = {"Circulatory", "Respiratory", "Endocrine"}
            df["primary_diag_high_risk"] = (
                df["diag_1_category"].isin(high_risk_diag)
            ).astype(int)

        # Drop raw diag codes (replaced by category)
        df = df.drop(
            columns=["diag_1", "diag_2", "diag_3"],
            errors="ignore"
        )
        logger.info("Diagnosis category features built")
        return df

    # Master transform

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering — input shape: {df.shape}")

        df = self._build_demographic_features(df)
        df = self._build_admission_features(df)
        df = self._build_clinical_features(df)
        df = self._build_lab_features(df)
        df = self._build_medication_features(df)
        df = self._build_diagnosis_features(df)
        df = self._build_interaction_features(df)

        logger.info(f"Feature engineering complete — output shape: {df.shape}")
        return df
    
    
class CategoricalEncoder:
    """
    Ordinal-encodes all remaining object columns after feature engineering.
    Fitted on train, applied to val + test (no leakage).
    """
    
    def __init__(self):
        self.encoder: OrdinalEncoder | None = None
        self.cat_cols: list[str] = []
        
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        self.cat_cols = df.select_dtypes(include= ['object']).columns.tolist()
        
        if not self.cat_cols:
            logger.info('No categorical columns to encode.')
            return df
        
        logger.info(f"Fitting ordinal encoder on columns: {self.cat_cols}")
        
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[self.cat_cols] = self.encoder.fit_transform(df[self.cat_cols])
        
        return df
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if not self.cat_cols or self.encoder is None:
            return df
        
        df[self.cat_cols] = self.encoder.transform(df[self.cat_cols])
        
        return df
    
    
    

class NullImputer:
    """
    Median imputation for numerical columns.
    Fitted on train, applied to val + test.
    """
    
    def __init__(self):
        self.medians: dict[str, float] = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        num_cols = df.select_dtypes(include= ['number']).columns.tolist()
        for col in num_cols:
            median = df[col].median()
            self.medians[col] = median
            df[col] = df[col].fillna(median)
            
        missing_left = df.isnull().sum().sum()
        logger.info(f"Null imputation complete. Missing values left: {missing_left}")
        
        return df
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        for col, median in self.medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
                
        return df
            
            

def run_feature_pipeline(cfg) -> dict[str, pd.DataFrame]:
    """
    Full feature engineering pipeline:
      1. Load raw data
      2. Clean
      3. Engineer features
      4. Encode categoricals (fit on train only)
      5. Impute nulls (fit on train only)
      6. Split → train / val / test
      7. Save parquet files

    Returns dict with keys: train, val, test
    """
    # Load raw
    raw_path = Path(cfg.data.raw_dir) / cfg.data.raw_filename
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    # Clean
    cleaner = DataCleaner(cfg)
    df = cleaner.fit_transform(df)

    # Separate target before engineering
    TARGET = cfg.data.target_binary_col
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Split BEFORE engineering to prevent leakage 
    # Train / temp
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg.splits.test_size,
        random_state=cfg.project.random_seed,
        stratify=y
    )
    # Train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=cfg.splits.val_size,
        random_state=cfg.project.random_seed,
        stratify=y_temp
    )

    logger.info(
        f"Split sizes — Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    # Feature engineer (fit on train, apply to all)
    engineer = FeatureEngineer(cfg)
    X_train = engineer.fit_transform(X_train)
    X_val   = engineer.fit_transform(X_val)     
    X_test  = engineer.fit_transform(X_test)  

    # Encode categoricals (fit on train only)
    encoder = CategoricalEncoder()
    X_train = encoder.fit_transform(X_train)
    X_val   = encoder.transform(X_val)
    X_test  = encoder.transform(X_test)

    # Impute nulls (fit on train only)
    imputer = NullImputer()
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    # Re-attach target
    train_df = X_train.copy(); train_df[TARGET] = y_train.values
    val_df   = X_val.copy();   val_df[TARGET]   = y_val.values
    test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

    # Save parquet
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.parquet"
    val_path   = processed_dir / "val.parquet"
    test_path  = processed_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path,   index=False)
    test_df.to_parquet(test_path,  index=False)

    logger.info(f"Saved - {train_path} ({len(train_df):,} rows)")
    logger.info(f"Saved - {val_path}   ({len(val_df):,} rows)")
    logger.info(f"Saved - {test_path}  ({len(test_df):,} rows)")
    logger.info(
        f"Final feature count: "
        f"{len([c for c in train_df.columns if c != TARGET])}"
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def main():
    cfg = load_config()
    run_feature_pipeline(cfg)
    logger.info("Feature engineering pipeline complete")


if __name__ == "__main__":
    main()