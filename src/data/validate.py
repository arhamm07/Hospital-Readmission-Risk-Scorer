import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__, log_file= 'logs/validate.log')


class DataValidator:
    """
    Runs a suite of data quality checks on the raw DataFrame.
    Raises ValueError on critical failures.
    Logs warnings for non-critical issues.
    """
    
    def __init__(self, df: pd.DataFrame, cfg):
        
        self.df = df
        self.cfg = cfg
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        
    def check_target_values(self):
        
        expected = {'<30', '>30', 'NO'}
        actual = set(self.df[self.cfg.data.target_col].unique())
        unexpected = actual - expected
        
        if unexpected:
            self.errors.append(
                f'Unexpected target values found: {unexpected}. Expected only {expected}.'
            )
            
        else:
            logger.info(f'Target values are valid and as expected: {expected}.')
            
            
    def check_leakage(self):
        """
        The raw target column must not appear in a form that
        leaks future information (e.g. numeric encoding of target
        that was accidentally left in).
        """
        
        leaky_cols = [c for c in self.df.columns
                      if c!= self.cfg.data.target_col and 'readmit' in c.lower()]
        
        if leaky_cols:
            self.erorrs.append(
                f'Potential target leakage columns found: {leaky_cols}.'
            )
        else:
            logger.info('NO target leakage columns found.')
            
    
    def check_duplicates(self):
        """encounter_id must be unique per row."""
        if 'encounter_id' not in self.df.columns:
            self.warnings.append('encounter_id column not found. Skipping duplicate check.')
            return
        
        dup_count = self.df.duplicated(subset= 'encounter_id').sum()
        
        if dup_count > 0:
            self.warnings.append(f'Found {dup_count} duplicate encounter_id values.')
        else:
            logger.info('No duplicated found.')
            
    
    def check_missing_values(self):
        """Audit missing / '?' values and warn if any column >35% missing."""
        
        df_clean = self.df.replace('?', np.nan)
        missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100).sort_values(ascending= False)
        high_missing = missing_pct[missing_pct > self.cfg.data.high_missing_threshold]
        
        if not high_missing.empty:
            self.warnings.append(
                f"Columns with >{self.cfg.data.high_missing_threshold}% missing "
                f"(will be dropped in preprocessing):\n{high_missing.to_dict()}"
            )
        else:
            logger.info("No columns exceed the missing-value threshold")
            
        logger.info(
            f"Top 5 missing columns:\n"
            f"{missing_pct.head(5).to_dict()}"
        )
            
            
    def check_numerical_ranges(self):
        """Key numerical columns must be within plausible clinical ranges."""
        range_checks = {
            "time_in_hospital": (1, 30),
            "num_lab_procedures": (0, 200),
            "num_procedures": (0, 10),
            "num_medications": (0, 100),
            "number_diagnoses": (0, 20),
        }

        for col, (lo, hi) in range_checks.items():
            if col not in self.df.columns:
                self.warnings.append(f"Column '{col}' not found - skipping range check")
                continue

            out_of_range = self.df[
                (self.df[col] < lo) | (self.df[col] > hi)
            ]

            if len(out_of_range) > 0:
                self.warnings.append(
                    f"'{col}': {len(out_of_range)} values outside [{lo}, {hi}]"
                )
            else:
                logger.info(f"'{col}' range OK [{lo}, {hi}]")
                

    def check_class_distribution(self):
        """
        Positive class (<30 day readmission) should be between 5–30%.
        Severe imbalance outside this range needs investigation.
        """
        pos_rate = (self.df[self.cfg.data.target_col] == self.cfg.data.positive_class).mean()
        logger.info(f"Positive class rate (<30d readmission): {pos_rate:.3f}")

        if pos_rate < 0.05:
            self.warnings.append(
                f"Very low positive rate ({pos_rate:.3f}) — "
                f"consider reviewing target definition"
            )
        elif pos_rate > 0.50:
            self.errors.append(
                f"Unexpectedly high positive rate ({pos_rate:.3f}) — "
                f"check for target leakage"
            )
        else:
            logger.info(f"Class distribution looks reasonable ({pos_rate:.3f} positive)")

    
    
    def check_minimum_row_count(self, min_rows: int = 50_000):
        """Dataset must have at least min_rows rows."""
        if len(self.df) < min_rows:
            self.errors.append(
                f"Only {len(self.df):,} rows — expected at least {min_rows:,}. "
                f"Download may be incomplete."
            )
        else:
            logger.info(f"Row count OK: {len(self.df):,}")
            
            
    def run_all(self) -> bool:
        """
        Runs all validation checks.
        Returns True if no critical errors, False otherwise.
        """
        logger.info("Running data validation checks...")

        self.check_minimum_row_count()
        self.check_target_values()
        self.check_leakage()
        self.check_duplicates()
        self.check_missing_values()
        self.check_numerical_ranges()
        self.check_class_distribution()

        #  Report
        logger.info("VALIDATION SUMMARY")

        if self.warnings:
            for w in self.warnings:
                logger.warning(w)

        if self.errors:
            for e in self.errors:
                logger.error(e)
            logger.error(f"\nValidation FAILED — {len(self.errors)} critical error(s)")
            return False

        logger.info(
            f"Validation PASSED "
            f"({len(self.warnings)} warning(s), 0 errors)"
        )
        return True
    
    
def main():
    cfg = load_config()
    
    raw_path = Path(cfg.data.raw_dir) / cfg.data.raw_filename
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{raw_path}'. ")
        
    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw data loaded with shape: {df.shape}")
    
    validator = DataValidator(df, cfg)
    passed = validator.run_all()
    
    if not passed:
        sys.exit(1)
        
        
if __name__ == "__main__":
    
    main()