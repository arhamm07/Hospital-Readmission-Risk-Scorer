import sys 
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file= 'logs/download.log')


def download_data(cfg) -> pd.DataFrame:
    """
    Fetches the Diabetes 130-US Hospitals dataset from UCI ML Repository.
    Dataset ID: 296
    Returns a merged DataFrame (features + target).
    """
    
    logger.info(f'Fetching UCI Dataset ID={cfg.data.dataset_id}...')
    
    dataset = fetch_ucirepo(id=cfg.data.dataset_id)
    
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    
    logger.info(f'Dataset has been fetched. Merging features and target...')
    
    df = pd.concat([features, targets], axis=1)
    
    logger.info(f'Dataset loaded with shape: {df.shape}. Columns: {df.columns.tolist()}')
    
    return df



def save_raw_data(df: pd.DataFrame, cfg) -> Path:
    """
    Saves the raw DataFrame as CSV to data/raw/<raw_filename>.
    Creates the directory if it does not exist.
    Returns the saved file path.
    """
    
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents= True, exist_ok= True)
    
    save_path = raw_dir / cfg.data.raw_filename
    df.to_csv(save_path, index= False)
    
    logger.info(f'Raw data saved to: {save_path}')
    
    return save_path


def create_icd10_lookup(cfg) -> Path:
    """
    Creates a minimal ICD-10 category lookup table and saves it
    to data/external/ — used later in feature engineering.
    """
    external_dir = Path(cfg.data.external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)

    icd10_map = {
        "Infectious":    "001-139",
        "Neoplasms":     "140-239",
        "Endocrine":     "240-279",
        "Blood":         "280-289",
        "Mental":        "290-319",
        "Nervous":       "320-389",
        "Circulatory":   "390-459",
        "Respiratory":   "460-519",
        "Digestive":     "520-579",
        "Genitourinary": "580-629",
        "Pregnancy":     "630-679",
        "Skin":          "680-709",
        "Musculoskeletal":"710-739",
        "Congenital":    "740-759",
        "Symptoms":      "780-799",
        "Injury":        "800-999",
        "External":      "E/V codes",
        "Other":         "Uncategorised",
    }

    lookup_df = pd.DataFrame([
        {"category": k, "icd_range": v}
        for k, v in icd10_map.items()
    ])

    lookup_path = external_dir / cfg.data.icd10_lookup_filename
    lookup_df.to_csv(lookup_path, index=False)

    logger.info(f"ICD-10 lookup saved → {lookup_path}")
    return lookup_path


def main():
    
    cfg = load_config()
    
    logger.info('Starting data download process...')
    
    df = download_data(cfg)
    
    raw_path = save_raw_data(df, cfg)
    
    icd_path = create_icd10_lookup(cfg)
    
    logger.info("Data download complete.")
    logger.info(f"  Raw data  → {raw_path}")
    logger.info(f"  ICD10 map → {icd_path}")
    

if __name__ == "__main__":
    
    main()