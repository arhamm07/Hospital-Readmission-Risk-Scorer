from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str = 'configs/config.yaml') -> DictConfig:
    """
    Loads the YAML config using OmegaConf.

    Args:
        config_path: path to config.yaml (relative to project root)

    Returns:
        OmegaConf DictConfig — access keys as cfg.data.raw_dir etc.

    Raises:
        FileNotFoundError if config path doesn't exist.
    """
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not Found at '{config_path}'. Make sure you are running from the project root."
        )
        
    cfg = OmegaConf.load(config_path)
    
    return cfg


def pretty_print_config(cfg: DictConfig) -> None:
    """Prints the full config in a readable YAML format."""
    print(OmegaConf.to_yaml(cfg))
    
    
    