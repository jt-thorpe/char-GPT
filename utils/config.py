import yaml


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load hyperparameters from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Hyperparameters for the model.    
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
