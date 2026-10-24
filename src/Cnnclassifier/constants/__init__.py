from pathlib import Path
CONFIG_FILE_PATH: Path = Path("config/config.yaml")
PARAMS_FILE_PATH: Path = Path("params.yaml")

# Same fraction must be used for training and evaluation, otherwise the
# "validation" images used in evaluation overlap with training images.
VALIDATION_SPLIT: float = 0.20
