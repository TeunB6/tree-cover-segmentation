from pathlib import Path
from src.utils.logger import Logger

DATA_URL = "https://owncloud.gwdg.de/index.php/s/H6MsR0wVGRuPPl3/download"
DATA_PATH = Path("data/")
DATA_HASH = "0ad723302caa8f34e3e9a451ce76973a399fb25035e5c244b490f606f3d7dbb9"
NEON_TREE_PATH = DATA_PATH / "neon_tree" / "NeonTreeEvaluation"
PT_DATA_PATH = DATA_PATH / "pt_data"
LOGGER = Logger("neon_tree")