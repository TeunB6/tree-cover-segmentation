from pathlib import Path
from src.utils.logger import Logger
from rich.console import Console
import torch

DATA_URL = "https://owncloud.gwdg.de/index.php/s/H6MsR0wVGRuPPl3/download"
DATA_PATH = Path("data/")
DATA_HASH = "0ad723302caa8f34e3e9a451ce76973a399fb25035e5c244b490f606f3d7dbb9"
NEON_TREE_PATH = DATA_PATH / "neon_tree" / "NeonTreeEvaluation"
PT_DATA_PATH = DATA_PATH / "pt_data"
LOGGER = Logger("neon_tree")
CHANNEL_MEANS = (0.5436624, 0.5296872, 0.4485631, 0.03097722)
CHANNEL_STDEVS = (0.5436624, 0.5296872, 0.4485631, 0.09546612)
CHM_MAX = 60.575721740722656
IMG_SIZE = (400, 400)
CONSOLE = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
