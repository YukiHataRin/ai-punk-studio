import os
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

# Ultralytics / matplotlib 的設定與快取放在專案內，不寫到使用者家目錄
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for _dir in (os.environ["YOLO_CONFIG_DIR"], os.environ["MPLCONFIGDIR"]):
    Path(_dir).mkdir(parents=True, exist_ok=True)


def resolve(path):
    """把設定檔中的相對路徑解析成專案根目錄下的絕對路徑。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    cfg["pose"]["model"] = str(resolve(cfg["pose"]["model"]))
    seg = cfg["segmentation"]
    seg["model"] = str(resolve(seg["model"]))
    seg["tracker"] = str(resolve(seg["tracker"]))
    return cfg
