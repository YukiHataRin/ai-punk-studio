"""把錄製的 skeleton.jsonl 轉成 CSV，方便用 Excel / pandas 分析。

    python -m aipunk_studio --export recordings/20261006_111500
    → recordings/20261006_111500/skeleton.csv（每列一個關節）
    → recordings/20261006_111500/metrics.csv（每列一位舞者一幀的九項指標）

欄位：frame, t, id, joint, joint_name, x, y, z, measured
（RGB 相機座標系：x 右、y 下、z 前，公尺；measured = 1 表示由深度實測）
"""

import csv
from pathlib import Path

from ..core.dance_metrics import METRIC_KEYS
from ..core.skeleton_format import FORMATS, MEDIAPIPE33
from .recorder import read_skeleton


def export(session_dir, out_path=None):
    session_dir = Path(session_dir)
    out_path = Path(out_path or session_dir / "skeleton.csv")
    rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "id", "joint", "joint_name", "x", "y", "z", "measured"])
        for rec in read_skeleton(session_dir / "skeleton.jsonl"):
            for person in rec["people"]:
                names = FORMATS[person.get("format", MEDIAPIPE33.name)].names
                for j, (xyz, m) in enumerate(zip(person.get("joints", []), person.get("measured", []))):
                    w.writerow([rec["frame"], rec["t"], person["id"], j, names[j], *xyz, m])
                    rows += 1
    return out_path, rows


def export_metrics(session_dir, out_path=None):
    """每列：frame, t, id, distance, distance_measured, 九項指標（沒有值的留空）。"""
    session_dir = Path(session_dir)
    out_path = Path(out_path or session_dir / "metrics.csv")
    rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "id", "distance", "distance_measured", *METRIC_KEYS])
        for rec in read_skeleton(session_dir / "skeleton.jsonl"):
            for person in rec["people"]:
                m = person.get("metrics")
                if m is None:
                    continue
                w.writerow([rec["frame"], rec["t"], person["id"], person["distance"], int(person["distance_measured"]),
                            *("" if m.get(k) is None else m[k] for k in METRIC_KEYS)])
                rows += 1
    return out_path, rows



def export_session(session_dir):
    """輸出 skeleton.csv 與 metrics.csv，回傳 [(路徑, 列數), ...]。"""
    if not (Path(session_dir) / "skeleton.jsonl").is_file():
        raise FileNotFoundError(f"{session_dir} 裡沒有 skeleton.jsonl")
    return [export(session_dir), export_metrics(session_dir)]
