"""把錄製的 skeleton.jsonl 轉成長表 CSV，方便用 Excel / pandas 分析。

    python tools/export_skeleton_csv.py recordings/20261006_111500
    → recordings/20261006_111500/skeleton.csv

欄位：frame, t, id, joint, joint_name, x, y, z, measured
（RGB 相機座標系：x 右、y 下、z 前，公尺；measured = 1 表示由深度實測）
"""

import argparse
import csv
from pathlib import Path

from astra_studio.io.recorder import read_skeleton

JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]


def export(session_dir, out_path=None):
    session_dir = Path(session_dir)
    out_path = Path(out_path or session_dir / "skeleton.csv")
    rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "id", "joint", "joint_name", "x", "y", "z", "measured"])
        for rec in read_skeleton(session_dir / "skeleton.jsonl"):
            for person in rec["people"]:
                for j, (xyz, m) in enumerate(zip(person.get("joints", []), person.get("measured", []))):
                    w.writerow([rec["frame"], rec["t"], person["id"], j, JOINT_NAMES[j], *xyz, m])
                    rows += 1
    return out_path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", help="錄製目錄（含 skeleton.jsonl）")
    parser.add_argument("-o", "--output", help="輸出 CSV 路徑（預設放在錄製目錄）")
    args = parser.parse_args()
    path, rows = export(args.session, args.output)
    print(f"已輸出 {rows} 列：{path}")


if __name__ == "__main__":
    main()
