"""WebSocket 串流的簡易用戶端：印出每幀的人物摘要（或原始 JSON）。

    python tools/ws_client.py                          # 連 ws://127.0.0.1:8765
    python tools/ws_client.py ws://192.168.1.20:8765 --frames 30
    python tools/ws_client.py --json > stream.jsonl    # 存原始訊息
"""

import argparse
import json

from websockets.sync.client import connect

from astra_studio import utf8_stdio


def summary(msg):
    people = []
    for p in msg["people"]:
        dist = "—" if p["distance"] is None else f"{'' if p['distance_measured'] else '≈'}{p['distance']:.2f} m"
        joints = f"{sum(p['measured'])}/{len(p['measured'])} 實測" if "measured" in p else "僅遮罩"
        energy = (p.get("metrics") or {}).get("energy")
        people.append(f"ID {p['id']} {dist}（{joints}）" + ("" if energy is None else f" 強度 {energy:.2f}"))
    return f"#{msg['frame']:5d}  {msg['fps']:4.1f} fps  " + ("、".join(people) if people else "沒有人")


def main():
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", default="ws://127.0.0.1:8765")
    parser.add_argument("--frames", type=int, help="收到幾幀後結束")
    parser.add_argument("--json", action="store_true", help="輸出原始 JSON（每行一則）")
    args = parser.parse_args()

    received = 0
    with connect(args.url, max_size=2**24) as ws:
        for raw in ws:
            msg = json.loads(raw)
            if args.json:
                print(raw, flush=True)
            elif msg["type"] == "hello":
                fmts = "、".join(f"{k}（{len(v['names'])} 點）" for k, v in msg["formats"].items())
                print(f"已連線：{msg['image']['width']}×{msg['image']['height']}，"
                      f"{'有深度' if msg['has_depth'] else '沒有深度（3D 為估計）'}，骨架格式：{fmts}", flush=True)
            elif msg["type"] == "frame":
                print(summary(msg), flush=True)
            if msg["type"] == "frame":
                received += 1
                if args.frames and received >= args.frames:
                    break


if __name__ == "__main__":
    main()
