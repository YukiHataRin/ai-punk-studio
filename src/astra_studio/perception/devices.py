"""PyTorch 推論裝置選擇：CUDA → Apple MPS → CPU（取自 Human Mask Studio）。"""

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Device:
    key: str
    label: str


def available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.extend(Device(f"cuda:{i}", f"CUDA · {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        devices.append(Device("mps", "Apple GPU · MPS"))
    devices.append(Device("cpu", f"CPU · {platform.machine()}"))
    return devices


def select_device(key="auto"):
    devices = available_devices()
    if key == "auto":
        return devices[0]
    for device in devices:
        if device.key == key:
            return device
    raise RuntimeError(f"找不到推論裝置：{key}")
