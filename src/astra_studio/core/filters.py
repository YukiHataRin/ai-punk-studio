"""One Euro Filter（Casiez et al. 2012），向量化版本。

慢速移動時強平滑（去抖動），快速移動時降低平滑（減少延遲）。
min_cutoff 越小越穩但越黏；beta 越大，快速動作時跟得越緊。
"""

import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t):
        x = np.asarray(x, np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()
