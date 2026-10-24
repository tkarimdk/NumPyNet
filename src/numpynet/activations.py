"""Nonlinear functions with hand-written derivatives."""

from __future__ import annotations

import numpy as np


class Tanh:
    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._output = np.tanh(inputs)
        return self._output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("forward() must run before backward()")
        if upstream.shape != self._output.shape:
            raise ValueError("Gradient shape differs from activation output")
        return upstream * (1.0 - self._output**2)


class Sigmoid:
    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Compute positive and negative branches separately to avoid overflow.
        output = np.empty_like(inputs, dtype=np.float64)
        positive = inputs >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-inputs[positive]))
        negative_exp = np.exp(inputs[~positive])
        output[~positive] = negative_exp / (1.0 + negative_exp)
        self._output = output
        return output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("forward() must run before backward()")
        if upstream.shape != self._output.shape:
            raise ValueError("Gradient shape differs from activation output")
        return upstream * self._output * (1.0 - self._output)
