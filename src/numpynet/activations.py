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


class ReLU:
    """Rectified linear activation: max(0, x), with derivative 0 at x == 0."""

    def __init__(self) -> None:
        self._positive: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._positive = inputs > 0
        return np.maximum(inputs, 0)

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if self._positive is None:
            raise RuntimeError("forward() must run before backward()")
        if upstream.shape != self._positive.shape:
            raise ValueError("Gradient shape differs from activation output")
        return upstream * self._positive


class Softmax:
    """Stable normalized exponential and explicit Jacobian-vector product."""

    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, logits: np.ndarray) -> np.ndarray:
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
            raise ValueError("Expected nonempty 2D logits with >= 2 classes")
        if not np.all(np.isfinite(logits)):
            raise ValueError("Logits must be finite")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._output = exp / np.sum(exp, axis=1, keepdims=True)
        return self._output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("forward() must run before backward()")
        if upstream.shape != self._output.shape:
            raise ValueError("Gradient shape differs from activation output")
        # (diag(p) - p p^T) @ upstream; computed without forming the Jacobian.
        return self._output * (upstream - np.sum(self._output * upstream, axis=1, keepdims=True))
