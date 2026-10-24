"""A dense layer and explicit, inspectable gradients."""

from __future__ import annotations

import numpy as np


class Dense:
    """Fully connected layer: output = inputs @ weights + bias.

    Shapes: inputs (batch, in_features), weights (in_features, out_features).
    Gradients are accumulated per entire batch by the upstream loss function.
    """

    def __init__(
        self, in_features: int, out_features: int, rng: np.random.Generator
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer widths must be positive")

        # Xavier uniform is a standard initialization method for tanh networks.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = rng.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros((1, out_features), dtype=np.float64)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._last_input: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Expected inputs shaped (batch, {self.weights.shape[0]}), "
                f"got {inputs.shape}"
            )
        self._last_input = inputs
        return inputs @ self.weights + self.bias

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        if self._last_input is None:
            raise RuntimeError("forward() must run before backward()")
        expected = (self._last_input.shape[0], self.weights.shape[1])
        if upstream.shape != expected:
            raise ValueError(f"Expected upstream gradient {expected}, got {upstream.shape}")

        self.grad_weights = self._last_input.T @ upstream
        self.grad_bias = np.sum(upstream, axis=0, keepdims=True)
        return upstream @ self.weights.T
