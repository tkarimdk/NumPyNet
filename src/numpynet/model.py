"""A simple sequential execution engine, intentionally without autograd."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class TrainableLayer(Protocol):
    def forward(self, inputs: np.ndarray) -> np.ndarray: ...
    def backward(self, upstream: np.ndarray) -> np.ndarray: ...


class Sequential:
    def __init__(self, layers: Sequence[TrainableLayer]) -> None:
        if not layers:
            raise ValueError("A model needs at least one layer")
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
        return loss_gradient
