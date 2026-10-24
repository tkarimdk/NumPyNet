"""Stochastic gradient descent (full-batch in the XOR example)."""

from __future__ import annotations

from .layers import Dense
from .model import Sequential


class SGD:
    def __init__(self, learning_rate: float) -> None:
        if not 0 < learning_rate <= 10:
            raise ValueError("learning_rate must be in (0, 10]")
        self.learning_rate = learning_rate

    def step(self, model: Sequential) -> None:
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias
