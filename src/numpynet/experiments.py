"""Deterministic training experiments and their measured histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import Sigmoid, Tanh
from .layers import Dense
from .losses import BinaryCrossEntropy
from .model import Sequential
from .optim import SGD


XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_TARGETS = np.array([[0.0], [1.0], [1.0], [0.0]])


@dataclass(frozen=True)
class EpochRecord:
    """Metrics after `epoch` parameter updates (epoch zero is initialization)."""

    epoch: int
    loss: float
    accuracy: float


@dataclass
class ExperimentResult:
    model: Sequential
    losses: list[float]  # retained for compatibility with existing tests
    predictions: np.ndarray
    accuracy: float
    history: list[EpochRecord]


def make_xor_model(seed: int = 7) -> Sequential:
    """Create the same initialized network for fair learning-rate comparisons."""
    rng = np.random.default_rng(seed)
    return Sequential([Dense(2, 4, rng), Tanh(), Dense(4, 1, rng), Sigmoid()])


def train_xor(epochs: int = 5000, learning_rate: float = 0.7, seed: int = 7) -> ExperimentResult:
    """Fit a 2 -> 4 -> 1 MLP to four XOR examples; not a generalization study."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    model = make_xor_model(seed)
    criterion = BinaryCrossEntropy()
    optimizer = SGD(learning_rate)
    history: list[EpochRecord] = []

    def measure(epoch: int) -> tuple[np.ndarray, float]:
        prediction = model.forward(XOR_INPUTS)
        loss = criterion.forward(prediction, XOR_TARGETS)
        accuracy = float(np.mean((prediction >= 0.5) == XOR_TARGETS))
        history.append(EpochRecord(epoch=epoch, loss=loss, accuracy=accuracy))
        return prediction, accuracy

    measure(0)
    for epoch in range(1, epochs + 1):
        # 'measure' has populated the forward and loss caches for this step.
        model.backward(criterion.backward())
        optimizer.step(model)
        final_prediction, accuracy = measure(epoch)

    return ExperimentResult(
        model=model,
        losses=[entry.loss for entry in history],
        predictions=final_prediction,
        accuracy=accuracy,
        history=history,
    )
