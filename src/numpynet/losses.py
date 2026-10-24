"""Binary cross-entropy, keeping its derivative visible for learning."""

from __future__ import annotations

import numpy as np


class BinaryCrossEntropy:
    def __init__(self, epsilon: float = 1e-12) -> None:
        if not 0 < epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")
        self.epsilon = epsilon
        self._predictions: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        if predictions.shape != targets.shape or predictions.ndim != 2:
            raise ValueError("Predictions and targets must be matching 2D arrays")
        if predictions.size == 0:
            raise ValueError("Cannot calculate a loss for an empty batch")
        if not np.all((targets == 0.0) | (targets == 1.0)):
            raise ValueError("Binary targets must be 0 or 1")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions must be finite")
        if np.any((predictions < 0.0) | (predictions > 1.0)):
            raise ValueError("Predictions must be probabilities in [0, 1]")

        probabilities = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        self._predictions = probabilities
        self._targets = targets
        loss = -(targets * np.log(probabilities) + (1.0 - targets) * np.log1p(-probabilities))
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        if self._predictions is None or self._targets is None:
            raise RuntimeError("forward() must run before backward()")
        probabilities, targets = self._predictions, self._targets
        return (probabilities - targets) / (
            probabilities * (1.0 - probabilities) * probabilities.size
        )
