"""Compare a network's hand-written gradients with central finite differences.

This is a diagnostic, not automatic differentiation. It deliberately uses
many forward passes, so it is best suited to small models and tiny batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layers import Dense
from .losses import BinaryCrossEntropy
from .model import Sequential


@dataclass(frozen=True)
class ParameterCheck:
    layer: int
    parameter: str
    count: int
    failed: int
    max_absolute_error: float
    worst_index: tuple[int, ...]
    analytical: float
    numerical: float

    @property
    def passed(self) -> bool:
        return self.failed == 0


@dataclass(frozen=True)
class GradientReport:
    checks: tuple[ParameterCheck, ...]
    epsilon: float
    atol: float
    rtol: float

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def checked_parameters(self) -> int:
        return sum(check.count for check in self.checks)


def check_gradients(
    model: Sequential,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    epsilon: float = 1e-5,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> GradientReport:
    """Check Dense weight/bias derivatives of BCE through a Sequential model.

    Numerical derivative: [L(theta + eps) - L(theta - eps)] / (2 * eps).
    Use non-saturated activations: BCE clipping introduces non-smooth points,
    where a finite difference need not agree with the implemented derivative.

    Restores all parameters even if evaluation raises. Does not update weights.
    Analytical gradients are snapshotted before the finite-difference passes.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be nonnegative and finite")

    dense_layers = [(index, layer) for index, layer in enumerate(model.layers) if isinstance(layer, Dense)]
    if not dense_layers:
        raise ValueError("Gradient checking requires at least one Dense layer")

    criterion = BinaryCrossEntropy()

    def get_loss() -> float:
        return criterion.forward(model.forward(inputs), targets)

    get_loss()
    model.backward(criterion.backward())
    parameters: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    for layer_index, layer in dense_layers:
        parameters.append((layer_index, "weights", layer.weights, layer.grad_weights.copy()))
        parameters.append((layer_index, "bias", layer.bias, layer.grad_bias.copy()))

    results: list[ParameterCheck] = []
    try:
        for layer_index, name, parameter, analytic in parameters:
            numeric = np.empty_like(parameter)
            for idx in np.ndindex(parameter.shape):
                original = float(parameter[idx])
                try:
                    parameter[idx] = original + epsilon
                    positive = get_loss()
                    parameter[idx] = original - epsilon
                    negative = get_loss()
                finally:
                    parameter[idx] = original
                numeric[idx] = (positive - negative) / (2.0 * epsilon)

            error = np.abs(numeric - analytic)
            tolerance = atol + rtol * np.maximum(np.abs(analytic), np.abs(numeric))
            failing = ~np.isfinite(error) | (error > tolerance)
            worst_flat = int(np.argmax(np.nan_to_num(error, nan=np.inf)))
            worst = tuple(int(i) for i in np.unravel_index(worst_flat, parameter.shape))
            results.append(
                ParameterCheck(
                    layer=layer_index,
                    parameter=name,
                    count=int(parameter.size),
                    failed=int(np.count_nonzero(failing)),
                    max_absolute_error=float(error[worst]),
                    worst_index=worst,
                    analytical=float(analytic[worst]),
                    numerical=float(numeric[worst]),
                )
            )
    finally:
        # Restore the normal forward/backward cache, too. The checker may be
        # followed immediately by an optimizer step without stale activations.
        get_loss()
        model.backward(criterion.backward())

    return GradientReport(tuple(results), epsilon, atol, rtol)
