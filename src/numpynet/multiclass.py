"""Synthetic multiclass classification experiments with separate holdout data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import Tanh
from .gradcheck import GradientReport, ParameterCheck
from .layers import Dense
from .losses import SoftmaxCrossEntropy
from .model import Sequential
from .optim import SGD


@dataclass(frozen=True)
class MulticlassRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class MulticlassResult:
    model: Sequential
    history: list[MulticlassRecord]
    validation_predictions: np.ndarray
    validation_targets: np.ndarray


def spiral_dataset(*, samples_per_class: int, noise: float, seed: int, classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Generate a labeled 2D spiral. Each seed produces new observations.

    Output order is permuted deterministically to avoid class-sorted batches.
    No data is loaded from external sources. This dataset is only a toy task.
    """
    if samples_per_class < 2 or classes < 2:
        raise ValueError("Need at least two samples per class and two classes")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("Noise must be nonnegative and finite")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), samples_per_class)
    radii = np.tile(np.linspace(0.05, 1.0, samples_per_class), classes)
    angles = (
        np.repeat(np.arange(classes), samples_per_class) * (2.0 * np.pi / classes)
        + 3.2 * radii
        + rng.normal(0, noise, size=labels.size)
    )
    data = np.column_stack((radii * np.sin(angles), radii * np.cos(angles)))
    order = rng.permutation(len(labels))
    return data[order].astype(np.float64), labels[order].astype(np.int64)


def make_multiclass_model(seed: int = 7, hidden_width: int = 24) -> Sequential:
    if hidden_width < 1:
        raise ValueError("hidden_width must be positive")
    rng = np.random.default_rng(seed)
    return Sequential([Dense(2, hidden_width, rng), Tanh(), Dense(hidden_width, 3, rng)])


def train_multiclass(
    *, epochs: int = 1200, learning_rate: float = 0.7, seed: int = 7,
    train_per_class: int = 100, validation_per_class: int = 60,
    noise: float = 0.18, hidden_width: int = 24,
) -> MulticlassResult:
    """Full-batch SGD on synthetic spirals, with a held-out generated set.

    Validation labels never enter gradient computation. This is a demonstrative
    split, not a claim about performance on independent real-world data.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train_x, train_y = spiral_dataset(samples_per_class=train_per_class, noise=noise, seed=seed + 1000)
    val_x, val_y = spiral_dataset(samples_per_class=validation_per_class, noise=noise, seed=seed + 2000)
    model = make_multiclass_model(seed=seed, hidden_width=hidden_width)
    criterion = SoftmaxCrossEntropy()
    optimizer = SGD(learning_rate)
    history: list[MulticlassRecord] = []

    def evaluate(epoch: int) -> None:
        train_logits = model.forward(train_x)
        train_loss = criterion.forward(train_logits, train_y)
        train_acc = float(np.mean(np.argmax(train_logits, axis=1) == train_y))
        val_logits = model.forward(val_x)
        # A separate criterion prevents validation evaluation from overwriting
        # the training gradient cache.
        val_loss = SoftmaxCrossEntropy().forward(val_logits, val_y)
        val_acc = float(np.mean(np.argmax(val_logits, axis=1) == val_y))
        history.append(MulticlassRecord(epoch, train_loss, train_acc, val_loss, val_acc))

    evaluate(0)
    for epoch in range(1, epochs + 1):
        logits = model.forward(train_x)
        criterion.forward(logits, train_y)
        model.backward(criterion.backward())
        optimizer.step(model)
        evaluate(epoch)

    probabilities = SoftmaxCrossEntropy().probabilities(model.forward(val_x))
    return MulticlassResult(model, history, probabilities, val_y.copy())


def check_multiclass_gradients(
    model: Sequential,
    inputs: np.ndarray,
    targets: np.ndarray,
    *, epsilon: float = 1e-5, atol: float = 1e-6, rtol: float = 1e-4,
) -> GradientReport:
    """Numerically validate Dense gradients through a logits-based CE loss.

    Recomputes network gradients after perturbation so the caches remain valid.
    Numerical checking is intended for small models only.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be nonnegative and finite")
    dense = [(i, layer) for i, layer in enumerate(model.layers) if isinstance(layer, Dense)]
    if not dense:
        raise ValueError("No Dense parameters to check")
    criterion = SoftmaxCrossEntropy()

    def get_loss() -> float:
        return criterion.forward(model.forward(inputs), targets)

    get_loss()
    model.backward(criterion.backward())
    params = []
    for index, layer in dense:
        params.extend([(index, "weights", layer.weights, layer.grad_weights.copy()),
                       (index, "bias", layer.bias, layer.grad_bias.copy())])
    results: list[ParameterCheck] = []
    try:
        for layer_index, name, parameter, analytical in params:
            numerical = np.zeros_like(parameter)
            for idx in np.ndindex(parameter.shape):
                old = float(parameter[idx])
                try:
                    parameter[idx] = old + epsilon
                    plus = get_loss()
                    parameter[idx] = old - epsilon
                    minus = get_loss()
                finally:
                    parameter[idx] = old
                numerical[idx] = (plus - minus) / (2 * epsilon)
            error = np.abs(numerical - analytical)
            tolerance = atol + rtol * np.maximum(np.abs(analytical), np.abs(numerical))
            failure = ~np.isfinite(error) | (error > tolerance)
            worst_flat = int(np.argmax(np.nan_to_num(error, nan=np.inf)))
            worst = tuple(int(x) for x in np.unravel_index(worst_flat, parameter.shape))
            results.append(ParameterCheck(layer_index, name, int(parameter.size),
                                          int(np.count_nonzero(failure)), float(error[worst]),
                                          worst, float(analytical[worst]), float(numerical[worst])))
    finally:
        get_loss()
        model.backward(criterion.backward())
    return GradientReport(tuple(results), epsilon, atol, rtol)
