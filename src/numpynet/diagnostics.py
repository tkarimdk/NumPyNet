"""Inspect how SGD training evolves, without automatic differentiation.

Every row describes ONE update. The gradient and parameter norms are taken just
before the update; the post-loss is measured after it. These observations are
not a diagnosis on their own and are not evidence of novel research.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np

from .activations import Tanh
from .experiments import XOR_INPUTS, XOR_TARGETS, EpochRecord, make_xor_model
from .layers import Dense
from .losses import BinaryCrossEntropy
from .optim import SGD


@dataclass(frozen=True)
class StepRecord:
    epoch: int
    pre_loss: float
    post_loss: float
    gradient_l2: float
    parameter_l2: float
    update_l2: float
    relative_update: float
    hidden_saturation: float
    accuracy: float

    @property
    def loss_increased(self) -> bool:
        return self.post_loss > self.pre_loss + 1e-12


@dataclass
class DiagnosticRun:
    learning_rate: float
    seed: int
    steps: list[StepRecord]
    history: list[EpochRecord]

    @property
    def first_loss(self) -> float:
        return self.history[0].loss

    @property
    def last_loss(self) -> float:
        return self.history[-1].loss

    @property
    def last_accuracy(self) -> float:
        return self.history[-1].accuracy

    @property
    def increasing_steps(self) -> int:
        return sum(row.loss_increased for row in self.steps)

    @property
    def max_relative_update(self) -> float:
        return max(row.relative_update for row in self.steps)

    @property
    def max_gradient_l2(self) -> float:
        return max(row.gradient_l2 for row in self.steps)


def _parameter_and_gradient_norms(model) -> tuple[float, float]:
    parameter_squares = 0.0
    gradient_squares = 0.0
    for layer in model.layers:
        if isinstance(layer, Dense):
            for param, gradient in ((layer.weights, layer.grad_weights),
                                    (layer.bias, layer.grad_bias)):
                parameter_squares += float(np.sum(param * param))
                gradient_squares += float(np.sum(gradient * gradient))
    return sqrt(parameter_squares), sqrt(gradient_squares)


def _hidden_saturation(model, threshold: float = 0.95) -> float:
    """Fraction of first Tanh layer activations with |activation| >= threshold."""
    for layer in model.layers:
        if isinstance(layer, Tanh):
            if layer._output is None:  # a forward pass must run first
                raise RuntimeError("Run a forward pass before checking saturation")
            return float(np.mean(np.abs(layer._output) >= threshold))
    raise ValueError("Expected a Tanh layer for this XOR diagnostic")


def diagnose_xor(*, epochs: int = 2000, learning_rate: float = 2.0, seed: int = 7) -> DiagnosticRun:
    """Train the same full-batch XOR model as ``train_xor`` with instrumentation.

    The input set is only 4 rows; accuracy is training accuracy, NOT held-out
    generalization. An increase in loss after an update is one observed sign
    of instability, not proof of its underlying cause.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    model = make_xor_model(seed)
    loss = BinaryCrossEntropy()
    optimizer = SGD(learning_rate)
    initial_prediction = model.forward(XOR_INPUTS)
    initial_loss = loss.forward(initial_prediction, XOR_TARGETS)
    history = [EpochRecord(0, initial_loss,
                           float(np.mean((initial_prediction >= 0.5) == XOR_TARGETS)))]
    steps: list[StepRecord] = []

    for epoch in range(1, epochs + 1):
        # The previous measurement has populated forward and loss caches.
        pre_loss = history[-1].loss
        saturation = _hidden_saturation(model)
        model.backward(loss.backward())
        parameter_l2, gradient_l2 = _parameter_and_gradient_norms(model)
        update_l2 = optimizer.learning_rate * gradient_l2
        relative_update = update_l2 / max(parameter_l2, 1e-12)
        optimizer.step(model)

        post_prediction = model.forward(XOR_INPUTS)
        post_loss = loss.forward(post_prediction, XOR_TARGETS)
        accuracy = float(np.mean((post_prediction >= 0.5) == XOR_TARGETS))
        if not all(np.isfinite(value) for value in
                   (pre_loss, post_loss, saturation, parameter_l2, gradient_l2,
                    update_l2, relative_update)):
            raise FloatingPointError(f"Non-finite training statistic at epoch {epoch}")
        history.append(EpochRecord(epoch, post_loss, accuracy))
        steps.append(StepRecord(epoch, pre_loss, post_loss, gradient_l2,
                                parameter_l2, update_l2, relative_update,
                                saturation, accuracy))
    return DiagnosticRun(learning_rate, seed, steps, history)


def write_steps_csv(path: str | Path, run: DiagnosticRun) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("epoch", "pre_loss", "post_loss", "loss_increased",
                         "gradient_l2", "parameter_l2", "update_l2",
                         "relative_update", "hidden_saturation", "accuracy"))
        for s in run.steps:
            writer.writerow((s.epoch, f"{s.pre_loss:.12g}", f"{s.post_loss:.12g}",
                             int(s.loss_increased), f"{s.gradient_l2:.12g}",
                             f"{s.parameter_l2:.12g}", f"{s.update_l2:.12g}",
                             f"{s.relative_update:.12g}", f"{s.hidden_saturation:.12g}",
                             f"{s.accuracy:.12g}"))
    return destination


def write_diagnostic_summary(path: str | Path, runs: list[DiagnosticRun]) -> Path:
    if not runs:
        raise ValueError("Need at least one run")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("learning_rate", "seed", "epochs", "first_loss", "last_loss",
                         "last_training_accuracy", "loss_increasing_steps",
                         "max_relative_update", "max_gradient_l2", "last_pre_update_hidden_saturation"))
        for run in runs:
            writer.writerow((run.learning_rate, run.seed, len(run.steps),
                             f"{run.first_loss:.12g}", f"{run.last_loss:.12g}",
                             f"{run.last_accuracy:.12g}", run.increasing_steps,
                             f"{run.max_relative_update:.12g}",
                             f"{run.max_gradient_l2:.12g}",
                             f"{run.steps[-1].hidden_saturation:.12g}"))
    return destination
