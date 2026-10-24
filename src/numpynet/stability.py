"""Controlled full-batch XOR experiments for step-size safeguards.

This module uses explicit gradients. Established stability interventions
are evaluated from identically initialized models, with losses measured on
only four *training* examples. 'Backtracking' evaluates post-update loss on
those same four examples (full-batch), so it is not a held-out validation test.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path

import numpy as np

from .diagnostics import _hidden_saturation
from .experiments import EpochRecord, XOR_INPUTS, XOR_TARGETS, make_xor_model
from .layers import Dense
from .losses import BinaryCrossEntropy
from .optim import SGD

MODES = ("baseline", "clip", "decay", "backtrack", "clip-decay", "clip-backtrack")


@dataclass(frozen=True)
class StabilityConfig:
    mode: str = "baseline"
    learning_rate: float = 10.0
    gradient_max_norm: float = 0.3
    decay_factor: float = 0.995
    min_learning_rate: float = 0.05
    backtrack_factor: float = 0.5
    max_backtracks: int = 8

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        for key, value in (
            ("learning_rate", self.learning_rate),
            ("gradient_max_norm", self.gradient_max_norm),
            ("decay_factor", self.decay_factor),
            ("min_learning_rate", self.min_learning_rate),
            ("backtrack_factor", self.backtrack_factor),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.learning_rate > 10:
            raise ValueError("learning_rate must not exceed 10 (existing SGD limit)")
        if not 0 < self.decay_factor <= 1:
            raise ValueError("decay_factor must be <= 1")
        if not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must be < 1")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if isinstance(self.max_backtracks, bool) or not isinstance(self.max_backtracks, int) or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")

    @property
    def clips(self) -> bool:
        return "clip" in self.mode

    @property
    def decays(self) -> bool:
        return "decay" in self.mode

    @property
    def backtracks(self) -> bool:
        return "backtrack" in self.mode

    def scheduled_lr(self, update_number: int) -> float:
        if update_number <= 0:
            raise ValueError("update_number must be >= 1")
        if not self.decays:
            return self.learning_rate
        return max(self.min_learning_rate, self.learning_rate * self.decay_factor ** (update_number - 1))


@dataclass(frozen=True)
class StabilityStep:
    epoch: int
    pre_loss: float
    post_loss: float
    raw_gradient_l2: float
    applied_gradient_l2: float
    clip_factor: float
    scheduled_lr: float
    effective_lr: float
    trial_count: int
    accepted: bool
    parameter_l2: float
    update_l2: float
    relative_update: float
    hidden_saturation: float
    accuracy: float

    @property
    def increased(self) -> bool:
        return self.post_loss > self.pre_loss + 1e-12


@dataclass
class StabilityRun:
    config: StabilityConfig
    seed: int
    history: list[EpochRecord]
    steps: list[StabilityStep]

    @property
    def final_loss(self) -> float:
        return self.history[-1].loss

    @property
    def final_accuracy(self) -> float:
        return self.history[-1].accuracy

    @property
    def increasing_steps(self) -> int:
        return sum(step.increased for step in self.steps)

    @property
    def rejected_updates(self) -> int:
        return sum(not step.accepted for step in self.steps)

    @property
    def clipped_steps(self) -> int:
        return sum(step.clip_factor < 1.0 for step in self.steps)

    @property
    def total_extra_trials(self) -> int:
        return sum(step.trial_count - 1 for step in self.steps)


def _parameters(model) -> list[tuple[np.ndarray, np.ndarray]]:
    """Writable model parameter arrays paired with their gradient arrays."""
    result = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            result.extend(((layer.weights, layer.grad_weights), (layer.bias, layer.grad_bias)))
    return result


def _norm(arrays: list[np.ndarray]) -> float:
    return sqrt(sum(float(np.sum(array * array)) for array in arrays))


def clip_global_norm(model, max_norm: float) -> tuple[float, float, float]:
    """In-place global L2 clipping of ALL Dense gradients, preserving direction.

    Returns (raw_norm, clipped_norm, scale); no scaling below max_norm.
    The gradients must have been computed by `backward` immediately beforehand.
    """
    if not isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    params = _parameters(model)
    grads = [gradient for _, gradient in params]
    if not all(np.all(np.isfinite(gradient)) for gradient in grads):
        raise FloatingPointError("non-finite gradient")
    raw = _norm(grads)
    factor = min(1.0, max_norm / raw) if raw > 0 else 1.0
    if factor < 1.0:
        for gradient in grads:
            gradient *= factor
    return raw, _norm(grads), factor


def run_stability(*, config: StabilityConfig, seed: int = 7, epochs: int = 2000) -> StabilityRun:
    if epochs <= 0 or isinstance(epochs, bool) or not isinstance(epochs, int):
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    model = make_xor_model(seed)
    loss = BinaryCrossEntropy()

    def evaluate(epoch: int) -> EpochRecord:
        prediction = model.forward(XOR_INPUTS)
        value = loss.forward(prediction, XOR_TARGETS)
        accuracy = float(np.mean((prediction >= 0.5) == XOR_TARGETS))
        return EpochRecord(epoch, value, accuracy)

    history = [evaluate(0)]
    steps: list[StabilityStep] = []
    for epoch in range(1, epochs + 1):
        pre_loss = history[-1].loss
        saturation = _hidden_saturation(model)
        model.backward(loss.backward())
        params = _parameters(model)
        parameter_l2 = _norm([arr for arr, _ in params])
        if config.clips:
            raw_gradient_l2, applied_gradient_l2, factor = clip_global_norm(
                model, config.gradient_max_norm
            )
        else:
            raw_gradient_l2 = _norm([gradient for _, gradient in params])
            applied_gradient_l2, factor = raw_gradient_l2, 1.0
        proposed_lr = config.scheduled_lr(epoch)
        snapshot = [array.copy() for array, _ in params]
        attempts = 0
        accepted = False
        actual_lr = 0.0
        last_row: EpochRecord | None = None
        max_attempts = 1 + (config.max_backtracks if config.backtracks else 0)
        for trial in range(max_attempts):
            attempts += 1
            rate = proposed_lr * config.backtrack_factor ** trial if config.backtracks else proposed_lr
            # Each trial must use identical pre-update weights.
            for (array, _), saved in zip(params, snapshot):
                array[...] = saved
            SGD(rate).step(model)
            try:
                candidate = evaluate(epoch)
                finite_candidate = isfinite(candidate.loss)
            except (ValueError, FloatingPointError, OverflowError):
                candidate = None
                finite_candidate = False
            if finite_candidate and (not config.backtracks or candidate.loss <= pre_loss + 1e-12):
                accepted = True
                actual_lr = rate
                last_row = candidate
                break
        if not accepted:
            # Revert rejected proposals: never silently keep the last bad trial.
            for (array, _), saved in zip(params, snapshot):
                array[...] = saved
            last_row = evaluate(epoch)
        assert last_row is not None
        if not isfinite(last_row.loss):
            raise FloatingPointError(f"non-finite loss after update {epoch}")
        update_l2 = actual_lr * applied_gradient_l2
        relative_update = update_l2 / max(parameter_l2, 1e-12)
        history.append(last_row)
        steps.append(StabilityStep(
            epoch, pre_loss, last_row.loss, raw_gradient_l2, applied_gradient_l2,
            factor, proposed_lr, actual_lr, attempts, accepted,
            parameter_l2, update_l2, relative_update, saturation, last_row.accuracy,
        ))
    return StabilityRun(config, seed, history, steps)


def write_stability_steps(path: str | Path, run: StabilityRun) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("epoch", "pre_loss", "post_loss", "loss_increased", "raw_gradient_l2",
                         "applied_gradient_l2", "clip_factor", "scheduled_lr", "effective_lr",
                         "trial_count", "accepted", "parameter_l2", "update_l2", "relative_update",
                         "hidden_saturation", "training_accuracy"))
        for s in run.steps:
            writer.writerow((s.epoch, s.pre_loss, s.post_loss, int(s.increased),
                             s.raw_gradient_l2, s.applied_gradient_l2, s.clip_factor,
                             s.scheduled_lr, s.effective_lr, s.trial_count, int(s.accepted),
                             s.parameter_l2, s.update_l2, s.relative_update,
                             s.hidden_saturation, s.accuracy))
    return destination


def write_stability_summary(path: str | Path, runs: list[StabilityRun]) -> Path:
    if not runs:
        raise ValueError("at least one run is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("mode", "seed", "initial_lr", "epochs", "initial_loss", "final_loss",
                         "final_training_accuracy", "loss_increasing_steps", "clipped_steps",
                         "rejected_updates", "extra_backtracking_trials", "max_relative_update"))
        for run in runs:
            writer.writerow((run.config.mode, run.seed, run.config.learning_rate,
                             len(run.steps), run.history[0].loss, run.final_loss,
                             run.final_accuracy, run.increasing_steps,
                             run.clipped_steps, run.rejected_updates,
                             run.total_extra_trials,
                             max(s.relative_update for s in run.steps)))
    return destination
