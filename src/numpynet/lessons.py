"""Small deterministic demonstrations that expose neural network arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import Sigmoid
from .layers import Dense
from .losses import BinaryCrossEntropy
from .model import Sequential
from .optim import SGD


@dataclass(frozen=True)
class NeuronTrace:
    """Values before and after one gradient-descent update for a single sample."""

    inputs: np.ndarray
    target: float
    weights_before: np.ndarray
    bias_before: float
    logit: float
    prediction_before: float
    loss_before: float
    gradient_weights: np.ndarray
    gradient_bias: float
    learning_rate: float
    weights_after: np.ndarray
    bias_after: float
    prediction_after: float
    loss_after: float


def trace_single_neuron(learning_rate: float = 0.1) -> NeuronTrace:
    """Trace a Dense(2, 1) -> Sigmoid -> BCE learning step, without autograd.

    Uses intentionally chosen weights, not random trained parameters, so the
    arithmetic can be reproduced on paper. A single sample is NOT a training
    benchmark; it merely makes the chain rule inspectable.
    """
    inputs = np.array([[2.0, -1.0]], dtype=np.float64)
    targets = np.array([[1.0]], dtype=np.float64)
    dense = Dense(2, 1, np.random.default_rng(0))
    dense.weights[:, 0] = [0.5, -0.25]
    dense.bias[0, 0] = 0.1
    model = Sequential([dense, Sigmoid()])
    loss_fn = BinaryCrossEntropy()
    optimizer = SGD(learning_rate)

    weights_before = dense.weights[:, 0].copy()
    bias_before = float(dense.bias[0, 0])
    logit = float((inputs @ dense.weights + dense.bias)[0, 0])
    prediction_before = float(model.forward(inputs)[0, 0])
    loss_before = loss_fn.forward(np.array([[prediction_before]]), targets)
    model.backward(loss_fn.backward())
    grad_weights = dense.grad_weights[:, 0].copy()
    grad_bias = float(dense.grad_bias[0, 0])

    optimizer.step(model)
    prediction_after = float(model.forward(inputs)[0, 0])
    loss_after = loss_fn.forward(np.array([[prediction_after]]), targets)

    return NeuronTrace(
        inputs=inputs[0].copy(), target=1.0,
        weights_before=weights_before, bias_before=bias_before,
        logit=logit, prediction_before=prediction_before, loss_before=loss_before,
        gradient_weights=grad_weights, gradient_bias=grad_bias,
        learning_rate=learning_rate,
        weights_after=dense.weights[:, 0].copy(), bias_after=float(dense.bias[0, 0]),
        prediction_after=prediction_after, loss_after=loss_after,
    )
