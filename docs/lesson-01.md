# Lesson 01 — One neuron, one learning step

**Objective:** explain and reproduce every value in a two-input neuron before working with an entire network. Allow 30–45 minutes to study and experiment; take longer if needed.

## Prerequisites

Python fundamentals (variables and functions); vectors and multiplication; the idea of a derivative. If derivatives are unfamiliar, review the chain rule in [`math-notes.md`](math-notes.md).

## A single, inspectable example

We choose a binary classifier with one neuron and one sample:

- Inputs: `x = [2.0, -1.0]`
- Weights: `w = [0.5, -0.25]`
- Bias: `b = 0.1`
- Correct target: `y = 1`
- Learning rate: `eta = 0.1`

### Part A: Forward pass

1. Weighted sum (logit): `z = x · w + b = 2(0.5) + (-1)(-0.25) + 0.1 = 1.35`
2. Sigmoid: `p = 1 / (1 + exp(-z)) ≈ 0.79413`
3. Binary cross-entropy (for `y=1`): `L = -ln(p) ≈ 0.23051`

So the neuron predicts about 79.4% probability of class 1. This is a numerical output, not a calibrated probability for a real dataset.

### Part B: Backward pass

The sigmoid derivative is `dp/dz = p(1-p)`.

For `y=1`, the binary cross-entropy derivative is `dL/dp = -1/p`. Combining them with the chain rule:

`dL/dz = (dL/dp)(dp/dz) = -(1/p) p(1-p) = p - 1`.

More generally, for binary target `y`, `dL/dz = p - y` when sigmoid is paired with BCE (away from clipping extremes).

Since `z = x · w + b`, the parameter gradients are:

- `dL/dw_1 = x_1 (p-y)`
- `dL/dw_2 = x_2 (p-y)`
- `dL/db = p-y`

Notice that the second input is negative. Its gradient changes sign accordingly.

### Part C: Parameter update

SGD uses `new_parameter = old_parameter - eta * gradient`.

We compute the gradients using the **current parameters**, update weights and bias once, and then run another forward pass. For this chosen example, the updated loss is smaller; that is not guaranteed for every learning rate and every example.

## Run the trace

```bash
python -m numpynet trace-neuron
python -m numpynet trace-neuron --lr 0.01
python -m numpynet trace-neuron --lr 1.0
```

Tests:

```bash
python -m unittest discover -s tests -v
```

If you do not want to install the package yet, run it from the repository root using Python's module path:

- PowerShell: `$env:PYTHONPATH='src'; python -m numpynet trace-neuron`
- macOS/Linux: `PYTHONPATH=src python -m numpynet trace-neuron`

## Your tasks — try before inspecting the printed answers

1. Calculate `z` from the four values above. Why do we add a bias?
2. Calculate `p` using the sigmoid formula (calculator allowed).
3. Explain why `dL/dz = p-y` in this example.
4. Calculate both weight gradients and the bias gradient to **4 decimal places**.
5. Calculate all **three** updated parameters at `eta=0.1`.
6. Run with `--lr 0.01` and `--lr 1.0`. Record the resulting losses, and explain the difference.

**Write your observations** in `docs/learning-log.md` after completing the exercise. Code and mathematics come first; publishing comes after you can explain them.

## Where to read the implementation

- `src/numpynet/lessons.py` orchestrates the example and stores trace values.
- `src/numpynet/layers.py` implements the Dense layer and its gradient.
- `src/numpynet/activations.py` implements Sigmoid and its derivative.
- `src/numpynet/losses.py` calculates BCE and its derivative.
- `src/numpynet/optim.py` makes the parameter update.
- `tests/test_lesson_one.py` compares the implementation against the mathematical formulas.

## Common mistakes

- Thinking a low loss on **one training example** means the model generalizes.
- Treating the gradient as the weight update; the update also multiplies by the learning rate and uses a negative sign.
- Mixing up `dL/dp` with `dL/dz`; the chain rule connects them.
- Forgetting the negative sign on the second input when calculating `dL/dw_2`.
