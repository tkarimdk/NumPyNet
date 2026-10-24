# Math notes — v0.1

These notes connect each equation to a line of code. Work through them before modifying the implementation.

## 1. Dense layer

Let `X` be a batch of size `N × d`, `W` be `d × h`, and `b` be `1 × h`.

```text
Z = XW + b
```

With incoming derivative `G = dL/dZ`:

```text
dL/dW = X.T @ G
dL/db = sum(G, axis=0)
dL/dX = G @ W.T
```

Because our BCE loss takes the mean, the upstream gradient already contains its division by the number of elements. **Do not divide by batch size a second time in `Dense.backward`.**

## 2. Tanh activation

```text
A = tanh(Z)
dA/dZ = 1 - A^2
```

Therefore `dL/dZ = (dL/dA) * (1 - A^2)` where `*` is elementwise multiplication.

## 3. Sigmoid activation

```text
A = 1 / (1 + exp(-Z))
dA/dZ = A(1-A)
```

The implementation handles positive and negative inputs separately to avoid overflow for very large magnitudes.

## 4. Binary cross-entropy (BCE)

For predicted probability `p` and label `y ∈ {0,1}`:

```text
L = -mean(y * log(p) + (1-y) * log(1-p))
dL/dp = (p-y) / (p * (1-p) * number_of_elements)
```

We clip probabilities away from exactly 0 and 1 to keep logarithms finite. This separate sigmoid+BCE teaching implementation is most accurate away from clipped extremes; a future release should use a fused `BCEWithLogits` for better extreme-value gradient behavior.

## 5. SGD

```text
W_next = W - learning_rate * dL/dW
b_next = b - learning_rate * dL/db
```

The optimizer modifies only Dense parameters. Activation layers have no trainable parameters.

## 6. Why XOR?

XOR cannot be separated by a single linear decision boundary. A hidden nonlinear layer lets the network transform inputs into a representation in which the classes can be separated. A correct implementation should fit these four samples with an appropriate configuration.

## 7. Gradient checking

Estimate one weight's derivative numerically:

```text
dL/dw ≈ (L(w + ε) - L(w - ε)) / (2ε)
```

Compare it with the analytical derivative from backpropagation. The numerical check is slower but useful for catching incorrect signs, shapes, and missing factors.
