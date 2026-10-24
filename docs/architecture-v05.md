# Multiclass architecture note

The new `Softmax` and `ReLU` classes implement explicit activation derivatives. `SoftmaxCrossEntropy` accepts logits plus integer class indices and computes the stable log-sum-exp cross-entropy gradient directly. `multiclass.py` owns deterministic synthetic dataset generation, the small training loop, and numerical parameter-gradient validation. `multiclass_reports.py` writes portable CSV and SVG experiment outputs.

The existing `Dense`, `Sequential`, and `SGD` components are reused unchanged. The original binary XOR CLI commands and analyses remain supported. New multiclass commands do not alter the saved state or behavior of the XOR experiments.

The training loop re-executes a forward pass on training inputs immediately before the gradient step. Measuring validation performance does **not** share loss-gradient state with the training loss. The validation set is independently generated and is never used for weight updates.

A separately implemented finite-difference checker verifies every weight and bias derivative in a 27-parameter 2–4–3 network; it restores all perturbed parameters and leaves the model caches valid.
