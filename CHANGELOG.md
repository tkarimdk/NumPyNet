# Changelog

## v0.5.0 — Three-class classification

- Added stable Softmax and its explicit Jacobian-vector derivative and ReLU activation.
- Added fused logits-based multiclass cross-entropy and direct class-logit gradients.
- Added seeded synthetic spiral training with separately generated validation data.
- Added 27-parameter numerical gradient verification, train/validation CSVs, and a browser-viewable SVG.
- Added CLI commands `train-multiclass` and `check-multiclass-gradients`.
- Added 16 tests; preserved 48 existing tests from the published v0.4.0 repository.
- Documented synthetic evaluation limits and learning exercises.


## v0.4.0 — Lesson 04: stability interventions

- Added global L2 gradient clipping across all Dense parameters, including gradients for biases.
- Added bounded exponential learning-rate decay and loss-checked backtracking with explicit rollback after rejected candidates.
- Added per-step CSV logs recording clipping, scheduled vs. actual rates, accepted/rejected updates, loss, saturation and parameter update ratios.
- Added `stabilize-xor` CLI supporting controlled interventions across three seeds, with SVG plots and summary CSV.
- Added 11 new automated tests, including a regression check that the baseline matches v0.3.
- Documented positive and negative results with explicit limitations and exercises.
- Preserved all previous commands and existing 35 tests.

## v0.3.0 — Lesson 03: instrument SGD

- Added deterministic per-update training diagnostics for gradient norms, parameter norms, relative update magnitude, loss-increase events, and hidden Tanh saturation.
- Added `diagnose-lr` CLI command for controlled comparisons across learning rates and seeds, with exported per-step and summary CSVs and browser-viewable loss plots for each seed.
- Added tests that verify instrumentation produces exactly the same loss history as the v0.2 training engine.
- Documented observed failures for rate 10 without making universal or causal claims.
- Preserved all v0.2 commands.

## v0.2.0 — Lesson 02: verify and measure

- Added a full-network central finite-difference gradient checker for Dense weights and biases, with configurable tolerances and error exit status.
- Added a negative-control test that verifies the checker rejects deliberately wrong derivatives.
- Added epoch-zero and post-update loss/accuracy records, CSV logs, deterministic learning-rate comparisons, and browser-viewable SVG plots.
- Added controlled experiment results, a new lesson, and experiment documentation.
- Preserved Lesson 01 commands and behavior.

## v0.1.1 — Lesson 01: inspect a learning step

- Added `trace-neuron` command to reveal a single neuron's forward pass, BCE loss, analytic gradients, and parameter update.
- Added a hands-on lesson with manually reproducible arithmetic.
- Added tests comparing the trace against closed-form results.
- Kept the existing XOR experiment and CLI behavior unchanged.

## v0.1.0 — Foundations

- Dense layer with explicit forward/backward propagation.
- Tanh and Sigmoid activation functions.
- Binary cross-entropy loss and SGD updates.
- XOR experiment, reproducibility checks, tests, documentation and GitHub Actions configuration.
