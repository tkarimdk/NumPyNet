# Lesson 03 — Diagnosing optimization behavior

**Question:** In our deterministic XOR experiment, why does SGD with learning rate 10 fail to fit the training examples while learning rate 2 succeeds?

This is **not a research claim** or a universal learning-rate recommendation. The network has only four training examples, and we inspect only a few randomly initialized runs.

## Learning goals

1. Distinguish a correct gradient from a productive optimization step.
2. Understand **gradient magnitude**, **update magnitude**, and **relative parameter change**.
3. See why a loss spike and saturated hidden activations can be useful diagnostics.
4. Use controlled comparisons across random seeds to avoid generalizing from one lucky initialization.
5. Describe observations separately from causal explanations.

## Step 1 — Generate the experiment

From the inner `NumPyNet` directory containing `pyproject.toml`, run:

```powershell
.\.venv\Scripts\python.exe -m numpynet diagnose-lr --rates 2 10 --seeds 7 8 9 --epochs 2000
```

The command writes:

- `results/lr_diagnostics/diagnostic_summary.csv`: one row per (learning rate, seed) experiment.
- `results/lr_diagnostics/steps_lr_2_seed_7.csv` and other matching paths: one row per parameter update, with pre- and post-update metrics.
- `results/lr_diagnostics/loss_seed_7.svg` (and a graph for every chosen seed): browser-viewable comparison of the selected rates on the same initialization.

The `results/` folder is ignored by Git. A small curated reference summary is in `examples/reference_diagnostic_summary.csv`.

**Fairness:** the network, full-batch XOR training data, loss function, optimizer, number of updates, and each seed are kept fixed for a pair of learning rates. Only the learning rate changes within each seed pair.

## Step 2 — Understand the diagnostics

Let `theta` be **all** Dense layer weights and biases concatenated into one vector, and `g` their corresponding gradient vector, both calculated immediately *before* each SGD step.

| Measurement | Definition | Interpretation |
| --- | --- | --- |
| `gradient_l2` | `sqrt(sum(g_i²))` | Overall gradient magnitude |
| `parameter_l2` | `sqrt(sum(theta_i²))` | Overall model parameter magnitude before update |
| `update_l2` | `eta * gradient_l2` | Length of SGD's requested parameter change |
| `relative_update` | `update_l2 / max(parameter_l2, 1e-12)` | Scale of the change compared with current parameters |
| `loss_increased` | `post_loss > pre_loss + 1e-12` | Whether this specific update made training loss worse |
| `hidden_saturation` | Fraction of Tanh hidden activations with `abs(activation) >= 0.95` | A heuristic signal of near-saturated hidden units |

For example, on **seed 7, first update**:

- The gradient norm is `0.07916935` for both learning rates, because they begin from the same weights and data.
- At learning rate **2**, `update_l2 ≈ 0.15834` and `relative_update ≈ 0.0767`.
- At learning rate **10**, `update_l2 ≈ 0.79169` and `relative_update ≈ 0.3836`.

The initial *direction* is the same; the *distance* moved is five times greater at rate 10. Future gradients differ because the parameters now occupy different positions.

## Step 3 — Read the evidence

In the reference run, for seed 7:

- With `eta=2`, **zero** of 2,000 steps increased loss; final training loss was about **0.000523**.
- With `eta=10`, **1,330** steps increased loss; final training loss was about **3.112075**.
- At rate 10, update 4 raised loss from about **0.4836** to **1.3431**.
- At rate 10, update 5 had a relative parameter update around **2.22**, meaning the update vector was more than twice the parameter vector's norm.
- Late in that run, the hidden-layer Tanh activations were almost entirely at or beyond the `0.95` threshold.

Other seeds (8 and 9) also struggled at rate 10, but the outcomes and number of loss-increasing updates differed. These are **observations**, not a proof that a certain update ratio or saturation threshold always causes failure.

## Step 4 — Form an explanation, then challenge it

**Working hypothesis:** Very large SGD steps can overshoot directions of lower loss. As parameters change drastically, activations may move toward extreme Tanh values, and training may oscillate or get trapped in unproductive regimes.

To test the hypothesis more rigorously, we would need more seeds, architectures, datasets, and **interventions** such as reducing the rate after a spike. This project does not yet implement an adaptive intervention or a predictive warning system.

**Important caveat:** gradients in this basic library propagate through Sigmoid and a clipped binary cross-entropy separately, which can behave poorly under extreme saturation. That numerical implementation choice itself warrants investigation. A fused, stable sigmoid-with-logits loss is a good future experiment.

## Your exercises (complete these before publishing)

1. Open `steps_lr_10_seed_7.csv`. At which update does the first significant loss increase occur? Write down its `pre_loss` and `post_loss`.
2. Calculate the **first update magnitude** for rate 10 yourself using `eta * gradient_l2`. Check it against the CSV.
3. Contrast `loss_increased` with `gradient_l2`. Why doesn't a valid gradient guarantee every finite step reduces the loss?
4. Compare rate 10 on seeds 7, 8, and 9. Is the result identical? What does that tell you about seed selection?
5. Write a short hypothesis in your own words. What experiment could *disprove* it?

Record observations and hypotheses in `docs/learning-log.md`. Conclusions should be supported by measured results and appropriate baselines.
