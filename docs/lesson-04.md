# Lesson 04 — Why training can fail, and how to test possible fixes

## Question

At initial learning rate **10**, our fixed XOR network often fails to fit its four training examples after 2,000 full-batch updates. Can standard training safeguards improve it, using **the same data, architecture, initialization seeds, and update budgets**?

These are controlled experiments on the XOR training set. They do not establish performance on unseen data or constitute a new optimization method.

## 1. The three interventions

### A. Global gradient-norm clipping

Let each model parameter have a derivative, collected into a long gradient vector `g`. Calculate its global L2 magnitude

`||g||₂ = sqrt(sum_i (g_i²))`.

If the gradient norm exceeds the cap `c`, rescale **all** gradient arrays by the same factor:

`g_clipped = g * min(1, c / ||g||₂)`.

This limits the *gradient magnitude* and preserves the gradient direction. It **does not** directly guarantee that the next loss decreases, because step size also depends on the learning rate and local loss curvature. This implementation defaults to `c = 0.3`.

### B. Exponential learning-rate decay

For the `k`th update, with k starting at 1:

`eta_k = max(eta_min, eta_initial * decay_factor ** (k - 1))`.

Default: initial `eta = 10`, factor `0.995`, minimum `0.05`. A scheduler can progressively reduce update size but **cannot undo early damage** by itself.

### C. Backtracking with rejected updates restored

Compute a candidate update. Evaluate the **full-batch training loss**. If the candidate loss exceeds the pre-update loss by more than `1e-12`, revert model parameters and try a smaller learning rate (multiply by `0.5`, up to eight additional attempts). If no candidate passes, restore the original parameters and record a skipped update.

This is a deliberately simple teaching version of monotonic backtracking, **not** a general training strategy for stochastic minibatches. Re-evaluating the full training loss repeatedly adds computation; loss monotonicity on the training batch does not guarantee good validation accuracy.

## 2. Reproduce the experiment

Inside the `NumPyNet` folder with Python/NumPy installed:

```powershell
python -m numpynet stabilize-xor --lr 10 --modes baseline clip decay backtrack --seeds 7 8 9 --epochs 2000
```

Try additional combinations:

```powershell
python -m numpynet stabilize-xor --modes baseline clip decay backtrack clip-backtrack --seeds 7 8 9 --epochs 2000
```

The command produces `results/stability/stability_summary.csv`, per-step CSV logs, and loss-curve SVG files for each seed. Open `results/stability/loss_seed_7.svg` to compare curves. Check the printed **training** accuracy as well as the loss, since high training accuracy can coexist with quite different losses.

## 3. Reference results

Full batch: 4 XOR examples; 2 -> 4 -> 1 network; fixed seeds; 2,000 updates per run. The `baseline` in this release was checked against v0.3 to preserve its loss history.

| Method | Seed 7: loss / accuracy | Seed 8: loss / accuracy | Seed 9: loss / accuracy |
| --- | --- | --- | --- |
| Unmodified SGD | 3.112075 / 50% | 1.533912 / 75% | 5.473079 / 50% |
| Gradient clipping (norm 0.3) | 0.000106 / 100% | 0.000105 / 100% | 0.000069 / 100% |
| Exponential decay (0.995, min 0.05) | 0.477392 / 75% | 0.000847 / 100% | 0.693151 / 50% |
| Backtracking (0.5, 8 retries) | 0.000102 / 100% | 0.000102 / 100% | 0.000101 / 100% |

These results apply only to the specified settings and seeds. Reproducibility is built in; statistical representativeness is **not** established by three runs. Both clipping and backtracking are known techniques, not inventions of this project.

### What the per-step logs reveal

- Unmodified rate 10 had **1,330, 1,281, 1,000** loss-increasing updates across seeds 7, 8, and 9.
- Global clipping still allowed occasional loss-increasing updates (**5, 11, 19**), but markedly improved final fit in these runs. This illustrates why clipping is not a monotonic-loss guarantee.
- Backtracking had **no measured increasing updates** by design because it rejects updates that increase the same full-batch training loss; it incurred **23, 29, 41 extra candidate evaluations** respectively.
- Exponential decay helped seed 8 but did not fully recover seeds 7 and 9. An intervention should not be declared successful based on just one favorable seed.

## 4. Exercises (do these yourself)

1. Increase the clipping threshold from `0.3` to `1.0`, then to `3.0` with `--max-norm`. How many updates are clipped? Does training accuracy stay consistent across seeds?
2. Change exponential decay from `0.995` to `0.99`. Does the earlier reduction in update size help? What happens when you change the minimum from `0.05` to `0.1`?
3. Try `--max-backtracks 0`. Count the rejected steps and explain why their `effective_lr` and `update_l2` are recorded as zero.
4. Compare `clip` and `clip-backtrack` under the same settings. Which reaches low loss sooner? Which requires more evaluations of the loss?
5. Write 150–200 words on why a monotonic training-loss curve does **not** guarantee generalization to new data.

## 5. Keep these distinctions clear

- A *parameter update* is different from an extra *candidate loss evaluation*; backtracking can cost more wall-clock time at the same epoch budget.
- Gradient clipping bounds a vector norm but does not ensure the learning rate is appropriate.
- A scheduler may not repair damage from high learning rates at early steps.
- The metric is **training accuracy on four samples**, not a held-out test score.
- Next, evaluate on genuinely nontrivial data, expand seed counts, and compare compute/evaluation overhead. The aim is correct experimental reasoning, not automatically claiming a publishable contribution.
