# Lesson 02 — Verify gradients and compare learning rates

**What you learn:** numerical differentiation, analytical gradients, why measurement matters, and the difference between fitting training data and generalization.

## Part A — Can we trust our gradients?

Our neural network calculates gradients through the chain rule. A gradient checker performs a separate computation using **only two loss evaluations**, for each parameter:

`dL/dw ≈ [L(w + ε) − L(w − ε)] / (2ε)`.

This is a **central finite difference**. We temporarily perturb one weight while keeping **all other weights unchanged**, measure both losses, then restore the original weight.

Example: if `L(w+ε)=0.230507` and `L(w−ε)=0.230511` with `ε=0.00001`, the estimated gradient is `(−0.000004)/(0.00002)=−0.2` (illustrative rounded values, **not** the actual network output).

A small error between the hand-derived and estimated gradient supports correctness. We compare using

`|analytic − numeric| <= atol + rtol * max(|analytic|, |numeric|)`.

```powershell
.\.venv\Scripts\python.exe -m numpynet check-gradients
```

The 2→4→1 network has 17 trainable numbers: `8+4` for the first Dense layer and `4+1` for the second. All 17 should pass with the default non-saturated initialization.

**Read:** `src/numpynet/gradcheck.py`, especially the `try/finally` that restores a perturbed parameter. The deliberately incorrect `Dense.backward()` test in `tests/test_gradcheck.py` proves the checker can reject an erroneous derivative.

**Important limitation:** `ε` that is too large introduces approximation error; `ε` that is too small amplifies floating-point cancellation. Sigmoid/BCE clipping and non-smooth activations can also produce disagreeing derivatives at special points.

### Experiment A (your own notes)

1. Run `check-gradients` with `--epsilon 0.01`, then `1e-5`, then `1e-10`.
2. Which epsilon has the smallest *maximum absolute error* on this model?
3. Explain why agreement is evidence supporting the implementation, but cannot prove its correctness in all cases.

## Part B — What does the learning rate change?

The SGD update is `w_next = w_current − η * (dL/dw)`. The learning rate `η` controls the step size. Too small can cause slow progress; too large can skip useful regions or destabilize learning.

To compare fairly, we hold constant the model architecture (2→4→1), four XOR examples, initialization **seed 7**, full-batch training, loss function, and **2000 update steps**. Only `η` changes.

```powershell
.\.venv\Scripts\python.exe -m numpynet compare-lr --rates 0.01 0.1 0.7 2 10 --epochs 2000 --seed 7
```

Outputs:

- `results/lr_comparison/summary.csv`: one summary row per learning rate.
- `results/lr_comparison/run_XX_lr_X.csv`: **epoch 0** and every post-update training loss/accuracy.
- `results/lr_comparison/loss_curves.svg`: logarithmic training-loss curves; open it in a browser without installing plotting libraries.

### Our reference run (Python 3.13 / NumPy, seed 7)

| Learning rate | Initial loss | Loss after 2000 steps | Training accuracy |
| ---: | ---: | ---: | ---: |
| 0.01 | 0.704652 | 0.490246 | 100% |
| 0.1 | 0.704652 | 0.014330 | 100% |
| 0.7 | 0.704652 | 0.001564 | 100% |
| 2 | 0.704652 | 0.000523 | 100% |
| 10 | 0.704652 | 3.112075 | 50% |

**Interpret carefully.** In this particular deterministic experiment, `η=2` fits the training points faster than `η=0.01`, whereas `η=10` fails to fit them within the same update budget. That does **not** imply `2` is generally optimal or `10` always fails; we'd need repeated seeds and different data to establish a broader result.

Notice that `η=0.01` already classifies all four examples correctly while retaining comparatively high loss. **Accuracy** uses only a 0.5 threshold, while **cross-entropy** also measures confidence.

### Experiment B (your own notes)

1. Pick one metric: final loss or number of updates until loss first falls below 0.1. Which is more meaningful for comparing speed of convergence, and why?
2. Run seeds `7`, `8`, and `9` for learning rates `0.1` and `2`. Does the ordering always hold?
3. Explain why testing on all four training examples is not a generalization evaluation.
4. Propose a change for v0.3: can a new activation or optimizer improve reproducibility across random seeds?

## Before merging into GitHub

- Verify the tests run.
- Add at least one observation written **in your own words** to `docs/learning-log.md`.
- Commit the meaningful change after understanding it. Do not change commit dates or manufacture activity.
