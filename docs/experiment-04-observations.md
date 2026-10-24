# Reproducible experiment notebook — v0.4

**Hypothesis:** For the deliberately unstable full-batch XOR setting (starting LR 10), clipping or rejecting overly aggressive steps can lead to more reliable training convergence than plain SGD.

**Independent variable:** intervention (`baseline`, `clip`, `decay`, `backtrack`, optional combinations).

**Controls:** seed in {7,8,9}, XOR inputs/targets, network architecture 2→4→1, sigmoid/BCE, gradient implementation, fixed 2,000 update budget, same initialization within each seed.

**Outcomes:** final binary cross-entropy on four training rows, final training accuracy, loss-increasing steps, clipping events, rejected steps, extra backtracking trials, update norms. **No validation/test data** and **no compute benchmarking** are included here.

**Command:**

```text
python -m numpynet stabilize-xor --epochs 2000 --lr 10 --modes baseline clip decay backtrack clip-backtrack --seeds 7 8 9
```

Detailed per-run results in `examples/lesson04_reference_summary.csv` and viewable `examples/lesson04_seed7_loss.svg`. The generated references can be checked by running the command again. The interventions are established methods; the experiments compare their behavior under the specified conditions.

**Observed:** Plain SGD converged poorly for LR 10. Clip and backtracking fit all four training rows for the tested seeds, whereas the chosen decay schedule was inconsistent. The baseline results reproduce v0.3.

**Not proved:** Better downstream or test-set accuracy, reduced wall-clock time, statistical advantage beyond three seeds, or general reliability for deeper models. Repeated evaluation for backtracking incurs compute overhead.

**Next experiment design:** Run 20–50 seeds, summarize median and dispersion, measure wall-clock and function evaluations, add a real dataset with train/validation/test splits, and pre-register comparison settings before tuning.
