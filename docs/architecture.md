# Architecture — a small, inspectable learning engine

```text
                      ┌───────────────┐
CLI: train / compare ──┤ experiments.py├──> ExperimentResult.history
                      └───────┬───────┘                |
                              v                        v
                        Sequential                 reports.py
                       forward/backward           CSV + SVG
                              |
                              v
                       Dense → Tanh → Dense → Sigmoid
                              |
                              v
                         BCE loss + SGD
                              |
        check-gradients ───────┘
            |   |   |
     central-difference probe of every Dense weight/bias
```

## Design choices

- NumPy array operations only; no graph-based autograd.
- Layers own their caches and explicit backward methods. This makes the chain rule visible, but the same model is **not** safe for concurrent training/inference calls.
- The checker temporarily mutates one parameter, restores it in a `finally` block, and restores the model's forward/backward caches after completing checks. It should be run on small, non-saturated networks.
- `history[0]` records metrics before any updates. `history[k]` records metrics after **k** SGD steps, so x-axis labels can be interpreted exactly.
- CSV and SVG are written with Python's standard library; training depends on NumPy only.
- The `compare-lr` command builds a **fresh model using the same seed** for every rate. This controls initialization but does not address variation across seeds.
