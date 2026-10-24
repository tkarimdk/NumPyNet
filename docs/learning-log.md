# Learning log

Record observed measurements, experiment settings, and hypotheses separately so results remain reproducible.

## v0.1 — Built the foundation

- Implemented Dense, Tanh, Sigmoid, BCE, SGD without autograd.
- Verified deterministic XOR learning and some analytic derivatives against finite differences.
- XOR has only four training examples; training accuracy doesn't demonstrate generalization.

## v0.1.1 — Lesson 01: one update

A fixed neuron example traces `x=[2,-1]`, `w=[0.5,-0.25]`, `b=0.1`, `target=1`, and one SGD step. The loss decreases from `0.230509` to `0.206296` for `η=0.1`.

**Follow-up exercise:** derive both weight gradients and explain why their signs differ.

## v0.2.0 — Numerical checking and learning-rate controls

**Engineering changes:** a reusable central-finite-difference checker, a negative-control test with a deliberately wrong gradient, structured post-update training history, CSV exports, and a self-contained SVG comparison plot.

**Verification:** all 17 weights and biases of the seeded 2→4→1 network pass the numerical gradient check at `ε=1e-5`, `atol=1e-6`, `rtol=1e-4`. Automated tests cover parameter restoration and demonstrate detection of an intentionally wrong derivative.

**Controlled run (2000 full-batch updates, seed 7):**

| Learning rate | Initial loss | Final loss | Accuracy on 4 training points |
| ---: | ---: | ---: | ---: |
| 0.01 | 0.704652 | 0.490246 | 100% |
| 0.1 | 0.704652 | 0.014330 | 100% |
| 0.7 | 0.704652 | 0.001564 | 100% |
| 2 | 0.704652 | 0.000523 | 100% |
| 10 | 0.704652 | 3.112075 | 50% |

**Observation:** larger learning rates improved optimization speed in this single controlled run up to a point, while `η=10` made the result worse. This is **not** a general result across models or datasets.

### Your observations (complete after running the lesson)

- [ ] Explain in your own words how central finite differences work.
- [ ] Try three epsilon values. Record which produces the smallest absolute discrepancy.
- [ ] Compare learning rates with seed 8. What stayed consistent? What changed?
- [ ] Explain why 100% XOR training accuracy with a loss near 0.49 is mathematically possible.
- [ ] State one possible limitation in our current experiment design.

**Next question:** How can we test whether an optimizer behaves consistently across multiple initializations rather than trusting one convenient seed?


## Lesson 03 — October 2026: learning-rate diagnostics (reference evidence)

Experiment parameters: XOR data (four training samples), 2→4→1 network, Tanh hidden layer, Sigmoid output, full-batch SGD, 2,000 updates, seeds 7, 8, 9. Learning rates 2 versus 10.

Results: rate 2 reached 100% training accuracy in all three runs. Rate 10 reached 50%, 75%, 50%. It also generated many more loss-increasing steps and much larger relative updates in these experiments.

**Your observations (write these in your own words):**

- Which two columns in the CSV best support your explanation?
- Which observed behavior surprised you?
- What test could falsify your hypothesis?

Do not claim this demonstrates generalization, novelty, or a causal explanation.


## Lesson 04: recovery experiments

We deliberately start with the unstable LR=10 setting (seeds 7, 8, 9). Gradient clipping (max norm 0.3) and monotonic backtracking (up to eight retries) both reach 100% training accuracy in all three runs, whereas the chosen decay schedule helps seed 8 but not seeds 7 and 9. Measured values and limitations are in `docs/lesson-04.md`. This is a reproducible educational result, not evidence of superior generalization.

**Write in your own words:** Why can clipping help even when it doesn't guarantee a loss decrease? Why should you count candidate evaluations when comparing backtracking to plain SGD?
