# Roadmap — evidence-driven, incremental development

Development follows working, testable milestones. Release dates depend on implementation and verification.

| Version | Engineering milestone | Learning objective | Status |
| --- | --- | --- | --- |
| v0.1 | Dense network, manual gradients, XOR | Chain rule, tensor shapes, optimization | Complete |
| v0.1.1 | Single neuron trace | Explain a complete SGD update | Complete |
| v0.2 | Numerical gradient checker, CSV metrics, learning-rate visualizations | Validate derivatives and compare optimization under controls | Complete |
| v0.3 | Learning-rate diagnostics, per-update CSV logs and three-seed comparison | Interpret gradient norms, step sizes, loss spikes and saturation | Complete |
| v0.4 | Gradient clipping, learning-rate decay and backtracking | Test whether stability interventions improve controlled failures | Complete |
| v0.5 | ReLU, softmax, logits-based multiclass cross-entropy, synthetic validation | Stable classification and derivative contracts | Complete |
| v0.6 | Mini-batching, momentum, Adam | Different optimization methods | Planned |
| v0.7 | Stable logits-based BCE and broader experiments on real data | Numerical stability, variation, selection bias, overfitting | Planned |
| v0.8 | Save/load model parameters and CLI inference | Reproducibility and usable inference | Planned |
| v1.0 | Documented public-dataset benchmark with baselines | Honest evaluation and limitations | Planned |

A separate diagnostic research project, **NeuroTrace**, is a possible longer-term direction. Any claims of novelty or performance require a literature review and controlled benchmark comparisons.
