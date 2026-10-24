# Evaluation history

## v0.4.0 — 2026-10-24

Published GitHub Actions run on Python 3.12 passed 48 tests, including a console encoding regression and a valid-SVG-label regression. The fixed-seed XOR experiment reached 100% training accuracy on its four examples; this is not a generalization estimate.

## v0.5.0 — local verification

- 64 automated tests passed in the development environment, including all 48 prior tests.
- A 2–4–3 multiclass network passed numerical checking for 27 parameters.
- The default 2–24–3 model trained successfully on 300 synthetic training examples and was evaluated on 180 separately generated synthetic validation examples.
- For seed 7 and 1,200 updates, both training and validation accuracy were 100%. This is a synthetic toy task, not a production accuracy claim.
- XML parsing tests validated SVG output, and the CLI produced CSV and SVG artifacts.

Validation of the new version on the GitHub Actions runner must be repeated after publication. This repository does not yet include a real-world multiclass benchmark.
