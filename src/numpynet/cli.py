"""CLI for inspectable learning experiments and correctness checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .diagnostics import diagnose_xor, write_steps_csv, write_diagnostic_summary
from .experiments import XOR_INPUTS, XOR_TARGETS, make_xor_model, train_xor
from .gradcheck import check_gradients
from .lessons import trace_single_neuron
from .reports import write_comparison_csv, write_history_csv, write_loss_svg
from .stability import MODES, StabilityConfig, run_stability, write_stability_steps, write_stability_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPyNet: neural networks from fundamentals")
    parser.add_argument("--version", action="version", version=f"NumPyNet {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    xor_parser = subcommands.add_parser("train-xor", help="Train a small network on XOR")
    xor_parser.add_argument("--epochs", type=int, default=5000)
    xor_parser.add_argument("--lr", type=float, default=0.7, help="Learning rate")
    xor_parser.add_argument("--seed", type=int, default=7)
    xor_parser.add_argument("--csv", type=Path, help="Optionally write every epoch to CSV")
    xor_parser.add_argument("--svg", type=Path, help="Optionally render the loss curve as SVG")

    trace_parser = subcommands.add_parser(
        "trace-neuron", help="Show one neuron's forward, backward, and SGD update"
    )
    trace_parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")

    check_parser = subcommands.add_parser(
        "check-gradients", help="Verify all Dense weight/bias derivatives numerically"
    )
    check_parser.add_argument("--seed", type=int, default=7)
    check_parser.add_argument("--epsilon", type=float, default=1e-5)
    check_parser.add_argument("--atol", type=float, default=1e-6)
    check_parser.add_argument("--rtol", type=float, default=1e-4)

    compare_parser = subcommands.add_parser(
        "compare-lr", help="Run controlled XOR experiments with a shared seed"
    )
    compare_parser.add_argument("--rates", nargs="+", type=float, default=[0.01, 0.1, 0.7, 2.0, 10.0])
    compare_parser.add_argument("--epochs", type=int, default=2000)
    compare_parser.add_argument("--seed", type=int, default=7)
    compare_parser.add_argument("--out-dir", type=Path, default=Path("results/lr_comparison"))

    diagnose_parser = subcommands.add_parser(
        "diagnose-lr", help="Trace gradient norms, update sizes, and saturation during XOR training"
    )
    diagnose_parser.add_argument("--rates", nargs="+", type=float, default=[2.0, 10.0])
    diagnose_parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    diagnose_parser.add_argument("--epochs", type=int, default=2000)
    diagnose_parser.add_argument("--out-dir", type=Path, default=Path("results/lr_diagnostics"))

    stability_parser = subcommands.add_parser(
        "stabilize-xor", help="Lesson 04: compare step-size stabilization strategies"
    )
    stability_parser.add_argument("--modes", nargs="+", choices=MODES,
                                  default=["baseline", "clip", "decay", "backtrack"])
    stability_parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    stability_parser.add_argument("--epochs", type=int, default=2000)
    stability_parser.add_argument("--lr", type=float, default=10.0)
    stability_parser.add_argument("--max-norm", type=float, default=0.3)
    stability_parser.add_argument("--decay", type=float, default=0.995)
    stability_parser.add_argument("--min-lr", type=float, default=0.05)
    stability_parser.add_argument("--backtrack-factor", type=float, default=0.5)
    stability_parser.add_argument("--max-backtracks", type=int, default=8)
    stability_parser.add_argument("--out-dir", type=Path, default=Path("results/stability"))

    arguments = parser.parse_args()

    if arguments.command == "trace-neuron":
        trace = trace_single_neuron(learning_rate=arguments.lr)
        print(f"NumPyNet v{__version__} | Lesson 01: one neuron, one update")
        print(f"Inputs x:          {trace.inputs.tolist()}")
        print(f"Target y:          {trace.target:.0f}")
        print(f"Initial weights:   {trace.weights_before.tolist()}")
        print(f"Initial bias:      {trace.bias_before:.6f}")
        print(f"1. Logit z=x@w+b:  {trace.logit:.6f}")
        print(f"2. Probability p:   {trace.prediction_before:.6f}")
        print(f"3. BCE loss:        {trace.loss_before:.6f}")
        print(f"4. dLoss/dWeights:  {trace.gradient_weights.tolist()}")
        print(f"   dLoss/dBias:     {trace.gradient_bias:.6f}")
        print(f"5. Learning rate:   {trace.learning_rate:.6f}")
        print(f"   Updated weights: {trace.weights_after.tolist()}")
        print(f"   Updated bias:    {trace.bias_after:.6f}")
        print(f"6. New prediction:  {trace.prediction_after:.6f}")
        print(f"   New BCE loss:    {trace.loss_after:.6f}")
        print("Tip: reproduce steps 1-5 on paper, then compare your answers.")

    elif arguments.command == "train-xor":
        result = train_xor(epochs=arguments.epochs, learning_rate=arguments.lr, seed=arguments.seed)
        print(f"NumPyNet v{__version__} | 2 -> 4 -> 1 MLP | seed={arguments.seed}")
        print(f"Initial loss: {result.losses[0]:.6f}")
        print(f"Final loss:   {result.losses[-1]:.6f}")
        print(f"Accuracy:     {result.accuracy:.0%} on 4 XOR samples (not a generalization metric)")
        print("Inputs -> target -> predicted probability")
        for inputs, target, probability in zip(XOR_INPUTS, XOR_TARGETS, result.predictions):
            print(f"{inputs.astype(int).tolist()} -> {int(target[0])} -> {probability[0]:.6f}")
        if arguments.csv:
            print(f"History CSV:  {write_history_csv(arguments.csv, result.history)}")
        if arguments.svg:
            print(f"Loss plot SVG: {write_loss_svg(arguments.svg, [(arguments.lr, result.history)])}")

    elif arguments.command == "check-gradients":
        report = check_gradients(make_xor_model(arguments.seed), XOR_INPUTS, XOR_TARGETS,
                                 epsilon=arguments.epsilon, atol=arguments.atol, rtol=arguments.rtol)
        print(f"NumPyNet v{__version__} | central finite-difference gradient check")
        print(f"epsilon={report.epsilon:g}, atol={report.atol:g}, rtol={report.rtol:g}")
        for row in report.checks:
            status = "PASS" if row.passed else "FAIL"
            print(f"{status} | layer {row.layer} {row.parameter}: {row.count} params, "
                  f"{row.failed} mismatches, max absolute error={row.max_absolute_error:.3g}")
            if not row.passed:
                print(f"       worst {row.worst_index}: analytical={row.analytical:.9g}, "
                      f"numerical={row.numerical:.9g}")
        print(f"Checked {report.checked_parameters} parameters: "
              f"{'ALL PASS' if report.passed else 'MISMATCHES FOUND'}")
        if not report.passed:
            raise SystemExit(1)

    elif arguments.command == "diagnose-lr":
        if len(arguments.rates) > 8:
            parser.error("At most 8 rates per plot")
        if len(arguments.rates) * len(arguments.seeds) > 24:
            parser.error("Limit diagnostics to 24 combinations of rate and seed")
        runs = []
        print(f"NumPyNet v{__version__} | Lesson 03: learning-rate diagnostics")
        print("Rate | Seed | Final loss | Train accuracy | Loss increases | Max relative update")
        for rate in arguments.rates:
            for seed in arguments.seeds:
                run = diagnose_xor(epochs=arguments.epochs, learning_rate=rate, seed=seed)
                runs.append(run)
                output = arguments.out_dir / f"steps_lr_{rate:g}_seed_{seed}.csv"
                write_steps_csv(output, run)
                print(f"{rate:>4g} | {seed:>4d} | {run.last_loss:>10.6f} | "
                      f"{run.last_accuracy:>14.0%} | {run.increasing_steps:>14d} | "
                      f"{run.max_relative_update:>19.4f}")
        summary = write_diagnostic_summary(arguments.out_dir / "diagnostic_summary.csv", runs)
        for seed in arguments.seeds:
            series = [(run.learning_rate, run.history) for run in runs if run.seed == seed]
            plot = write_loss_svg(arguments.out_dir / f"loss_seed_{seed}.svg", series)
            print(f"Plot for seed {seed}: {plot}")
        print(f"Summary CSV: {summary}")
        print("Per-step logs are in the same directory; see docs/lesson-03.md.")
        print("Note: all accuracy values are from four training samples, not a test set.")

    elif arguments.command == "stabilize-xor":
        if len(set(arguments.modes)) != len(arguments.modes):
            parser.error("Each mode can only appear once")
        if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
            parser.error("Provide one or more distinct seeds")
        if len(arguments.modes) > 8 or len(arguments.modes) * len(arguments.seeds) > 24:
            parser.error("Maximum eight modes and 24 total experiments")
        try:
            configurations = [
                StabilityConfig(mode=mode, learning_rate=arguments.lr,
                                gradient_max_norm=arguments.max_norm,
                                decay_factor=arguments.decay,
                                min_learning_rate=arguments.min_lr,
                                backtrack_factor=arguments.backtrack_factor,
                                max_backtracks=arguments.max_backtracks)
                for mode in arguments.modes
            ]
        except ValueError as error:
            parser.error(str(error))
        if arguments.epochs < 1:
            parser.error("epochs must be positive")
        print(f"NumPyNet v{__version__} | Lesson 04: stability interventions")
        print("Mode | Seed | Final loss | Training accuracy | Loss increases | Clipped | Rejected | Extra trials")
        runs = []
        for config in configurations:
            for seed in arguments.seeds:
                run = run_stability(config=config, seed=seed, epochs=arguments.epochs)
                runs.append(run)
                write_stability_steps(arguments.out_dir / f"steps_{config.mode}_seed_{seed}.csv", run)
                print(f"{config.mode:>14} | {seed:>4} | {run.final_loss:>10.6f} | "
                      f"{run.final_accuracy:>17.0%} | {run.increasing_steps:>14} | "
                      f"{run.clipped_steps:>7} | {run.rejected_updates:>8} | "
                      f"{run.total_extra_trials:>12}")
        summary = write_stability_summary(arguments.out_dir / "stability_summary.csv", runs)
        for seed in arguments.seeds:
            histories = [(run.config.mode, run.history) for run in runs if run.seed == seed]
            plot = write_loss_svg(arguments.out_dir / f"loss_seed_{seed}.svg", histories)
            print(f"Loss plot (seed {seed}): {plot}")
        print(f"Summary CSV: {summary}")
        print("SVG legend names correspond to intervention modes.")
        print("All metrics use four XOR training examples; no test-set claims.")

    elif arguments.command == "compare-lr":
        if len(arguments.rates) > 8:
            parser.error("compare-lr accepts at most 8 learning rates for the plot")
        comparisons = []
        print(f"NumPyNet v{__version__} | same XOR data, same initialization seed={arguments.seed}")
        print("Learning rate | Initial loss | Final loss | Final training accuracy")
        for index, rate in enumerate(arguments.rates):
            result = train_xor(epochs=arguments.epochs, learning_rate=rate, seed=arguments.seed)
            comparisons.append((rate, result.history))
            csv_path = arguments.out_dir / f"run_{index:02d}_lr_{rate:g}.csv"
            write_history_csv(csv_path, result.history)
            print(f"{rate:>13g} | {result.losses[0]:>12.6f} | {result.losses[-1]:>10.6f} | {result.accuracy:>22.0%}")
        summary_path = write_comparison_csv(arguments.out_dir / "summary.csv", comparisons)
        plot_path = write_loss_svg(arguments.out_dir / "loss_curves.svg", comparisons)
        print(f"CSV summary: {summary_path}")
        print(f"Open in a browser: {plot_path}")
        print("Reminder: XOR has only 4 training points. This is not a generalization comparison.")


if __name__ == "__main__":
    main()
