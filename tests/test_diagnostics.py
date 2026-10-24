import csv
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import math
import tempfile
import unittest
from pathlib import Path

from numpynet.diagnostics import diagnose_xor, write_steps_csv, write_diagnostic_summary
from numpynet.experiments import train_xor


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_match_original_training_loss_at_every_epoch(self):
        baseline = train_xor(epochs=20, learning_rate=0.7, seed=11)
        traced = diagnose_xor(epochs=20, learning_rate=0.7, seed=11)
        self.assertEqual([r.loss for r in traced.history], baseline.losses)
        self.assertEqual(traced.last_accuracy, baseline.accuracy)

    def test_each_step_has_pre_and_post_loss(self):
        report = diagnose_xor(epochs=10)
        self.assertEqual(len(report.steps), 10)
        self.assertAlmostEqual(report.steps[0].pre_loss, report.first_loss)
        self.assertAlmostEqual(report.steps[-1].post_loss, report.last_loss)
        for prev, next_ in zip(report.steps, report.steps[1:]):
            self.assertEqual(prev.post_loss, next_.pre_loss)

    def test_gradient_norm_matches_sgd_update_relation(self):
        run = diagnose_xor(epochs=5, learning_rate=2.0)
        for step in run.steps:
            self.assertAlmostEqual(step.update_l2, 2.0 * step.gradient_l2)
            self.assertAlmostEqual(step.relative_update,
                                   step.update_l2 / step.parameter_l2)
            self.assertTrue(all(math.isfinite(value) for value in
                                (step.gradient_l2, step.parameter_l2,
                                 step.update_l2, step.relative_update)))

    def test_hidden_saturation_bounded(self):
        report = diagnose_xor(epochs=20)
        self.assertTrue(all(0 <= s.hidden_saturation <= 1 for s in report.steps))

    def test_repeated_run_deterministic(self):
        first = diagnose_xor(epochs=20, seed=9)
        second = diagnose_xor(epochs=20, seed=9)
        self.assertEqual(first.steps, second.steps)

    def test_bad_arguments_rejected(self):
        with self.assertRaises(ValueError):
            diagnose_xor(epochs=0)
        with self.assertRaises(ValueError):
            diagnose_xor(epochs=-1)
        with self.assertRaises(ValueError):
            diagnose_xor(epochs=5, learning_rate=11)
        with self.assertRaises(ValueError):
            diagnose_xor(epochs=5, seed=1.5)

    def test_csv_has_expected_rows_and_columns(self):
        run = diagnose_xor(epochs=3)
        with tempfile.TemporaryDirectory() as temp:
            step_path = write_steps_csv(Path(temp) / 'nested' / 'steps.csv', run)
            summary_path = write_diagnostic_summary(Path(temp) / 'summary.csv', [run])
            with step_path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            with summary_path.open(newline='', encoding='utf-8') as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(summary), 1)
            self.assertIn('relative_update', rows[0])
            self.assertEqual(int(summary[0]['epochs']), 3)

    def test_cli_writes_seed_plot_and_summary(self):
        from numpynet.cli import main
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / 'logs'
            argv = ['numpynet', 'diagnose-lr', '--rates', '2', '10', '--seeds',
                    '7', '--epochs', '5', '--out-dir', str(output_dir)]
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((output_dir / 'loss_seed_7.svg').exists())
            self.assertTrue((output_dir / 'diagnostic_summary.csv').exists())
            self.assertTrue((output_dir / 'steps_lr_10_seed_7.csv').exists())

    def test_empty_summary_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                write_diagnostic_summary(Path(temp) / 'empty.csv', [])

    def test_loss_increase_definition(self):
        # 1e-12 tolerance avoids treating tiny floating point changes as spikes.
        from numpynet.diagnostics import StepRecord
        step = StepRecord(1, .5, .6, .1, 1, .1, .1, 0, .5)
        self.assertTrue(step.loss_increased)
        step = StepRecord(1, .5, .5 + 1e-14, .1, 1, .1, .1, 0, .5)
        self.assertFalse(step.loss_increased)

if __name__ == '__main__':
    unittest.main()
