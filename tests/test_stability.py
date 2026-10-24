"""Stability features should be correct, reproducible and not change the old engine."""

import csv
import io
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from numpynet.diagnostics import diagnose_xor
from numpynet.experiments import XOR_INPUTS, XOR_TARGETS, make_xor_model
from numpynet.losses import BinaryCrossEntropy
from numpynet.stability import (StabilityConfig, clip_global_norm,
                               run_stability, write_stability_steps, write_stability_summary)


class StabilityTests(unittest.TestCase):
    def test_clipper_preserves_direction_and_caps_global_norm(self):
        model = make_xor_model(7)
        criterion = BinaryCrossEntropy()
        criterion.forward(model.forward(XOR_INPUTS), XOR_TARGETS)
        model.backward(criterion.backward())
        gradients = [array.copy() for layer in (model.layers[0], model.layers[2])
                     for array in (layer.grad_weights, layer.grad_bias)]
        raw_norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        target = raw_norm / 5
        raw, applied, factor = clip_global_norm(model, target)
        self.assertAlmostEqual(raw, raw_norm)
        self.assertAlmostEqual(applied, target)
        self.assertAlmostEqual(factor, 0.2)
        actual = [array for layer in (model.layers[0], model.layers[2])
                  for array in (layer.grad_weights, layer.grad_bias)]
        for old, new in zip(gradients, actual):
            np.testing.assert_allclose(new, old * factor)
        _, again, unchanged = clip_global_norm(model, target)
        self.assertAlmostEqual(again, target)
        self.assertAlmostEqual(unchanged, 1)

    def test_zero_gradients_are_safe(self):
        model = make_xor_model(7)
        self.assertEqual(clip_global_norm(model, .01), (0, 0, 1))

    def test_legacy_baseline_matches_v03(self):
        legacy = diagnose_xor(epochs=22, learning_rate=10, seed=7)
        current = run_stability(config=StabilityConfig(), epochs=22, seed=7)
        np.testing.assert_allclose([r.loss for r in legacy.history],
                                   [r.loss for r in current.history], rtol=0, atol=1e-13)
        self.assertEqual(legacy.increasing_steps, sum(r.increased for r in current.steps))

    def test_decaying_schedule_is_bounded_and_decreasing(self):
        cfg = StabilityConfig(mode="decay", learning_rate=10, decay_factor=.5, min_learning_rate=.5)
        self.assertEqual([cfg.scheduled_lr(i) for i in range(1, 7)], [10, 5, 2.5, 1.25, .625, .5])
        run = run_stability(config=cfg, epochs=5)
        self.assertEqual([s.scheduled_lr for s in run.steps], [10, 5, 2.5, 1.25, .625])

    def test_backtracking_avoids_loss_increases_for_full_batch(self):
        run = run_stability(config=StabilityConfig(mode="backtrack"), epochs=50)
        self.assertTrue(all(step.post_loss <= step.pre_loss + 1e-12 for step in run.steps))
        self.assertGreater(run.total_extra_trials, 0)
        self.assertTrue(all(1 <= step.trial_count <= 9 for step in run.steps))

    def test_rejected_steps_restore_parameters_and_record_zero_step(self):
        cfg = StabilityConfig(mode="backtrack", max_backtracks=0, learning_rate=10)
        run = run_stability(config=cfg, seed=7, epochs=10)
        self.assertGreater(run.rejected_updates, 0)
        for step in run.steps:
            if not step.accepted:
                self.assertAlmostEqual(step.pre_loss, step.post_loss)
                self.assertEqual(step.update_l2, 0)
                self.assertEqual(step.effective_lr, 0)
                self.assertEqual(step.trial_count, 1)

    def test_clip_and_backtrack_can_coexist(self):
        run = run_stability(config=StabilityConfig(mode="clip-backtrack"), epochs=60)
        self.assertTrue(all(s.post_loss <= s.pre_loss + 1e-12 for s in run.steps))
        self.assertTrue(all(s.applied_gradient_l2 <= .3 + 1e-12 for s in run.steps))

    def test_runs_are_reproducible(self):
        cfg = StabilityConfig(mode="clip", learning_rate=10)
        first = run_stability(config=cfg, seed=8, epochs=50)
        second = run_stability(config=cfg, seed=8, epochs=50)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.steps, second.steps)

    def test_invalid_configurations_and_bad_epochs(self):
        for kwargs in ({'mode': 'unknown'}, {'learning_rate': 0},
                       {'gradient_max_norm': float('nan')}, {'gradient_max_norm': 0},
                       {'decay_factor': 1.2}, {'backtrack_factor': 1},
                       {'min_learning_rate': 11}, {'max_backtracks': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StabilityConfig(**kwargs)
        with self.assertRaises(ValueError):
            run_stability(config=StabilityConfig(), epochs=0)
        with self.assertRaises(ValueError):
            clip_global_norm(make_xor_model(), 0)

    def test_csv_exports_include_tracking_columns(self):
        run = run_stability(config=StabilityConfig(mode="clip"), epochs=7)
        with tempfile.TemporaryDirectory() as d:
            steps_path = write_stability_steps(Path(d) / 'a' / 'steps.csv', run)
            summary_path = write_stability_summary(Path(d) / 'summary.csv', [run])
            with steps_path.open(newline='', encoding='utf-8') as f:
                steps = list(csv.DictReader(f))
            with summary_path.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(steps), 7)
            self.assertIn('effective_lr', steps[0])
            self.assertIn('clip_factor', steps[0])
            self.assertEqual(rows[0]['mode'], 'clip')
            with self.assertRaises(ValueError):
                write_stability_summary(Path(d) / 'fail.csv', [])

    def test_cli_produces_labeled_svg_and_summary(self):
        from numpynet.cli import main
        with tempfile.TemporaryDirectory() as d:
            argv = ['numpynet', 'stabilize-xor', '--modes', 'baseline', 'clip',
                    '--seeds', '7', '--epochs', '7', '--out-dir', d]
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((Path(d) / 'stability_summary.csv').exists())
            plot = Path(d) / 'loss_seed_7.svg'
            self.assertTrue(plot.exists())
            ET.parse(plot)
            content = plot.read_text(encoding='utf8')
            self.assertIn('>baseline</text>', content)
            self.assertIn('>clip</text>', content)


if __name__ == '__main__':
    unittest.main()
