"""Numerical correctness, reproducibility, and independent holdout checks."""

import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from numpynet.activations import ReLU, Softmax
from numpynet.layers import Dense
from numpynet.losses import SoftmaxCrossEntropy
from numpynet.model import Sequential
from numpynet.multiclass import (
    check_multiclass_gradients,
    make_multiclass_model,
    spiral_dataset,
    train_multiclass,
)
from numpynet.multiclass_reports import write_multiclass_csv, write_multiclass_svg


class SoftmaxTests(unittest.TestCase):
    def test_stable_on_large_logits_and_normalized(self):
        softmax = Softmax()
        x = np.array([[10000.0, 10001.0, 9999.0], [-20000, -20000, -20000]])
        pred = softmax.forward(x)
        self.assertTrue(np.all(np.isfinite(pred)))
        np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(int(np.argmax(pred[0])), 1)

    def test_backward_matches_numerical_jvp(self):
        layer = Softmax()
        x = np.array([[0.4, -0.2, 1.2], [0.5, 0.6, -0.7]])
        upstream = np.array([[0.2, -0.8, 0.1], [-0.4, 0.3, 0.7]])
        layer.forward(x)
        analytic = layer.backward(upstream)
        numeric = np.empty_like(x)
        for idx in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[idx] += 1e-6
            minus[idx] -= 1e-6
            numeric[idx] = (np.sum(Softmax().forward(plus) * upstream)
                            - np.sum(Softmax().forward(minus) * upstream)) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_reject_invalid_shapes_and_backward_without_forward(self):
        with self.assertRaises(RuntimeError):
            Softmax().backward(np.ones((1, 2)))
        with self.assertRaises(ValueError):
            Softmax().forward(np.array([1., 2.]))
        layer = Softmax()
        layer.forward(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            layer.backward(np.ones((1, 3)))

    def test_relu_and_zero_derivative(self):
        layer = ReLU()
        np.testing.assert_array_equal(layer.forward(np.array([[-1., 0., 2.]])), [[0., 0., 2.]])
        np.testing.assert_array_equal(layer.backward(np.ones((1, 3))), [[0., 0., 1.]])


class MulticlassLossTests(unittest.TestCase):
    def test_extreme_logits_and_correct_targets(self):
        loss = SoftmaxCrossEntropy()
        result = loss.forward(np.array([[2000., -2000.], [-2000., 2000.]]), np.array([0, 1]))
        self.assertAlmostEqual(result, 0.0)
        self.assertTrue(np.all(np.isfinite(loss.backward())))

    def test_stable_for_wrong_confident_prediction(self):
        result = SoftmaxCrossEntropy().forward(np.array([[1000., -1000.]]), np.array([1]))
        self.assertAlmostEqual(result, 2000.0)

    def test_gradient_matches_finite_difference(self):
        logits = np.array([[0.3, -0.7, 1.0], [0.0, 0.4, -0.3]])
        targets = np.array([2, 0])
        c = SoftmaxCrossEntropy()
        c.forward(logits, targets)
        analytic = c.backward()
        numeric = np.zeros_like(logits)
        for idx in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[idx] += 1e-6
            minus[idx] -= 1e-6
            numeric[idx] = (SoftmaxCrossEntropy().forward(plus, targets)
                            - SoftmaxCrossEntropy().forward(minus, targets)) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_reject_targets_out_of_range_and_wrong_type(self):
        criterion = SoftmaxCrossEntropy()
        with self.assertRaises(RuntimeError):
            criterion.backward()
        for targets in [np.array([-1, 2]), np.array([0, 3]), np.array([0., 1.]),
                        np.array([[0], [1]])]:
            with self.assertRaises(ValueError):
                criterion.forward(np.zeros((2, 3)), targets)


class MulticlassExperimentsTests(unittest.TestCase):
    def test_spiral_is_deterministic_and_balanced(self):
        x1, y1 = spiral_dataset(samples_per_class=10, noise=0.18, seed=10)
        x2, y2 = spiral_dataset(samples_per_class=10, noise=0.18, seed=10)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(np.bincount(y1), [10, 10, 10])
        x3, _ = spiral_dataset(samples_per_class=10, noise=0.18, seed=11)
        self.assertFalse(np.array_equal(x1, x3))

    def test_multilayer_gradients_match(self):
        inputs, targets = spiral_dataset(samples_per_class=3, noise=0.15, seed=5)
        model = make_multiclass_model(seed=2, hidden_width=4)
        layers = [layer for layer in model.layers if isinstance(layer, Dense)]
        original = [(layer.weights.copy(), layer.bias.copy()) for layer in layers]
        report = check_multiclass_gradients(model, inputs, targets)
        self.assertTrue(report.passed)
        self.assertEqual(report.checked_parameters, 27)
        for layer, (weights, bias) in zip(layers, original):
            np.testing.assert_array_equal(layer.weights, weights)
            np.testing.assert_array_equal(layer.bias, bias)

    def test_incorrect_derivative_is_detected(self):
        class IncorrectDense(Dense):
            def backward(self, gradient):
                out = super().backward(gradient)
                self.grad_weights[0, 0] += 0.5
                return out
        model = Sequential([IncorrectDense(2, 3, np.random.default_rng(2))])
        x, y = spiral_dataset(samples_per_class=2, noise=0.1, seed=8)
        report = check_multiclass_gradients(model, x, y)
        self.assertFalse(report.passed)

    def test_three_class_holdout_learning(self):
        run = train_multiclass(epochs=1200, seed=7)
        self.assertEqual(len(run.history), 1201)
        self.assertGreater(run.history[-1].validation_accuracy, 0.9)
        self.assertLess(run.history[-1].validation_loss, run.history[0].validation_loss)
        np.testing.assert_allclose(run.validation_predictions.sum(axis=1), 1, atol=1e-12)
        self.assertEqual(run.validation_targets.shape, (180,))

    def test_training_is_reproducible(self):
        a = train_multiclass(epochs=20, seed=10)
        b = train_multiclass(epochs=20, seed=10)
        self.assertEqual(a.history, b.history)
        np.testing.assert_array_equal(a.validation_predictions, b.validation_predictions)

    def test_reports_parse(self):
        history = train_multiclass(epochs=10).history
        with TemporaryDirectory() as directory:
            csvpath = write_multiclass_csv(Path(directory) / 'history.csv', history)
            svgpath = write_multiclass_svg(Path(directory) / 'loss.svg', history)
            with csvpath.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 11)
            self.assertEqual(rows[0]['epoch'], '0')
            tree = ET.parse(svgpath)
            self.assertEqual(sum(n.tag.endswith('polyline') for n in tree.iter()), 2)

    def test_cli_runs_and_exports(self):
        with TemporaryDirectory() as directory:
            cmd = [sys.executable, '-m', 'numpynet', 'train-multiclass', '--epochs', '6',
                   '--out-dir', directory]
            r = subprocess.run(cmd, text=True, capture_output=True, check=False)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(Path(directory, 'history.csv').exists())
            self.assertTrue(Path(directory, 'loss_curves.svg').exists())
        r = subprocess.run([sys.executable, '-m', 'numpynet', 'check-multiclass-gradients'],
                           text=True, capture_output=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('Checked 27 parameters: ALL PASS', r.stdout)

    def test_reject_invalid_arguments(self):
        with self.assertRaises(ValueError):
            train_multiclass(epochs=0)
        with self.assertRaises(ValueError):
            spiral_dataset(samples_per_class=1, noise=0.2, seed=7)
        with self.assertRaises(ValueError):
            check_multiclass_gradients(make_multiclass_model(), np.zeros((2, 2)), np.array([0, 1]), epsilon=0)


if __name__ == '__main__':
    unittest.main()
