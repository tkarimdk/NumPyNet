"""A good numerical checker must also detect intentionally bad derivatives."""

import unittest
import os
import subprocess
import sys

import numpy as np

from numpynet.activations import Sigmoid
from numpynet.experiments import XOR_INPUTS, XOR_TARGETS, make_xor_model
from numpynet.gradcheck import check_gradients
from numpynet.layers import Dense
from numpynet.model import Sequential


class IncorrectDense(Dense):
    def backward(self, upstream: np.ndarray) -> np.ndarray:
        output = super().backward(upstream)
        self.grad_weights[0, 0] += 0.25  # simulate a genuine implementation error
        return output


class GradientCheckTests(unittest.TestCase):
    def test_cli_works_with_windows_console_encoding(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "numpynet", "check-gradients"],
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},
            capture_output=True, encoding="cp1252", check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Checked 17 parameters: ALL PASS", result.stdout)

    def test_entire_multilayer_network_passes(self) -> None:
        model = make_xor_model(seed=3)
        report = check_gradients(model, XOR_INPUTS, XOR_TARGETS)
        self.assertTrue(report.passed)
        self.assertEqual(report.checked_parameters, 17)  # (2x4)+4+(4x1)+1
        self.assertEqual(len(report.checks), 4)
        for check in report.checks:
            self.assertLess(check.max_absolute_error, 1e-6)

    def test_check_never_changes_parameters(self) -> None:
        model = make_xor_model(seed=7)
        layers = [layer for layer in model.layers if isinstance(layer, Dense)]
        originals = [(layer.weights.copy(), layer.bias.copy()) for layer in layers]
        check_gradients(model, XOR_INPUTS, XOR_TARGETS)
        for layer, (weights, bias) in zip(layers, originals):
            np.testing.assert_array_equal(layer.weights, weights)
            np.testing.assert_array_equal(layer.bias, bias)

    def test_deliberately_wrong_gradient_is_flagged(self) -> None:
        model = Sequential([IncorrectDense(2, 1, np.random.default_rng(2)), Sigmoid()])
        report = check_gradients(model, np.array([[0.2, 0.6]]), np.array([[1.0]]))
        self.assertFalse(report.passed)
        self.assertGreater(report.checks[0].failed, 0)

    def test_reject_invalid_epsilon(self) -> None:
        with self.assertRaises(ValueError):
            check_gradients(make_xor_model(), XOR_INPUTS, XOR_TARGETS, epsilon=0)

    def test_reject_model_without_trainable_params(self) -> None:
        with self.assertRaises(ValueError):
            check_gradients(Sequential([Sigmoid()]), XOR_INPUTS, XOR_TARGETS)


if __name__ == "__main__":
    unittest.main()
