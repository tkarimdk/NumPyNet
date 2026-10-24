"""Lesson 01 should agree with the closed-form single-neuron math."""

import unittest

import numpy as np

from numpynet.lessons import trace_single_neuron


class LessonOneTests(unittest.TestCase):
    def test_known_forward_values(self) -> None:
        result = trace_single_neuron()
        self.assertAlmostEqual(result.logit, 1.35)
        p = 1.0 / (1.0 + np.exp(-1.35))
        self.assertAlmostEqual(result.prediction_before, p)
        self.assertAlmostEqual(result.loss_before, -np.log(p))

    def test_gradients_match_closed_form(self) -> None:
        result = trace_single_neuron()
        dz = result.prediction_before - result.target
        np.testing.assert_allclose(result.gradient_weights, result.inputs * dz)
        self.assertAlmostEqual(result.gradient_bias, dz)

    def test_parameter_update_matches_sgd(self) -> None:
        result = trace_single_neuron(learning_rate=0.1)
        np.testing.assert_allclose(
            result.weights_after,
            result.weights_before - 0.1 * result.gradient_weights,
        )
        self.assertAlmostEqual(
            result.bias_after, result.bias_before - 0.1 * result.gradient_bias
        )

    def test_one_step_reduces_loss_on_this_example(self) -> None:
        result = trace_single_neuron()
        self.assertLess(result.loss_after, result.loss_before)

    def test_invalid_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            trace_single_neuron(0.0)


if __name__ == "__main__":
    unittest.main()
