"""Unit tests including a numerical check of a manually derived gradient."""

import unittest

import numpy as np

from numpynet.activations import Sigmoid, Tanh
from numpynet.layers import Dense


class LayerTests(unittest.TestCase):
    def test_dense_shapes(self) -> None:
        layer = Dense(3, 2, np.random.default_rng(1))
        self.assertEqual(layer.forward(np.ones((4, 3))).shape, (4, 2))
        self.assertEqual(layer.backward(np.ones((4, 2))).shape, (4, 3))
        self.assertEqual(layer.grad_weights.shape, (3, 2))
        self.assertEqual(layer.grad_bias.shape, (1, 2))

    def test_dense_rejects_wrong_input_shape(self) -> None:
        layer = Dense(3, 2, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            layer.forward(np.ones((4, 5)))

    def test_dense_requires_forward_before_backward(self) -> None:
        layer = Dense(3, 2, np.random.default_rng(1))
        with self.assertRaises(RuntimeError):
            layer.backward(np.ones((4, 2)))

    def test_dense_weight_gradient_matches_finite_difference(self) -> None:
        layer = Dense(2, 1, np.random.default_rng(3))
        inputs = np.array([[0.2, -0.4], [1.1, 0.9]])
        upstream = np.array([[0.7], [-0.2]])
        layer.forward(inputs)
        layer.backward(upstream)
        analytic = layer.grad_weights.copy()

        delta = 1e-6
        for index in np.ndindex(layer.weights.shape):
            original = layer.weights[index]
            layer.weights[index] = original + delta
            plus = float(np.sum(layer.forward(inputs) * upstream))
            layer.weights[index] = original - delta
            minus = float(np.sum(layer.forward(inputs) * upstream))
            layer.weights[index] = original
            numeric = (plus - minus) / (2.0 * delta)
            self.assertAlmostEqual(analytic[index], numeric, places=7)

    def test_sigmoid_extreme_inputs_remain_finite(self) -> None:
        values = Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
        self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_allclose(values, np.array([[0.0, 0.5, 1.0]]), atol=1e-12)

    def test_tanh_derivative(self) -> None:
        activation = Tanh()
        activation.forward(np.zeros((2, 3)))
        np.testing.assert_allclose(activation.backward(np.ones((2, 3))), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
