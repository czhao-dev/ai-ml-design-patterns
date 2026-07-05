"""Tests for tf_cells.py and tf_models.py.

All tests are skipped if TensorFlow is not installed, so the existing
test suite continues to run in environments without TF.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestLSTMCell(unittest.TestCase):
    def setUp(self) -> None:
        from climate_modeling.tf_cells import LSTMCell

        self.input_size = 8
        self.hidden_size = 16
        self.batch = 4
        self.cell = LSTMCell(self.input_size, self.hidden_size, name="test_cell")

    def test_output_shapes(self) -> None:
        x = tf.zeros([self.batch, self.input_size])
        h0 = tf.zeros([self.batch, self.hidden_size])
        c0 = tf.zeros([self.batch, self.hidden_size])

        h_new, (h_ret, c_new) = self.cell(x, (h0, c0))

        self.assertEqual(h_new.shape, (self.batch, self.hidden_size))
        self.assertEqual(c_new.shape, (self.batch, self.hidden_size))
        # Returned h from state tuple must match h_new
        np.testing.assert_array_equal(h_new.numpy(), h_ret.numpy())

    def test_state_changes_with_input(self) -> None:
        """Hidden state must differ for different inputs."""
        h0 = tf.zeros([self.batch, self.hidden_size])
        c0 = tf.zeros([self.batch, self.hidden_size])

        x_a = tf.ones([self.batch, self.input_size])
        x_b = tf.ones([self.batch, self.input_size]) * 2.0

        h_a, _ = self.cell(x_a, (h0, c0))
        h_b, _ = self.cell(x_b, (h0, c0))

        # Different inputs → different hidden states
        self.assertFalse(np.allclose(h_a.numpy(), h_b.numpy()))

    def test_variable_count(self) -> None:
        """LSTMCell should have exactly 2 trainable variables (W and b)."""
        self.assertEqual(len(self.cell.trainable_variables), 2)

    def test_forget_gate_bias_init(self) -> None:
        """First hidden_size entries of bias should equal 1.0 (forget gate)."""
        b = self.cell.b.numpy()
        np.testing.assert_allclose(b[: self.hidden_size], 1.0)
        np.testing.assert_allclose(b[self.hidden_size :], 0.0)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestLayerNorm(unittest.TestCase):
    def test_output_normalised(self) -> None:
        from climate_modeling.tf_cells import LayerNorm

        ln = LayerNorm(16)
        x = tf.constant(np.random.randn(8, 16).astype(np.float32))
        out = ln(x)
        # After layer-norm (with default gamma=1, beta=0) the output should
        # have near-zero mean and near-unit std along the last axis.
        mean = out.numpy().mean(axis=-1)
        std = out.numpy().std(axis=-1)
        np.testing.assert_allclose(mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(std, 1.0, atol=1e-4)

    def test_output_shape_preserved(self) -> None:
        from climate_modeling.tf_cells import LayerNorm

        ln = LayerNorm(32)
        x = tf.zeros([5, 10, 32])
        out = ln(x)
        self.assertEqual(out.shape, (5, 10, 32))


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self) -> None:
        from climate_modeling.tf_cells import MultiHeadAttention

        self.d_model = 16
        self.n_heads = 2
        self.batch = 3
        self.seq = 10
        self.attn = MultiHeadAttention(self.d_model, self.n_heads)

    def test_output_shapes(self) -> None:
        x = tf.random.normal([self.batch, self.seq, self.d_model])
        out, weights = self.attn(x)
        self.assertEqual(out.shape, (self.batch, self.seq, self.d_model))
        self.assertEqual(
            weights.shape,
            (self.batch, self.n_heads, self.seq, self.seq),
        )

    def test_attention_weights_sum_to_one(self) -> None:
        x = tf.random.normal([self.batch, self.seq, self.d_model])
        _, weights = self.attn(x)
        row_sums = tf.reduce_sum(weights, axis=-1).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_invalid_n_heads_raises(self) -> None:
        from climate_modeling.tf_cells import MultiHeadAttention

        with self.assertRaises(ValueError):
            MultiHeadAttention(d_model=15, n_heads=4)  # 15 not divisible by 4


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestFeedForward(unittest.TestCase):
    def test_output_shape(self) -> None:
        from climate_modeling.tf_cells import FeedForward

        ff = FeedForward(d_model=16, d_ff=32)
        x = tf.random.normal([4, 10, 16])
        out = ff(x)
        self.assertEqual(out.shape, (4, 10, 16))

    def test_variable_count(self) -> None:
        from climate_modeling.tf_cells import FeedForward

        ff = FeedForward(d_model=8, d_ff=16)
        # W1, b1, W2, b2 → 4 trainable variables
        self.assertEqual(len(ff.trainable_variables), 4)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestAdamOptimizer(unittest.TestCase):
    def test_param_moves_toward_target(self) -> None:
        """Gradient descent on a simple quadratic should decrease loss."""
        from climate_modeling.tf_cells import AdamOptimizer

        param = tf.Variable([[2.0, -3.0]], dtype=tf.float32)
        opt = AdamOptimizer(lr=0.1)
        initial_loss = float(tf.reduce_sum(tf.square(param)))

        for _ in range(50):
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(tf.square(param))
            (grad,) = tape.gradient(loss, [param])
            opt.apply_gradients([(grad, param)])

        final_loss = float(tf.reduce_sum(tf.square(param)))
        self.assertLess(final_loss, initial_loss)

    def test_step_counter_increments(self) -> None:
        from climate_modeling.tf_cells import AdamOptimizer

        param = tf.Variable([1.0])
        opt = AdamOptimizer(lr=0.01)
        for _ in range(3):
            with tf.GradientTape() as tape:
                loss = tf.square(param)
            (grad,) = tape.gradient(loss, [param])
            opt.apply_gradients([(grad, param)])

        self.assertEqual(int(opt._step), 3)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestLSTMForecaster(unittest.TestCase):
    def setUp(self) -> None:
        from climate_modeling.tf_models import LSTMForecaster

        self.batch = 4
        self.lookback = 10
        self.horizon = 3
        self.n_features = 9
        self.n_targets = 3
        self.model = LSTMForecaster(
            input_size=self.n_features,
            hidden_size=16,
            n_layers=2,
            horizon=self.horizon,
        )

    def test_output_shape(self) -> None:
        x = tf.random.normal([self.batch, self.lookback, self.n_features])
        out = self.model(x, training=False)
        self.assertEqual(out.shape, (self.batch, self.horizon, self.n_targets))

    def test_output_shape_training(self) -> None:
        x = tf.random.normal([self.batch, self.lookback, self.n_features])
        out = self.model(x, training=True)
        self.assertEqual(out.shape, (self.batch, self.horizon, self.n_targets))

    def test_deterministic_inference(self) -> None:
        """Two inference passes with the same input should produce identical output."""
        x = tf.constant(np.random.randn(self.batch, self.lookback, self.n_features).astype(np.float32))
        out1 = self.model(x, training=False).numpy()
        out2 = self.model(x, training=False).numpy()
        np.testing.assert_array_equal(out1, out2)

    def test_has_trainable_variables(self) -> None:
        self.assertGreater(len(self.model.trainable_variables), 0)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestTransformerForecaster(unittest.TestCase):
    def setUp(self) -> None:
        from climate_modeling.tf_models import TransformerForecaster

        self.batch = 4
        self.lookback = 10
        self.horizon = 3
        self.n_features = 9
        self.n_targets = 3
        self.model = TransformerForecaster(
            input_size=self.n_features,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            horizon=self.horizon,
        )

    def test_output_shape(self) -> None:
        x = tf.random.normal([self.batch, self.lookback, self.n_features])
        pred, attn_weights = self.model(x, training=False)
        self.assertEqual(pred.shape, (self.batch, self.horizon, self.n_targets))

    def test_attn_weights_count(self) -> None:
        """Should return one attention tensor per encoder block."""
        x = tf.random.normal([self.batch, self.lookback, self.n_features])
        _, attn_weights = self.model(x, training=False)
        self.assertEqual(len(attn_weights), 2)  # n_layers=2

    def test_attn_weights_shape(self) -> None:
        x = tf.random.normal([self.batch, self.lookback, self.n_features])
        _, attn_weights = self.model(x, training=False)
        for w in attn_weights:
            self.assertEqual(
                w.shape,
                (self.batch, 2, self.lookback, self.lookback),  # n_heads=2
            )

    def test_deterministic_inference(self) -> None:
        x = tf.constant(np.random.randn(self.batch, self.lookback, self.n_features).astype(np.float32))
        pred1, _ = self.model(x, training=False)
        pred2, _ = self.model(x, training=False)
        np.testing.assert_array_equal(pred1.numpy(), pred2.numpy())

    def test_has_trainable_variables(self) -> None:
        self.assertGreater(len(self.model.trainable_variables), 0)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed")
class TestWeightSerialization(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        import tempfile
        from pathlib import Path

        from climate_modeling.tf_models import LSTMForecaster
        from climate_modeling.tf_trainer import load_weights, save_weights

        model = LSTMForecaster(input_size=9, hidden_size=8, n_layers=1, horizon=3)
        x = tf.random.normal([2, 10, 9])
        pred_before = model(x, training=False).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.json"
            save_weights(model, path)

            # Corrupt the weights
            for v in model.trainable_variables:
                v.assign(tf.zeros_like(v))

            load_weights(model, path)

        pred_after = model(x, training=False).numpy()
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
