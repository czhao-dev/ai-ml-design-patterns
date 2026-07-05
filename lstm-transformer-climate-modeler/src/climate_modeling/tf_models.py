"""Pure-TensorFlow multi-step weather forecasting models.

Two architectures are provided, both predicting the next ``horizon`` days
of [PRCP, SNOW, TOBS] from a ``lookback``-day input window.

Classes
-------
LSTMForecaster        Stacked LSTM with a dense output head.
TransformerBlock      Encoder block: pre-norm attention + feed-forward.
TransformerForecaster Transformer encoder with mean pooling and output head.

Neither class imports anything from ``tf.keras``.  All weights are plain
``tf.Variable`` objects and the forward pass uses only ``tf.Module``,
``tf.linalg``, ``tf.nn``, and core TF ops.
"""

from __future__ import annotations

import tensorflow as tf

from climate_modeling.sequence_dataset import N_FEATURES, N_TARGETS
from climate_modeling.tf_cells import (
    AdamOptimizer,
    FeedForward,
    LayerNorm,
    LSTMCell,
    MultiHeadAttention,
)

__all__ = ["LSTMForecaster", "TransformerForecaster", "AdamOptimizer"]


# ---------------------------------------------------------------------------
# LSTM forecaster
# ---------------------------------------------------------------------------


class LSTMForecaster(tf.Module):
    """Stacked LSTM multi-step, multi-output weather forecaster.

    Architecture:
        input (batch, lookback, N_FEATURES)
            → 2 × LSTMCell (unrolled over ``lookback`` time-steps)
            → final hidden state (batch, hidden_size)
            → dropout (training only)
            → dense  (batch, horizon × N_TARGETS)
            → reshape (batch, horizon, N_TARGETS)

    Args:
        input_size:    Number of input features per time-step.
        hidden_size:   LSTM hidden dimension (default 64).
        n_layers:      Number of stacked LSTM cells (default 2).
        horizon:       Forecast horizon in days (default 7).
        dropout_rate:  Dropout probability applied between LSTM layers and
                       before the output head during training (default 0.3).
    """

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 64,
        n_layers: int = 2,
        horizon: int = 7,
        dropout_rate: float = 0.3,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.horizon = horizon
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Stacked LSTM cells – store as individual named attributes so
        # tf.Module's attribute tracker finds them in all TF versions.
        self.cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                name=f"lstm_cell_{i}",
            )
            for i in range(n_layers)
        ]

        # Output projection: hidden_size → horizon × N_TARGETS
        out_size = horizon * N_TARGETS
        limit = (6.0 / (hidden_size + out_size)) ** 0.5
        self.W_out = tf.Variable(
            tf.random.uniform([hidden_size, out_size], -limit, limit),
            name="W_out",
        )
        self.b_out = tf.Variable(tf.zeros([out_size]), name="b_out")

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass.

        Args:
            x:        ``(batch, lookback, N_FEATURES)``
            training: Enables dropout when ``True``.

        Returns:
            predictions: ``(batch, horizon, N_TARGETS)``
        """
        batch_size = tf.shape(x)[0]

        # Initialise hidden and cell states to zero
        states: list[tuple[tf.Tensor, tf.Tensor]] = [
            (
                tf.zeros([batch_size, self.hidden_size]),
                tf.zeros([batch_size, self.hidden_size]),
            )
            for _ in range(self.n_layers)
        ]

        # Unroll over the sequence using tf.unstack (works in eager + graph mode)
        for inp in tf.unstack(x, axis=1):                 # inp: (batch, N_FEATURES)
            for layer_idx, cell in enumerate(self.cells):
                h_new, new_state = cell(inp, states[layer_idx])
                if training and layer_idx < self.n_layers - 1:
                    h_new = tf.nn.dropout(h_new, rate=self.dropout_rate)
                inp = h_new
                states[layer_idx] = new_state

        # Take the final hidden state of the last layer
        h_final = states[-1][0]                           # (batch, hidden_size)

        if training:
            h_final = tf.nn.dropout(h_final, rate=self.dropout_rate)

        out = tf.matmul(h_final, self.W_out) + self.b_out  # (batch, horizon*N_TARGETS)
        return tf.reshape(out, [-1, self.horizon, N_TARGETS])


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------


class TransformerBlock(tf.Module):
    """Pre-norm Transformer encoder block.

    Layout (pre-norm style):
        x → LayerNorm → MultiHeadAttention → residual
          → LayerNorm → FeedForward        → residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.attn = MultiHeadAttention(d_model, n_heads, name="attn")
        self.ff = FeedForward(d_model, d_ff, name="ff")
        self.norm1 = LayerNorm(d_model, name="norm1")
        self.norm2 = LayerNorm(d_model, name="norm2")

    def __call__(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one encoder block.

        Returns:
            x:            Updated hidden states ``(batch, seq, d_model)``.
            attn_weights: Attention maps ``(batch, n_heads, seq, seq)``.
        """
        attn_out, attn_weights = self.attn(self.norm1(x), training=training)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Transformer forecaster
# ---------------------------------------------------------------------------


class TransformerForecaster(tf.Module):
    """Transformer-encoder multi-step, multi-output weather forecaster.

    Architecture:
        input (batch, lookback, N_FEATURES)
            → linear projection  (batch, lookback, d_model)
            → + sinusoidal positional encoding
            → dropout (training only)
            → N × TransformerBlock
            → mean pool over sequence   (batch, d_model)
            → dropout (training only)
            → dense                     (batch, horizon × N_TARGETS)
            → reshape                   (batch, horizon, N_TARGETS)

    Args:
        input_size:   Number of input features per time-step.
        d_model:      Transformer hidden dimension (default 32).
        n_heads:      Number of attention heads (default 2).
        n_layers:     Number of encoder blocks (default 3).
        d_ff:         Feed-forward inner dimension (default 64).
        horizon:      Forecast horizon in days (default 7).
        dropout_rate: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_size: int = N_FEATURES,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 3,
        d_ff: int = 64,
        horizon: int = 7,
        dropout_rate: float = 0.3,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.d_model = d_model
        self.horizon = horizon
        self.dropout_rate = dropout_rate

        # Input projection: N_FEATURES → d_model
        limit_in = (6.0 / (input_size + d_model)) ** 0.5
        self.W_in = tf.Variable(
            tf.random.uniform([input_size, d_model], -limit_in, limit_in),
            name="W_in",
        )
        self.b_in = tf.Variable(tf.zeros([d_model]), name="b_in")

        # Encoder blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, name=f"block_{i}")
            for i in range(n_layers)
        ]

        # Output head: d_model → horizon × N_TARGETS
        out_size = horizon * N_TARGETS
        limit_out = (6.0 / (d_model + out_size)) ** 0.5
        self.W_out = tf.Variable(
            tf.random.uniform([d_model, out_size], -limit_out, limit_out),
            name="W_out",
        )
        self.b_out = tf.Variable(tf.zeros([out_size]), name="b_out")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Forward pass.

        Args:
            x:        ``(batch, lookback, N_FEATURES)``
            training: Enables dropout when ``True``.

        Returns:
            predictions:   ``(batch, horizon, N_TARGETS)``
            attn_weights:  list of attention maps from each encoder block,
                           each ``(batch, n_heads, seq, seq)``
        """
        # Project input features into model dimension
        h = tf.matmul(x, self.W_in) + self.b_in    # (batch, seq, d_model)

        # Add sinusoidal positional encoding (no learnable parameters)
        seq_len = tf.shape(x)[1]
        pe = self._positional_encoding(seq_len)     # (seq, d_model)
        h = h + pe[tf.newaxis, :, :]               # broadcast over batch

        if training:
            h = tf.nn.dropout(h, rate=self.dropout_rate)

        # Encoder blocks
        attn_weights_list: list[tf.Tensor] = []
        for block in self.blocks:
            h, attn_weights = block(h, training=training)
            attn_weights_list.append(attn_weights)

        # Mean pooling: collapse sequence dimension → (batch, d_model)
        h_pooled = tf.reduce_mean(h, axis=1)

        if training:
            h_pooled = tf.nn.dropout(h_pooled, rate=self.dropout_rate)

        out = tf.matmul(h_pooled, self.W_out) + self.b_out   # (batch, horizon*N_TARGETS)
        predictions = tf.reshape(out, [-1, self.horizon, N_TARGETS])
        return predictions, attn_weights_list

    # ------------------------------------------------------------------
    # Positional encoding (fixed sinusoidal, no learnable params)
    # ------------------------------------------------------------------

    def _positional_encoding(self, seq_len: tf.Tensor) -> tf.Tensor:
        """Return sinusoidal positional encoding of shape ``(seq_len, d_model)``.

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
        """
        d = self.d_model
        half_d = d // 2

        positions = tf.cast(tf.range(seq_len), tf.float32)        # (seq,)
        i = tf.cast(tf.range(half_d), tf.float32)                 # (half_d,)
        div = tf.pow(10000.0, 2.0 * i / tf.cast(d, tf.float32))  # (half_d,)

        # (seq, half_d)
        angles = positions[:, tf.newaxis] / div[tf.newaxis, :]

        pe = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  # (seq, 2*half_d)

        # Trim to exactly d_model (handles odd d_model safely)
        return pe[:, :d]
