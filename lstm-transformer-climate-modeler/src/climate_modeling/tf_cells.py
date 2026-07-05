"""Pure TensorFlow primitive building blocks.

Every class here is implemented from first principles using ``tf.Module``
and ``tf.Variable`` — **no ``tf.keras`` API is used anywhere in this file**.

Classes
-------
LSTMCell            Single-step LSTM gate computation.
LayerNorm           Layer normalisation with learnable scale/bias.
MultiHeadAttention  Scaled dot-product multi-head self-attention.
FeedForward         Two-layer position-wise feed-forward network (GELU).
AdamOptimizer       Adam optimiser with bias-corrected moment estimates.
"""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# LSTM cell
# ---------------------------------------------------------------------------


class LSTMCell(tf.Module):
    """Single LSTM recurrent cell implemented with raw TensorFlow ops.

    All four gates (forget, input, cell, output) are computed in one
    ``tf.matmul`` call using a fused weight matrix ``W`` of shape
    ``[input_size + hidden_size, 4 * hidden_size]``.

    The forget-gate bias is initialised to 1 to encourage gradient flow
    at the start of training (Jozefowicz et al., 2015).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.hidden_size = hidden_size

        # Xavier-uniform initialisation for the fused gate weight
        fan_in = input_size + hidden_size
        fan_out = 4 * hidden_size
        limit = (6.0 / (fan_in + fan_out)) ** 0.5
        self.W = tf.Variable(
            tf.random.uniform([fan_in, fan_out], minval=-limit, maxval=limit),
            name="W",
        )
        # Forget gate bias = 1, all others = 0
        bias_init = tf.concat(
            [tf.ones([hidden_size]), tf.zeros([3 * hidden_size])],
            axis=0,
        )
        self.b = tf.Variable(bias_init, name="b")

    def __call__(
        self,
        x: tf.Tensor,
        state: tuple[tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """One time-step of the LSTM.

        Args:
            x:     Input tensor of shape ``(batch, input_size)``.
            state: ``(h_prev, c_prev)`` each ``(batch, hidden_size)``.

        Returns:
            h_new:          ``(batch, hidden_size)``
            (h_new, c_new): Updated state tuple.
        """
        h_prev, c_prev = state
        xh = tf.concat([x, h_prev], axis=-1)          # (batch, input+hidden)
        gates = tf.matmul(xh, self.W) + self.b        # (batch, 4*hidden)

        f_raw, i_raw, g_raw, o_raw = tf.split(gates, 4, axis=-1)

        f = tf.sigmoid(f_raw)   # forget gate
        i = tf.sigmoid(i_raw)   # input gate
        g = tf.tanh(g_raw)      # cell candidate
        o = tf.sigmoid(o_raw)   # output gate

        c_new = f * c_prev + i * g
        h_new = o * tf.tanh(c_new)
        return h_new, (h_new, c_new)


# ---------------------------------------------------------------------------
# Layer normalisation
# ---------------------------------------------------------------------------


class LayerNorm(tf.Module):
    """Layer normalisation over the last axis (Ba et al., 2016).

    Uses ``tf.nn.moments`` instead of ``tf.keras``.
    """

    def __init__(self, size: int, eps: float = 1e-5, name: str | None = None) -> None:
        super().__init__(name=name)
        self.eps = eps
        self.gamma = tf.Variable(tf.ones([size]), name="gamma")
        self.beta = tf.Variable(tf.zeros([size]), name="beta")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(tf.Module):
    """Scaled dot-product multi-head self-attention (Vaswani et al., 2017).

    Q, K, V and output projections are learnable ``tf.Variable`` matrices.
    No ``tf.keras`` layers are used.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        limit = (6.0 / (d_model + d_model)) ** 0.5
        self.W_q = tf.Variable(
            tf.random.uniform([d_model, d_model], -limit, limit), name="W_q"
        )
        self.W_k = tf.Variable(
            tf.random.uniform([d_model, d_model], -limit, limit), name="W_k"
        )
        self.W_v = tf.Variable(
            tf.random.uniform([d_model, d_model], -limit, limit), name="W_v"
        )
        self.W_o = tf.Variable(
            tf.random.uniform([d_model, d_model], -limit, limit), name="W_o"
        )

    def __call__(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Self-attention forward pass.

        Args:
            x:        ``(batch, seq_len, d_model)``
            training: Unused here; present for API consistency.

        Returns:
            output:   ``(batch, seq_len, d_model)``
            weights:  ``(batch, n_heads, seq_len, seq_len)``  – attention map
        """
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = tf.matmul(x, self.W_q)  # (batch, seq, d_model)
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)

        def split_heads(t: tf.Tensor) -> tf.Tensor:
            # (batch, seq, d_model) → (batch, n_heads, seq, d_head)
            t = tf.reshape(t, [batch, seq_len, self.n_heads, self.d_head])
            return tf.transpose(t, perm=[0, 2, 1, 3])

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        scores = tf.linalg.matmul(q, k, transpose_b=True) / scale  # (b, h, s, s)
        weights = tf.nn.softmax(scores, axis=-1)

        context = tf.linalg.matmul(weights, v)                     # (b, h, s, d_head)
        context = tf.transpose(context, perm=[0, 2, 1, 3])         # (b, s, h, d_head)
        context = tf.reshape(context, [batch, seq_len, self.d_model])

        output = tf.matmul(context, self.W_o)
        return output, weights


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------


class FeedForward(tf.Module):
    """Position-wise two-layer feed-forward network with GELU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        limit1 = (6.0 / (d_model + d_ff)) ** 0.5
        limit2 = (6.0 / (d_ff + d_model)) ** 0.5
        self.W1 = tf.Variable(
            tf.random.uniform([d_model, d_ff], -limit1, limit1), name="W1"
        )
        self.b1 = tf.Variable(tf.zeros([d_ff]), name="b1")
        self.W2 = tf.Variable(
            tf.random.uniform([d_ff, d_model], -limit2, limit2), name="W2"
        )
        self.b2 = tf.Variable(tf.zeros([d_model]), name="b2")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        h = tf.matmul(x, self.W1) + self.b1
        # GELU approximation: x * sigmoid(1.702 * x)
        h = h * tf.sigmoid(1.702 * h)
        return tf.matmul(h, self.W2) + self.b2


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------


class AdamOptimizer(tf.Module):
    """Adam optimiser (Kingma & Ba, 2015) built from pure TensorFlow ops.

    Moment variables are lazily initialised the first time a gradient is
    seen, so the optimiser does not need to be constructed with knowledge
    of the model's variables.

    .. note::
        Moment variables are stored in a Python ``dict`` keyed by
        ``id(variable)`` and are therefore **not** tracked by the
        ``tf.Module`` variable-collection machinery.  This is intentional:
        optimiser state is ephemeral and does not need to be restored.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        # Step counter is a tracked tf.Variable so its value persists
        self._step = tf.Variable(0, dtype=tf.int32, trainable=False, name="step")
        # Moment buffers – lazily created, keyed by variable id
        self._m: dict[int, tf.Variable] = {}
        self._v: dict[int, tf.Variable] = {}

    def apply_gradients(
        self, grads_and_vars: list[tuple[tf.Tensor | None, tf.Variable]]
    ) -> None:
        """Apply one Adam update step to *grads_and_vars*."""
        self._step.assign_add(1)
        t = tf.cast(self._step, tf.float32)

        # Bias-corrected learning rate
        lr_t = (
            self.lr
            * tf.sqrt(1.0 - tf.pow(self.beta2, t))
            / (1.0 - tf.pow(self.beta1, t))
        )

        for grad, var in grads_and_vars:
            if grad is None:
                continue
            vid = id(var)
            if vid not in self._m:
                self._m[vid] = tf.Variable(
                    tf.zeros_like(var), trainable=False, name=f"m_{var.name}"
                )
                self._v[vid] = tf.Variable(
                    tf.zeros_like(var), trainable=False, name=f"v_{var.name}"
                )

            m = self._m[vid]
            v = self._v[vid]
            m.assign(self.beta1 * m + (1.0 - self.beta1) * grad)
            v.assign(self.beta2 * v + (1.0 - self.beta2) * tf.square(grad))
            var.assign_sub(lr_t * m / (tf.sqrt(v) + self.eps))
