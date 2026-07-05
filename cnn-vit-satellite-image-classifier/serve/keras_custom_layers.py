"""Custom Keras layers used by the CNN-ViT hybrid model.

Importing this module registers AddPositionEmbedding and TransformerBlock
with the Keras serialisation registry, which is required before calling
tf.keras.models.load_model() on the CNN-ViT .keras checkpoint.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
    """Adds a learned positional embedding to a sequence of patch tokens."""

    def __init__(self, num_patches: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    """Single Transformer encoder block: pre-norm MHA + MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.mha = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        x = self.norm1(x + self.mha(x, x))
        return self.norm2(x + self.mlp(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return config
