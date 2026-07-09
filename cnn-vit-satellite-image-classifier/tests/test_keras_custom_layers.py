"""Tests for serve/keras_custom_layers.py."""

import numpy as np
import tensorflow as tf

from serve.keras_custom_layers import AddPositionEmbedding, TransformerBlock


def test_add_position_embedding_output_shape_and_effect():
    layer = AddPositionEmbedding(num_patches=5, embed_dim=8)
    tokens = tf.zeros((2, 5, 8))

    output = layer(tokens)

    assert output.shape == (2, 5, 8)
    # Adding a (generally nonzero) learned embedding to zero tokens should
    # not just return the input unchanged.
    assert not np.allclose(output.numpy(), 0.0)


def test_add_position_embedding_config_round_trip():
    layer = AddPositionEmbedding(num_patches=5, embed_dim=8, name="pos_embed")
    restored = AddPositionEmbedding.from_config(layer.get_config())

    assert restored.num_patches == 5
    assert restored.embed_dim == 8


def test_transformer_block_preserves_sequence_shape():
    layer = TransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    tokens = tf.random.normal((2, 5, 8))

    output = layer(tokens)

    assert output.shape == (2, 5, 8)


def test_transformer_block_config_round_trip():
    layer = TransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16, dropout=0.2)
    restored = TransformerBlock.from_config(layer.get_config())

    assert restored.embed_dim == 8
    assert restored.num_heads == 2
    assert restored.mlp_dim == 16
    assert restored.dropout == 0.2
