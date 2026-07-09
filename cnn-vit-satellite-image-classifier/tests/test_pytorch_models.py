"""Tests for serve/pytorch_models.py."""

import torch

from serve.pytorch_models import CNN_ViT_Hybrid, build_satellite_cnn


def test_build_satellite_cnn_forward_shape_default_classes():
    torch.manual_seed(0)
    model = build_satellite_cnn().eval()
    x = torch.randn(2, 3, 64, 64)

    logits = model(x)

    assert logits.shape == (2, 2)


def test_build_satellite_cnn_respects_num_classes():
    model = build_satellite_cnn(num_classes=5).eval()
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 5)


def test_cnn_vit_hybrid_forward_shape():
    torch.manual_seed(0)
    model = CNN_ViT_Hybrid(num_classes=2, embed_dim=32, depth=1, heads=2).eval()
    x = torch.randn(2, 3, 64, 64)

    logits = model(x)

    assert logits.shape == (2, 2)


def test_cnn_vit_hybrid_backbone_weights_are_addressable_under_cnn_prefix():
    model = CNN_ViT_Hybrid(embed_dim=32, depth=1, heads=2)
    state_dict_keys = model.state_dict().keys()
    assert any(k.startswith("cnn.features.") for k in state_dict_keys)
