import torch

from tinyllm.attention import CausalSelfAttention


def test_output_shape():
    attn = CausalSelfAttention(hidden_size=32, n_heads=4, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_causal_mask():
    attn = CausalSelfAttention(hidden_size=32, n_heads=4, dropout=0.0)
    attn.eval()
    x = torch.randn(1, 10, 32)
    out1 = attn(x)

    x_perturbed = x.clone()
    x_perturbed[:, 5:, :] += torch.randn_like(x_perturbed[:, 5:, :])
    out2 = attn(x_perturbed)

    # Positions before the perturbation must be unaffected by future tokens.
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)
    # Positions at/after the perturbation should generally change.
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-6)
