import torch

from tinyllm.generation import generate
from tinyllm.model import GPT

TINY_CFG = dict(vocab_size=4096, context_length=32, n_layers=4, n_heads=4, hidden_size=256, dropout=0.1)


def make_model():
    torch.manual_seed(0)
    return GPT(**TINY_CFG)


def test_forward_shape_and_loss():
    model = make_model()
    x = torch.randint(0, TINY_CFG["vocab_size"], (2, 16))
    y = torch.randint(0, TINY_CFG["vocab_size"], (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, TINY_CFG["vocab_size"])
    assert torch.isfinite(loss)


def test_backward_populates_all_grads():
    model = make_model()
    x = torch.randint(0, TINY_CFG["vocab_size"], (2, 16))
    y = torch.randint(0, TINY_CFG["vocab_size"], (2, 16))
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"


def test_param_count_near_expected():
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < n_params < 6_000_000


def test_generation_length_and_context_crop():
    model = make_model()
    model.eval()
    prompt_len = TINY_CFG["context_length"]  # at context-length boundary
    idx = torch.randint(0, TINY_CFG["vocab_size"], (1, prompt_len))
    out = generate(model, idx, max_new_tokens=5, context_length=TINY_CFG["context_length"])
    assert out.shape[1] == prompt_len + 5
