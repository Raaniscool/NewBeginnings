"""Model tests: shapes, causal masking, loss, backward, CPU-only behavior,
and the param-count accounting used in milestone reports.
"""

from __future__ import annotations

import torch

from core.model import GPT
from core.config import ModelConfig


def tiny_model(**over):
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4,
                      d_model=48, dropout=0.0, bias=False)
    for k, v in over.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return GPT(cfg), cfg


def test_forward_shapes_and_loss():
    model, cfg = tiny_model()
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss.item() == loss.item()  # not NaN
    assert torch.isfinite(loss)


def test_logits_only_without_targets():
    model, cfg = tiny_model()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_causal_masking_is_exact():
    """Change ONLY future tokens; logits at earlier positions must be
    bit-identical. This proves no information leaks backward in time."""
    model, cfg = tiny_model()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        base, _ = model(x)
    x_future_changed = x.clone()
    x_future_changed[0, 10:] = (x_future_changed[0, 10:] + 3) % cfg.vocab_size
    with torch.no_grad():
        changed, _ = model(x_future_changed)
    assert torch.equal(base[0, :10], changed[0, :10]), \
        "causal mask broken: earlier logits changed when the future changed"


def test_backward_produces_finite_grads_everywhere():
    model, cfg = tiny_model(dropout=0.1)
    x = torch.randint(0, cfg.vocab_size, (4, 20))
    y = torch.randint(0, cfg.vocab_size, (4, 20))
    model.train()
    _, loss = model(x, y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_param_count_matches_architecture_accounting():
    model, cfg = tiny_model()
    n = model.num_params()
    est = cfg.n_params_estimate()
    tokens = cfg.vocab_size * cfg.d_model
    assert n == est, f"reported {n} params but estimate {est}; report math is wrong"
    # tied embeddings: lm_head shares wte -> only one embedding matrix counted
    assert model.lm_head.weight is model.wte.weight
    assert tokens == cfg.vocab_size * cfg.d_model


def test_position_limit_enforced():
    model, cfg = tiny_model()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 5))
    try:
        model(x)
    except (ValueError, AssertionError, IndexError, RuntimeError):
        return  # loud failure is the contract
    raise AssertionError("model accepted sequences longer than block_size")


def test_next_token_logprobs_shape():
    model, cfg = tiny_model()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        lp = model.next_token_logprobs(x)
    assert lp.shape == (2, cfg.vocab_size)
    # log-softmax sanity: each row sums (in prob space) to ~1
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2), atol=1e-4)
