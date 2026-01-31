import torch
import torch.nn as nn

from src.train.loops import _flatten_grads


def test_flatten_grads_skips_sparse_and_returns_dense_only():
    emb = nn.Embedding(10, 4, sparse=True)
    lin = nn.Linear(4, 1)

    # forward: take a single index, pass through emb then linear
    idx = torch.tensor([1], dtype=torch.int64)
    out = lin(emb(idx))
    out.sum().backward()

    params = [("emb.weight", emb.weight), ("lin.weight", lin.weight), ("lin.bias", lin.bias)]
    flat, dense_cnt, sparse_skipped = _flatten_grads(params)
    assert sparse_skipped == 1  # embedding grad skipped
    assert dense_cnt == 2       # linear weight + bias
    assert flat.numel() == lin.weight.grad.numel() + lin.bias.grad.numel()
    assert flat.is_sparse is False
