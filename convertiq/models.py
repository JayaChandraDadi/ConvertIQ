"""Click and conversion models in PyTorch.

All variants share one input layer (hashed field embeddings + dense features) and
a DCN v2 cross network. They differ in how the two tasks share parameters:

  single     one DCN v2 model per task (the CVR model sees only clicked rows)
  shared     shared-bottom MLP, one tower per task
  mmoe       Multi-gate Mixture-of-Experts: experts shared, a softmax gate per task

and in the conversion head:

  esmm=True  pCTCVR = pCTR * pCVR, both trained on ALL impressions (Ma et al. 2018),
             which removes the sample-selection bias of training CVR on clicks only.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Inputs(nn.Module):
    def __init__(self, buckets: list[int], n_dense: int, emb: int = 12):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(b, emb) for b in buckets)
        self.dense_bn = nn.BatchNorm1d(n_dense)
        self.out_dim = emb * len(buckets) + n_dense
        for e in self.embs:
            nn.init.normal_(e.weight, 0, 0.01)

    def forward(self, sparse, dense):
        return torch.cat([e(sparse[:, i]) for i, e in enumerate(self.embs)] + [self.dense_bn(dense)], 1)


class CrossNet(nn.Module):
    """DCN v2 cross layers: x_{l+1} = x0 * (W x_l + b) + x_l, low-rank W = U V to keep it cheap."""

    def __init__(self, d: int, layers: int = 3, rank: int = 64):
        super().__init__()
        self.U = nn.ModuleList(nn.Linear(d, rank, bias=False) for _ in range(layers))
        self.V = nn.ModuleList(nn.Linear(rank, d) for _ in range(layers))

    def forward(self, x0):
        x = x0
        for u, v in zip(self.U, self.V):
            x = x0 * v(u(x)) + x
        return x


def mlp(d_in: int, dims=(256, 128), dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for h in dims:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    return nn.Sequential(*layers)


class DCNv2(nn.Module):
    """Single-task model: cross network in parallel with a deep MLP."""

    def __init__(self, buckets, n_dense, emb=12):
        super().__init__()
        self.inp = Inputs(buckets, n_dense, emb)
        d = self.inp.out_dim
        self.cross = CrossNet(d)
        self.deep = mlp(d)
        self.head = nn.Linear(d + 128, 1)

    def forward(self, sparse, dense):
        x0 = self.inp(sparse, dense)
        return self.head(torch.cat([self.cross(x0), self.deep(x0)], 1)).squeeze(-1)  # logit


class MultiTask(nn.Module):
    """Two-task model returning logits for CTR and for CVR (or CTCVR without ESMM)."""

    def __init__(self, buckets, n_dense, kind: str = "mmoe", n_experts: int = 4, emb: int = 12):
        super().__init__()
        assert kind in ("shared", "mmoe")
        self.kind = kind
        self.inp = Inputs(buckets, n_dense, emb)
        d = self.inp.out_dim
        self.cross = CrossNet(d)
        if kind == "shared":
            self.bottom = mlp(d, (256, 128))
        else:
            self.experts = nn.ModuleList(mlp(d, (256, 128)) for _ in range(n_experts))
            self.gates = nn.ModuleList(nn.Linear(d, n_experts) for _ in range(2))
        self.towers = nn.ModuleList(nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(2))

    def forward(self, sparse, dense):
        x = self.cross(self.inp(sparse, dense))
        if self.kind == "shared":
            h = self.bottom(x)
            reps = [h, h]
        else:
            e = torch.stack([ex(x) for ex in self.experts], 1)            # [B, E, 128]
            reps = [(torch.softmax(g(x), -1).unsqueeze(-1) * e).sum(1) for g in self.gates]
        return self.towers[0](reps[0]).squeeze(-1), self.towers[1](reps[1]).squeeze(-1)

    def gate_weights(self, sparse, dense) -> torch.Tensor:
        """Mean gate distribution per task: shows how much the tasks share experts."""
        x = self.cross(self.inp(sparse, dense))
        return torch.stack([torch.softmax(g(x), -1).mean(0) for g in self.gates])
