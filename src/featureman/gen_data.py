import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ModularArithmeticDataset(Dataset):
    def __init__(self, p=113, n_samples=1000, operation="add"):
        self.p = p
        self.data = []
        for _ in range(n_samples):
            a, b = np.random.randint(0, p, size=2)
            self.data.append(
                {
                    "inputs": torch.tensor([a, p, b, p + 1], dtype=torch.long),
                    "labels": torch.tensor((a + b) % p, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class OneLayerTransformer(nn.Module):
    def __init__(self, p=113, d_model=128, nheads=4):
        super().__init__()
        vocab_size = p + 3  # 0 to p-1, +, =, pad

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.01)

        self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

        self.unembed = nn.Linear(d_model, p)

    def forward(self, x):
        # xshape: (batch_size, seq_len)
        resid = self.embedding(x) + self.pos_emb[:, : x.shape[1], :]

        attn_out, _ = self.attn(resid, resid, resid)
        resid = resid + attn_out

        resid = self.mlp(resid) + resid

        logits = self.unembed(resid[:, -1, :])
        return logits
