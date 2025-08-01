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

    def forward(self, x, return_activations=False):
        # xshape: (batch_size, seq_len)
        resid = self.embedding(x) + self.pos_emb[:, : x.shape[1], :]

        attn_out, _ = self.attn(resid, resid, resid)
        resid = resid + attn_out

        mlp_pre = self.mlp[0](resid)
        mlp_act = self.mlp[1](mlp_pre)
        mlp_out = self.mlp[2](mlp_act)
        resid = mlp_out + resid

        logits = self.unembed(resid[:, -1, :])

        if return_activations:
            return logits, mlp_act
        return logits


class BatchedSAE(nn.Module):
    """
    This is a modified version of BatchedSAE that uses a different initialization method.
    Link to the updates to SAE: https://transformer-circuits.pub/2024/april-update/index.html#training-saes

    Key changes:
    - Decoder weights (W_d) are no longer L2 normalized to unit norm
    - L1 regularization is weighted by the L2 norm of decoder columns
    - The decoder weights (W_d) are initialized with random directions and fixed L2 norm
    - The encoder weights (W_e) are initialized as the transpose of W_d
    - The decoder biases (b_d) are initialized to zero
    - The encoder biases (b_e) are initialized to zero
    """

    def __init__(self, input_dim, n_models, width_ratio=4, activation=nn.ReLU()):
        super().__init__()
        self.n_models = n_models
        self.sae_hidden = input_dim * width_ratio

        # Initialize W_d with random directions and fixed L2 norm
        W_d_init = torch.randn(n_models, self.sae_hidden, input_dim)
        # Normalize columns to have L2 norm of 0.1
        W_d_init = 0.1 * W_d_init / W_d_init.norm(p=2, dim=2, keepdim=True)

        # Initialize W_e as W_d transpose
        W_e_init = W_d_init.transpose(1, 2)

        # Shape: [n_models, input_dim, sae_hidden]
        self.W_e = nn.Parameter(W_e_init)

        # Shape: [n_models, sae_hidden]
        self.b_e = nn.Parameter(torch.zeros(n_models, self.sae_hidden))

        # Shape: [n_models, sae_hidden, input_dim]
        self.W_d = nn.Parameter(W_d_init)

        # Shape: [n_models, input_dim]
        self.b_d = nn.Parameter(torch.zeros(n_models, input_dim))
        self.nonlinearity = activation

    def forward(self, x):
        # x shape is already: [n_models, batch_size, input_dim]

        # Compute activations f(x) for each model
        # bmm for batched matrix multiply
        acts = self.nonlinearity(
            torch.bmm(x, self.W_e)
            + self.b_e.unsqueeze(1)  # [n_models, batch_size, sae_hidden]
        )

        # Calculate L1 regularization weighted by decoder norm for each feature
        # [n_models, batch_size, sae_hidden] * [n_models, sae_hidden, 1] -> [n_models]
        decoder_norms = self.W_d.norm(p=2, dim=2)  # [n_models, sae_hidden]
        l1_regularization = (acts.abs() * decoder_norms.unsqueeze(1)).sum(
            dim=[1, 2]
        )  # [n_models]

        # Calculate L0 sparsity metric
        l0 = (acts > 0).sum(dim=2).float().mean(dim=1)  # [n_models]

        # Reconstruct input for each model
        reconstruction = torch.bmm(acts, self.W_d) + self.b_d.unsqueeze(
            1
        )  # [n_models, batch_size, input_dim]

        return l0, l1_regularization, reconstruction

    def train(
        self,
        train_data,
        test_data,
        batch_size=128,
        n_epochs=10000,
        l1_lam=3e-5,
        weight_decay=1e-4,
        output_epoch=False,
        patience=10,
        min_improvement=1e-4,
    ):
        # Ensure train_data shape is [n_models, n_samples, input_dim]
        assert train_data.dim() == 3 and train_data.size(0) == self.n_models, (
            f"Expected train_data shape [n_models, n_samples, input_dim], got {train_data.shape}"
        )

        # Convert l1_lam to tensor if it's not already
        if isinstance(l1_lam, (int, float)):
            l1_lam = torch.full((self.n_models,), l1_lam, device=train_data.device)
        else:
            l1_lam = torch.tensor(l1_lam, device=train_data.device)

        assert l1_lam.shape == (self.n_models,), (
            f"l1_lam must have shape [n_models], got {l1_lam.shape}"
        )

        n_samples = train_data.size(1)  # Use size(1) to get number of samples
        batch_size = min(batch_size, n_samples)  # Ensure batch_size <= n_samples
        n_batches = n_samples // batch_size

        # Initialize tracking variables
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        # Initialize early stopping trackers for each model
        best_losses = torch.full(
            (self.n_models,), float("inf"), device=train_data.device
        )
        patience_counters = torch.zeros(self.n_models, dtype=torch.int)
        active_models = torch.ones(
            self.n_models, dtype=torch.bool, device=train_data.device
        )

        for epoch in range(n_epochs):
            # Skip iteration if all models have converged
            if not active_models.any():
                break

            indices = torch.randperm(n_samples)
            total_mse_loss = torch.zeros(self.n_models, device=train_data.device)
            total_l1_loss = torch.zeros(self.n_models, device=train_data.device)
            total_l0 = torch.zeros(self.n_models, device=train_data.device)

            # Process batches
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch = train_data[:, batch_indices]  # Select samples for all models

                optimizer.zero_grad()
                l0, l1, recon_hiddens = self(batch)

                # Calculate loss for each model separately - simplified
                recon_loss = nn.MSELoss(reduction="none")(recon_hiddens, batch).mean(
                    dim=[1, 2]
                )  # [n_models]

                sparsity_loss = l1_lam * l1
                loss = recon_loss + sparsity_loss

                # Sum losses across all models for backward
                loss.sum().backward()
                optimizer.step()

                total_mse_loss += recon_loss.detach()
                total_l1_loss += sparsity_loss.detach()
                total_l0 += l0.detach()

            # Early stopping check for each model
            avg_loss = total_mse_loss / n_batches

            for m in range(self.n_models):
                if not active_models[m]:
                    continue

                if avg_loss[m] < best_losses[m] - min_improvement:
                    best_losses[m] = avg_loss[m]
                    patience_counters[m] = 0
                else:
                    patience_counters[m] += 1
                    if patience_counters[m] >= patience:
                        active_models[m] = False
                        if output_epoch:
                            print(
                                f"Model {m} converged at epoch {epoch} with loss {best_losses[m]:.4f}"
                            )

            if (epoch % 10 == 0) and output_epoch:
                avg_l1_loss = total_l1_loss / n_batches
                avg_l0 = total_l0 / n_batches

                for m in range(self.n_models):
                    if active_models[m]:
                        print(
                            f"Model {m}, Epoch {epoch}, Loss: {avg_loss[m]:.4f}, "
                            f"L1: {avg_l1_loss[m]:.4f}, "
                            f"L0: {avg_l0[m]:.4f}"
                        )

        return [
            {
                "mse": total_mse_loss[m].item() / n_batches,
                "L0": total_l0[m].item() / n_batches,
                "L1 lambda": l1_lam[m].item(),
                "weights": self.W_d[m, :, :].detach().cpu().numpy(),
                "biases": self.b_d[m, :].detach().cpu().numpy(),
            }
            for m in range(self.n_models)
        ]
