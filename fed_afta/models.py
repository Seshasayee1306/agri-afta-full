import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleTorchEncoder(nn.Module):
    """
    A corrected PyTorch encoder that:
    - inherits nn.Module
    - supports .eval()
    - supports correct state_dict() and load_state_dict()
    - produces embeddings for XGBoost in the AFTA pipeline
    """

    def __init__(self, input_dim, embedding_dim=16, device='cpu'):
        super(SimpleTorchEncoder, self).__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Encoder network
        self.network_ = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),
            nn.ReLU()
        )

        # Local classifier (OPTIONAL – used only during federated training)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.to(self.device)

    # -------------------------------
    # Utility to convert arrays
    # -------------------------------
    def _to_tensor(self, arr):
        return torch.tensor(np.asarray(arr), dtype=torch.float32, device=self.device)

    # -------------------------------
    # Training loop (local)
    # -------------------------------
    def fit(self, X, y, epochs=30, batch_size=256, lr=1e-3, verbose=False):

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        params = list(self.network_.parameters()) + list(self.classifier.parameters())
        opt = optim.Adam(params, lr=lr)
        loss_fn = nn.BCELoss()

        for ep in range(epochs):
            epoch_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                emb = self.network_(xb)
                out = self.classifier(emb)
                loss = loss_fn(out, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item() * xb.size(0)

            if verbose:
                print(f"[encoder] Epoch {ep+1}/{epochs} loss: {epoch_loss/len(dataset):.4f}")

    # -------------------------------
    # Probability prediction
    # -------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)

        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            emb = self.network_(t)
            out = self.classifier(emb).cpu().numpy().reshape(-1)
        return out

    # -------------------------------
    # Binary prediction
    # -------------------------------
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    # -------------------------------
    # Embedding extraction (used by AFTA + XGBoost head)
    # -------------------------------
    def get_embeddings(self, X):
        X = np.asarray(X, dtype=np.float32)

        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            emb = self.network_(t).cpu().numpy()
        return emb

    # -------------------------------
    # Return self as classifier
    # -------------------------------
    @property
    def clf(self):
        return self

    # -------------------------------
    # Custom state_dict to match saved format
    # -------------------------------
    def state_dict(self):
        sd = {}

        # Save encoder weights
        for k, v in self.network_.state_dict().items():
            sd[f"network_{k}"] = v.detach().cpu()

        # Save classifier weights
        for k, v in self.classifier.state_dict().items():
            sd[f"classifier_{k}"] = v.detach().cpu()

        return sd

    # -------------------------------
    # Load saved weights back
    # -------------------------------
    def load_state_dict(self, state_dict):
        net_sd = {}
        cls_sd = {}

        for k, v in state_dict.items():
            if k.startswith("network_"):
                net_sd[k.replace("network_", "")] = (
                    v if isinstance(v, torch.Tensor) else torch.tensor(v)
                )
            elif k.startswith("classifier_"):
                cls_sd[k.replace("classifier_", "")] = (
                    v if isinstance(v, torch.Tensor) else torch.tensor(v)
                )

        # Load with strict=False so missing keys won't fail
        self.network_.load_state_dict(net_sd, strict=False)
        self.classifier.load_state_dict(cls_sd, strict=False)
