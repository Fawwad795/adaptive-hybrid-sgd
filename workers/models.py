"""
Model definitions for the adaptive hybrid SGD framework.

LogReg  — pure NumPy logistic regression (784 → 10 for MNIST).
SmallCNN — LeNet-5 style PyTorch CNN (~62 K params) for CIFAR-10.

Both expose a common interface used by the trainer and the distributed engines:
    model.forward(X)              → logits (np.ndarray or torch.Tensor)
    model.loss(logits, y)         → scalar loss
    model.gradients(X, y)         → dict[str, np.ndarray]  (detached, CPU)
    model.get_params()            → dict[str, np.ndarray]
    model.set_params(param_dict)  → None
    model.update(grads, lr)       → None  (in-place SGD step)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility ────────────────────────────────────────────────────────────────────

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-12)
    return -log_p.mean()


# ── Logistic Regression (NumPy) ────────────────────────────────────────────────

class LogReg:
    """
    Multinomial logistic regression.
    Input : (N, 784) float32 arrays (flattened MNIST images, already normalised)
    Output: 10 class logits
    """

    def __init__(self, n_features: int = 784, n_classes: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / n_features)
        self.W = rng.normal(0.0, scale, (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b          # (N, 10)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return _cross_entropy(self.predict_proba(X), y)

    # ── Gradients ─────────────────────────────────────────────────────────────

    def gradients(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """
        Returns {"W": grad_W, "b": grad_b} for a mini-batch (X, y).
        Gradient is averaged over the batch.
        """
        n = len(y)
        probs = self.predict_proba(X)       # (N, 10)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1.0      # (N, 10)
        delta /= n
        grad_W = X.T @ delta               # (784, 10)
        grad_b = delta.sum(axis=0)         # (10,)
        return {"W": grad_W, "b": grad_b}

    # ── Parameter access ──────────────────────────────────────────────────────

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"].astype(np.float32)
        self.b = params["b"].astype(np.float32)

    def update(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.W -= lr * grads["W"]
        self.b -= lr * grads["b"]


# ── Small CNN (PyTorch / LeNet-5 style) ────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    LeNet-5 style network for CIFAR-10 (3×32×32 → 10 classes, ~62 K params).

    Architecture:
        Conv1(3→6, 5×5) → ReLU → AvgPool(2×2)     # → 6×14×14
        Conv2(6→16, 5×5) → ReLU → AvgPool(2×2)    # → 16×5×5
        FC(400→120) → ReLU
        FC(120→84)  → ReLU
        FC(84→10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # ── Gradient dict ─────────────────────────────────────────────────────────

    def gradients(self) -> dict[str, np.ndarray]:
        """Return a detached CPU copy of all parameter gradients (after backward)."""
        return {
            name: param.grad.detach().cpu().numpy().copy()
            for name, param in self.named_parameters()
            if param.grad is not None
        }

    # ── Parameter access ──────────────────────────────────────────────────────

    def get_params(self) -> dict[str, np.ndarray]:
        return {
            name: param.detach().cpu().numpy().copy()
            for name, param in self.named_parameters()
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(torch.from_numpy(params[name]))

    def update(self, grads: dict[str, np.ndarray], lr: float) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in grads:
                    param -= lr * torch.from_numpy(grads[name]).to(param.device)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(name: str, seed: int = 0):
    """Return a freshly initialised model for the given name."""
    name = name.lower()
    if name == "logreg":
        return LogReg(seed=seed)
    if name == "cnn":
        torch.manual_seed(seed)
        return SmallCNN()
    raise ValueError(f"Unknown model: {name!r}. Choose 'logreg' or 'cnn'.")
