from __future__ import annotations
"""
Model definition for:
Chandroth, J.; Ali, J. "Lightweight MLP-Based Feature Extraction with Linear
Classifier for Intrusion Detection System in Internet of Things."
Electronics 2026, 15(8), 1604. https://doi.org/10.3390/electronics15081604

Architecture per Section 3.2-3.3:
  Input (d features)
    -> Linear(d, 128) -> ReLU              (Eq. 2, first hidden layer h1)
    -> Linear(128, 64)                     (Eq. 3, embedding z)
    -> Linear(64, C)                       (Eq. 4, logits o)
    -> SoftMax                             (Eq. 5, applied via CrossEntropyLoss
                                             during training / explicitly at
                                             inference time)

NOTE (ambiguity to confirm with authors):
  Section 3.2 states "The hidden layers employ the ReLU activation function"
  (plural), which would suggest ReLU is also applied after the embedding
  layer. However, Eq. (3) is written as a plain affine transform with no
  sigma(.) term:  z = W2 h1 + b2
  This implementation follows Eq. (3) literally (no ReLU on the embedding
  layer). If the authors confirm ReLU should be applied there too, flip
  `self.embedding_activation` to True.
"""

import torch
import torch.nn as nn


class LightweightMLP_IDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden1: int = 128, embedding_dim: int = 64,
                 embedding_activation: bool = True):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.embedding = nn.Linear(hidden1, embedding_dim)
        self.embedding_activation = embedding_activation
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        h1 = self.relu(self.hidden1(x))               # Eq. 2
        z = self.embedding(h1)                          # Eq. 3
        if self.embedding_activation:
            z = self.relu(z)
        logits = self.classifier(z)                      # Eq. 4 (pre-softmax)
        if return_embedding:
            return logits, z
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)              # Eq. 5


if __name__ == "__main__":
    # quick sanity check + parameter count for a given (d, C) pair
    d, C = 78, 7  # example: CICIDS2017-like dims, adjust to your real d/C
    model = LightweightMLP_IDS(input_dim=d, num_classes=C)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"input_dim={d}, num_classes={C}, trainable_params={n_params}")
    x = torch.randn(4, d)
    out = model(x)
    print("output shape:", out.shape)
