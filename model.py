from __future__ import annotations
"""
Model definition for:
Chandroth, J.; Ali, J. "Lightweight MLP-Based Feature Extraction with Linear
Classifier for Intrusion Detection System in Internet of Things."
Electronics 2026, 15(8), 1604. https://doi.org/10.3390/electronics15081604

Architecture per Section 3.2-3.3 and Figure 1:
  Input (d features)
    -> Linear(d, 128) -> ReLU -> LayerNorm  (Eq. 2, first hidden layer h1)
    -> Linear(128, 64) -> ReLU -> LayerNorm (Eq. 3, embedding z)
    -> Linear(64, C)                        (Eq. 4, logits o)
    -> SoftMax                              (Eq. 5, applied via CrossEntropyLoss
                                              during training / explicitly at
                                              inference time)

NOTE:
  - Section 3.2 states "The hidden layers employ the ReLU activation function"
    (plural), which conflicts with Eq. (3)'s plain affine transform (no
    sigma(.) term) for the embedding layer. CONFIRMED by corresponding
    author Jehad Ali via email (2026-09-28): both hidden layers (128 and 64)
    use ReLU. `embedding_activation` therefore defaults to True. Pass
    `embedding_activation=False` (train.py: `--no_embedding_activation`) for
    Eq. (3)'s literal no-activation reading instead.
  - Figure 1 depicts each MLP block as "Dense+ReLU" followed by "LayerNorm",
    but neither Eq. (2)/(3) nor the Section 3.2 text mention normalization at
    all -- LayerNorm only appears in the figure, and the authors' reply did
    not address it (this question was asked separately, after the reply had
    already been sent). Since Figure 1 most plausibly reflects the actual
    implementation (and would also explain why the authors' training curves
    in Figure 2/3 are much smoother than lr=0.003 with no LR scheduler would
    otherwise produce), `layer_norm` defaults to True here. Pass
    `layer_norm=False` (train.py: `--no_layer_norm`) to reproduce the
    equations/text literally instead.
"""

import torch
import torch.nn as nn


class LightweightMLP_IDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden1: int = 128, embedding_dim: int = 64,
                 embedding_activation: bool = True,
                 layer_norm: bool = True):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.embedding = nn.Linear(hidden1, embedding_dim)
        self.embedding_activation = embedding_activation
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = nn.LayerNorm(hidden1)
            self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        h1 = self.relu(self.hidden1(x))               # Eq. 2
        if self.layer_norm:
            h1 = self.ln1(h1)                          # Figure 1 block 1
        z = self.embedding(h1)                          # Eq. 3
        if self.embedding_activation:
            z = self.relu(z)
        if self.layer_norm:
            z = self.ln2(z)                             # Figure 1 block 2
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
