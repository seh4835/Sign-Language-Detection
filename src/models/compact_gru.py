"""Small GRU classifier for fixed-length sign landmark sequences."""
from __future__ import annotations

import torch
from torch import nn


class CompactGRUClassifier(nn.Module):
    """One-layer GRU with masked-feature input and mean/max temporal pooling."""

    def __init__(
        self,
        input_dim: int = 180,
        hidden_dim: int = 48,
        n_classes: int = 100,
        input_dropout: float = 0.15,
        output_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if input_dim != 180:
            raise ValueError("CompactGRUClassifier expects 180 features per frame.")
        if hidden_dim < 1 or n_classes < 2:
            raise ValueError("hidden_dim must be positive and n_classes must be at least two.")
        self.input_dropout = nn.Dropout(input_dropout)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(output_dropout)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1:] != (30, 180):
            raise ValueError("Expected input shape (batch, 30, 180).")
        if not torch.isfinite(sequence).all():
            raise ValueError("GRU input contains non-finite values.")
        states, _ = self.gru(self.input_dropout(sequence))
        states = self.norm(states)
        pooled = torch.cat((states.mean(dim=1), states.max(dim=1).values), dim=1)
        logits = self.classifier(self.output_dropout(pooled))
        if not torch.isfinite(logits).all():
            raise ValueError("GRU produced non-finite logits.")
        return logits
