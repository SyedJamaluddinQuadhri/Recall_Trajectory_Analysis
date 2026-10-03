"""
sequential_classifiers.py — Lightweight sequential models for the Phase 2
richer-representation experiment (Stage 3 improvement).

These models treat the 32-layer trajectory as a TIME SEQUENCE of per-layer
signals rather than flattening everything into a single vector.  The design
priority is correctness and comparability with the existing flat MLP, NOT
maximum capacity — both models are kept small intentionally.

Models
------
CNN1DClassifier
    Two 1-D convolutional layers over the 32-layer sequence, followed by
    global average pooling and a linear head.  Captures local patterns
    (e.g. "confidence rises sharply then plateaus") without any recurrence.

GRUClassifier
    One GRU layer that reads the 32-layer sequence left-to-right (layer 0
    to layer 31), then uses the final hidden state to predict.  Captures
    cumulative / sequential dependencies across the full depth.

Input convention (both models)
-------------------------------
    x  : torch.Tensor, shape [batch, n_layers_seq, n_channels]
            n_layers_seq — number of transformer layers (32 for Phi-2)
            n_channels   — number of signals per layer (e.g. 2 = prob+entropy,
                           4 = prob+entropy+mlp_norm+rank_norm)
    Returns: logit tensor, shape [batch], NO sigmoid — use BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    Lightweight 1-D CNN for sequential trajectory classification.

    Architecture (default, 2-channel input):
        Conv1d(n_channels → 16, kernel=3, pad=1) → ReLU
        Conv1d(16 → 32, kernel=3, pad=1)          → ReLU
        GlobalAvgPool over the 32-layer dimension
        Linear(32 → 1)  →  raw logit (no sigmoid)

    Padding=1 on kernel=3 preserves the sequence length so global-avg-pool
    always operates over exactly n_layers_seq positions regardless of input
    size — safe for any n_layers value.

    Parameter count (n_channels=2): 2*16*3 + 16 + 16*32*3 + 32 + 32 + 1 ≈ 1,681
    (intentionally comparable in capacity to the flat MLP baseline's ~289 params)
    """

    def __init__(self, n_channels: int = 2, n_layers_seq: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers_seq = n_layers_seq

        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.head  = nn.Linear(32, 1)

        # Kaiming init for conv layers (ReLU activations)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_layers_seq, n_channels]
        Returns:
            logits: [batch]
        """
        # Conv1d expects [batch, channels, length]
        x = x.permute(0, 2, 1)               # → [batch, n_channels, n_layers_seq]
        x = torch.relu(self.conv1(x))         # → [batch, 16, n_layers_seq]
        x = torch.relu(self.conv2(x))         # → [batch, 32, n_layers_seq]
        x = x.mean(dim=-1)                    # global avg pool → [batch, 32]
        return self.head(x).squeeze(-1)        # → [batch]


class GRUClassifier(nn.Module):
    """
    Small GRU for sequential trajectory classification.

    Architecture (default, 2-channel input):
        GRU(input_size=n_channels, hidden_size=hidden_size, batch_first=True)
        Linear(hidden_size → 1)  →  raw logit (no sigmoid)

    The GRU reads layers 0..31 left-to-right (shallow to deep), so the
    final hidden state summarizes the cumulative evidence across all layers.

    Parameter count (n_channels=2, hidden=32):
        GRU: 3 * (hidden*(n_channels+hidden) + 2*hidden) ≈ 3,264
        Head: 32+1 = 33
    Again intentionally small — comparable to or slightly larger than the
    flat MLP, not a fundamentally different capacity class.
    """

    def __init__(self, n_channels: int = 2, hidden_size: int = 32):
        super().__init__()
        self.n_channels  = n_channels
        self.hidden_size = hidden_size

        self.gru  = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_layers_seq, n_channels]
        Returns:
            logits: [batch]
        """
        # GRU with batch_first=True: output [batch, seq, hidden], h_n [1, batch, hidden]
        _, h_n = self.gru(x)                  # h_n: [1, batch, hidden_size]
        h_last = h_n.squeeze(0)               # → [batch, hidden_size]
        return self.head(h_last).squeeze(-1)  # → [batch]
