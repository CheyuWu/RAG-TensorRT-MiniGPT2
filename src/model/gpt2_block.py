from typing import Optional
import torch
from torch import nn


# GPT2Block implements a single transformer block as used in GPT-2 models.
# It consists of a multi-head self-attention layer, followed by a feed-forward network,
# each with layer normalization and dropout. This block is the core building unit of transformer-based language models.
class GPT2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.ff: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size),
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(hidden_size)
        self.norm2: nn.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            attn_mask (Optional[torch.Tensor]): Attention mask for self-attention
        Returns:
            torch.Tensor: Output tensor of the same shape as input
        """
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
