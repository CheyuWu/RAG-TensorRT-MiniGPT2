from torch import nn
from .gpt2_block import GPT2Block


class MiniGPT2(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [GPT2Block(hidden_size, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.head(x)
