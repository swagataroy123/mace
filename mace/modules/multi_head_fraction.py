import torch
import torch.nn as nn
import torch.nn.functional as F


class Attentionpoolingfraction(nn.Module):
    def __init__(self, *X, pool_type="avg", temperature=0.5):
        """
        Transformer-style attention for computing normalized delta1, delta2, ...
        Args:
            input_dim: Dimensionality of the input features.
            hidden_dim: Dimensionality of the attention projection space.
        """
        super(Attentionpoolingfraction, self).__init__()
        # self.query_proj = nn.Linear(input_dim, hidden_dim)
        # self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.X = X
        self.pool_type = pool_type
        self.temperature = temperature

    def forward(self, x):
        """
        Compute normalized attention probabilities for multiple inputs X1, X2, ...
        Args:
            x: Input vector of shape [batch_size, input_dim].
            *X: Variable number of input matrices (e.g., X1, X2, ...), each of shape [seq_len, input_dim].
            pool_type: Pooling type ('avg' or 'max').
            temperature: Temperature scaling for sharpening the softmax output.
        Returns:
            delta: Normalized attention probabilities of shape [batch_size, len(X)].
        """
        # Project query
        query = x  # self.query_proj(x)  # [batch_size, hidden_dim]
        query = F.normalize(query, p=2, dim=-1)
        pooled = []
        for Xi in self.X:
            # Project keys for each Xi
            key = Xi  # self.key_proj(Xi)  # [seq_len, hidden_dim]
            key = F.normalize(key, p=2, dim=-1)
            # Compute attention scores
            attn_scores = torch.matmul(query, key.T) / (
                key.size(-1) ** 0.5
            )  # [batch_size, seq_len]
            # Apply pooling
            if self.pool_type == "avg":
                pooled_score = torch.mean(
                    attn_scores, dim=-1, keepdim=True
                )  # [batch_size, 1]
            elif self.pool_type == "max":
                pooled_score, _ = torch.max(
                    attn_scores, dim=-1, keepdim=True
                )  # [batch_size, 1]
            else:
                raise ValueError(
                    f"Invalid pool_type: {pool_type}. Choose 'avg' or 'max'."
                )

            pooled.append(pooled_score)

        # Concatenate pooled scores and normalize with temperature-scaled softmax
        pooled = torch.cat(pooled, dim=-1)  # [batch_size, len(X)]

        delta = F.softmax(pooled / self.temperature, dim=-1)  # Normalize to sum to 1

        return delta


def mask_with_deltas(
    x: torch.Tensor, deltas: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """
    Create a mask for attention scores where the mask values correspond to deltas (probabilities) for each head.
    Args:
        x: Input tensor of shape [batch_size, seq_len * num_heads].
        deltas: Tensor of probabilities (deltas) for each head of shape [batch_size, num_heads].
        num_heads: Number of attention heads.
    Returns:
        masked_x: Masked tensor of the same shape as `x`, weighted by deltas.
    """
    # Reshape input tensor to [batch_size, seq_len, num_heads]
    seq_len = x.shape[1] // num_heads
    # x_reshaped = x.view(x.shape[0], seq_len, num_heads).permute(2,0,1)

    # Expand deltas to match the sequence length dimension
    deltas_expanded = deltas.unsqueeze(-1).repeat(1, 1, seq_len).reshape(x.shape)
    # Apply the mask (weight x by deltas for each head)
    masked_x = x * deltas_expanded

    # Reshape back to the original shape
    # masked_x = masked_x  # [batch_size, seq_len * num_heads]

    return masked_x
