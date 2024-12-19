import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import RobertaConfig, RobertaModel

from src.common.config import LANGJEPAConfig


def normalize_features(features: Tensor) -> Tensor:
    """L2 normalize features along embedding dimension."""
    return F.normalize(features, p=2, dim=-1)


class TextTransformer(nn.Module):
    """LANG-JEPA encoder using RoBERTa.

    Encodes text into a dense representation using a RoBERTa model.

    Example:
        >>> encoder = TextTransformer(config)
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch_size, seq_len]
        >>> attention_mask = torch.ones(1, 5)            # [batch_size, seq_len]
        >>> features = encoder(input_ids, attention_mask) # [batch_size, seq_len, embed_dim]
    """

    def __init__(self, config: LANGJEPAConfig):
        super().__init__()
        roberta_config = RobertaConfig(
            vocab_size=len(config.data.tokenizer),
            hidden_size=config.model.embed_dim,
            num_hidden_layers=config.model.num_layers,
            num_attention_heads=config.model.num_heads,
            intermediate_size=int(config.model.embed_dim * config.model.mlp_ratio),
            hidden_dropout_prob=config.model.dropout,
            attention_probs_dropout_prob=config.model.dropout,
            max_position_embeddings=config.model.max_length,
            type_vocab_size=1,
        )
        self.encoder = RobertaModel(roberta_config)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Transform input tokens into contextual embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Mask for padded tokens [batch_size, seq_len] (1 for real tokens, 0 for padding)

        Returns:
            Contextual embeddings [batch_size, seq_len, embed_dim]
        """
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state


class TextPredictor(nn.Module):
    """Predicts target sentence embeddings from context embeddings.

    Uses attention to process context sequence into a single predicted embedding
    for the target (next) sentence.

    Example:
        >>> predictor = TextPredictor(input_dim=768, pred_dim=384)
        >>> context_feats = torch.randn(2, 10, 768)     # [batch_size, seq_len, input_dim]
        >>> padding_mask = torch.ones(2, 10)            # [batch_size, seq_len]
        >>> predictions = predictor(context_feats, padding_mask)  # [batch_size, pred_dim]
    """

    def __init__(self, input_dim: int, pred_dim: int, num_heads: int = 8):
        super().__init__()
        # Attention to aggregate context sequence
        self.context_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # Learnable query for attention
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Projection to prediction space
        self.projection = nn.Sequential(
            nn.Linear(input_dim, pred_dim * 2),
            nn.GELU(),
            nn.LayerNorm(pred_dim * 2),
            nn.Linear(pred_dim * 2, pred_dim),
            nn.LayerNorm(pred_dim),
        )

    def project_targets(self, features: Tensor) -> Tensor:
        """Project target features to prediction space.

        Args:
            features: Target features [batch_size, embed_dim]

        Returns:
            Projected features [batch_size, pred_dim]
        """
        return self.projection(features)

    def forward(
        self, context_feats: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        """Generate predictions from context embeddings.

        Args:
            context_feats: Context token features [batch_size, seq_len, input_dim]
            padding_mask: Optional mask for padded positions [batch_size, seq_len]

        Returns:
            Predicted target embeddings [batch_size, pred_dim]
        """
        batch_size = context_feats.size(0)

        # Expand query for batch size
        query = self.query.expand(batch_size, -1, -1)

        # Convert padding mask for attention if provided
        key_padding_mask = None if padding_mask is None else (padding_mask == 0)

        # Attend over context sequence
        attended_context, _ = self.context_attention(
            query=query,
            key=context_feats,
            value=context_feats,
            key_padding_mask=key_padding_mask,
        )

        # Project to prediction space
        return self.projection(attended_context.squeeze(1))
