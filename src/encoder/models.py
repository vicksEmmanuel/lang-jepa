import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel

from src.common.config import LANGJEPAConfig


class TextTransformer(nn.Module):
    """Text encoder based on pre-trained transformer models."""

    def __init__(self, config: LANGJEPAConfig):
        super().__init__()
        # Load base model config and update with our settings
        model_config = AutoConfig.from_pretrained(config.data.tokenizer_path)
        model_config.update(
            {
                "hidden_size": config.model.embed_dim,
                "num_hidden_layers": config.model.num_layers,
                "num_attention_heads": config.model.num_heads,
                "intermediate_size": int(
                    config.model.embed_dim * config.model.mlp_ratio
                ),
                "hidden_dropout_prob": config.model.dropout,
                "attention_probs_dropout_prob": config.model.dropout,
                "vocab_size": len(config.data.tokenizer),
                "gradient_checkpointing": config.meta.use_gradient_checkpointing,
            }
        )
        self.encoder = AutoModel.from_config(model_config)

        # After creating encoder, enable gradient checkpointing
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Get contextual embeddings for input tokens."""
        outputs = self.encoder(input_ids, attention_mask, return_dict=True)
        return outputs.last_hidden_state


class TextPredictor(nn.Module):
    """Predicts next sentence embeddings from context embeddings."""

    def __init__(self, input_dim: int, pred_dim: int, num_heads: int = 8):
        super().__init__()
        # Attention to aggregate context sequence
        self.context_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Project to prediction space
        self.projection = nn.Sequential(
            nn.Linear(input_dim, pred_dim), nn.LayerNorm(pred_dim)
        )

    def forward(
        self, context_feats: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Generate predictions from context."""
        # Prepare attention inputs
        query = self.query.expand(context_feats.size(0), -1, -1)
        key_padding_mask = (
            ~attention_mask.bool() if attention_mask is not None else None
        )

        # Get context representation
        context, _ = self.context_attention(
            query=query,
            key=context_feats,
            value=context_feats,
            key_padding_mask=key_padding_mask,
        )

        # Project and normalize
        predictions = self.projection(context.squeeze(1))
        return F.normalize(predictions, p=2, dim=-1)

    def project_targets(self, features: Tensor) -> Tensor:
        """Project target features to prediction space."""
        predictions = self.projection(features)
        return F.normalize(predictions, p=2, dim=-1)
