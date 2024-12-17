import torch
import torch.nn as nn

from src.config import LANGJEPAConfig

initializer_range: float = 0.02
layer_norm_eps = 1e-5


class TextTransformer(nn.Module):
    def __init__(self, config: LANGJEPAConfig):
        super().__init__()
        self.config = config.model  # Using model section of config

        # Token embeddings
        vocab_size = len(config.data.tokenizer)
        self.token_embedding = nn.Embedding(vocab_size, self.config.embed_dim)
        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.config.max_length, self.config.embed_dim)
        )
        self.dropout = nn.Dropout(self.config.dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.num_heads,
            dim_feedforward=int(self.config.embed_dim * self.config.mlp_ratio),
            dropout=self.config.dropout,
            activation="gelu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers
        )

        self.layer_norm = nn.LayerNorm(self.config.embed_dim, eps=layer_norm_eps)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=initializer_range)
        # Layers within transformer encoder are initialized by default

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [B, N] long tensor of token IDs
            attention_mask: [B, N] bool or int tensor (1 for tokens to attend, 0 for padded)
        Returns:
            features: [B, N, D] float tensor
        """
        B, N = input_ids.shape
        # Embed tokens
        x = self.token_embedding(input_ids)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embedding[:, :N, :]

        x = self.dropout(x)

        # If attention_mask is provided, convert to a key_padding_mask for TransformerEncoder
        # TransformerEncoder expects a bool mask with True meaning "not attend" or we can pass None if no masking needed.
        # PyTorch expects key_padding_mask with shape [B, N], True for tokens to ignore.
        # If attention_mask is 1 for valid and 0 for padding, we invert it.
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True where we have padding
        else:
            key_padding_mask = None

        # Forward through Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Final layer norm
        x = self.layer_norm(x)
        return x


def extract_features_for_masks(
    features: torch.Tensor, masks_batch: list
) -> torch.Tensor:
    """
    Extract features for masked tokens and return their mean representations.

    Args:
        features: [B, N, D] tensor of encoded features
        masks_batch: List of lists of tensors containing token indices

    Returns:
        masked_features: [M, D] tensor where M is total number of masked segments
    """
    all_features = []
    batch_size = features.size(0)

    # For each item in batch
    for b in range(batch_size):
        # For each masked segment in this batch item
        for mask in masks_batch[b]:
            # Get features for these indices and take their mean
            segment_features = features[b, mask].mean(0)
            all_features.append(segment_features)

    if not all_features:
        return torch.empty(0, device=features.device)

    return torch.stack(all_features)


class TextPredictor(nn.Module):
    def __init__(self, input_dim: int, pred_dim: int):
        super().__init__()
        # Project to concept space
        self.target_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, pred_dim),
        )

        # Predict in concept space
        hidden_dim = pred_dim * 2
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pred_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.target_projection, self.predictor]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0.0)
                    nn.init.constant_(m.weight, 1.0)

    def project_targets(self, features: torch.Tensor) -> torch.Tensor:
        """Project encoder features into concept space"""
        return self.target_projection(features)

    def forward(
        self, context_feats: torch.Tensor, enc_masks_batch: list, pred_masks_batch: list
    ) -> torch.Tensor:
        """
        Predict features for masked segments using context features.
        Should produce same number of predictions as masked segments.
        """
        all_pred_feats = []
        batch_size = context_feats.size(0)

        # For each item in batch
        for b in range(batch_size):
            # Get context features
            if len(enc_masks_batch[b]) > 0:
                context_feats_b = []
                for mask in enc_masks_batch[b]:
                    context_feats_b.append(context_feats[b, mask].mean(0))
                context_feats_b = torch.stack(context_feats_b).mean(0)
            else:
                # If no context, use mean of all non-padding features
                context_feats_b = context_feats[b][
                    context_feats[b].abs().sum(-1) > 0
                ].mean(0)

            # Make one prediction for each masked segment
            for _ in range(len(pred_masks_batch[b])):
                pred_feat = self.predictor(context_feats_b)
                all_pred_feats.append(pred_feat)

        if not all_pred_feats:
            return torch.empty(0, device=context_feats.device)

        return torch.stack(all_pred_feats)
