from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


@dataclass
class DecoderConfig:
    """Configuration for the concept decoder."""

    embed_dim: int  # Dimension of concept space
    hidden_dim: int  # Internal dimension of decoder
    vocab_size: int  # Set from tokenizer
    pad_token_id: int  # Set from tokenizer
    bos_token_id: int  # Set from tokenizer
    eos_token_id: int  # Set from tokenizer
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_length: int = 128

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizer,
        embed_dim: int,
        hidden_dim: int | None = None,
        **kwargs,
    ) -> "DecoderConfig":
        """Create config from tokenizer and embedding dimension."""
        return cls(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim or embed_dim * 2,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,  # This will override defaults if provided
        )


class ConceptDecoder(nn.Module):
    """Decoder for converting concept embeddings back to text."""

    def __init__(self, config: DecoderConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Project from concept space to decoder dimension
        self.concept_proj = nn.Linear(config.embed_dim, config.hidden_dim)

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_length, config.hidden_dim)
        )

        # Decoder
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)

        # Output projection
        self.out_proj = nn.Linear(config.hidden_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        concepts: torch.Tensor,  # [batch_size, embed_dim]
        target_ids: torch.Tensor | None = None,  # [batch_size, seq_len]
    ) -> torch.Tensor:
        batch_size = concepts.shape[0]
        device = concepts.device

        # Reshape concepts if needed
        if len(concepts.shape) > 2:
            concepts = concepts.reshape(batch_size, -1)  # Flatten to [B, D]

        # Project concept to decoder space
        memory = self.concept_proj(concepts)  # [B, H]

        if self.training and target_ids is not None:
            # Get sequence length for teacher forcing (excluding last token)
            seq_length = target_ids.size(1) - 1

            # Expand memory to match sequence length
            memory = memory.unsqueeze(1).expand(-1, seq_length, -1)  # [B, L-1, H]

            # Teacher forcing (exclude last token from input)
            tgt_emb = self.token_embedding(target_ids[:, :-1])  # [B, L-1, H]
            tgt_emb = tgt_emb + self.pos_embedding[:, :seq_length]

            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_length, device=device
            )

            # Decode
            hidden = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.out_proj(hidden)  # [B, L-1, V]

            return logits

        else:
            # Autoregressive generation
            curr_ids = torch.full(
                (batch_size, 1),
                self.config.bos_token_id,
                device=device,
                dtype=torch.long,
            )


            logits = []
            for _ in range(self.config.max_length - 1):
                # Embed current sequence
                tgt_emb = self.token_embedding(curr_ids)
                tgt_emb = tgt_emb + self.pos_embedding[:, : curr_ids.size(1)]

                # Decode one step
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    curr_ids.size(1), device=device
                )

                # Ensure memory is 3-dimensional
                memory_expanded = memory.unsqueeze(1).expand(-1, curr_ids.size(1), -1)  # [B, L, H]

                hidden = self.decoder(
                    tgt_emb, memory_expanded, tgt_mask=tgt_mask
                )
                step_logits = self.out_proj(hidden[:, -1:])  # [B, 1, V]
                logits.append(step_logits)

                # Sample next token
                next_token = step_logits.argmax(dim=-1)  # [B, 1]
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

                # Stop if we see end token
                if (next_token == self.config.eos_token_id).any():
                    break

            logits = torch.cat(logits, dim=1)

        return logits
    


    @torch.no_grad()
    def generate(
        self,
        concepts: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
    ) -> list[str]:
        """
        Generate text from concepts.

        Args:
            concepts: Concept embeddings to decode
            tokenizer: Tokenizer for decoding
            max_length: Optional override for maximum length

        Returns:
            List of generated strings
        """
        self.eval()
        logits = self.forward(concepts)
        sequences = logits.argmax(dim=-1)

        # Decode to text
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return texts
