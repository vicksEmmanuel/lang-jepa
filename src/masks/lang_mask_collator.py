from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer

from src.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


@dataclass
class MaskOutput:
    """Typed output from mask collation"""

    input_ids: torch.Tensor  # [batch_size, seq_len]
    attention_mask: torch.Tensor  # [batch_size, seq_len]
    enc_masks: list[list[int]]  # Indices of context tokens for each batch item
    pred_masks: list[list[int]]  # Indices of mask tokens for each batch item
    original_texts: list[str]  # Original texts before processing


class LangMaskCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mask_ratio: float,
        max_length: int,
        sentence_split_config: SentenceSplitterConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.max_length = max_length
        self.sentence_splitter = SentenceSplitter(
            sentence_split_config or SentenceSplitterConfig()
        )

    def __call__(self, batch: list[str]) -> MaskOutput:
        """Process a batch of texts into masked input sequences."""
        # Split texts into sentences
        batch_sentences = self.sentence_splitter(batch)

        # Initialize output holders
        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        enc_masks_batch: list[list[int]] = []
        pred_masks_batch: list[list[int]] = []

        for sentences in batch_sentences:
            # Choose sentences to mask
            num_to_mask = max(1, int(self.mask_ratio * len(sentences)))
            # print(f"Masking {num_to_mask} out of {len(sentences)} sentences")
            mask_indices = set(torch.randperm(len(sentences))[:num_to_mask].tolist())

            # Build masked sequence
            sequence: list[int] = [self.tokenizer.cls_token_id]
            current_idx = 1  # Start after CLS
            enc_masks: list[int] = []
            pred_masks: list[int] = []

            # Process each sentence
            for i, sent in enumerate(sentences):
                if i in mask_indices:
                    sequence.append(self.tokenizer.mask_token_id)
                    pred_masks.append(current_idx)
                    current_idx += 1
                else:
                    # Tokenize and add sentence
                    sent_ids = self.tokenizer.encode(sent, add_special_tokens=False)
                    sequence.extend(sent_ids)
                    enc_masks.extend(range(current_idx, current_idx + len(sent_ids)))
                    current_idx += len(sent_ids)

            sequence.append(self.tokenizer.sep_token_id)

            # Truncate if needed
            if len(sequence) > self.max_length:
                sequence = sequence[: self.max_length]
                # Adjust masks for truncation
                enc_masks = [idx for idx in enc_masks if idx < self.max_length]
                pred_masks = [idx for idx in pred_masks if idx < self.max_length]

            # Add padding
            attention_mask = [1] * len(sequence)
            padding_length = self.max_length - len(sequence)
            if padding_length > 0:
                sequence.extend([self.tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)

            # Add to batch
            input_ids_batch.append(sequence)
            attention_mask_batch.append(attention_mask)
            enc_masks_batch.append(enc_masks)
            pred_masks_batch.append(pred_masks)

        return MaskOutput(
            input_ids=torch.tensor(input_ids_batch),
            attention_mask=torch.tensor(attention_mask_batch),
            enc_masks=enc_masks_batch,
            pred_masks=pred_masks_batch,
            original_texts=batch,
        )
