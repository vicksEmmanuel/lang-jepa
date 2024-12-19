from dataclasses import dataclass

from torch import Tensor
from transformers import PreTrainedTokenizer

from src.common.datasets.fineweb_edu import DatasetOutput


@dataclass
class Batch:
    context_ids: Tensor  # Tokenized context sequences [batch_size, seq_len]
    padding_masks: Tensor  # Padding masks (1 for real tokens, 0 for padding)
    context_texts: list[str]  # Original context text before tokenization
    target_texts: list[str]  # Target sentences to predict


def create_collate_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    def collate_fn(batch: list[DatasetOutput]) -> Batch:
        contexts = [item.context for item in batch]
        tokens = tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return Batch(
            context_ids=tokens["input_ids"],
            padding_masks=tokens["attention_mask"],
            context_texts=contexts,
            target_texts=[item.target for item in batch],
        )

    return collate_fn
