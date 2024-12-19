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


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[DatasetOutput]) -> Batch:
        contexts = [item.context for item in batch]
        tokens = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return Batch(
            context_ids=tokens["input_ids"],
            padding_masks=tokens["attention_mask"],
            context_texts=contexts,
            target_texts=[item.target for item in batch],
        )
