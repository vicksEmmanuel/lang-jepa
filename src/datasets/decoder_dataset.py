from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from src.config import LANGJEPAConfig


@dataclass
class DecoderBatch:
    """Holds batched data for decoder training."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_texts: list[str]  # Original texts for evaluation


class DecoderDataset(Dataset):
    """Dataset for training the concept decoder."""

    def __init__(
        self, texts: list[str], tokenizer: PreTrainedTokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"text": self.texts[idx]}

    def collate_fn(self, batch: list[dict[str, str]]) -> DecoderBatch:
        """Collate batch of texts into tensors."""
        texts = [item["text"] for item in batch]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return DecoderBatch(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            input_texts=texts,
        )


def create_train_loader(
    config: LANGJEPAConfig, texts: list[str] | None = None
) -> DataLoader:
    """Create training data loader."""
    # If texts not provided, load from config's dataset
    if texts is None:
        from src.datasets.fineweb_edu import TextDataset

        dataset = TextDataset(
            train_file=config.data.train_file,
            limit=config.data.limit,
            min_length=config.data.min_length,
        )
        texts = dataset.samples

    # Create decoder dataset
    decoder_dataset = DecoderDataset(
        texts=texts, tokenizer=config.data.tokenizer, max_length=config.model.max_length
    )

    # Create loader
    return DataLoader(
        decoder_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=decoder_dataset.collate_fn,
        pin_memory=True,
    )


def create_eval_loader(
    config: "LANGJEPAConfig", texts: list[str] | None = None, eval_size: int = 1000
) -> DataLoader:
    """Create evaluation data loader."""
    if texts is None:
        # Load validation split if available, otherwise use subset of training
        try:
            from src.datasets.fineweb_edu import TextDataset

            eval_dataset = TextDataset(
                train_file=config.data.train_file,
                limit=eval_size,
                min_length=config.data.min_length,
                split="validation",  # Assuming this is added to TextDataset
            )
            texts = eval_dataset.samples
        except:
            # If no validation split, use subset of training data
            from src.datasets.fineweb_edu import TextDataset

            dataset = TextDataset(
                train_file=config.data.train_file,
                limit=eval_size,
                min_length=config.data.min_length,
            )
            texts = dataset.samples[:eval_size]

    # Create decoder dataset
    decoder_dataset = DecoderDataset(
        texts=texts, tokenizer=config.data.tokenizer, max_length=config.model.max_length
    )

    # Create loader without shuffling for consistent evaluation
    return DataLoader(
        decoder_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=decoder_dataset.collate_fn,
        pin_memory=True,
    )


# Utility function to split data for training and evaluation
def split_train_eval(
    texts: list[str], eval_ratio: float = 0.1, shuffle: bool = True, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split texts into training and evaluation sets."""
    if shuffle:
        import random

        random.seed(seed)
        texts = texts.copy()
        random.shuffle(texts)

    split_idx = int(len(texts) * (1 - eval_ratio))
    return texts[:split_idx], texts[split_idx:]
