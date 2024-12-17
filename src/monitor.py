from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import PreTrainedTokenizer

import wandb
from src.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


@dataclass
class MonitoringExample:
    original_text: str
    masked_sentences: list[str]
    context_sentences: list[str]
    target_embeddings: torch.Tensor
    predicted_embeddings: torch.Tensor
    similarity_score: float


class TrainingMonitor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        log_to_wandb: bool = True,
        num_examples: int = 3,
        log_every_n_epochs: int = 1,
    ):
        self.tokenizer = tokenizer
        self.log_to_wandb = log_to_wandb
        self.num_examples = num_examples
        self.log_every_n_epochs = log_every_n_epochs
        self.console = Console()
        self.sentence_splitter = SentenceSplitter(SentenceSplitterConfig())

    def _map_token_to_sentence_indices(
        self,
        sentences: list[str],
        input_ids: torch.Tensor,
        mask_token_indices: list[int],
    ) -> set[int]:
        """Maps token indices to sentence indices."""
        # Tokenize each sentence individually to create mapping
        sentence_boundaries = []
        current_pos = 1  # Start after CLS token

        for sent in sentences:
            tokens = self.tokenizer.encode(sent, add_special_tokens=False)
            sentence_boundaries.append((current_pos, current_pos + len(tokens)))
            current_pos += len(tokens)

        # Find which sentences contain the mask tokens
        masked_sentence_indices = set()
        for mask_idx in mask_token_indices:
            for sent_idx, (start, end) in enumerate(sentence_boundaries):
                if start <= mask_idx <= end:
                    masked_sentence_indices.add(sent_idx)
                    break

        return masked_sentence_indices

    def log_training_examples(
        self,
        epoch: int,
        batch_texts: list[str],
        mask_output,
        encoder,
        predictor,
        device: torch.device,
    ) -> None:
        if epoch % self.log_every_n_epochs != 0:
            return

        examples = []

        for idx in range(min(self.num_examples, len(batch_texts))):
            # Get original text and split into sentences
            original_text = batch_texts[idx]
            sentences = self.sentence_splitter([original_text])[0]

            # Debug prints
            self.console.print(f"\n[yellow]Debug for example {idx}:")
            self.console.print(f"Pred masks: {mask_output.pred_masks[idx]}")
            self.console.print(f"Input IDs shape: {mask_output.input_ids[idx].shape}")

            # Map token indices to sentence indices
            mask_indices = self._map_token_to_sentence_indices(
                sentences, mask_output.input_ids[idx], mask_output.pred_masks[idx]
            )

            # Separate masked and context sentences
            masked_sentences = [
                sent for i, sent in enumerate(sentences) if i in mask_indices
            ]
            context_sentences = [
                sent for i, sent in enumerate(sentences) if i not in mask_indices
            ]

            # More debug prints
            self.console.print(f"Total sentences: {len(sentences)}")
            self.console.print(f"Masked indices: {mask_indices}")
            self.console.print(f"Number of masked sentences: {len(masked_sentences)}")
            self.console.print(f"Number of context sentences: {len(context_sentences)}")

            # Rest of the code remains the same...
            with torch.no_grad():
                target_features = encoder(
                    mask_output.input_ids[idx : idx + 1].to(device),
                    mask_output.attention_mask[idx : idx + 1].to(device),
                )
                target_feats = target_features[0, mask_output.pred_masks[idx]]
                target_embeddings = predictor.project_targets(target_feats)

                context_features = encoder(
                    mask_output.input_ids[idx : idx + 1].to(device),
                    mask_output.attention_mask[idx : idx + 1].to(device),
                )
                predicted_embeddings = predictor(
                    context_features,
                    [mask_output.enc_masks[idx]],
                    [mask_output.pred_masks[idx]],
                )

            similarity = (
                F.cosine_similarity(predicted_embeddings, target_embeddings, dim=1)
                .mean()
                .item()
            )

            example = MonitoringExample(
                original_text=original_text,
                masked_sentences=masked_sentences,
                context_sentences=context_sentences,
                target_embeddings=target_embeddings.cpu(),
                predicted_embeddings=predicted_embeddings.cpu(),
                similarity_score=similarity,
            )
            examples.append(example)

        self._display_examples(epoch, examples)
        if self.log_to_wandb:
            self._log_to_wandb(epoch, examples)

    def _display_examples(self, epoch: int, examples: list[MonitoringExample]) -> None:
        """Display examples in a rich formatted table with horizontal lines."""
        self.console.print(f"\n=== Training Examples (Epoch {epoch}) ===\n")

        for i, example in enumerate(examples, 1):
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )
            table.add_column("Type", style="cyan", width=20)
            table.add_column("Content", style="green")

            # Add original text in smaller chunks for readability
            chunks = [
                example.original_text[i : i + 100]
                for i in range(0, len(example.original_text), 100)
            ]
            table.add_row("Original Text", "\n".join(chunks))

            # Add masked sentences with index numbers
            masked_text = "\n".join(
                f"{idx + 1}. {sent}"
                for idx, sent in enumerate(example.masked_sentences)
            )
            table.add_row("Masked Sentences", masked_text or "No masked sentences")

            # Add context sentences with index numbers
            context_text = "\n".join(
                f"{idx + 1}. {sent}"
                for idx, sent in enumerate(example.context_sentences)
            )
            table.add_row("Context Sentences", context_text or "No context sentences")

            table.add_row("Embedding Similarity", f"{example.similarity_score:.4f}")

            self.console.print(Panel(table, title=f"Example {i}", border_style="blue"))
        self.console.print("\n")

    def _log_to_wandb(self, epoch: int, examples: list[MonitoringExample]) -> None:
        """Log examples to Weights & Biases."""
        for i, example in enumerate(examples):
            wandb.log(
                {
                    f"examples/original_text_{i}": example.original_text,
                    f"examples/masked_sentences_{i}": "\n".join(
                        example.masked_sentences
                    ),
                    f"examples/context_sentences_{i}": "\n".join(
                        example.context_sentences
                    ),
                    f"examples/embedding_similarity_{i}": example.similarity_score,
                    "epoch": epoch,
                }
            )
