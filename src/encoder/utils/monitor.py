import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from transformers import PreTrainedTokenizer

import wandb


@dataclass
class MonitoringExample:
    """Holds information for a single monitoring example."""

    context_text: str
    target_text: str
    predicted_embedding: torch.Tensor
    target_embedding: torch.Tensor
    similarity_score: float


class TrainingMonitor:
    """Monitors and logs training progress for next-sentence prediction."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        log_dir: Path = Path("logs/monitor_logs"),
        num_examples: int = 3,
        log_every_n_epochs: int = 1,
        log_to_wandb: bool = True,
    ):
        self.tokenizer = tokenizer
        self.log_dir = Path(log_dir)
        self.num_examples = num_examples
        self.log_every_n_epochs = log_every_n_epochs
        self.log_to_wandb = log_to_wandb
        self.console = Console()

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up loggers
        self._setup_loggers()

    def _setup_loggers(self):
        """Initialize different loggers for console and file output."""
        # Console logger
        self.console_logger = logging.getLogger("console")
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
        self.console_logger.setLevel(logging.INFO)

        # File logger for training examples
        self.file_logger = logging.getLogger("training_examples")
        file_handler = logging.FileHandler(self.log_dir / "training_examples.log")
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        self.file_logger.setLevel(logging.INFO)

    def log_training_examples(
        self,
        epoch: int,
        batch_texts: list[str],
        target_texts: list[str],
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
    ) -> None:
        """Log training examples showing next-sentence prediction performance.

        Args:
            epoch: Current training epoch
            batch_texts: List of context texts
            target_texts: List of target (next sentence) texts
            predicted_features: Predicted embeddings from the model
            target_features: True target embeddings
            encoder: The encoder model (for additional analysis if needed)
            predictor: The predictor model (for additional analysis if needed)
        """
        if epoch % self.log_every_n_epochs != 0:
            return

        examples = []
        for idx in range(min(self.num_examples, len(batch_texts))):
            # Calculate cosine similarity for this example
            similarity = F.cosine_similarity(
                predicted_features[idx : idx + 1], target_features[idx : idx + 1], dim=1
            ).item()

            example = MonitoringExample(
                context_text=batch_texts[idx],
                target_text=target_texts[idx],
                predicted_embedding=predicted_features[idx].cpu(),
                target_embedding=target_features[idx].cpu(),
                similarity_score=similarity,
            )
            examples.append(example)

        self._display_examples(epoch, examples)
        if self.log_to_wandb:
            self._log_to_wandb(epoch, examples)

    def _display_examples(self, epoch: int, examples: list[MonitoringExample]) -> None:
        """Display training examples in a formatted table."""
        self.file_logger.info(f"\n=== Training Examples (Epoch {epoch}) ===")

        for i, example in enumerate(examples, 1):
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )
            table.add_column("Type", style="cyan", width=20)
            table.add_column("Content", style="green")

            # Format context text for display
            escaped_context = escape(example.context_text)
            context_chunks = [
                escaped_context[i : i + 100]
                for i in range(0, len(escaped_context), 100)
            ]
            table.add_row("Context", "\n".join(context_chunks))

            # Format target text
            escaped_target = escape(example.target_text)
            target_chunks = [
                escaped_target[i : i + 100] for i in range(0, len(escaped_target), 100)
            ]
            table.add_row("Target (Next Sentence)", "\n".join(target_chunks))

            # Add similarity score
            table.add_row("Cosine Similarity", f"{example.similarity_score:.4f}")

            # Add embedding statistics
            pred_norm = example.predicted_embedding.norm().item()
            target_norm = example.target_embedding.norm().item()
            table.add_row(
                "Embedding Stats",
                f"Predicted norm: {pred_norm:.4f}\nTarget norm: {target_norm:.4f}",
            )

            # Print the table to console and log to file
            panel = Panel(table, title=f"Example {i}", border_style="blue")
            self.file_logger.info(panel)
            self.file_logger.info(f"\nExample {i}:")
            self.file_logger.info(f"Context: {example.context_text}")
            self.file_logger.info(f"Target: {example.target_text}")
            self.file_logger.info(f"Similarity: {example.similarity_score:.4f}")
            self.file_logger.info("-" * 80)

    def _log_to_wandb(self, epoch: int, examples: list[MonitoringExample]) -> None:
        """Log examples to Weights & Biases."""
        for i, example in enumerate(examples):
            wandb.log(
                {
                    f"examples/context_{i}": example.context_text,
                    f"examples/target_{i}": example.target_text,
                    f"examples/similarity_{i}": example.similarity_score,
                    f"examples/pred_norm_{i}": example.predicted_embedding.norm().item(),
                    f"examples/target_norm_{i}": example.target_embedding.norm().item(),
                    "epoch": epoch,
                }
            )

            # Create similarity histogram for this batch
            if i == 0:  # Only do this once per batch
                wandb.log(
                    {
                        "similarity_distribution": wandb.Histogram(
                            [e.similarity_score for e in examples]
                        ),
                        "epoch": epoch,
                    }
                )
