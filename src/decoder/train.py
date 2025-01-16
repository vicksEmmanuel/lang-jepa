from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.logging import AverageMeter
from src.decoder.decoder_dataset import DecoderBatch
from src.decoder.models import ConceptDecoder
from src.decoder.utils.evaluation import ConceptMetrics, SampleGenerator
from src.encoder.models import TextTransformer


@dataclass
class DecoderTrainingConfig:
    """Configuration for decoder training."""

    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    grad_clip: float
    weight_decay: float
    eval_steps: int
    save_steps: int
    output_dir: str


class DecoderTrainer:
    def __init__(
        self,
        config: DecoderTrainingConfig,
        encoder: TextTransformer,
        decoder: ConceptDecoder,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move models to device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Setup optimizer
        self.optimizer = AdamW(
            self.decoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup metrics
        self.metrics = ConceptMetrics(self.decoder.tokenizer, self.device)
        self.sample_generator = SampleGenerator(
            self.encoder, self.decoder, self.decoder.tokenizer, self.device
        )

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _process_batch(self, batch: DecoderBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of data."""
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        return input_ids, attention_mask

    def train(self) -> None:
        """Main training loop."""
        best_loss = float("inf")
        global_step = 0
        loss_meter = AverageMeter()

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            self.decoder.train()

            for batch in tqdm(self.train_loader, desc="Training"):
                # Process batch
                input_ids, attention_mask = self._process_batch(batch)

                # Get concept embeddings from encoder
                with torch.no_grad():
                    concepts = self.encoder(input_ids, attention_mask)
                    if len(concepts.shape) > 2:
                        concepts = concepts.mean(dim=1)  # Average sequence dimension

                # Forward pass through decoder
                # input_ids[:, :-1] as input, input_ids[:, 1:] as targets
                logits = self.decoder(concepts, target_ids=input_ids)  # [B, L-1, V]

                # Prepare targets (excluding first token)
                targets = input_ids[:, 1:].reshape(-1)  # [B*(L-1)]

                # Reshape logits to match target shape
                logits = logits.reshape(-1, logits.size(-1))  # [B*(L-1), V]

                # Verify shapes match
                assert logits.size(0) == targets.size(0), (
                    f"Shape mismatch: logits {logits.shape}, targets {targets.shape}. "
                    f"Batch size: {input_ids.size(0)}, Sequence length: {input_ids.size(1)}"
                )

                # Compute loss
                loss = F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=self.decoder.config.pad_token_id,
                )

                # Update meter
                loss_meter.update(loss.item())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.decoder.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

                # Increment step
                global_step += 1

                # Evaluate if needed
                if global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"\nStep {global_step} - Eval loss: {eval_loss:.4f}")

                    # Save if best
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self.save_checkpoint(
                            self.output_dir / "best_decoder.pt",
                            global_step,
                            best_loss,
                        )

                    # Generate samples
                    if self.eval_loader is not None:
                        eval_batch = next(iter(self.eval_loader))
                        samples = self.sample_generator.generate_samples(
                            eval_batch.input_texts, num_samples=2
                        )
                        self._print_samples(samples)

                # Save checkpoint if needed
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(
                        self.output_dir / f"decoder_step_{global_step}.pt",
                        global_step,
                        loss_meter.avg,
                    )

                # Log progress
                if global_step % 10 == 0:
                    print(f"\nStep {global_step} - Loss: {loss_meter.avg:.4f}")

            # End of epoch
            print(f"Epoch {epoch + 1} finished. Avg loss: {loss_meter.avg:.4f}")
            loss_meter.reset()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the decoder."""
        if self.eval_loader is None:
            return float("inf")

        self.decoder.eval()
        total_loss = 0
        num_batches = 0

        for i, batch in enumerate(self.eval_loader):

            if i >= 10:
                break

            # Process batch
            input_ids, attention_mask = self._process_batch(batch)
            # Get concepts
            concepts = self.encoder(input_ids, attention_mask)

            if len(concepts.shape) > 2:
                concepts = concepts.mean(dim=1)  # Average sequence dimension

            # Generate
            logits = self.decoder(concepts, target_ids=input_ids)

            print(f"logits: {logits.shape}")

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                ignore_index=self.decoder.config.pad_token_id,
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.decoder.train()
        return avg_loss

    def save_checkpoint(self, path: Path, global_step: int, loss: float) -> None:
        """Save a checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": global_step,
                "model_state_dict": self.decoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "config": self.decoder.config,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path}")

    def _print_samples(self, samples: list) -> None:
        """Print generated samples."""
        print("\nGenerated Samples:")
        print("-" * 50)
        for i, sample in enumerate(samples, 1):
            print(f"Sample {i}:")
            print(f"Original : {sample['original']}")
            print(f"Generated: {sample['generated']}")
            print(f"BLEU     : {sample['bleu']:.4f}")
            print()
