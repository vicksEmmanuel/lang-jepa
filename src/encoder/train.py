import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import wandb
from src.common.config import LANGJEPAConfig
from src.common.datasets.fineweb_edu import TextDataset, worker_init_fn
from src.common.logging import AverageMeter, CSVLogger
from src.encoder.collator import Batch, Collator
from src.encoder.models import TextPredictor, TextTransformer
from src.encoder.utils.helper import init_optimizer, load_checkpoint, save_checkpoint
from src.encoder.utils.monitor import TrainingMonitor


def train(config: LANGJEPAConfig) -> None:
    """Main training function for LANG-JEPA next-sentence prediction."""

    # Initialize wandb
    load_dotenv()
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project="lang-jepa",
        config=config.model_dump(),
        name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    os.makedirs(config.logging.log_dir, exist_ok=True)
    log_file = os.path.join(config.logging.log_dir, "training.csv")
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.6f", "lr"),
        ("%.2f", "time(ms)"),
    )

    # Initialize dataset and dataloader
    dataset = TextDataset(
        train_file=config.data.train_file,
        limit=config.data.limit,
        min_length=config.data.min_length,
        min_sentences=config.data.min_sentences,
    )

    collator = Collator(
        tokenizer=config.data.tokenizer, max_length=config.model.max_length
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
    )

    # Initialize models
    encoder = TextTransformer(config=config).to(device)
    predictor = TextPredictor(
        input_dim=config.model.embed_dim,
        pred_dim=config.model.pred_dim,
    ).to(device)

    # Initialize optimizer and schedulers
    optimizer, scaler, scheduler, wd_scheduler = init_optimizer(
        encoder=encoder,
        predictor=predictor,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        warmup=config.optimization.warmup,
        total_epochs=config.optimization.epochs,
        steps_per_epoch=len(dataloader),
        use_bfloat16=config.meta.use_bfloat16,
    )

    # Initialize training monitor
    monitor = TrainingMonitor(
        tokenizer=config.data.tokenizer,
        log_dir=Path(config.logging.log_dir),
        log_to_wandb=True,
    )

    # Load checkpoint if specified
    start_epoch = 0
    if config.meta.load_checkpoint:
        start_epoch = load_checkpoint(
            checkpoint_path=config.meta.checkpoint_path,
            encoder=encoder,
            predictor=predictor,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    # Training loop
    loss_meter = AverageMeter()
    encoder.train()
    predictor.train()

    for epoch in range(start_epoch, config.optimization.epochs):
        epoch_start = time.time()
        loss_meter.reset()

        for itr, batch in enumerate(dataloader):
            batch: Batch
            # Move batch to device
            context_ids = batch.context_ids.to(device)
            context_mask = batch.padding_masks.to(device)

            # Process target sentences
            target_tokens = config.data.tokenizer(
                batch.target_texts,
                padding=True,
                truncation=True,
                max_length=config.model.max_length,
                return_tensors="pt",
            ).to(device)

            # Get embeddings for both context and target
            with torch.cuda.amp.autocast(enabled=config.meta.use_bfloat16):
                # Get target embeddings
                with torch.no_grad():
                    target_features = encoder(
                        target_tokens.input_ids,
                        target_tokens.attention_mask,
                    )
                    # Average token embeddings for sentence representation
                    target_features = target_features.mean(dim=1)
                    target_features = predictor.project_targets(target_features)
                    target_features = F.normalize(target_features, p=2, dim=-1)

                # Get context embeddings and predict
                context_features = encoder(context_ids, context_mask)
                predicted_features = predictor(context_features, context_mask)
                predicted_features = F.normalize(predicted_features, p=2, dim=-1)

                # Compute loss using cosine similarity
                loss = (
                    1 - F.cosine_similarity(predicted_features, target_features).mean()
                )

            # Optimize
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Update schedulers
            lr = scheduler.step()
            wd_scheduler.step()

            # Logging
            loss_val = loss.item()
            loss_meter.update(loss_val)

            if itr % config.logging.log_freq == 0:
                elapsed = (time.time() - epoch_start) * 1000.0
                csv_logger.log(epoch + 1, itr, loss_val, lr, elapsed)
                print(
                    f"[Epoch {epoch+1}/{config.optimization.epochs}, Itr {itr}] "
                    f"loss: {loss_meter.avg:.4f}, lr: {lr:.2e}"
                )

                # Log metrics
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/learning_rate": lr,
                        "train/iteration": itr + epoch * len(dataloader),
                        "stats/target_features_norm": target_features.norm(dim=1)
                        .mean()
                        .item(),
                        "stats/predicted_features_norm": predicted_features.norm(dim=1)
                        .mean()
                        .item(),
                        "stats/cosine_similarity": F.cosine_similarity(
                            predicted_features, target_features
                        )
                        .mean()
                        .item(),
                    }
                )

                # Monitor training examples
                monitor.log_training_examples(
                    epoch=epoch,
                    batch_texts=batch.context_texts,
                    target_texts=batch.target_texts,
                    predicted_features=predicted_features.detach(),
                    target_features=target_features.detach(),
                    encoder=encoder,
                    predictor=predictor,
                )
                monitor.log_validation_metrics(
                    epoch=epoch,
                    pred_embeddings=predicted_features,
                    target_embeddings=target_features,
                )

        # End of epoch
        if (epoch + 1) % config.logging.checkpoint_freq == 0:
            ckpt_path = os.path.join(
                config.logging.log_dir, f"checkpoint-epoch{epoch+1}.pth"
            )
            save_checkpoint(
                ckpt_path,
                encoder,
                predictor,
                optimizer,
                scaler,
                epoch + 1,
                loss_meter.avg,
            )

        wandb.log(
            {
                "epoch/loss": loss_meter.avg,
                "epoch/time": time.time() - epoch_start,
                "epoch/number": epoch + 1,
            }
        )

    print("Training completed successfully.")
    wandb.finish()
