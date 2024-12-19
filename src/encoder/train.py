import os
import time

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import wandb
from src.common.config import LANGJEPAConfig
from src.common.datasets.fineweb_edu import TextDataset
from src.common.logging import AverageMeter, CSVLogger
from src.encoder.collator import Batch, create_collate_fn
from src.encoder.models import TextPredictor, TextTransformer
from src.encoder.utils.helper import init_optimizer, load_checkpoint, save_checkpoint


def train(config: LANGJEPAConfig) -> None:
    """Main training function with type-safe config."""

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
    )

    collate_fn = create_collate_fn(config.data.tokenizer, config.model.max_length)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
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
            padding_masks = batch.padding_masks.to(device)

            # Get target embeddings
            with torch.no_grad():
                # Tokenize targets just for getting embeddings
                target_tokens = config.data.tokenizer(
                    batch.target_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                target_features = encoder(
                    target_tokens["input_ids"], target_tokens["attention_mask"]
                )
                target_features = F.normalize(target_features.mean(dim=1), p=2, dim=-1)
                target_features = predictor.project_targets(target_features)

            # Get context embeddings and predict
            context_features = encoder(context_ids, padding_masks)
            predicted_features = predictor(context_features, padding_masks)

            # Compute loss
            loss = F.smooth_l1_loss(predicted_features, target_features)

            # Optimize
            optimizer.zero_grad()
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

                wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/learning_rate": lr,
                        "train/iteration": itr + epoch * len(dataloader),
                        "stats/target_features_mean": target_features.mean().item(),
                        "stats/target_features_std": target_features.std().item(),
                        "stats/predicted_features_mean": predicted_features.mean().item(),
                        "stats/predicted_features_std": predicted_features.std().item(),
                    }
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
