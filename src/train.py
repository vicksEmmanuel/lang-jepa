import math
import os
import time

import torch
import torch.nn.functional as F
from devtools import debug
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import wandb
from src.config import LANGJEPAConfig
from src.datasets.fineweb_edu import TextDataset
from src.helper import (
    init_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from src.masks.lang_mask_collator import LangMaskCollator
from src.models.text_transformer import (
    TextPredictor,
    TextTransformer,
    extract_features_for_masks,
)
from src.utils.logging import AverageMeter, CSVLogger

load_dotenv()

wandb.login(key=os.environ["WANDB_API_KEY"])


def train(config: LANGJEPAConfig) -> None:
    """
    Main training function with type-safe config

    Args:
        config: Validated LANGJEPAConfig instance
    """
    # Initialize wandb
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

    # Initialize dataset
    dataset = TextDataset(
        train_file=config.data.train_file,
        limit=config.data.limit,
        min_length=config.data.min_length,
    )

    # Initialize collator
    collator = LangMaskCollator(
        tokenizer=config.data.tokenizer,
        mask_ratio=config.mask.mask_ratio,
        max_length=config.model.max_length,
    )

    # Initialize dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    # --------------------
    # Initialize model, optimizer, schedulers
    # --------------------
    use_bfloat16 = config.meta.use_bfloat16

    encoder = TextTransformer(config=config).to(device)
    predictor = TextPredictor(
        input_dim=config.model.embed_dim, pred_dim=config.model.pred_dim
    ).to(device)

    optimizer, scaler, scheduler, wd_scheduler = init_optimizer(
        encoder=encoder,
        predictor=predictor,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        warmup=config.optimization.warmup,
        total_epochs=config.optimization.epochs,
        steps_per_epoch=len(dataloader),
        use_bfloat16=use_bfloat16,
    )

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

    # --------------------
    # Training loop
    # --------------------
    epochs = config.optimization.epochs
    log_freq = config.logging.log_freq
    checkpoint_freq = config.logging.checkpoint_freq

    loss_meter = AverageMeter()
    encoder.train()
    predictor.train()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        loss_meter.reset()

        # Track epoch metrics
        epoch_metrics = {"epoch": epoch + 1, "epoch_loss": 0.0, "epoch_time": 0.0}

        for itr, (input_dict, enc_masks_batch, pred_masks_batch) in enumerate(
            dataloader
        ):
            iter_start = time.time()

            # Track iteration timing metrics
            timing_metrics = {}

            # Data loading time
            data_load_time = time.time() - iter_start
            timing_metrics["data_loading_time"] = data_load_time
            print(f"Data loading took: {data_load_time:.3f}s")

            # Move to device
            move_start = time.time()
            input_ids = input_dict["input_ids"].to(device)
            attention_mask = input_dict["attention_mask"].to(device)
            move_time = time.time() - move_start
            timing_metrics["device_transfer_time"] = move_time
            print(f"Moving to device took: {move_time:.3f}s")

            # Forward passes and timing tracking
            # Step 1: Forward target encoder
            target_start = time.time()
            with torch.no_grad():
                target_features = encoder(input_ids, attention_mask)
                target_features = normalize_features(target_features)
                target_feats = extract_features_for_masks(
                    target_features, pred_masks_batch
                )
                target_feats = predictor.project_targets(target_feats)
            target_time = time.time() - target_start
            timing_metrics["target_encoding_time"] = target_time
            print(f"Target encoding took: {target_time:.3f}s")

            # Step 2: Forward context encoder and predictor
            context_start = time.time()
            context_features = encoder(input_ids, attention_mask)
            context_time = time.time() - context_start
            timing_metrics["context_encoding_time"] = context_time
            print(f"Context encoding took: {context_time:.3f}s")

            pred_start = time.time()
            predicted_feats = predictor(
                context_features, enc_masks_batch, pred_masks_batch
            )
            pred_time = time.time() - pred_start
            timing_metrics["prediction_time"] = pred_time
            print(f"Prediction took: {pred_time:.3f}s")

            # Step 3: Compute loss
            loss_start = time.time()
            loss = F.smooth_l1_loss(predicted_feats, target_feats)
            loss_time = time.time() - loss_start
            timing_metrics["loss_computation_time"] = loss_time
            print(f"Loss computation took: {loss_time:.3f}s")

            # Step 4: Optimization
            opt_start = time.time()
            optimizer.zero_grad()
            if use_bfloat16 and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            opt_time = time.time() - opt_start
            timing_metrics["optimization_time"] = opt_time
            print(f"Optimization took: {opt_time:.3f}s")

            # Scheduler updates
            sched_start = time.time()
            lr = scheduler.step()
            wd_scheduler.step()
            sched_time = time.time() - sched_start
            timing_metrics["scheduler_time"] = sched_time
            print(f"Scheduler updates took: {sched_time:.3f}s")

            # Total iteration time
            iter_time = time.time() - iter_start
            timing_metrics["total_iteration_time"] = iter_time
            print(f"Total iteration took: {iter_time:.3f}s")
            print("-" * 50)

            # Log metrics to wandb
            loss_val = loss.item()
            loss_meter.update(loss_val)

            wandb_metrics = {
                "train/loss": loss_val,
                "train/learning_rate": lr,
                "train/iteration": itr + epoch * len(dataloader),
                **timing_metrics,
            }
            debug(wandb_metrics)

            # Add some tensor statistics
            wandb_metrics.update(
                {
                    "stats/target_features_mean": target_features.mean().item(),
                    "stats/target_features_std": target_features.std().item(),
                    "stats/predicted_features_mean": predicted_feats.mean().item(),
                    "stats/predicted_features_std": predicted_feats.std().item(),
                }
            )

            wandb.log(wandb_metrics)

            # Regular logging
            if (itr % log_freq == 0) or math.isnan(loss_val) or math.isinf(loss_val):
                elapsed = (time.time() - epoch_start) * 1000.0
                csv_logger.log(epoch + 1, itr, loss_val, lr, elapsed)
                print(
                    f"[Epoch {epoch+1}/{epochs}, Itr {itr}] loss: {loss_meter.avg:.4f}, lr: {lr:.2e}"
                )

        # End of epoch logging
        epoch_time = time.time() - epoch_start
        epoch_metrics["epoch_loss"] = loss_meter.avg
        epoch_metrics["epoch_time"] = epoch_time
        wandb.log(
            {
                "epoch/loss": loss_meter.avg,
                "epoch/time": epoch_time,
                "epoch/number": epoch + 1,
            }
        )

        print(f"Epoch {epoch+1}/{epochs} finished, avg loss: {loss_meter.avg:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            print(f"Saving checkpoint to {config.logging.log_dir}")
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
            print(f"Saved checkpoint to {ckpt_path}")
            # Log checkpoint to wandb
            # wandb.save(ckpt_path)

    print("Training completed successfully.")
    wandb.finish()

    print("Training completed successfully.")


# Placeholder utility functions that we will implement later:
def normalize_features(features):
    # For example, L2 normalization over the feature dimension
    return F.normalize(features, p=2, dim=-1)
