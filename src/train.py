import math
import os
import time

import torch
import torch.nn.functional as F
from devtools import debug
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import wandb
from src.datasets.fineweb_edu import TextDataset
from src.helper import (
    init_model,
    init_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from src.masks.lang_mask_collator import LangMaskCollator
from src.models.text_transformer import extract_features_for_masks
from src.utils.logging import AverageMeter, CSVLogger

load_dotenv()

wandb.login(key=os.environ["WANDB_API_KEY"])

def train(cfg):
    """
    Main training function for LANG-JEPA.

    Args:
        cfg (dict): Configuration dictionary with required keys:
            - data: { 'train_file': str, 'batch_size': int, 'num_workers': int, ... }
            - mask: { 'mask_ratio': float, ... }
            - model: { 'model_name': str, 'max_length': int, ... }
            - optimization: { 'epochs': int, 'lr': float, 'warmup': int, 'weight_decay': float, ... }
            - logging: { 'log_dir': str, 'log_freq': int, 'checkpoint_freq': int, ... }
            - meta: { 'use_bfloat16': bool, 'load_checkpoint': bool, 'checkpoint_path': str, ... }
    """
    # Initialize wandb at the start, after device setup
    wandb.init(
        project="lang-jepa",
        config=cfg,
        name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Setup logging
    # --------------------
    log_dir = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.csv")
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.2f", "lr"),
        ("%.2f", "time(ms)"),
    )

    # --------------------
    # Load dataset and create dataloader
    # --------------------
    train_file = cfg["data"]["train_file"]
    dataset = TextDataset(
        train_file=train_file,
        min_length=cfg["data"].get("min_length", 10),
        limit=cfg["data"].get("limit", None),
    )
    # Collator handles sentence splitting and masking
    collator = LangMaskCollator(
        tokenizer=cfg["data"][
            "tokenizer"
        ],  # assuming tokenizer object is passed in cfg
        mask_ratio=cfg["mask"]["mask_ratio"],
        max_length=cfg["model"]["max_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collator,
    )

    # --------------------
    # Initialize model, optimizer, schedulers
    # --------------------
    use_bfloat16 = cfg["meta"].get("use_bfloat16", False)
    debug(cfg["model"]["model_name"])
    encoder, predictor = init_model(
        max_length=cfg["model"]["max_length"],
        pred_dim=cfg["model"].get("pred_dim", 384),  # dimension for predictor
        device=device,
    )

    optimizer, scaler, scheduler, wd_scheduler = init_optimizer(
        encoder=encoder,
        predictor=predictor,
        lr=cfg["optimization"]["lr"],
        weight_decay=cfg["optimization"]["weight_decay"],
        warmup=cfg["optimization"]["warmup"],
        total_epochs=cfg["optimization"]["epochs"],
        steps_per_epoch=len(dataloader),
        use_bfloat16=use_bfloat16,
    )

    start_epoch = 0
    if cfg["meta"].get("load_checkpoint", False):
        checkpoint_path = cfg["meta"]["checkpoint_path"]
        start_epoch = load_checkpoint(
            checkpoint_path=checkpoint_path,
            encoder=encoder,
            predictor=predictor,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    # --------------------
    # Training loop
    # --------------------
    epochs = cfg["optimization"]["epochs"]
    log_freq = cfg["logging"].get("log_freq", 50)
    checkpoint_freq = cfg["logging"].get("checkpoint_freq", 1)

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
            print(f"Saving checkpoint to {log_dir}")
            ckpt_path = os.path.join(log_dir, f"checkpoint-epoch{epoch+1}.pth")
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
            wandb.save(ckpt_path)

    print("Training completed successfully.")
    wandb.finish()

    print("Training completed successfully.")


# Placeholder utility functions that we will implement later:
def normalize_features(features):
    # For example, L2 normalization over the feature dimension
    return F.normalize(features, p=2, dim=-1)
