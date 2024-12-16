import math
import os
import time

import torch
import torch.nn.functional as F
from devtools import debug
from torch.utils.data import DataLoader

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
        train_file=train_file, min_length=cfg["data"].get("min_length", 10)
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

        for itr, (input_dict, enc_masks_batch, pred_masks_batch) in enumerate(
            dataloader
        ):
            # input_dict: {"input_ids": ..., "attention_mask": ...}
            # enc_masks_batch: list of lists of tensors (context sentences' token idx)
            # pred_masks_batch: list of lists of tensors (predicted sentences' token idx)

            input_ids = input_dict["input_ids"].to(device)
            attention_mask = input_dict["attention_mask"].to(device)


            # Step 1: Forward target encoder to get target features
            with torch.no_grad():
                target_features = encoder(input_ids, attention_mask)
                target_features = normalize_features(target_features)
                # Extract and project features for masked segments
                target_feats = extract_features_for_masks(
                    target_features, pred_masks_batch
                )
                target_feats = predictor.project_targets(target_feats)

            # Step 2: Forward context encoder and predictor to get predictions
            context_features = encoder(input_ids, attention_mask)
            predicted_feats = predictor(
                context_features, enc_masks_batch, pred_masks_batch
            )

            # They should now have the same first dimension
            assert (
                target_feats.size(0) == predicted_feats.size(0)
            ), f"Mismatch in number of predictions ({predicted_feats.size(0)}) vs targets ({target_feats.size(0)})"

            # Step 3: Compute loss in concept space
            loss = F.smooth_l1_loss(predicted_feats, target_feats)

            # Step 4: Backprop and optimization
            optimizer.zero_grad()
            if use_bfloat16 and scaler is not None:
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

            if (itr % log_freq == 0) or math.isnan(loss_val) or math.isinf(loss_val):
                elapsed = (time.time() - epoch_start) * 1000.0
                csv_logger.log(epoch + 1, itr, loss_val, lr, elapsed)
                print(
                    f"[Epoch {epoch+1}/{epochs}, Itr {itr}] loss: {loss_meter.avg:.4f}, lr: {lr:.2e}"
                )

            # Optional: break early if debugging
            # if itr > 10:
            #     break

        # End of epoch logging
        print(f"Epoch {epoch+1}/{epochs} finished, avg loss: {loss_meter.avg:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
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

    print("Training completed successfully.")


# Placeholder utility functions that we will implement later:
def normalize_features(features):
    # For example, L2 normalization over the feature dimension
    return F.normalize(features, p=2, dim=-1)

