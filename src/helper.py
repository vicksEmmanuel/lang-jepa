import logging
import os

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)


def init_optimizer(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    lr: float,
    weight_decay: float,
    warmup: int,
    total_epochs: int,
    steps_per_epoch: int,
    final_wd: float = 0.0,
    final_lr: float = 0.0,
    use_bfloat16: bool = False,
):
    """
    I initialize optimizer, schedulers, and scaler.
    """
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0.0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    total_steps = steps_per_epoch * total_epochs

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=int(warmup * steps_per_epoch),
        start_lr=lr * 0.1,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=total_steps,
    )

    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer, ref_wd=weight_decay, final_wd=final_wd, T_max=total_steps
    )

    scaler = GradScaler() if use_bfloat16 and torch.cuda.is_available() else None

    return optimizer, scaler, scheduler, wd_scheduler


def load_checkpoint(
    checkpoint_path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> int:
    """
    I load a checkpoint if available.
    """
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch.")
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        optimizer.load_state_dict(checkpoint["opt"])

        if (
            scaler is not None
            and "scaler" in checkpoint
            and checkpoint["scaler"] is not None
        ):
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {start_epoch}).")
        return start_epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        return 0


def save_checkpoint(
    checkpoint_path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss: float,
):
    """
    I save a checkpoint.
    """
    state = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "loss": loss,
    }

    try:
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at {checkpoint_path}: {e}")
