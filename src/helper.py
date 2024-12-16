import logging
import os

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from src.models.predictor_head import build_predictor
from src.models.text_transformer import build_text_encoder
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)

def init_model(model_name: str, max_length: int, pred_dim: int, device: torch.device):
    """
    Initialize the encoder and predictor models.

    Args:
        model_name (str): Model identifier (e.g., "text-base").
        max_length (int): Maximum sequence length for the encoder input.
        pred_dim (int): Dimension of predictor output embeddings.
        device (torch.device): Device on which to load models.

    Returns:
        (encoder: nn.Module, predictor: nn.Module)
    """
    # Build encoder
    encoder, embed_dim = build_text_encoder(model_name=model_name, max_length=max_length, device=device)
    # Build predictor
    predictor = build_predictor(embed_dim=embed_dim, pred_dim=pred_dim, device=device)

    encoder.to(device)
    predictor.to(device)

    return encoder, predictor


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
    use_bfloat16: bool = False
):
    """
    Initialize optimizer, schedulers, and scaler.

    Args:
        encoder, predictor (nn.Module): The initialized models.
        lr (float): Base learning rate.
        weight_decay (float): Initial weight decay.
        warmup (int): Number of warmup epochs.
        total_epochs (int): Total training epochs.
        steps_per_epoch (int): Number of iterations per epoch.
        final_wd (float): Final weight decay after cosine schedule. Default: 0.0
        final_lr (float): Final learning rate after cosine schedule. Default: 0.0
        use_bfloat16 (bool): Whether to use bfloat16 mixed-precision.

    Returns:
        optimizer (torch.optim.Optimizer), scaler (GradScaler or None),
        scheduler (WarmupCosineSchedule), wd_scheduler (CosineWDSchedule)
    """
    # Create parameter groups
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0.0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # Total steps including warmup and decay
    total_steps = steps_per_epoch * total_epochs

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=int(warmup*steps_per_epoch),
        start_lr=lr*0.1,   # Can adjust start_lr if needed
        ref_lr=lr,
        final_lr=final_lr,
        T_max=total_steps
    )

    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer,
        ref_wd=weight_decay,
        final_wd=final_wd,
        T_max=total_steps
    )

    scaler = GradScaler() if use_bfloat16 and torch.cuda.is_available() else None

    return optimizer, scaler, scheduler, wd_scheduler


def load_checkpoint(
    checkpoint_path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> int:
    """
    Load checkpoint if available.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        encoder, predictor: model instances.
        optimizer: optimizer instance.
        scaler: GradScaler instance (can be None).
        device: torch.device

    Returns:
        start_epoch (int): Epoch to resume training from.
    """
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch.")
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        predictor.load_state_dict(checkpoint['predictor'])
        optimizer.load_state_dict(checkpoint['opt'])

        if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])

        start_epoch = checkpoint['epoch']
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
    loss: float
):
    """
    Save checkpoint.

    Args:
        checkpoint_path (str): Path to save the checkpoint file.
        encoder, predictor: models.
        optimizer: optimizer instance.
        scaler: GradScaler instance (can be None).
        epoch (int): Current epoch number.
        loss (float): Last epoch's average loss.
    """
    state = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'opt': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'loss': loss
    }

    try:
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at {checkpoint_path}: {e}")
