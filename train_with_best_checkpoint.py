"""
Improved training with best model checkpointing
- Saves best model based on validation loss
- Shows detailed logging
- Validates training is working
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from train import (
    LabelSmoothingLoss, run_epoch, evaluate_bleu, collate_fn,
    save_checkpoint, load_checkpoint
)


def train_with_best_checkpoint():
    """Train with best model selection."""

    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 1000,
        "batch_size": 64,
        "num_epochs": 15,  # Increased from 10
        "lr": 1.0,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    wandb.init(
        entity="arshit1-mankodi-iit-madras",
        project="DL_3",
        name="train_best_checkpoint",
        config=config,
    )
    cfg = wandb.config

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    train_ds = Multi30kDataset(split="train", min_freq=cfg.min_freq)
    val_ds = Multi30kDataset(
        split="validation",
        min_freq=cfg.min_freq,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )
    test_ds = Multi30kDataset(
        split="test",
        min_freq=cfg.min_freq,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

    print(f"✓ Train samples: {len(train_ds)}")
    print(f"✓ Val samples: {len(val_ds)}")
    print(f"✓ Test samples: {len(test_ds)}")
    print(f"✓ Source vocab: {len(train_ds.src_vocab['itos'])}")
    print(f"✓ Target vocab: {len(train_ds.tgt_vocab['itos'])}")
    print(f"✓ Pad index: {train_ds.pad_idx}")

    pad_idx = train_ds.pad_idx

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✓ Device: {device}\n")

    # Create model
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        tokenizer_de=train_ds._tokenize_de,
        tokenizer_en=train_ds._tokenize_en,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}\n")

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    # Loss function
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab["itos"]),
        pad_idx=pad_idx,
        smoothing=cfg.label_smoothing,
    )

    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        # Training
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            is_train=True,
            device=device,
        )

        # Validation
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            is_train=False,
            device=device,
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to W&B
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch": epoch,
        })

        # Print progress
        print(f"Epoch {epoch:2d} | "
              f"train_loss: {train_loss:7.4f} | "
              f"val_loss: {val_loss:7.4f} | "
              f"lr: {current_lr:.2e}", end="")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                "checkpoint_best.pt"
            )
            print(" ✓ BEST")
        else:
            print()

    print("\n" + "=" * 70)
    print(f"Best model saved at epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    print("=" * 70 + "\n")

    # Evaluate with best model
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load best checkpoint
    model_best = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    load_checkpoint("checkpoint_best.pt", model_best)
    print("✓ Loaded best checkpoint\n")

    # Test BLEU
    print("Computing test BLEU (this may take a few minutes)...")
    bleu_best = evaluate_bleu(
        model_best,
        test_loader,
        train_ds.tgt_vocab,
        device=device,
    )

    # Also compute for final model
    bleu_final = evaluate_bleu(
        model,
        test_loader,
        train_ds.tgt_vocab,
        device=device,
    )

    print(f"\nTest BLEU (best model):  {bleu_best:.2f}")
    print(f"Test BLEU (final model): {bleu_final:.2f}")

    wandb.log({
        "test_bleu_best": bleu_best,
        "test_bleu_final": bleu_final,
        "best_epoch": best_epoch,
    })

    wandb.finish()

    return bleu_best


if __name__ == "__main__":
    train_with_best_checkpoint()
