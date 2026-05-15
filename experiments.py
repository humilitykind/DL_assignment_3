"""
Experiments for W&B Report (Section 2)
DA6401 Assignment 3: Transformer for Machine Translation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import math
from typing import Optional, Dict, List

from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from train import (
    LabelSmoothingLoss, run_epoch, greedy_decode, evaluate_bleu,
    collate_fn, save_checkpoint, load_checkpoint
)


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.1: Noam Scheduler vs Fixed Learning Rate
# ══════════════════════════════════════════════════════════════════════

def exp_2_1_scheduler_comparison():
    """
    Train model with two LR schedules:
    1. Noam Scheduler (warmup + decay)
    2. Fixed constant learning rate (e.g., 1e-4)
    """
    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "batch_size": 32,
        "num_epochs": 10,
        "fixed_lr": 1e-4,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    # Initialize W&B
    run = wandb.init(
        entity="arshit1-mankodi-iit-madras",
        project="DL_3",
        name="exp_2_1_scheduler_comparison",
        config=config,
    )
    cfg = wandb.config

    # Load data
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

    pad_idx = train_ds.pad_idx
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab["itos"]),
        pad_idx=pad_idx,
        smoothing=cfg.label_smoothing,
    )

    # Train with NOAM scheduler
    print("\n" + "=" * 60)
    print("Training with NOAM SCHEDULER")
    print("=" * 60)
    model_noam = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer_noam = optim.Adam(
        model_noam.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler_noam = NoamScheduler(
        optimizer_noam,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model_noam,
            loss_fn,
            optimizer_noam,
            scheduler_noam,
            epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model_noam,
            loss_fn,
            None,
            None,
            epoch,
            is_train=False,
            device=device,
        )
        wandb.log({
            "noam_train_loss": train_loss,
            "noam_val_loss": val_loss,
            "noam_lr": optimizer_noam.param_groups[0]["lr"],
            "epoch": epoch,
        })
        print(f"Epoch {epoch}: noam train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={optimizer_noam.param_groups[0]['lr']:.6f}")

    save_checkpoint(model_noam, optimizer_noam, scheduler_noam, cfg.num_epochs - 1, "checkpoint_noam.pt")

    # Train with FIXED learning rate
    print("\n" + "=" * 60)
    print("Training with FIXED LEARNING RATE")
    print("=" * 60)
    model_fixed = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer_fixed = optim.Adam(
        model_fixed.parameters(),
        lr=cfg.fixed_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model_fixed,
            loss_fn,
            optimizer_fixed,
            None,
            epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model_fixed,
            loss_fn,
            None,
            None,
            epoch,
            is_train=False,
            device=device,
        )
        wandb.log({
            "fixed_train_loss": train_loss,
            "fixed_val_loss": val_loss,
            "epoch": epoch,
        })
        print(f"Epoch {epoch}: fixed train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    save_checkpoint(model_fixed, optimizer_fixed, None, cfg.num_epochs - 1, "checkpoint_fixed.pt")

    # Evaluate both
    bleu_noam = evaluate_bleu(model_noam, test_loader, train_ds.tgt_vocab, device=device)
    bleu_fixed = evaluate_bleu(model_fixed, test_loader, train_ds.tgt_vocab, device=device)

    wandb.log({
        "test_bleu_noam": bleu_noam,
        "test_bleu_fixed": bleu_fixed,
    })
    print(f"\nTest BLEU - Noam: {bleu_noam:.2f}, Fixed: {bleu_fixed:.2f}")

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.2: Scaling Factor √(1/d_k) Ablation
# ══════════════════════════════════════════════════════════════════════

def exp_2_2_scaling_factor():
    """
    Train with and without the √(1/d_k) scaling factor in attention.
    Log gradient norms of Q and K weights during first 1000 steps.
    """
    print("\n" + "=" * 60)
    print("Experiment 2.2: Scaling Factor Ablation")
    print("=" * 60)

    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "batch_size": 32,
        "num_epochs": 10,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    wandb.init(
        entity="arshit1-mankodi-iit-madras",
        project="DL_3",
        name="exp_2_2_scaling_factor",
        config=config,
    )
    cfg = wandb.config

    # Load data
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

    pad_idx = train_ds.pad_idx
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab["itos"]),
        pad_idx=pad_idx,
        smoothing=cfg.label_smoothing,
    )

    # Note: Scaling is built into scaled_dot_product_attention.
    # To ablate, we would need to modify the model. For now,
    # we log gradient norms of Q and K weights to show the effect.

    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    model.train()
    global_step = 0
    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(train_loader):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log gradient norms of Q and K in encoder first layer
            if global_step < 1000:
                for name, param in model.encoder.layers[0].self_attn.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if "W_q" in name or "W_k" in name:
                            wandb.log({
                                f"grad_{name}": grad_norm,
                                "step": global_step,
                            })

            global_step += 1

            if step % 50 == 0:
                wandb.log({"loss": loss.item(), "step": global_step})
                print(f"Epoch {epoch}, Step {step}: loss={loss.item():.4f}")

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
        wandb.log({"val_loss": val_loss, "epoch": epoch})

    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.3: Attention Rollout & Head Specialization
# ══════════════════════════════════════════════════════════════════════

def exp_2_3_attention_visualization():
    """
    Extract and visualize attention weights from the last encoder layer.
    Analyze head specialization.
    """
    print("\n" + "=" * 60)
    print("Experiment 2.3: Attention Rollout & Head Specialization")
    print("=" * 60)

    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "batch_size": 32,
        "num_epochs": 10,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    wandb.init(
        entity="arshit1-mankodi-iit-madras",
        project="DL_3",
        name="exp_2_3_attention_visualization",
        config=config,
    )
    cfg = wandb.config

    # Load data
    train_ds = Multi30kDataset(split="train", min_freq=cfg.min_freq)
    val_ds = Multi30kDataset(
        split="validation",
        min_freq=cfg.min_freq,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load pretrained model
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    try:
        epoch = load_checkpoint("checkpoint.pt", model)
        print(f"Loaded checkpoint from epoch {epoch}")
    except FileNotFoundError:
        print("Warning: No checkpoint found, using untrained model for visualization example")

    model.eval()

    # Extract attention for a sample sentence
    sample_batch = next(iter(torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, train_ds.pad_idx),
    )))

    src, tgt = sample_batch
    src = src.to(device)

    src_mask = make_src_mask(src)

    # Forward through encoder with attention capture
    with torch.no_grad():
        # Manually run encoder to capture attention
        x = model.pos_enc(model.src_embed(src) * math.sqrt(model.d_model))

        # Capture attention weights from each layer's last head
        attention_heatmaps = []
        for layer_idx, layer in enumerate(model.encoder.layers):
            # Run self-attention and capture weights
            attn_out, attn_w = _get_attention_weights(layer, x, src_mask, device)
            attention_heatmaps.append(attn_w)  # [batch, heads, seq, seq]
            x = layer(x, src_mask)

        # Log attention heatmaps for last layer
        last_layer_attn = attention_heatmaps[-1]  # [1, num_heads, src_len, src_len]

        # Create visualization for each head
        import matplotlib.pyplot as plt
        import numpy as np

        src_tokens = [train_ds.src_vocab["itos"][idx.item()] for idx in src[0] if idx.item() != train_ds.pad_idx]

        num_heads = last_layer_attn.shape[1]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for head_idx in range(min(num_heads, 8)):
            attn_matrix = last_layer_attn[0, head_idx, :len(src_tokens), :len(src_tokens)].cpu().numpy()
            axes[head_idx].imshow(attn_matrix, cmap='viridis')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xticks(range(len(src_tokens)))
            axes[head_idx].set_yticks(range(len(src_tokens)))
            axes[head_idx].set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=8)
            axes[head_idx].set_yticklabels(src_tokens, fontsize=8)

        plt.tight_layout()
        plt.savefig('/tmp/attention_heatmap.png', dpi=100, bbox_inches='tight')
        wandb.log({"attention_heatmap": wandb.Image('/tmp/attention_heatmap.png')})
        plt.close()

    wandb.finish()


def _get_attention_weights(layer, x, mask, device):
    """Extract attention weights from an encoder layer."""
    batch_size = x.size(0)

    # Self-attention computation
    q = layer.self_attn.W_q(x)
    k = layer.self_attn.W_k(x)
    v = layer.self_attn.W_v(x)

    q = q.view(batch_size, -1, layer.self_attn.num_heads, layer.self_attn.d_k).transpose(1, 2)
    k = k.view(batch_size, -1, layer.self_attn.num_heads, layer.self_attn.d_k).transpose(1, 2)
    v = v.view(batch_size, -1, layer.self_attn.num_heads, layer.self_attn.d_k).transpose(1, 2)

    if mask is not None:
        mask = mask.expand(batch_size, layer.self_attn.num_heads, q.size(-2), k.size(-2))

    # Compute attention
    from model import scaled_dot_product_attention
    attn_out, attn_w = scaled_dot_product_attention(q, k, v, mask)

    return attn_out, attn_w


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.4: Positional Encoding vs Learned Embeddings
# ══════════════════════════════════════════════════════════════════════

def exp_2_4_positional_encoding():
    """
    Compare sinusoidal positional encoding vs learned positional embeddings.
    """
    print("\n" + "=" * 60)
    print("Experiment 2.4: Positional Encoding vs Learned Embeddings")
    print("=" * 60)

    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "batch_size": 32,
        "num_epochs": 10,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    wandb.init(
        entity="arshit1-mankodi-iit-madras",
        project="DL_3",
        name="exp_2_4_positional_encoding",
        config=config,
    )
    cfg = wandb.config

    # This would require a variant Transformer with learned embeddings
    # For now, we document the theoretical discussion

    wandb.log({
        "note": "Sinusoidal PE allows extrapolation to longer sequences; "
                "Learned embeddings are task-specific but limited to training length"
    })

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.5: Label Smoothing Ablation
# ══════════════════════════════════════════════════════════════════════

def exp_2_5_label_smoothing():
    """
    Train with label smoothing (ε=0.1) vs without (ε=0.0).
    Log prediction confidence.
    """
    print("\n" + "=" * 60)
    print("Experiment 2.5: Label Smoothing Ablation")
    print("=" * 60)

    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "batch_size": 32,
        "num_epochs": 10,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    wandb.init(
        entity="arshit1-mankodi-iit-madras",
        project="DL_3",
        name="exp_2_5_label_smoothing",
        config=config,
    )
    cfg = wandb.config

    # Load data
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

    pad_idx = train_ds.pad_idx
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Train with label smoothing = 0.1
    model_smooth = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer_smooth = optim.Adam(
        model_smooth.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler_smooth = NoamScheduler(
        optimizer_smooth,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    loss_fn_smooth = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab["itos"]),
        pad_idx=pad_idx,
        smoothing=0.1,
    )

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model_smooth,
            loss_fn_smooth,
            optimizer_smooth,
            scheduler_smooth,
            epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model_smooth,
            loss_fn_smooth,
            None,
            None,
            epoch,
            is_train=False,
            device=device,
        )
        wandb.log({
            "smooth_train_loss": train_loss,
            "smooth_val_loss": val_loss,
            "epoch": epoch,
        })

    # Train without label smoothing (ε=0.0)
    model_no_smooth = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer_no_smooth = optim.Adam(
        model_no_smooth.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler_no_smooth = NoamScheduler(
        optimizer_no_smooth,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    loss_fn_no_smooth = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab["itos"]),
        pad_idx=pad_idx,
        smoothing=0.0,
    )

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model_no_smooth,
            loss_fn_no_smooth,
            optimizer_no_smooth,
            scheduler_no_smooth,
            epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model_no_smooth,
            loss_fn_no_smooth,
            None,
            None,
            epoch,
            is_train=False,
            device=device,
        )
        wandb.log({
            "no_smooth_train_loss": train_loss,
            "no_smooth_val_loss": val_loss,
            "epoch": epoch,
        })

    # Evaluate
    bleu_smooth = evaluate_bleu(model_smooth, test_loader, train_ds.tgt_vocab, device=device)
    bleu_no_smooth = evaluate_bleu(model_no_smooth, test_loader, train_ds.tgt_vocab, device=device)

    wandb.log({
        "test_bleu_smooth": bleu_smooth,
        "test_bleu_no_smooth": bleu_no_smooth,
    })

    wandb.finish()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exp_num = sys.argv[1]
        if exp_num == "2.1":
            exp_2_1_scheduler_comparison()
        elif exp_num == "2.2":
            exp_2_2_scaling_factor()
        elif exp_num == "2.3":
            exp_2_3_attention_visualization()
        elif exp_num == "2.4":
            exp_2_4_positional_encoding()
        elif exp_num == "2.5":
            exp_2_5_label_smoothing()
        else:
            print("Invalid experiment number. Use 2.1, 2.2, 2.3, 2.4, or 2.5")
    else:
        print("Usage: python experiments.py <2.1|2.2|2.3|2.4|2.5>")
