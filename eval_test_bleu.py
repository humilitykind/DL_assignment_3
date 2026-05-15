#!/usr/bin/env python3
"""
Quick Test BLEU Evaluation
Run after training to get test set BLEU score
"""

import torch
from torch.utils.data import DataLoader
import sys

from dataset import Multi30kDataset
from model import Transformer
from train import evaluate_bleu, collate_fn, load_checkpoint


def eval_test_bleu(checkpoint_path="checkpoint.pt"):
    """Evaluate test BLEU from a saved checkpoint."""

    print("=" * 60)
    print("TEST BLEU EVALUATION")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    train_ds = Multi30kDataset(split="train", min_freq=2)
    test_ds = Multi30kDataset(
        split="test",
        min_freq=2,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

    print(f"✓ Dataset loaded: {len(test_ds)} test samples")
    print(f"  Source vocab: {len(train_ds.src_vocab['itos'])}")
    print(f"  Target vocab: {len(train_ds.tgt_vocab['itos'])}")

    # Create test loader
    pad_idx = train_ds.pad_idx
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )

    # Device selection
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        model = Transformer(
            src_vocab_size=len(train_ds.src_vocab["itos"]),
            tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
            d_model=512,
            N=6,
            num_heads=8,
            d_ff=2048,
            dropout=0.1,
        ).to(device)

        epoch = load_checkpoint(checkpoint_path, model)
        print(f"✓ Loaded checkpoint from epoch {epoch}")
    except FileNotFoundError:
        print(f"❌ Error: {checkpoint_path} not found!")
        print("   Please train the model first: python train.py")
        sys.exit(1)

    # Evaluate
    print("\nEvaluating on test set (this may take a few minutes)...")
    model.eval()
    bleu = evaluate_bleu(
        model,
        test_loader,
        train_ds.tgt_vocab,
        device=device,
        max_len=100,
    )

    print("\n" + "=" * 60)
    print(f"TEST BLEU SCORE: {bleu:.2f}")
    print("=" * 60)

    return bleu


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pt",
        help="Path to checkpoint file (default: checkpoint.pt)"
    )
    args = parser.parse_args()

    bleu = eval_test_bleu(args.checkpoint)
