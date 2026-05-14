import torch
from torch.utils.data import DataLoader

from dataset import Multi30kDataset
from train import evaluate_bleu, collate_fn, load_checkpoint
from model import Transformer


def main() -> None:
    train_ds = Multi30kDataset(split="train", min_freq=2)
    val_ds = Multi30kDataset(
        split="validation",
        min_freq=2,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

    loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, train_ds.pad_idx),
    )

    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i == 1000:
            break

    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=512,
        N=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
    )

    load_checkpoint("checkpoint.pt", model)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    bleu = evaluate_bleu(model, batches, train_ds.tgt_vocab, device=device, max_len=50)
    print("BLEU (2 batches)", bleu)


if __name__ == "__main__":
    main()
